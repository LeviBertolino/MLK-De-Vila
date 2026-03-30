"""
DPO Training com MLX usando modelo fp16 (não quantizado).

Modelos quantizados (Q4) não suportam backpropagation no MLX.
Este script usa o modelo em fp16 e pré-computa os log-probs de
referência pra caber na memória do M2 8GB.

Estratégia de memória:
  1. Carrega modelo fp16 (~3.4 GB)
  2. Pré-computa ref log-probs pra todos os pares → salva em disco
  3. Descarrega ref model (libera ~3.4 GB)
  4. Aplica LoRA e treina DPO (~4-5 GB total)

Uso:
    python scripts/04_dpo_train.py --iters 50
    python scripts/04_dpo_train.py --iters 50 --beta 0.1
"""

import json
import argparse
import gc
import random
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm import load
from mlx_lm.tuner.lora import LoRALinear

# ============================================================
# CONFIGURAÇÃO
# ============================================================

PROJECT_ROOT = Path(__file__).parent.parent
DPO_DATASET = PROJECT_ROOT / "data" / "preference" / "dpo_dataset.jsonl"
SFT_ADAPTER = PROJECT_ROOT / "adapters" / "sft"
DPO_ADAPTER_PATH = PROJECT_ROOT / "adapters" / "dpo"
REF_CACHE = PROJECT_ROOT / "data" / "ref_log_probs.json"

# Modelo fp16 (não quantizado!) — necessário pra backprop funcionar
MODEL_FP16 = "mlx-community/gemma-3-1b-it-bf16"
# Modelo SFT merged (base + SFT fundidos) — usado como ref e base pro DPO
MODEL_SFT_MERGED = str(PROJECT_ROOT / "models" / "sft_merged")

SYSTEM_MESSAGE = (
    "Você é o 'Mlk de Vila', um educador financeiro que fala a língua da periferia. "
    "Explica finanças de forma simples, direta e usando exemplos do dia a dia. "
    "Usa gírias como mano, parceiro, trampo, grana, padoca, bico. "
    "É motivador e honesto."
)


def load_dpo_data() -> list[dict]:
    entries = []
    with open(DPO_DATASET, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def truncate_text(text: str, max_chars: int = 500) -> str:
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_period = truncated.rfind(".")
    if last_period > max_chars // 2:
        return truncated[:last_period + 1]
    return truncated


def format_prompt(prompt: str) -> str:
    return f"{SYSTEM_MESSAGE}\n\nPergunta: {prompt}\n\nResposta:"


def tokenize_pair(tokenizer, prompt_text, response_text, max_len=256):
    full_text = prompt_text + " " + response_text
    tokens = tokenizer.encode(full_text)
    prompt_tokens = tokenizer.encode(prompt_text)
    if len(tokens) > max_len:
        tokens = tokens[:max_len]
    input_ids = mx.array(tokens[:-1])[None, :]
    target_ids = mx.array(tokens[1:])
    prompt_len = min(len(prompt_tokens) - 1, len(tokens) - 2)
    return input_ids, target_ids, prompt_len


def compute_response_log_probs(logits, target_ids, prompt_len):
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    token_log_probs = mx.take_along_axis(
        log_probs, target_ids[:, None], axis=-1
    ).squeeze(-1)
    if prompt_len < len(token_log_probs):
        return mx.mean(token_log_probs[prompt_len:])
    return mx.mean(token_log_probs)


# ============================================================
# FASE 1: Pré-computar ref log-probs
# ============================================================

def precompute_ref_log_probs(data, args):
    """Carrega modelo, computa ref log-probs, salva e descarrega."""
    if REF_CACHE.exists():
        print("📦 Ref log-probs já computados, carregando do cache...")
        with open(REF_CACHE, "r") as f:
            cached = json.load(f)
        print(f"   {len(cached)} pares carregados do cache")
        return cached

    print("📦 Carregando modelo de referência (SFT merged)...")
    ref_model, tokenizer = load(MODEL_SFT_MERGED)
    ref_model.freeze()
    print("   Modelo SFT merged carregado como referência ✅")
    print()

    print("🔢 Pré-computando ref log-probs para todos os pares...")
    ref_data = []

    for i, entry in enumerate(data):
        prompt_text = format_prompt(entry["prompt"])
        chosen_text = truncate_text(entry["chosen"])
        rejected_text = truncate_text(entry["rejected"])

        c_ids, c_tgt, c_plen = tokenize_pair(tokenizer, prompt_text, chosen_text)
        r_ids, r_tgt, r_plen = tokenize_pair(tokenizer, prompt_text, rejected_text)

        ref_c_logits = ref_model(c_ids)[0]
        ref_c_lp = compute_response_log_probs(ref_c_logits, c_tgt, c_plen)

        ref_r_logits = ref_model(r_ids)[0]
        ref_r_lp = compute_response_log_probs(ref_r_logits, r_tgt, r_plen)

        mx.eval(ref_c_lp, ref_r_lp)

        ref_data.append({
            "prompt": entry["prompt"],
            "chosen": entry["chosen"],
            "rejected": entry["rejected"],
            "ref_chosen_lp": ref_c_lp.item(),
            "ref_rejected_lp": ref_r_lp.item(),
        })

        if (i + 1) % 10 == 0:
            print(f"   [{i+1}/{len(data)}] computados")
            mx.clear_cache()

    # Salvar cache
    with open(REF_CACHE, "w", encoding="utf-8") as f:
        json.dump(ref_data, f, ensure_ascii=False)
    print(f"   ✅ Salvos em {REF_CACHE}")
    print()

    # Descarregar modelo de referência
    print("🗑️  Descarregando modelo de referência pra liberar memória...")
    del ref_model
    del tokenizer
    gc.collect()
    mx.clear_cache()
    print("   Memória liberada ✅")
    print()

    return ref_data


# ============================================================
# FASE 2: DPO Training
# ============================================================

def train_dpo(ref_data, args):
    """Treina DPO com modelo fp16 + LoRA do SFT (mesmas camadas!)."""

    print("📦 Carregando modelo policy (SFT merged) pra treinar...")
    model, tokenizer = load(MODEL_SFT_MERGED)
    print("   ✅ Modelo SFT merged carregado (SFT já embutido nos pesos)")

    # Aplicar LoRA FRESCOS (o SFT já está nos pesos base do merged)
    print("   Aplicando LoRA frescos pro DPO...")
    layers = model.model.layers
    total_layers = len(layers)
    num_lora_layers = 4
    start = max(0, total_layers - num_lora_layers)
    lora_count = 0

    for i in range(start, total_layers):
        # Attention projections
        attn = layers[i].self_attn
        for proj_name in ["q_proj", "v_proj", "o_proj"]:
            if hasattr(attn, proj_name):
                original = getattr(attn, proj_name)
                if isinstance(original, nn.Linear):
                    setattr(attn, proj_name, LoRALinear.from_base(original, r=4))
                    lora_count += 1
        # MLP projections
        mlp = layers[i].mlp
        for proj_name in ["gate_proj", "up_proj", "down_proj"]:
            if hasattr(mlp, proj_name):
                original = getattr(mlp, proj_name)
                if isinstance(original, nn.Linear):
                    setattr(mlp, proj_name, LoRALinear.from_base(original, r=4))
                    lora_count += 1

    # Freeze tudo, unfreeze só LoRA (agora não tem linear.weight problema)
    model.freeze()
    for _, module in model.named_modules():
        if isinstance(module, LoRALinear):
            module.unfreeze()
            module.linear.freeze()

    trainable = sum(arr.size for _, arr in tree_flatten(model.trainable_parameters()))
    print(f"   {lora_count} projeções LoRA | {trainable:,} parâmetros treináveis")

    peak_mem = mx.get_peak_memory() / 1e9
    print(f"   Peak mem após carregar: {peak_mem:.3f} GB")
    print()

    # Optimizer
    optimizer = optim.Adam(learning_rate=args.lr)
    DPO_ADAPTER_PATH.mkdir(parents=True, exist_ok=True)

    # Loss function — recebe o model como 1o arg pra nn.value_and_grad
    def loss_fn(model, c_ids, c_tgt, c_plen, r_ids, r_tgt, r_plen, ref_c_lp, ref_r_lp):
        pi_c_logits = model(c_ids)[0]
        pi_c_lp = compute_response_log_probs(pi_c_logits, c_tgt, c_plen)

        pi_r_logits = model(r_ids)[0]
        pi_r_lp = compute_response_log_probs(pi_r_logits, r_tgt, r_plen)

        logit = args.beta * ((pi_c_lp - pi_r_lp) - (ref_c_lp - ref_r_lp))
        loss = -mx.log(mx.sigmoid(logit) + 1e-8)
        return loss

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    # Training loop — epoch-based com shuffle
    print("🚀 Iniciando DPO training...")
    print()

    iteration = 0
    epoch = 0
    indices = list(range(len(ref_data)))
    running_loss = 0.0
    loss_count = 0

    while iteration < args.iters:
        # Shuffle no início de cada época
        random.shuffle(indices)
        epoch += 1

        for idx in indices:
            if iteration >= args.iters:
                break
            iteration += 1
            entry = ref_data[idx]

            prompt_text = format_prompt(entry["prompt"])
            chosen_text = truncate_text(entry["chosen"])
            rejected_text = truncate_text(entry["rejected"])

            c_ids, c_tgt, c_plen = tokenize_pair(tokenizer, prompt_text, chosen_text)
            r_ids, r_tgt, r_plen = tokenize_pair(tokenizer, prompt_text, rejected_text)

            ref_c_lp = mx.array(entry["ref_chosen_lp"])
            ref_r_lp = mx.array(entry["ref_rejected_lp"])

            loss, grads = loss_and_grad(
                model, c_ids, c_tgt, c_plen, r_ids, r_tgt, r_plen, ref_c_lp, ref_r_lp
            )
            # Gradient clipping para evitar colapso
            grads, _ = optim.clip_grad_norm(grads, max_norm=1.0)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            mx.clear_cache()

            running_loss += loss.item()
            loss_count += 1

            if iteration % 10 == 0:
                avg_loss = running_loss / loss_count
                peak_mem = mx.get_peak_memory() / 1e9
                print(f"  Iter {iteration:3d} (epoch {epoch}): avg_loss = {avg_loss:.4f}, Peak mem = {peak_mem:.3f} GB")
                running_loss = 0.0
                loss_count = 0

            if iteration % args.save_every == 0:
                lora_weights = {k: v for k, v in tree_flatten(model.trainable_parameters()) if "lora_" in k}
                mx.save_safetensors(str(DPO_ADAPTER_PATH / f"{iteration:06d}_adapters.safetensors"), dict(lora_weights))

    # Salvar final — só pesos LoRA
    print()
    print("💾 Salvando adapter DPO final...")
    lora_weights = {k: v for k, v in tree_flatten(model.trainable_parameters()) if "lora_" in k}
    mx.save_safetensors(str(DPO_ADAPTER_PATH / "adapters.safetensors"), dict(lora_weights))
    print(f"   Salvo em: {DPO_ADAPTER_PATH / 'adapters.safetensors'}")
    print(f"   {len(lora_weights)} tensores LoRA salvos")

    # Copiar adapter_config do SFT (mesma arquitetura LoRA)
    import shutil
    sft_config = SFT_ADAPTER / "adapter_config.json"
    dpo_config = DPO_ADAPTER_PATH / "adapter_config.json"
    if sft_config.exists():
        shutil.copy2(sft_config, dpo_config)
        print("   📋 adapter_config.json copiado do SFT")

    print()
    print("✅ DPO training concluído!")


def main():
    parser = argparse.ArgumentParser(description="DPO Training com MLX (fp16)")
    parser.add_argument("--iters", type=int, default=564)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--recompute-refs", action="store_true", help="Forçar recomputação dos ref log-probs")
    args = parser.parse_args()

    print("🏘️  DPO Training - Educação Financeira da Periferia")
    print(f"   Modelo: {MODEL_FP16} (fp16)")
    print(f"   Beta: {args.beta} | LR: {args.lr} | Iters: {args.iters}")
    print()

    data = load_dpo_data()
    random.seed(42)
    random.shuffle(data)
    print(f"📊 {len(data)} pares DPO carregados")
    print()

    if args.recompute_refs and REF_CACHE.exists():
        REF_CACHE.unlink()

    # Fase 1: pré-computar refs (carrega modelo, computa, descarrega)
    ref_data = precompute_ref_log_probs(data, args)

    # Fase 2: treinar DPO (carrega modelo novamente, agora com LoRA)
    train_dpo(ref_data, args)


if __name__ == "__main__":
    main()
