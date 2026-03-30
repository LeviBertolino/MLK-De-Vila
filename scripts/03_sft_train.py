"""
Script para treinar SFT com LoRA usando mlx-lm.

Etapa 1: Converte o dataset SFT para o formato chat do mlx-lm
Etapa 2: Roda o treinamento LoRA

Uso:
    python scripts/03_sft_train.py --prepare     # só prepara os dados
    python scripts/03_sft_train.py --train        # prepara + treina
    python scripts/03_sft_train.py --train --grad-checkpoint  # menos memória
"""

import json
import argparse
import random
from pathlib import Path

# ============================================================
# CONFIGURAÇÃO
# ============================================================

PROJECT_ROOT = Path(__file__).parent.parent
SFT_DATASET = PROJECT_ROOT / "data" / "instruction" / "sft_dataset.jsonl"
TRAIN_DIR = PROJECT_ROOT / "data" / "sft_chat"  # dados formatados pro mlx-lm
ADAPTER_PATH = PROJECT_ROOT / "adapters" / "sft"

# Para pipeline com CPT: usar modelo CPT-fused
# Para pipeline sem CPT: usar "mlx-community/gemma-3-1b-it-bf16"
MODEL = str(PROJECT_ROOT / "models" / "gemma-3-1b-cpt")

SYSTEM_MESSAGE = (
    "Você é o 'Mlk de Vila', um educador financeiro que fala a língua da periferia. "
    "Explica finanças de forma simples, direta e usando exemplos do dia a dia. "
    "Usa gírias como mano, parceiro, trampo, grana, padoca, bico. "
    "É motivador e honesto."
)

TRAIN_SPLIT = 0.85  # 85% treino, 10% validação, 5% teste


def load_sft_data() -> list[dict]:
    """Carrega o dataset SFT."""
    entries = []
    with open(SFT_DATASET, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def truncate_text(text: str, max_chars: int = 800) -> str:
    """Trunca texto para evitar sequências longas demais que causam NaN/OOM."""
    if len(text) <= max_chars:
        return text
    # Corta no último parágrafo completo dentro do limite
    truncated = text[:max_chars]
    last_period = truncated.rfind(".")
    if last_period > max_chars // 2:
        return truncated[:last_period + 1]
    return truncated


def convert_to_chat_format(entries: list[dict]) -> list[dict]:
    """Converte para o formato chat que o mlx-lm espera.

    Formato: {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}

    Filtra respostas vazias ou muito curtas que ensinam o modelo a gerar <end_of_turn>.
    """
    chat_entries = []
    skipped = 0
    for entry in entries:
        response = truncate_text(entry["response"])
        # Filtrar respostas vazias ou muito curtas (< 50 chars)
        if not response or len(response.strip()) < 50:
            skipped += 1
            continue
        chat_entry = {
            "messages": [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": entry["instruction"]},
                {"role": "assistant", "content": response},
            ]
        }
        chat_entries.append(chat_entry)
    if skipped:
        print(f"   ⚠️  {skipped} exemplos com resposta vazia/curta removidos")
    return chat_entries


def split_and_save(chat_entries: list[dict]):
    """Divide em train/valid/test e salva."""
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)

    random.seed(42)
    random.shuffle(chat_entries)

    n = len(chat_entries)
    train_end = int(n * TRAIN_SPLIT)
    valid_end = int(n * (TRAIN_SPLIT + 0.10))

    splits = {
        "train": chat_entries[:train_end],
        "valid": chat_entries[train_end:valid_end],
        "test": chat_entries[valid_end:],
    }

    for name, data in splits.items():
        filepath = TRAIN_DIR / f"{name}.jsonl"
        with open(filepath, "w", encoding="utf-8") as f:
            for entry in data:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"   {name}: {len(data)} exemplos → {filepath}")

    return splits


def prepare_data():
    """Prepara os dados para treinamento."""
    print("📦 Preparando dados para SFT...")
    print()

    entries = load_sft_data()
    print(f"   {len(entries)} entradas carregadas do dataset SFT")

    chat_entries = convert_to_chat_format(entries)
    print(f"   {len(chat_entries)} convertidas para formato chat")
    print()

    splits = split_and_save(chat_entries)
    print()

    # Preview
    print("📋 Preview de um exemplo de treino:")
    sample = splits["train"][0]
    for msg in sample["messages"]:
        role = msg["role"].upper()
        content = msg["content"][:100]
        print(f"   [{role}] {content}...")
    print()

    return splits


def train(args):
    """Executa o treinamento SFT com LoRA."""
    import subprocess

    ADAPTER_PATH.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python3", "-m", "mlx_lm", "lora",
        "--model", MODEL,
        "--train",
        "--data", str(TRAIN_DIR),
        "--fine-tune-type", "lora",
        "--batch-size", "1",
        "--num-layers", "12",
        "-c", str(PROJECT_ROOT / "configs" / "lora_config.yaml"),
        "--grad-checkpoint",
        "--iters", str(args.iters),
        "--learning-rate", "1e-5",
        "--adapter-path", str(ADAPTER_PATH),
        "--max-seq-length", "1024",
        "--steps-per-report", "5",
        "--steps-per-eval", "25",
        "--save-every", "25",
        "--mask-prompt",
    ]

    if args.grad_checkpoint:
        cmd.append("--grad-checkpoint")

    if args.resume:
        cmd.extend(["--resume-adapter-file", str(ADAPTER_PATH / "adapters.safetensors")])

    print("🚀 Iniciando treinamento SFT com LoRA")
    print(f"   Modelo base: {MODEL}")
    print(f"   Dados: {TRAIN_DIR}")
    print(f"   Adapter: {ADAPTER_PATH}")
    print(f"   Iterações: {args.iters}")
    print(f"   Grad checkpoint: {args.grad_checkpoint}")
    print()
    print(f"   Comando: {' '.join(cmd)}")
    print()

    subprocess.run(cmd, check=True)

    print()
    print(f"✅ Treinamento SFT concluído!")
    print(f"   Adapter salvo em: {ADAPTER_PATH}")


def main():
    parser = argparse.ArgumentParser(description="SFT Training com LoRA via mlx-lm")
    parser.add_argument("--prepare", action="store_true", help="Só prepara os dados")
    parser.add_argument("--train", action="store_true", help="Prepara + treina")
    parser.add_argument("--iters", type=int, default=1950, help="Número de iterações (default: 1950, evita overfitting do iter 2000)")
    parser.add_argument("--grad-checkpoint", action="store_true", help="Gradient checkpointing (menos memória)")
    parser.add_argument("--resume", action="store_true", help="Retomar do último checkpoint")
    args = parser.parse_args()

    if not args.prepare and not args.train:
        print("Use --prepare (só dados) ou --train (dados + treino)")
        return

    splits = prepare_data()

    if args.train:
        train(args)


if __name__ == "__main__":
    main()
