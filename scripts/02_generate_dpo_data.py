"""
Script V2 para gerar dataset DPO com contraste cultural real.

Gera pares de preferência onde:
- chosen: usa gírias no contexto CORRETO, linguagem natural da periferia
- rejected: linguagem formal/corporativa OU gírias usadas ERRADO

Uso:
    python scripts/02_generate_dpo_data.py --provider mlx --model mlx-community/Qwen3-4B-4bit
"""

import json
import re
import gc
import random
import argparse
from pathlib import Path

# ============================================================
# CONFIGURAÇÃO
# ============================================================

PROJECT_ROOT = Path(__file__).parent.parent
ARTIGOS_DIR = PROJECT_ROOT / "data" / "raw" / "artigos"
TRANSCRICOES_DIR = PROJECT_ROOT / "data" / "raw" / "transcricoes"
SFT_DATASET = PROJECT_ROOT / "data" / "instruction" / "sft_dataset.jsonl"
DPO_OUTPUT = PROJECT_ROOT / "data" / "preference" / "dpo_dataset.jsonl"

# Gírias com uso correto
GIRIAS_CONTEXTO = [
    {"giria": "trampo", "significa": "trabalho", "uso_errado": "trampo de investir (sem sentido)"},
    {"giria": "grana", "significa": "dinheiro", "uso_errado": "grana de conhecimento (sem sentido)"},
    {"giria": "bico", "significa": "trabalho extra", "uso_errado": "bico de poupança (sem sentido)"},
    {"giria": "quebrada", "significa": "periferia, bairro", "uso_errado": "quebrada financeira (confuso)"},
    {"giria": "correr atrás", "significa": "se esforçar", "uso_errado": "correr atrás do juros (estranho)"},
    {"giria": "passa a visão", "significa": "dar conselho", "uso_errado": "visão do investimento (corporativo)"},
    {"giria": "a fita é", "significa": "a questão é", "uso_errado": "a fita do mercado (forçado)"},
    {"giria": "moiado", "significa": "deu errado", "uso_errado": "o investimento moiou (uso ok, mas sem contexto)"},
    {"giria": "veneno", "significa": "sufoco, dificuldade", "uso_errado": "veneno financeiro (mistura registros)"},
    {"giria": "seloko", "significa": "expressão de surpresa", "uso_errado": "seloko o rendimento (forçado)"},
    {"giria": "embaçado", "significa": "difícil", "uso_errado": "taxa embaçada (estranho)"},
    {"giria": "treta", "significa": "problema", "uso_errado": "treta do IPCA (sem naturalidade)"},
]


def clear_memory():
    gc.collect()
    try:
        import mlx.core as mx
        mx.clear_cache()
    except (ImportError, AttributeError):
        pass


def load_artigos() -> str:
    artigos = []
    for filepath in sorted(ARTIGOS_DIR.glob("*.md")):
        content = filepath.read_text(encoding="utf-8")
        artigos.append(f"=== {filepath.stem} ===\n{content}")
    return "\n\n".join(artigos)


def load_transcricao_sample() -> str:
    trans_files = list(TRANSCRICOES_DIR.glob("*.txt")) if TRANSCRICOES_DIR.exists() else []
    if not trans_files:
        return ""
    f = random.choice(trans_files)
    return f.read_text(encoding="utf-8")[:600]


def get_girias_sample(n=5) -> str:
    selected = random.sample(GIRIAS_CONTEXTO, min(n, len(GIRIAS_CONTEXTO)))
    lines = []
    for g in selected:
        lines.append(f'- "{g["giria"]}" = {g["significa"]}')
    return "\n".join(lines)


def generate_with_mlx(prompt: str, system: str, model: str) -> str:
    import requests
    response = requests.post(
        "http://localhost:8080/v1/chat/completions",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.7,
            "max_tokens": 600,
            "top_p": 0.9,
        },
        timeout=300,
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    content = re.sub(r"<\|im_end\|>", "", content)
    return content.strip()


def generate_with_ollama(prompt: str, system: str, model: str) -> str:
    import requests
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model, "prompt": prompt, "system": system,
            "stream": False, "options": {"temperature": 0.7, "num_predict": 600},
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["response"]


def generate_with_anthropic(prompt: str, system: str, model: str) -> str:
    import anthropic
    client = anthropic.Anthropic()
    message = client.messages.create(
        model=model, max_tokens=800, system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


def gen(prompt: str, system: str, provider: str, model: str) -> str:
    generators = {
        "mlx": generate_with_mlx,
        "ollama": generate_with_ollama,
        "anthropic": generate_with_anthropic,
    }
    return generators[provider](prompt, system, model)


def load_sft_instructions() -> list[str]:
    instructions = []
    if SFT_DATASET.exists():
        with open(SFT_DATASET, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    instructions.append(entry["instruction"])
    return instructions


def generate_dpo_pair(instruction: str, artigos: str, provider: str, model: str) -> dict:
    """Gera par DPO com contraste cultural real."""

    girias = get_girias_sample(5)
    transcricao = load_transcricao_sample()

    # --- CHOSEN: resposta autêntica da periferia ---
    system_chosen = f"""Você é o "Mlk de Vila", educador financeiro da periferia de SP.
Nasceu na zona leste, escola pública, fez MBA em finanças. Fala a língua da quebrada.

GÍRIAS (use no contexto CORRETO):
{girias}

Regras: use gírias naturalmente, exemplos com R$ reais, direto e motivador, 2-3 parágrafos.
Comece direto, sem título."""

    prompt_chosen = f"""Referência de estilo:
{artigos[:1200]}

Fala real da periferia:
{transcricao[:500] if transcricao else ''}

PERGUNTA: {instruction}"""

    chosen = gen(prompt_chosen, system_chosen, provider, model)

    # --- REJECTED: resposta formal/corporativa ---
    system_rejected = """Você é um consultor financeiro formal de uma grande empresa.
Use linguagem técnica, termos em inglês (cash flow, budget, portfolio, asset allocation).
Tom profissional, distante e impessoal. Cite conceitos acadêmicos.
NÃO use gírias, NÃO use linguagem informal. 2-3 parágrafos."""

    prompt_rejected = f"""PERGUNTA: {instruction}

Responda de forma técnica e profissional:"""

    rejected = gen(prompt_rejected, system_rejected, provider, model)

    return {"prompt": instruction, "chosen": chosen, "rejected": rejected}


def load_existing_dpo() -> list:
    entries = []
    if DPO_OUTPUT.exists():
        with open(DPO_OUTPUT, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
    return entries


def save_dpo_entry(entry: dict):
    with open(DPO_OUTPUT, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Gera dataset DPO V2 com contraste cultural")
    parser.add_argument("--provider", choices=["ollama", "mlx", "anthropic"], default="mlx")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--fresh", action="store_true", help="Limpar dataset existente")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    default_models = {
        "ollama": "qwen2.5:7b",
        "mlx": "mlx-community/Qwen3-4B-4bit",
        "anthropic": "claude-sonnet-4-20250514",
    }
    model = args.model or default_models[args.provider]

    print(f"🏘️  Gerador de Dataset DPO V2 - Contraste Cultural")
    print(f"   Provider: {args.provider} | Modelo: {model}")
    print(f"   Output: {DPO_OUTPUT}")
    print()

    artigos = load_artigos()
    instructions = load_sft_instructions()

    if args.fresh and DPO_OUTPUT.exists():
        DPO_OUTPUT.unlink()
        print("🗑️  Dataset anterior removido")

    existing = load_existing_dpo()
    existing_prompts = {e["prompt"] for e in existing}

    new_instructions = [i for i in instructions if i not in existing_prompts]
    if args.limit:
        new_instructions = new_instructions[:args.limit]

    print(f"📄 {len(list(ARTIGOS_DIR.glob('*.md')))} artigos como contexto")
    print(f"📊 {len(existing)} pares DPO existentes")
    print(f"🎯 {len(new_instructions)} novos pares a gerar")
    print()

    if args.dry_run:
        for i, inst in enumerate(new_instructions, 1):
            print(f"  {i}. {inst[:60]}")
        return

    generated = 0
    errors = 0

    for i, instruction in enumerate(new_instructions, 1):
        print(f"  [{i}/{len(new_instructions)}] {instruction[:50]}...", end=" ", flush=True)
        try:
            entry = generate_dpo_pair(instruction, artigos, args.provider, model)
            save_dpo_entry(entry)
            generated += 1
            print("✅")
        except Exception as e:
            errors += 1
            print(f"❌ {e}")
        finally:
            clear_memory()

    print()
    print(f"=== RESULTADO ===")
    print(f"✅ Gerados: {generated}")
    print(f"❌ Erros: {errors}")
    print(f"📊 Total DPO: {len(existing) + generated}")


if __name__ == "__main__":
    main()
