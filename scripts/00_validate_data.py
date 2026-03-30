"""
Script para validar e mostrar estatísticas dos datasets SFT e DPO.

Uso:
    python scripts/00_validate_data.py
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
SFT_FILE = PROJECT_ROOT / "data" / "instruction" / "sft_dataset.jsonl"
DPO_FILE = PROJECT_ROOT / "data" / "preference" / "dpo_dataset.jsonl"


def validate_jsonl(filepath: Path, required_fields: list[str]) -> tuple[list, list]:
    """Valida um arquivo JSONL e retorna (válidos, erros)."""
    valid = []
    errors = []

    if not filepath.exists():
        print(f"  ❌ Arquivo não encontrado: {filepath}")
        return valid, errors

    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                missing = [f for f in required_fields if f not in entry]
                if missing:
                    errors.append((i, f"Campos faltando: {missing}"))
                elif any(len(str(entry[f])) < 10 for f in required_fields):
                    errors.append((i, "Campo muito curto (< 10 chars)"))
                else:
                    valid.append(entry)
            except json.JSONDecodeError as e:
                errors.append((i, f"JSON inválido: {e}"))

    return valid, errors


def avg_length(entries: list, field: str) -> int:
    if not entries:
        return 0
    return sum(len(e.get(field, "")) for e in entries) // len(entries)


def print_sample(entry: dict, fields: list):
    """Mostra uma amostra formatada."""
    for field in fields:
        value = entry.get(field, "")
        if len(value) > 150:
            value = value[:150] + "..."
        print(f"    {field}: {value}")


def main():
    print("=" * 60)
    print("🏘️  VALIDAÇÃO DOS DATASETS - Periferia Finance LLM")
    print("=" * 60)
    print()

    # === SFT Dataset ===
    print("📋 DATASET SFT (Supervised Fine-Tuning)")
    print("-" * 40)
    sft_valid, sft_errors = validate_jsonl(SFT_FILE, ["instruction", "response"])

    print(f"  ✅ Entradas válidas: {len(sft_valid)}")
    print(f"  ❌ Entradas com erro: {len(sft_errors)}")
    if sft_valid:
        print(f"  📏 Tamanho médio instrução: {avg_length(sft_valid, 'instruction')} chars")
        print(f"  📏 Tamanho médio resposta: {avg_length(sft_valid, 'response')} chars")

    if sft_errors:
        print(f"\n  Erros encontrados:")
        for line_num, error in sft_errors[:5]:
            print(f"    Linha {line_num}: {error}")

    if sft_valid:
        print(f"\n  📌 Amostra (primeira entrada):")
        print_sample(sft_valid[0], ["instruction", "response"])

    parse_errors = sum(1 for e in sft_valid if e.get("_parse_error"))
    if parse_errors:
        print(f"\n  ⚠️  {parse_errors} entradas com erro de parse (revisar manualmente)")

    print()

    # === DPO Dataset ===
    print("📋 DATASET DPO (Direct Preference Optimization)")
    print("-" * 40)
    dpo_valid, dpo_errors = validate_jsonl(DPO_FILE, ["prompt", "chosen", "rejected"])

    print(f"  ✅ Entradas válidas: {len(dpo_valid)}")
    print(f"  ❌ Entradas com erro: {len(dpo_errors)}")
    if dpo_valid:
        print(f"  📏 Tamanho médio prompt: {avg_length(dpo_valid, 'prompt')} chars")
        print(f"  📏 Tamanho médio chosen: {avg_length(dpo_valid, 'chosen')} chars")
        print(f"  📏 Tamanho médio rejected: {avg_length(dpo_valid, 'rejected')} chars")

    if dpo_errors:
        print(f"\n  Erros encontrados:")
        for line_num, error in dpo_errors[:5]:
            print(f"    Linha {line_num}: {error}")

    if dpo_valid:
        print(f"\n  📌 Amostra (primeira entrada):")
        print_sample(dpo_valid[0], ["prompt", "chosen", "rejected"])

    print()

    # === Resumo ===
    print("=" * 60)
    print("📊 RESUMO")
    print("=" * 60)
    total_sft = len(sft_valid)
    total_dpo = len(dpo_valid)

    print(f"  SFT: {total_sft} entradas válidas")
    print(f"  DPO: {total_dpo} entradas válidas")
    print()

    # Recomendações
    print("💡 RECOMENDAÇÕES:")
    if total_sft < 500:
        print(f"  ⚠️  SFT tem {total_sft} entradas. Recomendado: mínimo 500.")
        print(f"     Execute: python scripts/01_generate_sft_data.py --provider ollama")
    else:
        print(f"  ✅ SFT com {total_sft} entradas — bom pra começar!")

    if total_dpo < 200:
        print(f"  ⚠️  DPO tem {total_dpo} entradas. Recomendado: mínimo 200.")
        print(f"     Execute: python scripts/02_generate_dpo_data.py --provider ollama")
    else:
        print(f"  ✅ DPO com {total_dpo} entradas — bom pra começar!")

    if total_sft >= 500 and total_dpo >= 200:
        print(f"\n  🚀 Datasets prontos! Próximo passo:")
        print(f"     python scripts/03_sft_train.py")


if __name__ == "__main__":
    main()
