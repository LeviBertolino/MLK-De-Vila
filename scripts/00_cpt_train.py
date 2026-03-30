"""
Script para Continuous Pre-Training (CPT) com LoRA usando mlx-lm.

O CPT treina o modelo em texto bruto (next-token prediction) para internalizar
o dialeto da periferia ANTES do SFT ensinar O QUE falar.

Pipeline completo: CPT → fuse → SFT → fuse → DPO → benchmark

Uso:
    python scripts/00_cpt_train.py --train              # treina CPT
    python scripts/00_cpt_train.py --train --iters 500  # custom iters
"""

import argparse
import subprocess
from pathlib import Path

# ============================================================
# CONFIGURAÇÃO
# ============================================================

PROJECT_ROOT = Path(__file__).parent.parent
CPT_DATA_DIR = PROJECT_ROOT / "data" / "cpt"
ADAPTER_PATH = PROJECT_ROOT / "adapters" / "cpt"

MODEL = "mlx-community/gemma-3-1b-it-bf16"
CPT_MODEL_FUSED = PROJECT_ROOT / "models" / "gemma-3-1b-cpt"

# CPT precisa de LR mais alto que SFT — queremos internalizar vocabulário novo
# 100 iters com 1e-6 = 0.42 epochs, insuficiente. 500 iters com 2e-5 = ~2 epochs.
DEFAULT_ITERS = 500
DEFAULT_LR = "2e-5"


def train(args):
    """Executa o CPT com LoRA."""
    ADAPTER_PATH.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python3", "-m", "mlx_lm", "lora",
        "--model", MODEL,
        "--train",
        "--data", str(CPT_DATA_DIR),
        "--fine-tune-type", "lora",
        "--batch-size", "1",
        "--num-layers", "10",
        "-c", str(PROJECT_ROOT / "configs" / "lora_config_cpt.yaml"),
        "--grad-checkpoint",
        "--iters", str(args.iters),
        "--learning-rate", DEFAULT_LR,
        "--adapter-path", str(ADAPTER_PATH),
        "--max-seq-length", "1024",
        "--steps-per-report", "10",
        "--steps-per-eval", "50",
        "--save-every", "100",
        # SEM --mask-prompt: CPT treina em TODO o texto (next-token prediction)
    ]

    if args.resume:
        cmd.extend(["--resume-adapter-file", str(ADAPTER_PATH / "adapters.safetensors")])

    print("🧠 Iniciando Continuous Pre-Training (CPT) com LoRA")
    print(f"   Modelo base: {MODEL}")
    print(f"   Dados: {CPT_DATA_DIR}")
    print(f"   Adapter: {ADAPTER_PATH}")
    print(f"   Iterações: {args.iters}")
    print(f"   Learning rate: {DEFAULT_LR}")
    print(f"   Max seq length: 1024")
    print(f"   SEM --mask-prompt (treina em todo o texto)")
    print()
    print(f"   Comando: {' '.join(cmd)}")
    print()

    subprocess.run(cmd, check=True)

    print()
    print(f"✅ CPT concluído!")
    print(f"   Adapter salvo em: {ADAPTER_PATH}")
    print()
    print("Próximos passos:")
    print(f"   1. Fuse: python3 -m mlx_lm fuse --model {MODEL} --adapter-path {ADAPTER_PATH} --save-path {CPT_MODEL_FUSED}")
    print(f"   2. SFT:  python scripts/03_sft_train.py --train  (apontar MODEL para {CPT_MODEL_FUSED})")


def main():
    parser = argparse.ArgumentParser(description="Continuous Pre-Training (CPT) com LoRA via mlx-lm")
    parser.add_argument("--train", action="store_true", help="Executa o treinamento CPT")
    parser.add_argument("--iters", type=int, default=DEFAULT_ITERS, help=f"Número de iterações (default: {DEFAULT_ITERS})")
    parser.add_argument("--resume", action="store_true", help="Retomar do último checkpoint")
    args = parser.parse_args()

    if not args.train:
        print("Use --train para iniciar o treinamento CPT")
        print()
        print("Antes de treinar, gere os dados com:")
        print("   python scripts/00_prepare_cpt_data.py")
        return

    # Verificar se dados existem
    train_file = CPT_DATA_DIR / "train.jsonl"
    valid_file = CPT_DATA_DIR / "valid.jsonl"

    if not train_file.exists() or not valid_file.exists():
        print("❌ Dados CPT não encontrados!")
        print("   Rode primeiro: python scripts/00_prepare_cpt_data.py")
        return

    # Contar exemplos
    with open(train_file) as f:
        n_train = sum(1 for _ in f)
    with open(valid_file) as f:
        n_valid = sum(1 for _ in f)

    print(f"📊 Dados: {n_train} treino, {n_valid} validação")
    print()

    train(args)


if __name__ == "__main__":
    main()
