#!/bin/bash
# ============================================================
# Setup do Projeto: Periferia Finance LLM
# MacBook Pro M2 8GB
# ============================================================

set -e

echo "🏘️  Setup - Periferia Finance LLM"
echo "=================================="
echo ""

# 1. Criar virtual environment
echo "📦 Criando virtual environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "   ✅ .venv criado"
else
    echo "   ⏭️  .venv já existe"
fi

source .venv/bin/activate

# 2. Instalar dependências
echo ""
echo "📦 Instalando dependências..."
pip install --upgrade pip -q

pip install -q \
    mlx \
    mlx-lm \
    requests \
    jsonlines

echo "   ✅ Dependências instaladas"

# 3. Baixar o modelo (isso vai demorar na primeira vez ~4.5GB)
echo ""
echo "🧠 Baixando Qwen3-8B-4bit para MLX..."
echo "   (Primeira vez: ~4.5GB de download, pode demorar)"
echo ""
python3 -c "
from mlx_lm import load
print('   Baixando e carregando modelo...')
model, tokenizer = load('mlx-community/Qwen3-8B-4bit')
print('   ✅ Modelo baixado e verificado!')
del model, tokenizer
"

echo ""
echo "============================================"
echo "✅ Setup completo!"
echo ""
echo "Próximos passos:"
echo ""
echo "  1. Ativar o environment:"
echo "     source .venv/bin/activate"
echo ""
echo "  2. Subir o server MLX:"
echo "     python3 -m mlx_lm.server --model mlx-community/Qwen3-8B-4bit --port 8080"
echo ""
echo "  3. Em OUTRO terminal, gerar os dados:"
echo "     source .venv/bin/activate"
echo "     python3 scripts/01_generate_sft_data.py --provider mlx --model mlx-community/Qwen3-8B-4bit"
echo ""
echo "============================================"
