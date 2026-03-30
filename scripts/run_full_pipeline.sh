#!/bin/bash
# =============================================================================
# Pipeline completo v2: CPT agressivo + SFT sem respostas vazias
#
# Mudanças vs v1:
#   CPT: 500 iters (antes 100), LR 2e-5 (antes 1e-6), rank 32, 10 layers
#   SFT: 1950 iters (antes 2000, evita overfitting), filtro de respostas vazias
#   DPO: 300 iters (igual), recomputa refs
#
# Uso:
#   chmod +x scripts/run_full_pipeline.sh
#   ./scripts/run_full_pipeline.sh 2>&1 | tee results/pipeline_v2_log.txt
# =============================================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

BASE_MODEL="mlx-community/gemma-3-1b-it-bf16"
CPT_ADAPTER="$PROJECT_ROOT/adapters/cpt"
CPT_MODEL="$PROJECT_ROOT/models/gemma-3-1b-cpt"
SFT_ADAPTER="$PROJECT_ROOT/adapters/sft"
SFT_MERGED="$PROJECT_ROOT/models/sft_merged"
DPO_ADAPTER="$PROJECT_ROOT/adapters/dpo"

echo ""
echo -e "${BLUE}================================================================${NC}"
echo -e "${BLUE}  PIPELINE v2: CPT agressivo + SFT limpo + DPO + Benchmark${NC}"
echo -e "${BLUE}================================================================${NC}"
echo ""
echo "  Mudanças vs v1:"
echo "    CPT: 500 iters, LR 2e-5, rank 32, scale 20, 10 layers"
echo "    SFT: 1950 iters, filtro de respostas vazias/curtas"
echo "    DPO: 300 iters, recomputa refs"
echo ""

# Verificar dados
echo -e "${YELLOW}[0/6] Verificando dados...${NC}"
for f in "data/cpt/train.jsonl" "data/instruction/sft_dataset.jsonl" "data/preference/dpo_dataset.jsonl"; do
    if [ ! -f "$PROJECT_ROOT/$f" ]; then
        echo -e "${RED}ERRO: $f não encontrado!${NC}"
        exit 1
    fi
done
echo "  Todos os dados OK"
echo ""

# Limpar artefatos
echo -e "${YELLOW}[0/6] Limpando artefatos antigos...${NC}"
rm -rf "$CPT_ADAPTER" "$SFT_ADAPTER" "$DPO_ADAPTER"
rm -rf "$CPT_MODEL" "$SFT_MERGED"
rm -f "$PROJECT_ROOT/data/ref_log_probs.json"
rm -rf "$PROJECT_ROOT/data/sft_chat"
echo "  Limpo"
echo ""

# =============================================================================
# STEP 1: CPT Training (agressivo)
# =============================================================================
echo -e "${GREEN}================================================================${NC}"
echo -e "${GREEN}  [1/6] CPT Training — 500 iters, LR 2e-5, rank 32, 10 layers${NC}"
echo -e "${GREEN}================================================================${NC}"
echo ""

python scripts/00_cpt_train.py --train --iters 500

echo ""
echo -e "${GREEN}  CPT concluído!${NC}"
echo ""

# =============================================================================
# STEP 2: Fuse CPT
# =============================================================================
echo -e "${GREEN}================================================================${NC}"
echo -e "${GREEN}  [2/6] Fusing CPT adapter${NC}"
echo -e "${GREEN}================================================================${NC}"
echo ""

python3 -m mlx_lm fuse \
    --model "$BASE_MODEL" \
    --adapter-path "$CPT_ADAPTER" \
    --save-path "$CPT_MODEL"

echo -e "${GREEN}  Fused em: $CPT_MODEL${NC}"
echo ""

# =============================================================================
# STEP 3: SFT Training (sobre CPT, sem respostas vazias)
# =============================================================================
echo -e "${GREEN}================================================================${NC}"
echo -e "${GREEN}  [3/6] SFT Training — 1950 iters, dados limpos${NC}"
echo -e "${GREEN}================================================================${NC}"
echo ""

python scripts/03_sft_train.py --train --iters 1950

echo ""
echo -e "${GREEN}  SFT concluído!${NC}"
echo ""

# =============================================================================
# STEP 4: Fuse SFT
# =============================================================================
echo -e "${GREEN}================================================================${NC}"
echo -e "${GREEN}  [4/6] Fusing SFT adapter${NC}"
echo -e "${GREEN}================================================================${NC}"
echo ""

python3 -m mlx_lm fuse \
    --model "$CPT_MODEL" \
    --adapter-path "$SFT_ADAPTER" \
    --save-path "$SFT_MERGED"

echo -e "${GREEN}  Fused em: $SFT_MERGED${NC}"
echo ""

# =============================================================================
# STEP 5: DPO Training
# =============================================================================
echo -e "${GREEN}================================================================${NC}"
echo -e "${GREEN}  [5/6] DPO Training — 300 iters${NC}"
echo -e "${GREEN}================================================================${NC}"
echo ""

python scripts/04_dpo_train.py --iters 300 --recompute-refs

echo ""
echo -e "${GREEN}  DPO concluído!${NC}"
echo ""

# =============================================================================
# STEP 6: Benchmark
# =============================================================================
echo -e "${GREEN}================================================================${NC}"
echo -e "${GREEN}  [6/6] Benchmark: Base vs CPT vs SFT vs DPO${NC}"
echo -e "${GREEN}================================================================${NC}"
echo ""

python scripts/05_benchmark.py \
    --output results/benchmark_v2_cpt_aggressive.json \
    --stages base cpt sft dpo

echo ""
echo -e "${BLUE}================================================================${NC}"
echo -e "${BLUE}  PIPELINE v2 COMPLETO!${NC}"
echo -e "${BLUE}================================================================${NC}"
echo ""
echo "  Resultados: results/benchmark_v2_cpt_aggressive.json"
echo "  Log:        results/pipeline_v2_log.txt"
echo ""
echo "  Comparação esperada:"
echo "    v1: CPT 100it/1e-6 → SFT 9/16, DPO 11/16"
echo "    v2: CPT 500it/2e-5 → SFT ?/16, DPO ?/16"
echo ""
