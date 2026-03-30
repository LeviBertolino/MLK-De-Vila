#!/bin/bash
# ============================================================
# Monitor de memória e progresso durante a geração de dados
# Uso: bash scripts/monitor.sh
# ============================================================

echo "🏘️  Monitor - Periferia Finance LLM"
echo "===================================="
echo "Pressione Ctrl+C para sair"
echo ""

while true; do
    clear
    echo "🏘️  Monitor - $(date '+%H:%M:%S')"
    echo "===================================="
    echo ""

    # Memória
    echo "💾 MEMÓRIA:"
    memory_pressure=$(memory_pressure 2>/dev/null | grep "System-wide memory free percentage" | awk '{print $NF}')
    if [ -n "$memory_pressure" ]; then
        echo "   Memória livre: ${memory_pressure}"
    fi

    # Swap
    swap_used=$(sysctl vm.swapusage 2>/dev/null | awk '{print $7}')
    echo "   Swap usado: ${swap_used:-N/A}"

    # Processo MLX
    echo ""
    echo "🧠 PROCESSO MLX:"
    mlx_pid=$(pgrep -f "mlx_lm.server" 2>/dev/null)
    if [ -n "$mlx_pid" ]; then
        mlx_mem=$(ps -p $mlx_pid -o rss= 2>/dev/null | awk '{printf "%.1f GB", $1/1048576}')
        echo "   PID: $mlx_pid | RAM: $mlx_mem | Status: ✅ Rodando"
    else
        echo "   Status: ❌ Server não está rodando"
    fi

    # Progresso dos datasets
    echo ""
    echo "📊 PROGRESSO:"
    if [ -f "data/instruction/sft_dataset.jsonl" ]; then
        sft_count=$(wc -l < "data/instruction/sft_dataset.jsonl" | tr -d ' ')
        echo "   SFT: ${sft_count} / 500 entradas ($(( sft_count * 100 / 500 ))%)"
    fi
    if [ -f "data/preference/dpo_dataset.jsonl" ]; then
        dpo_count=$(wc -l < "data/preference/dpo_dataset.jsonl" | tr -d ' ')
        echo "   DPO: ${dpo_count} / 200 entradas ($(( dpo_count * 100 / 200 ))%)"
    fi

    # Temperatura CPU (se disponível)
    echo ""
    echo "🌡️  Dica: se o Mac esquentar muito, é normal."
    echo "   O swap no SSD vai ser usado. Não feche outros apps."

    sleep 10
done
