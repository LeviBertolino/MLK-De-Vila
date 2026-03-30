"""
Benchmark comparativo: Base vs SFT vs SFT+DPO

Testa com dois tipos de perguntas:
1. FORMAL: linguagem padrão (testa resistência do alinhamento)
2. PERIFERIA: linguagem da quebrada (testa se o modelo reconhece o dialeto)

Uso:
    python scripts/05_benchmark.py
    python scripts/05_benchmark.py --output results/benchmark.json
"""

import json
import re
import argparse
from pathlib import Path
from datetime import datetime

from mlx_lm import load, generate

# ============================================================
# CONFIGURAÇÃO
# ============================================================

PROJECT_ROOT = Path(__file__).parent.parent
SFT_ADAPTER = PROJECT_ROOT / "adapters" / "sft"
DPO_ADAPTER = PROJECT_ROOT / "adapters" / "dpo"
RESULTS_DIR = PROJECT_ROOT / "results"

MODEL_BASE = "mlx-community/gemma-3-1b-it-bf16"
MODEL_CPT = str(PROJECT_ROOT / "models" / "gemma-3-1b-cpt")
MODEL_SFT_MERGED = str(PROJECT_ROOT / "models" / "sft_merged")

# ============================================================
# PERGUNTAS: FORMAL vs PERIFERIA
# ============================================================

QUESTIONS_FORMAL = [
    "O que é inflação e como ela afeta meu dinheiro?",
    "Como economizar ganhando um salário mínimo?",
    "Vale a pena fazer empréstimo pra pagar dívida?",
    "O que é juros compostos?",
    "Como começar a investir com R$ 50?",
    "Qual a diferença entre poupar e investir?",
    "O que é reserva de emergência e por que é importante?",
    "Como fazer um orçamento mensal simples?",
]

QUESTIONS_PERIFERIA = [
    "Mano, o que é essa tal de inflação? Tá comendo minha grana?",
    "Parceiro, como faz pra guardar uns trocado ganhando um salário mínimo?",
    "Fica ligado, vale a pena pegar empréstimo pra quitar uma dívida?",
    "Passa a visão sobre juros compostos, como funciona essa fita?",
    "Mano, dá pra começar a investir com 50 conto? Como faz?",
    "Qual a diferença entre guardar grana e investir? Me explica na moral.",
    "O que é reserva de emergência? Por que o pessoal da quebrada precisa disso?",
    "Como organizar a grana do mês de um jeito simples? Tô perdido, mano.",
]


def clean_response(text: str) -> str:
    """Remove blocos <think>...</think>, <end_of_turn>, <pad> e artefatos."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.replace("<end_of_turn>", "").replace("<pad>", "")
    text = re.sub(r"\n{3,}", "\n\n", text)  # colapsar linhas vazias excessivas
    return text.strip()


def generate_response(model, tokenizer, question: str, max_tokens: int = 300) -> str:
    """Gera resposta usando chat template. Sem system message (diagnóstico mostrou que piora)."""
    messages = [{"role": "user", "content": question}]

    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    else:
        prompt = f"Pergunta: {question}\n\nResposta:"

    response = generate(
        model, tokenizer, prompt=prompt,
        max_tokens=max_tokens,
    )

    return clean_response(response)


def load_model_stage(stage: str):
    """Carrega modelo no estágio especificado."""
    if stage == "base":
        print(f"   📦 Carregando modelo BASE (sem fine-tuning)...")
        model, tokenizer = load(MODEL_BASE)
    elif stage == "cpt":
        print(f"   📦 Carregando modelo CPT (base + CPT fused)...")
        model, tokenizer = load(MODEL_CPT)
    elif stage == "sft":
        print(f"   📦 Carregando modelo SFT (CPT + SFT adapter)...")
        model, tokenizer = load(MODEL_CPT, adapter_path=str(SFT_ADAPTER))
    elif stage == "dpo":
        print(f"   📦 Carregando modelo SFT+DPO (merged + DPO adapter)...")
        model, tokenizer = load(MODEL_SFT_MERGED, adapter_path=str(DPO_ADAPTER))
    else:
        raise ValueError(f"Estágio desconhecido: {stage}")

    # Adicionar <end_of_turn> como EOS token para parar geração corretamente
    tokenizer.add_eos_token("<end_of_turn>")

    return model, tokenizer


def run_benchmark(questions_formal, questions_periferia, stages, max_tokens):
    """Roda benchmark completo com ambos os tipos de perguntas."""
    results = []

    for stage in stages:
        model, tokenizer = load_model_stage(stage)
        print(f"   ✅ Modelo carregado (EOS tokens: {tokenizer.eos_token_ids})")
        print()

        # Perguntas formais
        print(f"   --- Perguntas FORMAIS ---")
        for i, question in enumerate(questions_formal, 1):
            print(f"   [{i}/{len(questions_formal)}] {question[:50]}...", end=" ", flush=True)
            response = generate_response(model, tokenizer, question, max_tokens)
            results.append({
                "stage": stage,
                "tipo": "formal",
                "question": question,
                "response": response,
            })
            status = "✅" if len(response) > 20 else "⚠️ (curta)"
            print(status)

        # Perguntas periferia
        print(f"   --- Perguntas PERIFERIA ---")
        for i, question in enumerate(questions_periferia, 1):
            print(f"   [{i}/{len(questions_periferia)}] {question[:50]}...", end=" ", flush=True)
            response = generate_response(model, tokenizer, question, max_tokens)
            results.append({
                "stage": stage,
                "tipo": "periferia",
                "question": question,
                "response": response,
            })
            status = "✅" if len(response) > 20 else "⚠️ (curta)"
            print(status)

        # Liberar memória entre estágios
        del model, tokenizer
        import gc
        gc.collect()
        print()

    return results


def print_comparison(results):
    """Imprime comparação lado a lado."""
    stages = ["base", "cpt", "sft", "dpo"]
    stage_labels = {
        "base": "🔵 BASE",
        "cpt": "🟠 CPT",
        "sft": "🟡 SFT",
        "dpo": "🟢 DPO",
    }

    for tipo in ["formal", "periferia"]:
        tipo_label = "📝 FORMAL" if tipo == "formal" else "🏘️ PERIFERIA"
        print()
        print("=" * 80)
        print(f"  {tipo_label}")
        print("=" * 80)

        # Get unique questions for this tipo
        questions = []
        seen = set()
        for r in results:
            if r["tipo"] == tipo and r["question"] not in seen:
                questions.append(r["question"])
                seen.add(r["question"])

        # Index results
        by_question = {}
        for r in results:
            if r["tipo"] == tipo:
                key = r["question"]
                if key not in by_question:
                    by_question[key] = {}
                by_question[key][r["stage"]] = r["response"]

        for q in questions:
            print()
            print(f"❓ {q}")
            print("-" * 80)

            for stage in stages:
                if stage in by_question.get(q, {}):
                    label = stage_labels[stage]
                    response = by_question[q][stage]
                    if not response:
                        response = "[VAZIO - modelo gerou <end_of_turn> imediato]"
                    preview = response[:400]
                    if len(response) > 400:
                        preview += "..."
                    print(f"\n{label}:")
                    print(f"   {preview}")

            print()
            print("=" * 80)

    # Resumo de qualidade
    print()
    print("=" * 80)
    print("  📊 RESUMO")
    print("=" * 80)
    actual_stages = sorted(set(r["stage"] for r in results), key=lambda s: stages.index(s) if s in stages else 99)
    for stage in actual_stages:
        stage_results = [r for r in results if r["stage"] == stage]
        total = len(stage_results)
        com_conteudo = sum(1 for r in stage_results if len(r["response"]) > 20)
        n_formal = sum(1 for r in stage_results if r["tipo"] == "formal")
        n_periferia = sum(1 for r in stage_results if r["tipo"] == "periferia")
        formal_ok = sum(1 for r in stage_results if r["tipo"] == "formal" and len(r["response"]) > 20)
        periferia_ok = sum(1 for r in stage_results if r["tipo"] == "periferia" and len(r["response"]) > 20)
        label = stage_labels.get(stage, stage)
        print(f"   {label}: {com_conteudo}/{total} com conteúdo (formal: {formal_ok}/{n_formal}, periferia: {periferia_ok}/{n_periferia})")
    print()


def save_results(results, output_path: Path):
    """Salva resultados em JSON."""
    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model": MODEL_BASE,
            "sft_adapter": str(SFT_ADAPTER),
            "dpo_adapter": str(DPO_ADAPTER),
            "nota": "Sem system message (diagnóstico mostrou que piora geração SFT/DPO). EOS inclui <end_of_turn>.",
        },
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Resultados salvos em: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark: Base vs SFT vs DPO (Formal + Periferia)")
    parser.add_argument("--output", type=str, default="results/benchmark.json")
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--stages", nargs="+", default=["base", "cpt", "sft", "dpo"],
                        choices=["base", "cpt", "sft", "dpo"])
    args = parser.parse_args()

    output_path = PROJECT_ROOT / args.output

    total_q = len(QUESTIONS_FORMAL) + len(QUESTIONS_PERIFERIA)
    print("📊 Benchmark: Base vs SFT vs SFT+DPO")
    print(f"   Modelo: {MODEL_BASE}")
    print(f"   Perguntas: {total_q} ({len(QUESTIONS_FORMAL)} formais + {len(QUESTIONS_PERIFERIA)} periferia)")
    print(f"   Estágios: {', '.join(args.stages)}")
    print(f"   Max tokens: {args.max_tokens}")
    print(f"   System message: NÃO (diagnóstico mostrou que piora)")
    print(f"   EOS tokens: <eos> + <end_of_turn>")
    print()

    results = run_benchmark(QUESTIONS_FORMAL, QUESTIONS_PERIFERIA, args.stages, args.max_tokens)
    print_comparison(results)
    save_results(results, output_path)


if __name__ == "__main__":
    main()
