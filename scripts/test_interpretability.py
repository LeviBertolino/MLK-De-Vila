#!/usr/bin/env python3
"""Teste de interpretabilidade: frases 100% gírias em cada estágio do modelo."""

import sys, json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from mlx_lm import load, generate

MODEL_BASE = "mlx-community/gemma-3-1b-it-bf16"
MODEL_CPT = str(PROJECT_ROOT / "models" / "gemma-3-1b-cpt")
MODEL_SFT_MERGED = str(PROJECT_ROOT / "models" / "sft_merged")
SFT_ADAPTER = str(PROJECT_ROOT / "adapters" / "sft")
DPO_ADAPTER = str(PROJECT_ROOT / "adapters" / "dpo")

STAGES = {
    "base": lambda: load(MODEL_BASE),
    "cpt": lambda: load(MODEL_CPT),
    "sft": lambda: load(MODEL_CPT, adapter_path=SFT_ADAPTER),
    "dpo": lambda: load(MODEL_SFT_MERGED, adapter_path=DPO_ADAPTER),
}

# Frases-teste com gírias profundas
TEST_PHRASES = [
    {
        "phrase": "Parça, to colando para goma, hoje o bagulho tá loko, se pá, vou dá a visão para minha coroa!",
        "meaning": "Amigo, estou indo para o trabalho, hoje as coisas estão difíceis, talvez eu vá informar minha mãe!",
        "question": "O que essa pessoa está dizendo? Explica pra mim o que ela quer dizer com essa frase: 'Parça, to colando para goma, hoje o bagulho tá loko, se pá, vou dá a visão para minha coroa!'",
    },
    {
        "phrase": "Mano, o bonde todo tá no corre, mas a firma tá de grau, se vacilá o bicho pega!",
        "meaning": "Amigo, todo o grupo está trabalhando/correndo atrás, mas o trabalho/emprego está bom, se vacilar vai ter problema!",
        "question": "Me explica essa frase como se eu não soubesse gírias: 'Mano, o bonde todo tá no corre, mas a firma tá de grau, se vacilá o bicho pega!'",
    },
    {
        "phrase": "A quebrada tá embaçada, o fluxo secou, tô pensando em fazer um bico pra não ficar na pindaíba.",
        "meaning": "A comunidade está difícil, o dinheiro acabou, estou pensando em fazer um trabalho informal para não ficar sem dinheiro.",
        "question": "Traduz essa frase pra linguagem formal: 'A quebrada tá embaçada, o fluxo secou, tô pensando em fazer um bico pra não ficar na pindaíba.'",
    },
]


def clean_response(text: str) -> str:
    import re
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.replace("<end_of_turn>", "").replace("<pad>", "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def generate_response(model, tokenizer, question: str, max_tokens: int = 400) -> str:
    messages = [{"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    response = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens)
    return clean_response(response)


def main():
    results = []

    for stage_name, loader in STAGES.items():
        print(f"\n{'='*60}")
        print(f"  Carregando estágio: {stage_name.upper()}")
        print(f"{'='*60}")

        model, tokenizer = loader()
        tokenizer.add_eos_token("<end_of_turn>")

        for test in TEST_PHRASES:
            print(f"\n  📝 Frase: {test['phrase'][:60]}...")
            response = generate_response(model, tokenizer, test["question"])
            print(f"  💬 Resposta: {response[:150]}...")

            results.append({
                "stage": stage_name,
                "phrase": test["phrase"],
                "meaning": test["meaning"],
                "question": test["question"],
                "response": response,
            })

        del model, tokenizer

    # Salvar
    output_path = PROJECT_ROOT / "results" / "interpretability_test.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n\n{'='*60}")
    print(f"  💾 Resultados salvos em: {output_path}")
    print(f"{'='*60}")

    # Print summary
    for test in TEST_PHRASES:
        print(f"\n{'─'*60}")
        print(f"  FRASE: {test['phrase']}")
        print(f"  SIGNIFICADO REAL: {test['meaning']}")
        for stage_name in STAGES:
            r = [x for x in results if x["stage"] == stage_name and x["phrase"] == test["phrase"]][0]
            print(f"\n  [{stage_name.upper()}]:")
            print(f"  {r['response'][:300]}")


if __name__ == "__main__":
    main()
