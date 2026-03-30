#!/usr/bin/env python3
"""Teste de compreensão dialetal: frase natural da periferia em cada estágio."""

import sys, json, re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
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

PHRASE = "Ae parceiro, to no busão chegando na quebrada, mano esse bagulho de grana é embaçado, você dá mó trampo e essa parada de inflação engole sua grana. Na moral, papo reto memo, você precisa pegar a visão de investir. Chegando ai, nois troca ideia."


def clean_response(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.replace("<end_of_turn>", "").replace("<pad>", "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def generate_response(model, tokenizer, question: str, max_tokens: int = 500) -> str:
    messages = [{"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    response = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens)
    return clean_response(response)


def main():
    results = []

    for stage_name, loader in STAGES.items():
        print(f"\n{'='*70}")
        print(f"  {stage_name.upper()}")
        print(f"{'='*70}")

        model, tokenizer = loader()
        tokenizer.add_eos_token("<end_of_turn>")

        response = generate_response(model, tokenizer, PHRASE)
        print(f"\n{response}\n")

        results.append({"stage": stage_name, "phrase": PHRASE, "response": response})
        del model, tokenizer

    output_path = PROJECT_ROOT / "results" / "dialect_comprehension_test.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Salvo em: {output_path}")


if __name__ == "__main__":
    main()
