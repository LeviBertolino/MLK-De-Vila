#!/usr/bin/env python3
"""
Mlk de Vila — Educador Financeiro da Periferia
Demo interativa com Gradio para testar o modelo fine-tuned.
"""

import re
import gradio as gr
from pathlib import Path
from mlx_lm import load, generate

PROJECT_ROOT = Path(__file__).resolve().parent

# Model paths
MODEL_SFT_MERGED = str(PROJECT_ROOT / "models" / "sft_merged")
DPO_ADAPTER = str(PROJECT_ROOT / "adapters" / "dpo")

# Load model once at startup
print("Carregando modelo Mlk de Vila (DPO)...")
model, tokenizer = load(MODEL_SFT_MERGED, adapter_path=DPO_ADAPTER)
tokenizer.add_eos_token("<end_of_turn>")
print("Modelo carregado!")


def clean_response(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.replace("<end_of_turn>", "").replace("<pad>", "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chat(message: str, history: list[dict]) -> str:
    messages = [{"role": h["role"], "content": h["content"]} for h in history]
    messages.append({"role": "user", "content": message})

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    response = generate(model, tokenizer, prompt=prompt, max_tokens=500)
    return clean_response(response)


EXAMPLES = [
    "Mano, o que é essa tal de inflação? Tá comendo minha grana?",
    "Parceiro, como faz pra guardar uns trocado ganhando um salário mínimo?",
    "Passa a visão sobre juros compostos, como funciona essa fita?",
    "Mano, dá pra começar a investir com 50 conto? Como faz?",
    "Ae parceiro, esse bagulho de grana é embaçado, essa parada de inflação engole a grana. Na moral, preciso pegar a visão de investir.",
    "O que é reserva de emergência? Por que o pessoal da quebrada precisa disso?",
    "Fica ligado, vale a pena pegar empréstimo pra quitar uma dívida?",
    "Como organizar a grana do mês de um jeito simples? Tô perdido, mano.",
]

demo = gr.ChatInterface(
    fn=chat,
    title="Mlk de Vila",
    description=(
        "**Educador financeiro que fala a língua da periferia.**\n\n"
        "Modelo Gemma 3 1B IT fine-tuned com CPT + SFT + DPO para educação "
        "financeira no dialeto da periferia brasileira. Treinado inteiramente "
        "em hardware com 8GB de memória.\n\n"
        "Manda a pergunta no dialeto ou em português formal — o Mlk de Vila responde."
    ),
    examples=EXAMPLES,
)

if __name__ == "__main__":
    demo.launch()
