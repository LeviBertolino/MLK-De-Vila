"""
Script de diagnóstico para identificar por que SFT/DPO geram <end_of_turn><pad>.

Testa:
1. Quais tokens são EOS (end-of-sequence) para o tokenizer
2. Se <end_of_turn> está nos EOS tokens
3. O prompt exato gerado pelo chat template
4. Os primeiros tokens gerados pelo modelo SFT
5. Comparação: com e sem <end_of_turn> como EOS token
"""

import re
from pathlib import Path
from mlx_lm import load, generate

PROJECT_ROOT = Path(__file__).parent.parent
SFT_ADAPTER = PROJECT_ROOT / "adapters" / "sft"
MODEL_BASE = "mlx-community/gemma-3-1b-it-bf16"

SYSTEM_MESSAGE = (
    "Você é o 'Mlk de Vila', um educador financeiro que fala a língua da periferia. "
    "Explica finanças de forma simples, direta e usando exemplos do dia a dia. "
    "Usa gírias como mano, parceiro, trampo, grana, padoca, bico. "
    "É motivador e honesto."
)

TEST_QUESTIONS = [
    "O que é inflação e como ela afeta meu dinheiro?",           # funciona no SFT
    "Como economizar ganhando um salário mínimo?",               # falha no SFT
    "Mano, o que é essa tal de inflação? Tá comendo minha grana?",  # falha no SFT
]


def diagnose():
    print("=" * 70)
    print("DIAGNÓSTICO DO BENCHMARK")
    print("=" * 70)

    # 1. Carregar modelo base e verificar tokenizer
    print("\n[1] Carregando modelo BASE...")
    model_base, tok_base = load(MODEL_BASE)

    print(f"\n   EOS token IDs: {tok_base.eos_token_ids}")

    # Verificar token IDs importantes
    vocab = tok_base._tokenizer.get_vocab()
    end_of_turn_id = vocab.get("<end_of_turn>", None)
    eos_id = vocab.get("<eos>", None)
    pad_id = vocab.get("<pad>", None)
    print(f"   <end_of_turn> ID: {end_of_turn_id}")
    print(f"   <eos> ID: {eos_id}")
    print(f"   <pad> ID: {pad_id}")
    print(f"   <end_of_turn> in eos_token_ids? {end_of_turn_id in tok_base.eos_token_ids}")
    print(f"   Tokenizer has thinking? {tok_base.has_thinking}")

    # 2. Verificar formato do prompt
    print("\n[2] Formato do prompt com chat template:")
    messages_with_sys = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": TEST_QUESTIONS[0]},
    ]
    messages_no_sys = [
        {"role": "user", "content": TEST_QUESTIONS[0]},
    ]

    prompt_with = tok_base.apply_chat_template(
        messages_with_sys, tokenize=False, add_generation_prompt=True,
    )
    prompt_without = tok_base.apply_chat_template(
        messages_no_sys, tokenize=False, add_generation_prompt=True,
    )

    print(f"\n   COM system message:")
    print(f"   ---")
    print(f"   {repr(prompt_with)}")
    print(f"   ---")
    print(f"\n   SEM system message:")
    print(f"   ---")
    print(f"   {repr(prompt_without)}")
    print(f"   ---")

    # 3. Verificar tokenização do prompt (string → tokens vs direto)
    print("\n[3] Comparação de tokenização:")
    tokens_from_string = tok_base._tokenizer.encode(prompt_with, add_special_tokens=False)
    tokens_direct = tok_base.apply_chat_template(
        messages_with_sys, tokenize=True, add_generation_prompt=True,
    )

    print(f"   Tokens via string (encode):        {len(tokens_from_string)} tokens")
    print(f"   Tokens via chat_template(tokenize): {len(tokens_direct)} tokens")
    print(f"   São iguais? {tokens_from_string == tokens_direct}")

    if tokens_from_string != tokens_direct:
        print(f"\n   DIFERENÇAS ENCONTRADAS!")
        for i, (a, b) in enumerate(zip(tokens_from_string, tokens_direct)):
            if a != b:
                print(f"   Pos {i}: string={a} ({tok_base._tokenizer.decode([a])!r}) vs direct={b} ({tok_base._tokenizer.decode([b])!r})")
        if len(tokens_from_string) != len(tokens_direct):
            print(f"   Comprimento diferente: {len(tokens_from_string)} vs {len(tokens_direct)}")

    # 4. Gerar com modelo BASE (deve funcionar)
    print("\n[4] Geração com modelo BASE (sem adapter):")
    for q in TEST_QUESTIONS:
        messages = [{"role": "user", "content": q}]
        prompt = tok_base.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        resp = generate(model_base, tok_base, prompt=prompt, max_tokens=50)
        preview = resp[:100].replace("\n", "\\n")
        print(f"   Q: {q[:50]}...")
        print(f"   R: {preview}")
        print()

    del model_base
    import gc; gc.collect()

    # 5. Carregar SFT e testar
    print("\n[5] Carregando modelo SFT (base + adapter)...")
    model_sft, tok_sft = load(MODEL_BASE, adapter_path=str(SFT_ADAPTER))

    print(f"   EOS token IDs (SFT): {tok_sft.eos_token_ids}")

    print("\n   Geração SFT SEM system message:")
    for q in TEST_QUESTIONS:
        messages = [{"role": "user", "content": q}]
        prompt = tok_sft.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        resp = generate(model_sft, tok_sft, prompt=prompt, max_tokens=50)
        preview = resp[:100].replace("\n", "\\n")
        print(f"   Q: {q[:50]}...")
        print(f"   R: {preview}")
        print()

    print("\n   Geração SFT COM system message:")
    for q in TEST_QUESTIONS:
        messages = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": q},
        ]
        prompt = tok_sft.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        resp = generate(model_sft, tok_sft, prompt=prompt, max_tokens=50)
        preview = resp[:100].replace("\n", "\\n")
        print(f"   Q: {q[:50]}...")
        print(f"   R: {preview}")
        print()

    # 6. Testar com <end_of_turn> como EOS token
    print("\n[6] Geração SFT COM system message + <end_of_turn> como EOS:")
    tok_sft.add_eos_token("<end_of_turn>")
    print(f"   EOS token IDs agora: {tok_sft.eos_token_ids}")

    for q in TEST_QUESTIONS:
        messages = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": q},
        ]
        prompt = tok_sft.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        resp = generate(model_sft, tok_sft, prompt=prompt, max_tokens=300)
        preview = resp[:200].replace("\n", "\\n")
        print(f"   Q: {q[:50]}...")
        print(f"   R: {preview}")
        print(f"   Len: {len(resp)} chars")
        print()

    # 7. Testar com temperature mais alta
    print("\n[7] Geração SFT COM system + EOS fix + temp=0.7:")
    for q in TEST_QUESTIONS:
        messages = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": q},
        ]
        prompt = tok_sft.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        resp = generate(model_sft, tok_sft, prompt=prompt, max_tokens=300, temp=0.7)
        preview = resp[:200].replace("\n", "\\n")
        print(f"   Q: {q[:50]}...")
        print(f"   R: {preview}")
        print(f"   Len: {len(resp)} chars")
        print()

    print("\n" + "=" * 70)
    print("DIAGNÓSTICO COMPLETO")
    print("=" * 70)


if __name__ == "__main__":
    diagnose()
