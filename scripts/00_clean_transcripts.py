"""
Limpa transcrições VTT do YouTube e converte em texto puro.
Salva textos limpos na pasta data/raw/transcricoes/

Uso:
    python scripts/00_clean_transcripts.py
"""

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
VTT_DIR = PROJECT_ROOT / "data" / "raw" / "youtube"
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "transcricoes"


def clean_vtt(vtt_text: str) -> str:
    """Remove timestamps, tags e duplicatas de um arquivo VTT."""
    lines = vtt_text.split("\n")
    clean_lines = []
    seen = set()

    for line in lines:
        # Pular headers VTT
        if line.startswith("WEBVTT") or line.startswith("Kind:") or line.startswith("Language:"):
            continue
        # Pular timestamps
        if re.match(r"^\d{2}:\d{2}:\d{2}", line):
            continue
        # Pular linhas com align/position
        if "align:" in line or "position:" in line:
            continue
        # Remover tags HTML/VTT inline
        line = re.sub(r"<[^>]+>", "", line)
        # Remover [Música], [Aplausos], etc.
        line = re.sub(r"\[.*?\]", "", line)

        line = line.strip()
        if not line:
            continue

        # Evitar linhas duplicadas consecutivas
        if line not in seen:
            clean_lines.append(line)
            seen.add(line)
        # Reset seen a cada 5 linhas pra permitir repetições naturais
        if len(seen) > 50:
            seen.clear()

    # Juntar em parágrafos (frases curtas do VTT viram texto corrido)
    text = " ".join(clean_lines)

    # Limpar espaços duplos
    text = re.sub(r"\s{2,}", " ", text)

    # Quebrar em parágrafos a cada ~300 chars no ponto mais próximo
    paragraphs = []
    while len(text) > 300:
        break_point = text[:350].rfind(" ")
        if break_point < 200:
            break_point = 300
        paragraphs.append(text[:break_point].strip())
        text = text[break_point:].strip()
    if text:
        paragraphs.append(text)

    return "\n\n".join(paragraphs)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    vtt_files = sorted(VTT_DIR.glob("*.vtt"))
    print(f"🎥 Limpando {len(vtt_files)} transcrições VTT...")
    print()

    total_chars = 0
    for i, vtt_file in enumerate(vtt_files, 1):
        content = vtt_file.read_text(encoding="utf-8")
        clean = clean_vtt(content)

        # Extrair nome limpo do episódio
        name = vtt_file.stem.replace(".pt", "")
        # Normalizar caracteres unicode estranhos do YouTube
        name = name.replace("？", "?").replace("：", ":").replace("｜", "|")

        output_file = OUTPUT_DIR / f"{name}.txt"
        output_file.write_text(clean, encoding="utf-8")

        total_chars += len(clean)
        if i % 20 == 0 or i == len(vtt_files):
            print(f"   [{i}/{len(vtt_files)}] processados")

    print()
    print(f"✅ {len(vtt_files)} transcrições limpas")
    print(f"📊 Total: {total_chars:,} caracteres (~{total_chars // 4:,} tokens)")
    print(f"📁 Salvos em: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
