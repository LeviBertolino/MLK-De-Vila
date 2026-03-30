"""
Script para preparar dados de Continuous Pre-Training (CPT).

Converte todas as fontes de texto bruto (transcrições YouTube, artigos,
gírias, expressões, situações do cotidiano) em formato JSONL para
treinamento de next-token prediction.

O CPT ensina o modelo a FALAR como a periferia antes do SFT ensinar
O QUE falar.

Uso:
    python scripts/00_prepare_cpt_data.py
"""

import json
import re
import random
from pathlib import Path

# ============================================================
# CONFIGURAÇÃO
# ============================================================

PROJECT_ROOT = Path(__file__).parent.parent
YOUTUBE_DIR = PROJECT_ROOT / "data" / "raw" / "youtube"
ARTIGOS_DIR = PROJECT_ROOT / "data" / "raw" / "artigos"
CPT_OUTPUT_DIR = PROJECT_ROOT / "data" / "cpt"

TRAIN_SPLIT = 0.90  # 90% treino, 10% validação

# ============================================================
# PARSERS
# ============================================================

def parse_vtt(filepath: Path) -> str:
    """Extrai texto limpo de arquivo VTT (legendas YouTube)."""
    text_lines = []
    seen_lines = set()

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # Pular cabeçalhos VTT, timestamps e linhas vazias
            if not line:
                continue
            if line.startswith("WEBVTT") or line.startswith("Kind:") or line.startswith("Language:"):
                continue
            if re.match(r"\d{2}:\d{2}:\d{2}\.\d{3}", line):
                continue
            if "align:" in line or "position:" in line:
                continue

            # Limpar tags de timing inline
            clean = re.sub(r"<\d{2}:\d{2}:\d{2}\.\d{3}>", "", line)
            clean = re.sub(r"</?c>", "", clean)
            clean = clean.strip()

            if clean and clean not in seen_lines:
                seen_lines.add(clean)
                text_lines.append(clean)

    # Juntar em texto corrido
    full_text = " ".join(text_lines)
    # Limpar espaços múltiplos
    full_text = re.sub(r"\s+", " ", full_text).strip()
    return full_text


def parse_markdown(filepath: Path) -> str:
    """Extrai texto de arquivo Markdown."""
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    # Remover headers markdown mas manter o texto
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Remover bold/italic markers
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
    # Remover links markdown [text](url)
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    # Limpar linhas horizontais
    text = re.sub(r"^---+$", "", text, flags=re.MULTILINE)
    # Limpar espaços excessivos
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ============================================================
# TEXTOS ADICIONAIS DE CULTURA DA PERIFERIA
# ============================================================

SITUACOES_COTIDIANO = [
    # Rotina matinal
    "Acordei cinco da manhã, tomei um rango rápido e saí correndo pro ponto. O busão tava lotado, mó jet até o centro. Cheguei no trampo atrasado, o patrão ficou pistola. Papo reto, todo dia é essa correria.",

    # Rolê com os parceiros
    "Sábado à noite o bonde se juntou na esquina. Tinha um fluxo lá embaixo, a revoada tava braba. O mano chegou todo mandrake, de pisante novo e kit pesado. Cola lá na goma do Dudu primeiro, depois a gente marcha e boa.",

    # Problemas com grana
    "Tô sem grana nenhuma, mano. O bico que eu tava fazendo acabou, e o trampo novo só paga mês que vem. Tô comendo rango na casa da coroa enquanto não dá pra se virar. Se pa vou ter que correr atrás de outro bico essa semana. O bagulho tá embaçado.",

    # Enquadro policial
    "Ontem o mano tomou um enquadro voltando do trampo. Os polícia mandou ele encostar no muro, revirou tudo. Ele ficou bolado, não tinha feito nada. Na quebrada é assim, se pa você tá de boa e do nada aparece viatura. Fica ligado, não dá pala, segue reto.",

    # Fofoca e conflito
    "A vizinha tava metendo um caô danado, falando que eu dei mole. Papo torto dela direto. Soltei a real na cara: pode pá, não tô pra treta não, mas se continuar sem visão vai dar bololô. O bonde ficou sabendo e já ligou nóis: tranquilo, é fechamento contigo.",

    # Procurando trampo
    "Faz duas semanas que tô correndo atrás de trampo. Fui num lugar lá no centro, o cara falou que ia ligar. Nunca ligou. Se pa vou fazer uns bico por enquanto, pintura, frete, o que aparecer. Não dá pra ficar parado, o rango não cai do céu. O mano lá da oficina falou que se pa tem uma vaga, liga nóis se rolar.",

    # Solidariedade na comunidade
    "Quando a mãe da Jessi ficou doente, a quebrada inteira se juntou. Cada um deu uma merreca, juntou rango, um mano emprestou o cavalo pra levar ela no hospital. É assim que funciona: quando um cai, o bonde levanta. Não tem 0800 do governo que resolva, então a gente resolve entre nóis.",

    # Educação financeira da quebrada
    "Fica ligado mano, quando tu recebe o salário, a primeira coisa é separar a grana do aluguel e das contas. O que sobrar, tenta guardar pelo menos um pouquinho. Eu sei que é difícil, a correria é todo dia, mas se tu não separar, o dinheiro some e tu nem sabe pra onde foi. Passa a visão pro seus parceiro também.",

    # Empreendendo na quebrada
    "O Marcão abriu uma barbearia no beco da quebrada. Começou com uma máquina e uma cadeira emprestada. Hoje tem três cadeiras e dois funcionários. A fita é: ele não esperou ter grana pra começar, foi na raça. Correu atrás de cliente, fez preço justo, e o bonde todo começou a cortar lá. Na quebrada, boca a boca é o melhor marketing.",

    # Sobre economizar
    "Mano, economizar ganhando um salário mínimo é embaçado, mas não é impossível. A fita é: anota tudo que tu gasta no mês. Tudo mesmo. Aí tu vai ver que tem um monte de grana indo embora em coisa que tu nem precisa. Corta o supérfluo e guarda o que sobrar. Pode ser dez conto, vinte conto, o que for. O importante é criar o hábito.",

    # Juros e dívidas
    "Parceiro, cartão de crédito é veneno se tu não souber usar. O cara gasta trezentos conto no cartão achando que tá de boa, mas quando chega a fatura com os juros, o bagulho tá embaçado. Fica ligado: se não pode pagar à vista, pensa duas vezes antes de parcelar. Juros composto é o bicho, ele come tua grana sem tu perceber.",

    # Investimento para iniciantes
    "Se pa tu acha que investir é coisa de rico, mas não é não, mano. Dá pra começar com cinquenta conto no Tesouro Direto. A fita é: em vez de deixar a grana parada na conta, tu coloca num lugar que rende. É pouco? É. Mas é melhor que nada. Com o tempo, aquele trocado vai crescendo e tu nem percebe.",

    # Sobre trabalho e renda
    "Na quebrada, a maioria faz bico pra complementar a renda. Tem gente que faz frete, vende bala no farol, trabalha de Uber, faz unha em casa. A correria é real. O trampo formal paga pouco e é difícil de conseguir. Mas o que importa é não ficar parado. Corre atrás, mano, que oportunidade aparece pra quem tá se movendo.",

    # Sobre moradia e custos
    "O aluguel na quebrada é mais barato que no centro, mas mesmo assim pesa no bolso. O cara ganha um salário mínimo e mete metade no aluguel. Sobra o quê? Quase nada. Por isso que muita gente divide casa, mora com a coroa, ou faz um puxadinho nos fundos. A fita é: moradia come a maior parte da grana.",

    # Sobre educação
    "Estudar morando na quebrada é correria dobrada. Tu acorda cedo, pega busão lotado, chega no trampo, volta tarde, e ainda tem que abrir o caderno. Mas é o caminho, mano. Estudo é investimento. Não é fácil, mas é o que pode mudar tua vida. Fica ligado nas bolsas de estudo, nos cursos online gratuitos. Tem muita coisa de graça na internet.",

    # Planejamento financeiro
    "Passa a visão: todo mês tu precisa saber exatamente quanto entra e quanto sai. Pega um caderno ou usa o celular mesmo. Anota: salário, bico, qualquer grana extra. Depois anota: aluguel, luz, água, gás, rango, transporte. O que sobrar é teu. Se não sobrar nada, tu precisa cortar alguma coisa. Papo reto, sem controle tu não sai do lugar.",
]

GIRIAS_EM_CONTEXTO = [
    "Mano, o trampo tá pesado mas a grana tá entrando. Fica ligado que mês que vem tem aumento.",
    "Fiz um bico de frete no fim de semana e consegui uma grana extra. Vou guardar pra emergência.",
    "A fita é: se tu não correr atrás, ninguém vai fazer por tu. Na quebrada é cada um por si, mas o bonde se ajuda.",
    "Tava embaçado o negócio, mas o parceiro me passou a visão e eu consegui resolver.",
    "O bagulho ficou moiado quando o cara meteu o preço lá em cima. Aí procurei outro fornecedor.",
    "Seloko, o preço do gás subiu de novo! A coroa ficou pistola quando viu a conta.",
    "Pode pá que investir é o caminho. Mesmo com merreca dá pra começar. O negócio é ter disciplina.",
    "Mano, não vai na onda de quem fala que é 0800 investir sem estudar. Tem que se ligar antes.",
    "O bonde todo tá correndo atrás de trampo. A quebrada tá difícil mas a gente não desiste.",
    "Tô bolado com essa dívida do cartão. Os juros tão comendo minha grana toda. Preciso quitar isso.",
    "A correria é todo dia, mano. Acordo cedo, pego o busão lotado, trabalho o dia todo. Mas é isso, né? Tem que correr atrás.",
    "Cria da quebrada sabe o valor do dinheiro. Desde moleque aprende que nada vem fácil.",
    "O fechamento do mês deu positivo! Consegui guardar duzentos conto. Tô no caminho certo.",
    "Soltar a real: se tu tá gastando mais do que ganha, tu tá na roubada. Para e reorganiza.",
    "Papo reto, parceiro: orçamento não é coisa de rico. É coisa de quem quer sair do sufoco.",
    "Brotar na padaria todo dia pra tomar café tá custando uma nota. Melhor fazer em casa e economizar.",
    "A nave do mano quebrou e ele ficou sem como ir pro trampo. Por isso que reserva de emergência é importante.",
    "Liga nóis quando tiver novidade do curso de finanças. Quero aprender a investir direito.",
    "Rango tá caro demais no mercado. A saída é pesquisar preço e comprar no atacadão.",
    "Se pa o melhor investimento pra quem tá começando é o Tesouro Direto. É seguro e rende mais que a poupança.",
]


# ============================================================
# MAIN
# ============================================================

def main():
    print("📦 Preparando dados para Continuous Pre-Training (CPT)")
    print()

    all_texts = []

    # 1. Transcrições YouTube
    vtt_files = sorted(YOUTUBE_DIR.glob("*.vtt"))
    print(f"📹 {len(vtt_files)} transcrições YouTube encontradas")
    for vtt in vtt_files:
        text = parse_vtt(vtt)
        if len(text) > 100:  # Pular transcrições muito curtas
            all_texts.append(text)
            print(f"   ✅ {vtt.name[:60]}... ({len(text)} chars)")
        else:
            print(f"   ⚠️  {vtt.name[:60]}... (muito curto, pulando)")

    # 2. Artigos Markdown
    md_files = sorted(ARTIGOS_DIR.glob("*.md"))
    print(f"\n📄 {len(md_files)} artigos encontrados")
    for md in md_files:
        text = parse_markdown(md)
        if len(text) > 100:
            all_texts.append(text)
            print(f"   ✅ {md.name[:60]}... ({len(text)} chars)")

    # 3. Situações do cotidiano
    print(f"\n🏘️  {len(SITUACOES_COTIDIANO)} situações do cotidiano")
    for sit in SITUACOES_COTIDIANO:
        all_texts.append(sit)

    # 4. Gírias em contexto
    print(f"💬 {len(GIRIAS_EM_CONTEXTO)} frases com gírias em contexto")
    for g in GIRIAS_EM_CONTEXTO:
        all_texts.append(g)

    print(f"\n📊 Total: {len(all_texts)} textos")

    # Quebrar textos longos em chunks de ~500 tokens (~2000 chars)
    MAX_CHARS = 2000
    chunks = []
    for text in all_texts:
        if len(text) <= MAX_CHARS:
            chunks.append(text)
        else:
            # Quebrar em parágrafos
            paragraphs = text.split("\n\n")
            current_chunk = ""
            for para in paragraphs:
                if len(current_chunk) + len(para) + 2 <= MAX_CHARS:
                    current_chunk += ("\n\n" + para if current_chunk else para)
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = para
            if current_chunk:
                chunks.append(current_chunk)

            # Se ainda tem chunks muito grandes, quebrar por sentenças
            final_chunks = []
            for chunk in chunks[-len(paragraphs):]:  # Só os recém-adicionados
                if len(chunk) > MAX_CHARS:
                    sentences = re.split(r'(?<=[.!?])\s+', chunk)
                    sub_chunk = ""
                    for sent in sentences:
                        if len(sub_chunk) + len(sent) + 1 <= MAX_CHARS:
                            sub_chunk += (" " + sent if sub_chunk else sent)
                        else:
                            if sub_chunk:
                                final_chunks.append(sub_chunk)
                            sub_chunk = sent
                    if sub_chunk:
                        final_chunks.append(sub_chunk)
                else:
                    final_chunks.append(chunk)

    # Filtrar chunks muito curtos
    chunks = [c for c in chunks if len(c) > 50]

    # Duplicar textos com gírias (data augmentation) — mais peso no dialeto
    giria_chunks = [c for c in chunks if any(g in c.lower() for g in
        ["mano", "trampo", "grana", "bico", "quebrada", "papo reto",
         "fica ligado", "correria", "embaçado", "bagulho", "correr atrás",
         "parceiro", "treta", "bolado", "se pa", "pode pá"])]

    print(f"📦 {len(chunks)} chunks totais")
    print(f"🏘️  {len(giria_chunks)} chunks com gírias (serão duplicados)")

    # Adicionar duplicatas dos chunks com gírias
    all_chunks = chunks + giria_chunks  # 2x peso nos textos com gírias
    random.seed(42)
    random.shuffle(all_chunks)

    print(f"📊 {len(all_chunks)} chunks finais (com augmentation)")

    # Salvar em formato JSONL
    CPT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    split_idx = int(len(all_chunks) * TRAIN_SPLIT)
    train_chunks = all_chunks[:split_idx]
    valid_chunks = all_chunks[split_idx:]

    train_path = CPT_OUTPUT_DIR / "train.jsonl"
    valid_path = CPT_OUTPUT_DIR / "valid.jsonl"

    with open(train_path, "w", encoding="utf-8") as f:
        for chunk in train_chunks:
            f.write(json.dumps({"text": chunk}, ensure_ascii=False) + "\n")

    with open(valid_path, "w", encoding="utf-8") as f:
        for chunk in valid_chunks:
            f.write(json.dumps({"text": chunk}, ensure_ascii=False) + "\n")

    print(f"\n✅ Dados CPT salvos:")
    print(f"   train: {len(train_chunks)} textos → {train_path}")
    print(f"   valid: {len(valid_chunks)} textos → {valid_path}")

    # Stats
    total_chars = sum(len(c) for c in all_chunks)
    print(f"\n📊 Estatísticas:")
    print(f"   Total de caracteres: {total_chars:,}")
    print(f"   Média por chunk: {total_chars // len(all_chunks):,} chars")
    print(f"   Estimativa tokens: ~{total_chars // 4:,}")


if __name__ == "__main__":
    main()
