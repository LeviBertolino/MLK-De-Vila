"""
Script V2 para gerar dataset SFT com dados culturais ricos.

Usa gírias, expressões, transcrições do Guetonomia e artigos do Mlk de Vila
para gerar pares instrução/resposta com uso CORRETO das gírias no contexto certo.

Uso:
    python scripts/01_generate_sft_data.py --provider mlx --model mlx-community/Qwen3-4B-4bit
"""

import json
import re
import random
import argparse
from pathlib import Path

# ============================================================
# CONFIGURAÇÃO
# ============================================================

PROJECT_ROOT = Path(__file__).parent.parent
ARTIGOS_DIR = PROJECT_ROOT / "data" / "raw" / "artigos"
TRANSCRICOES_DIR = PROJECT_ROOT / "data" / "raw" / "transcricoes"
SFT_OUTPUT = PROJECT_ROOT / "data" / "instruction" / "sft_dataset.jsonl"

# ============================================================
# DADOS CULTURAIS
# ============================================================

# Gírias com significado correto — o modelo PRECISA saber o que cada uma significa
GIRIAS_CONTEXTO = [
    {"giria": "trampo", "significa": "trabalho", "exemplo": "acabei de chegar do trampo"},
    {"giria": "grana", "significa": "dinheiro", "exemplo": "tô sem grana esse mês"},
    {"giria": "mano", "significa": "amigo ou vocativo", "exemplo": "mano, me ajuda aí"},
    {"giria": "parceiro", "significa": "amigo próximo", "exemplo": "é nóis parceiro"},
    {"giria": "bico", "significa": "trabalho extra temporário", "exemplo": "vou fazer um bico no fim de semana"},
    {"giria": "quebrada", "significa": "bairro, periferia", "exemplo": "sou lá da quebrada da zona leste"},
    {"giria": "correr atrás", "significa": "se esforçar, buscar", "exemplo": "tem que correr atrás pra conseguir"},
    {"giria": "passa a visão", "significa": "dar conselho, explicar", "exemplo": "me passa a visão sobre investimento"},
    {"giria": "a fita é", "significa": "o assunto é, a questão é", "exemplo": "a fita é que juros come seu dinheiro"},
    {"giria": "fica ligado", "significa": "presta atenção", "exemplo": "fica ligado que vou te explicar"},
    {"giria": "moiado", "significa": "deu errado, sem chance", "exemplo": "queria comprar mas moiou, acabou a grana"},
    {"giria": "veneno", "significa": "sufoco, dificuldade", "exemplo": "passei um veneno pra pagar as contas"},
    {"giria": "0800", "significa": "de graça", "exemplo": "esse curso é 0800, aproveita"},
    {"giria": "nave", "significa": "carro", "exemplo": "juntei grana e comprei minha nave"},
    {"giria": "coroa", "significa": "mãe", "exemplo": "minha coroa me ensinou a economizar"},
    {"giria": "seloko", "significa": "expressão de surpresa", "exemplo": "seloko mano, o juros tá alto demais"},
    {"giria": "embaçado", "significa": "difícil, complicado", "exemplo": "tá embaçado pagar tudo com salário mínimo"},
    {"giria": "bonado", "significa": "com dinheiro", "exemplo": "daqui uns anos tu vai tá bonado"},
    {"giria": "brotar", "significa": "aparecer, surgir", "exemplo": "brota lá no curso de finanças"},
    {"giria": "ramelar", "significa": "vacilar, errar", "exemplo": "ramelei gastando tudo no cartão"},
    {"giria": "desacerto", "significa": "deu errado, problema", "exemplo": "foi um desacerto pegar empréstimo com agiota"},
    {"giria": "relíquia", "significa": "pessoa sábia, experiente", "exemplo": "meu pai é relíquia, sabe das coisas"},
    {"giria": "liga nois", "significa": "conta comigo", "exemplo": "precisa de ajuda? liga nois"},
    {"giria": "treta", "significa": "problema, confusão", "exemplo": "cartão de crédito sem controle é treta"},
    {"giria": "bagulho", "significa": "coisa, assunto", "exemplo": "esse bagulho de investir é bom demais"},
    {"giria": "aliado", "significa": "parceiro, amigo de confiança", "exemplo": "esse é meu aliado, a gente se conhece desde menó"},
    {"giria": "goma", "significa": "casa", "exemplo": "bora lá na minha goma"},
    {"giria": "banca", "significa": "grupo de amigos, turma", "exemplo": "cola com a banca que tá suave"},
    {"giria": "kit", "significa": "roupas, conjunto de roupas", "exemplo": "comprei um kit nervoso pro rolê"},
    {"giria": "pisante", "significa": "tênis", "exemplo": "se liga nesse pisante que eu comprei"},
    {"giria": "goró", "significa": "bebida alcoólica", "exemplo": "bora tomar um goró hoje"},
    {"giria": "treta", "significa": "problema, confusão", "exemplo": "cartão de crédito sem controle é treta"},
    {"giria": "trombar", "significa": "encontrar alguém", "exemplo": "vamos nos trombar no metrô às 18h"},
    {"giria": "busão", "significa": "ônibus", "exemplo": "vou pegar o busão das 7h pro trampo"},
    {"giria": "bandeco", "significa": "comida, refeição", "exemplo": "bora colar no bandeco"},
    {"giria": "caminhada", "significa": "situação, rumo de vida", "exemplo": "cada um na sua caminhada"},
    {"giria": "monstrão", "significa": "corajoso, pessoa admirável", "exemplo": "aquele ali é monstrão, corre atrás demais"},
    {"giria": "é quente", "significa": "é verdade", "exemplo": "pode acreditar, é quente!"},
    {"giria": "na humildade", "significa": "com respeito, fazendo favor", "exemplo": "na humildade, me explica esse negócio de investir"},
    {"giria": "salve", "significa": "cumprimento", "exemplo": "salve quebrada, boa noite pá nois"},
    {"giria": "mó fita", "significa": "situação difícil ou complicada", "exemplo": "tirar documento no centro é mó fita"},
    {"giria": "pistola", "significa": "irritado, com raiva", "exemplo": "fiquei pistola quando vi o juros do cartão"},
    {"giria": "osso", "significa": "difícil, complicado", "exemplo": "tá osso pagar todas as contas esse mês"},
    {"giria": "zuado", "significa": "mal, não estar bem", "exemplo": "tô zuado, gastei tudo no começo do mês"},
    {"giria": "menó", "significa": "jovem, adolescente, criança", "exemplo": "os menó precisam aprender sobre grana desde cedo"},
    {"giria": "mina", "significa": "mulher, namorada", "exemplo": "minha mina me ajudou a organizar as contas"},
    {"giria": "brecado", "significa": "impedido, paralisado", "exemplo": "tô brecado, sem grana pra investir esse mês"},
    {"giria": "na bala", "significa": "disposto, animado", "exemplo": "tô na bala pra aprender a investir"},
    {"giria": "se pa", "significa": "talvez, pode ser", "exemplo": "se pa eu vou no curso de finanças hoje"},
    {"giria": "chavoso", "significa": "estiloso no jeito da quebrada", "exemplo": "o mano é chavoso, sempre de kit novo"},
    {"giria": "bolado", "significa": "preocupado ou irritado", "exemplo": "tô bolado com essa dívida do cartão"},
    {"giria": "cria", "significa": "pessoa que nasceu e cresceu na comunidade", "exemplo": "ele é cria do morro, conhece todo mundo"},
    {"giria": "fechamento", "significa": "pessoa de confiança", "exemplo": "pode confiar nele, é fechamento"},
    {"giria": "bonde", "significa": "turma, grupo de amigos", "exemplo": "o bonde todo vai no curso de educação financeira"},
    {"giria": "papo reto", "significa": "falar direto e franco", "exemplo": "papo reto mano, tu precisa parar de gastar assim"},
    {"giria": "pode pá", "significa": "pode confiar, com certeza", "exemplo": "pode pá, mês que vem eu começo a investir"},
    {"giria": "rango", "significa": "comida, refeição", "exemplo": "gasta mais com rango do que com aluguel"},
    {"giria": "rolê", "significa": "passeio, saída", "exemplo": "em vez de gastar tudo no rolê, guarda um pouco"},
    {"giria": "merreca", "significa": "pouco dinheiro", "exemplo": "me pagaram uma merreca nesse bico"},
    {"giria": "correria", "significa": "o dia a dia de quem corre atrás", "exemplo": "a correria é todo dia, mas vale a pena"},
    {"giria": "dar pala", "significa": "deixar perceber, não saber disfarçar", "exemplo": "não vai dar pala que tá juntando grana"},
    {"giria": "soltar a real", "significa": "falar a verdade", "exemplo": "solta a real, quanto tu deve no cartão?"},
    {"giria": "mandrake", "significa": "estiloso, descolado", "exemplo": "chegou todo mandrake com pisante novo"},
    {"giria": "caô", "significa": "mentira ou tá tudo certo (depende do contexto)", "exemplo": "esse investimento milagroso é caô"},
]

# Personas — diferentes perspectivas da periferia
PERSONAS = [
    {
        "nome": "Jovem de 17 anos",
        "contexto": "Estudante do ensino médio, mora com a mãe, faz bico nos fins de semana. Quer entender de dinheiro mas nunca teve acesso a educação financeira.",
        "tom": "curioso, usa muita gíria, faz perguntas diretas"
    },
    {
        "nome": "Mãe trabalhadora de 35 anos",
        "contexto": "Trabalha como diarista, cuida de 2 filhos, mora na periferia de SP. Precisa fazer o dinheiro render e quer ensinar os filhos sobre finanças.",
        "tom": "prática, preocupada com o futuro dos filhos, direta"
    },
    {
        "nome": "Trabalhador de 25 anos",
        "contexto": "Trabalha de carteira assinada ganhando salário mínimo, faz bico de entregador. Quer sair das dívidas e começar a investir.",
        "tom": "determinado, ansioso pra mudar de vida, fala gíria"
    },
    {
        "nome": "Empreendedor de quebrada",
        "contexto": "Tem uma barbearia na comunidade, é MEI, quer crescer o negócio e organizar as finanças.",
        "tom": "ambicioso, prático, linguagem de rua"
    },
    {
        "nome": "Adolescente de 15 anos",
        "contexto": "Usa muito celular, vê influencers ostentando, não entende nada de dinheiro mas quer aprender.",
        "tom": "linguagem jovem, usa gírias atuais, faz perguntas simples"
    },
]

# Temas expandidos com contextos específicos da periferia
TEMAS_V2 = [
    # Básico — situações reais
    {"tema": "o que é poupança", "contexto_periferia": "guardando dinheiro no cofrinho vs conta no banco"},
    {"tema": "diferença entre poupar e investir", "contexto_periferia": "guardar debaixo do colchão vs fazer o dinheiro trabalhar"},
    {"tema": "o que é juros", "contexto_periferia": "juros do cartão, juros do agiota, juros que come a grana"},
    {"tema": "juros compostos", "contexto_periferia": "como R$ 50 por mês vira uma grana boa em 5 anos"},
    {"tema": "reserva de emergência", "contexto_periferia": "quando o trampo acaba, quando a geladeira quebra"},
    {"tema": "inflação", "contexto_periferia": "o arroz que custava 15 conto agora tá 25, o busão que subia todo ano"},
    {"tema": "guardar dinheiro no colchão", "contexto_periferia": "a grana parada perde valor, a inflação come"},
    {"tema": "educação financeira desde cedo", "contexto_periferia": "ensinar os menó sobre dinheiro desde a escola"},

    # Dívidas — situações do dia a dia
    {"tema": "como sair das dívidas", "contexto_periferia": "nome sujo no SPC, não consegue nem comprar fiado"},
    {"tema": "nome sujo e como limpar", "contexto_periferia": "não consegue abrir conta, não consegue financiar nada"},
    {"tema": "empréstimo fácil e agiota", "contexto_periferia": "propaganda no WhatsApp, agiota da esquina, armadilhas"},
    {"tema": "cartão de crédito", "contexto_periferia": "parcelou tudo, mínimo do cartão, rotativo comendo tudo"},
    {"tema": "negociar dívida com banco", "contexto_periferia": "feirão limpa nome, negociação pelo app"},
    {"tema": "cheque especial", "contexto_periferia": "entrou no vermelho sem perceber, juros absurdo"},
    {"tema": "consignado", "contexto_periferia": "desconta direto do salário, parece bom mas tem pegadinha"},
    {"tema": "diferença entre dívida boa e ruim", "contexto_periferia": "financiar curso vs financiar tênis caro"},

    # Renda — realidade da periferia
    {"tema": "renda extra", "contexto_periferia": "uber, ifood, bico de pintor, fazer truffa pra vender"},
    {"tema": "ideias de bico", "contexto_periferia": "lavar carro, vender bala no trem, fazer unha, cortar cabelo"},
    {"tema": "ser MEI", "contexto_periferia": "formalizar o bico, emitir nota, ter CNPJ"},
    {"tema": "precificar seu trabalho", "contexto_periferia": "quanto cobrar pelo corte, pela faxina, pelo frete"},
    {"tema": "empreender na quebrada", "contexto_periferia": "abrir uma barbearia, lanchonete, loja de roupa"},

    # Investimentos — acessível
    {"tema": "tesouro direto", "contexto_periferia": "emprestar dinheiro pro governo e ganhar juros"},
    {"tema": "CDB", "contexto_periferia": "emprestar pro banco e ganhar mais que a poupança"},
    {"tema": "renda fixa vs variável", "contexto_periferia": "o seguro e o arriscado, cada um no seu nível"},
    {"tema": "ações", "contexto_periferia": "ser dono de um pedacinho da empresa"},
    {"tema": "começar a investir com pouco", "contexto_periferia": "dá pra começar com R$ 30, R$ 50"},
    {"tema": "PIX e conta digital", "contexto_periferia": "banco pelo celular, sem taxa, sem fila"},

    # Comportamento — cultura e pressão social
    {"tema": "pressão pra gastar e ostentação", "contexto_periferia": "comprar tênis caro, iPhone parcelado, roupa de marca"},
    {"tema": "consumismo na quebrada", "contexto_periferia": "todo mundo ostentando, pressão dos amigos, redes sociais"},
    {"tema": "conversar sobre dinheiro com família", "contexto_periferia": "ajudar em casa sem se endividar, dividir contas"},
    {"tema": "ensinar criança sobre dinheiro", "contexto_periferia": "mesada, cofre, ensinar o valor do trabalho"},
    {"tema": "paciência nos investimentos", "contexto_periferia": "resultado não vem rápido, disciplina todo mês"},
    {"tema": "golpes financeiros", "contexto_periferia": "pirâmide do WhatsApp, golpe do Pix, investimento milagroso"},

    # Planejamento — metas reais
    {"tema": "orçamento mensal", "contexto_periferia": "anotar tudo no caderno ou app, saber pra onde vai a grana"},
    {"tema": "metas financeiras", "contexto_periferia": "juntar pra comprar a moto, fazer uma viagem, sair do aluguel"},
    {"tema": "aposentadoria", "contexto_periferia": "INSS, previdência, pensar no futuro quando for mais velho"},
    {"tema": "imprevisto financeiro", "contexto_periferia": "filho ficou doente, perdeu o trampo, enchente levou tudo"},
    {"tema": "como economizar no dia a dia", "contexto_periferia": "feira vs mercado, cozinhar em casa, economizar luz e água"},
    {"tema": "organizar salário quando recebe", "contexto_periferia": "separar contas, guardar antes de gastar, pagar o essencial primeiro"},

    # Novos — vindos da pesquisa de referências reais
    {"tema": "racismo financeiro", "contexto_periferia": "banco nega crédito pro preto, juros mais alto na quebrada, acesso desigual"},
    {"tema": "ostentação vs investimento", "contexto_periferia": "iPhone parcelado em 12x vs R$ 50 por mês no Tesouro, o que vale mais"},
    {"tema": "microcrédito na favela", "contexto_periferia": "pegar empréstimo pequeno pra investir no negócio, sem agiota"},
    {"tema": "Tesouro Selic pra iniciante", "contexto_periferia": "o Favelado Investidor começou com R$ 100, qualquer um pode"},
    {"tema": "conta digital gratuita", "contexto_periferia": "Nubank, C6, Inter — banco sem taxa pelo celular, sem fila"},
    {"tema": "separar CPF do CNPJ", "contexto_periferia": "quem tem MEI precisa separar a grana do negócio da grana pessoal"},
    {"tema": "Pix e segurança", "contexto_periferia": "golpe do Pix, limite noturno, como se proteger"},
    {"tema": "como funciona o SPC e Serasa", "contexto_periferia": "score, nome sujo, como consultar de graça"},
    {"tema": "feirão limpa nome", "contexto_periferia": "oportunidade de negociar dívida com desconto, como funciona"},
    {"tema": "empreendedorismo feminino na periferia", "contexto_periferia": "mãe solteira que faz bolo pra vender, manicure em casa, loja no Instagram"},
    {"tema": "fluxo de caixa do pequeno negócio", "contexto_periferia": "quanto entra e quanto sai da barbearia, da lanchonete"},
    {"tema": "como investir sendo menor de idade", "contexto_periferia": "conta com responsável, aprender cedo, cofre digital"},
    {"tema": "relação emocional com dinheiro", "contexto_periferia": "gastar quando tá triste, comprar pra se sentir bem, consumo emocional"},
    {"tema": "educação financeira antirracista", "contexto_periferia": "entender que o sistema é desigual mas dá pra jogar o jogo"},
    {"tema": "rap e educação financeira", "contexto_periferia": "Racionais falando de dinheiro, funk ostentação vs consciência financeira"},
    {"tema": "morar de aluguel vs casa própria", "contexto_periferia": "financiar pelo Minha Casa Minha Vida, juntar pra entrada, vale a pena?"},
    {"tema": "economia circular na quebrada", "contexto_periferia": "comprar do vizinho, brechó, reciclagem como renda"},
    {"tema": "como lidar com pressão dos amigos", "contexto_periferia": "todo mundo gastando e tu economizando, como manter o foco"},
    {"tema": "planejamento financeiro pra quem ganha por dia", "contexto_periferia": "trabalhador informal, diarista, bico sem salário fixo"},
    {"tema": "importância de ter um sonho financeiro", "contexto_periferia": "sair do aluguel, abrir negócio, dar conforto pra coroa"},

    # Vocabulário e dialeto da periferia
    {"tema": "por que a periferia tem expressões próprias", "contexto_periferia": "a quebrada criou sua própria linguagem como forma de identidade e resistência"},
    {"tema": "vocabulário da periferia paulista", "contexto_periferia": "gírias como trampo, grana, mano, quebrada — palavras que a Faria Lima não entende"},
    {"tema": "dialeto da favela", "contexto_periferia": "a forma como se fala na favela não é erro, é identidade cultural e resistência"},
    {"tema": "expressões da periferia no dia a dia", "contexto_periferia": "como as gírias aparecem naturalmente nas conversas sobre dinheiro, trabalho e vida"},
    {"tema": "o que significa ser da quebrada", "contexto_periferia": "quebrada não é só lugar, é identidade, é comunidade, é modo de vida"},
    {"tema": "como a periferia fala sobre dinheiro", "contexto_periferia": "grana, verba, corre, trampo — a linguagem financeira do povo"},
    {"tema": "gírias financeiras da quebrada", "contexto_periferia": "bonado, moiado, veneno, correr atrás — termos que descrevem situação financeira"},
    {"tema": "a linguagem do rap e educação financeira", "contexto_periferia": "Racionais, Emicida, Djonga falando de dinheiro, desigualdade e consciência"},
    {"tema": "funk ostentação vs consciência financeira", "contexto_periferia": "MC cantando sobre nave e corrente de ouro vs planejamento real"},

    # A vida na favela — contexto social
    {"tema": "como é a vida financeira na favela", "contexto_periferia": "fazer malabarismo todo mês pra pagar conta, sem crédito, sem margem de erro"},
    {"tema": "o corre diário na periferia", "contexto_periferia": "acordar 4h da manhã, pegar busão lotado, trampar o dia todo, voltar de noite"},
    {"tema": "solidariedade financeira na comunidade", "contexto_periferia": "vizinho empresta arroz, vaquinha pra ajudar quem precisa, comunidade se ajuda"},
    {"tema": "desafios de empreender na favela", "contexto_periferia": "sem capital inicial, sem crédito, mas com criatividade e corre"},
    {"tema": "como a desigualdade afeta a quebrada", "contexto_periferia": "escola ruim, posto de saúde longe, banco não chega, oportunidade não aparece"},
    {"tema": "a cultura do bico e trabalho informal", "contexto_periferia": "maioria trabalha sem carteira, bico de tudo: pintor, eletricista, vendedor, entregador"},
    {"tema": "quando cai o salário na quebrada", "contexto_periferia": "pagar aluguel, conta de luz, feira, busão — e ver se sobra alguma coisa"},
    {"tema": "criança crescendo na periferia e dinheiro", "contexto_periferia": "menó vê os pais ralando, aprende cedo o valor da grana"},
    {"tema": "mulheres empreendedoras na favela", "contexto_periferia": "coroa que faz bolo, mina que faz unha, mãe que vende roupa no Instagram"},
    {"tema": "o papel da comunidade na educação financeira", "contexto_periferia": "quem aprendeu ensina quem não sabe, conhecimento circula na quebrada"},

    # Identidade e cultura financeira
    {"tema": "preto e dinheiro não são rivais", "contexto_periferia": "quebrando o mito de que pessoa negra não pode ser investidor"},
    {"tema": "por que educação financeira não chega na periferia", "contexto_periferia": "linguagem difícil, exemplos de rico, quem ensina não conhece a realidade"},
    {"tema": "a favela como potência econômica", "contexto_periferia": "R$ 119 bilhões em atividade econômica, a quebrada movimenta o país"},
    {"tema": "influenciadores financeiros da quebrada", "contexto_periferia": "Nath Finanças, Favelado Investidor, NoFront — gente da periferia ensinando periferia"},
    {"tema": "sair da pobreza começa pela informação", "contexto_periferia": "antes de sair da pobreza, saia da ignorância — conhecimento é poder"},
    {"tema": "como o sistema financeiro exclui a periferia", "contexto_periferia": "juros mais altos, crédito negado, sem agência no bairro, atendimento diferente"},
    {"tema": "autoestima financeira na quebrada", "contexto_periferia": "acreditar que dá pra investir mesmo ganhando pouco, parar de se achar incapaz"},
    {"tema": "sonhos financeiros da periferia", "contexto_periferia": "casa própria, carro, viagem, faculdade pro filho — sonhos reais e possíveis"},

    # Situações do cotidiano
    {"tema": "como dividir a grana quando mora com a família", "contexto_periferia": "ajudar nas contas de casa sem se endividar, dividir justo"},
    {"tema": "comprar fiado na venda do bairro", "contexto_periferia": "caderninho do mercadinho, dívida informal, como controlar"},
    {"tema": "casamento e dinheiro na periferia", "contexto_periferia": "casal juntando grana, dividindo contas, planejando junto"},
    {"tema": "como lidar com parente que pede dinheiro", "contexto_periferia": "irmão desempregado, tia que precisa, como ajudar sem se prejudicar"},
    {"tema": "primeiro emprego e como lidar com o salário", "contexto_periferia": "menó que começou a trampar e não sabe o que fazer com a grana"},
    {"tema": "Black Friday e armadilhas de consumo", "contexto_periferia": "desconto que não é desconto, parcelamento que vira dívida"},
    {"tema": "como sobreviver no fim do mês", "contexto_periferia": "quando a grana acaba antes do mês acabar, como se virar"},
    {"tema": "o que fazer quando perde o trampo", "contexto_periferia": "seguro desemprego, FGTS, como se organizar até achar outro corre"},
]


def load_artigos() -> str:
    """Carrega artigos como contexto."""
    artigos = []
    for filepath in sorted(ARTIGOS_DIR.glob("*.md")):
        content = filepath.read_text(encoding="utf-8")
        artigos.append(f"=== {filepath.stem} ===\n{content}")
    return "\n\n".join(artigos)


def load_transcricoes_sample(n=3) -> str:
    """Carrega N transcrições aleatórias como referência de fala."""
    trans_files = list(TRANSCRICOES_DIR.glob("*.txt"))
    if not trans_files:
        return ""
    selected = random.sample(trans_files, min(n, len(trans_files)))
    texts = []
    for f in selected:
        content = f.read_text(encoding="utf-8")[:800]  # primeiros 800 chars
        texts.append(f"=== {f.stem} ===\n{content}")
    return "\n\n".join(texts)


def get_girias_sample(n=5) -> str:
    """Seleciona N gírias aleatórias com significado e exemplo."""
    selected = random.sample(GIRIAS_CONTEXTO, min(n, len(GIRIAS_CONTEXTO)))
    lines = []
    for g in selected:
        lines.append(f'- "{g["giria"]}" = {g["significa"]}. Ex: "{g["exemplo"]}"')
    return "\n".join(lines)


def generate_with_mlx(messages: list, model: str, max_tokens: int = 800) -> str:
    """Gera resposta usando mlx-lm server local."""
    import requests

    response = requests.post(
        "http://localhost:8080/v1/chat/completions",
        json={
            "model": model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": max_tokens,
            "top_p": 0.9,
        },
        timeout=300,
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]

    # Limpar thinking mode do Qwen3 (fechado ou não)
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    content = re.sub(r"<think>.*", "", content, flags=re.DOTALL).strip()  # think não fechado
    content = re.sub(r"<\|im_end\|>", "", content)
    content = re.sub(r"<\|im_start\|>.*", "", content)
    content = re.sub(r"<\|endoftext\|>", "", content)

    return content.strip()


def generate_with_ollama(messages: list, model: str, max_tokens: int = 800) -> str:
    import requests
    # Convert messages to single prompt for ollama
    system = ""
    prompt = ""
    for m in messages:
        if m["role"] == "system":
            system = m["content"]
        elif m["role"] == "user":
            prompt = m["content"]

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {"temperature": 0.7, "num_predict": max_tokens},
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["response"]


def generate_with_anthropic(messages: list, model: str, max_tokens: int = 800) -> str:
    import anthropic
    client = anthropic.Anthropic()
    system = ""
    user_msgs = []
    for m in messages:
        if m["role"] == "system":
            system = m["content"]
        else:
            user_msgs.append(m)

    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=user_msgs,
    )
    return message.content[0].text


def generate(messages: list, provider: str, model: str, max_tokens: int = 800) -> str:
    generators = {
        "mlx": generate_with_mlx,
        "ollama": generate_with_ollama,
        "anthropic": generate_with_anthropic,
    }
    return generators[provider](messages, model, max_tokens)


def generate_sft_pair(tema_obj: dict, artigos: str, provider: str, model: str) -> dict:
    """Gera um par instrução/resposta com contexto cultural rico."""

    tema = tema_obj["tema"]
    contexto = tema_obj["contexto_periferia"]
    persona = random.choice(PERSONAS)
    girias = get_girias_sample(5)
    transcricao = load_transcricoes_sample(2)

    # --- CHAMADA 1: Gerar pergunta como a persona ---
    msgs_pergunta = [
        {"role": "system", "content": "Você gera perguntas realistas sobre educação financeira."},
        {"role": "user", "content": f"""Crie UMA pergunta curta e natural sobre "{tema}" ({contexto}).

A pergunta é feita por: {persona['nome']} — {persona['contexto']}
Tom: {persona['tom']}

A pergunta deve ser como se a pessoa estivesse conversando com um amigo.
Pode usar gírias se natural pro personagem.

Responda APENAS com a pergunta, sem aspas, sem explicação."""}
    ]

    pergunta = generate(msgs_pergunta, provider, model, max_tokens=150)
    # Limpar — pegar só a primeira linha relevante
    pergunta = pergunta.strip().split("\n")[0].strip().strip('"').strip("'")
    # Se ficou vazio ou só sobrou lixo, usar fallback
    if len(pergunta) < 10 or pergunta.startswith("<"):
        pergunta = f"Me explica sobre {tema}, como funciona isso?"

    # --- CHAMADA 2: Gerar resposta com contexto cultural completo ---
    msgs_resposta = [
        {"role": "system", "content": f"""Você é o "Mlk de Vila", educador financeiro da periferia de São Paulo.
Sua história: nasceu na zona leste, estudou em escola pública, trabalhou desde cedo, fez MBA em finanças.
Você fala a LÍNGUA DA QUEBRADA porque é de lá. Não é imitação — é sua identidade.

GÍRIAS QUE VOCÊ USA (com significado correto):
{girias}

REGRAS:
- Use as gírias APENAS no contexto correto (trampo = trabalho, grana = dinheiro, bico = trabalho extra)
- NUNCA use gíria como filler sem sentido
- Exemplos com valores reais: R$ 50, R$ 100, R$ 800, R$ 1.200
- Seja direto, motivador e honesto
- 2 a 4 parágrafos
- NÃO coloque título, cabeçalho ou "Resposta:"
- Comece direto respondendo"""},

        {"role": "user", "content": f"""Referência de como o Mlk de Vila escreve:
{artigos[:1500]}

Referência de como se fala na periferia (transcrições reais):
{transcricao[:1000] if transcricao else '(sem transcrição disponível)'}

A pessoa que está perguntando: {persona['nome']} — {persona['contexto']}

PERGUNTA: {pergunta}"""}
    ]

    resposta = generate(msgs_resposta, provider, model, max_tokens=600)

    return {
        "instruction": pergunta,
        "response": resposta,
        "metadata": {
            "tema": tema,
            "persona": persona["nome"],
            "girias_usadas": [g["giria"] for g in random.sample(GIRIAS_CONTEXTO, 5)]
        }
    }


def load_existing_dataset() -> list:
    entries = []
    if SFT_OUTPUT.exists():
        with open(SFT_OUTPUT, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
    return entries


def save_entry(entry: dict):
    with open(SFT_OUTPUT, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Gera dataset SFT V2 com contexto cultural")
    parser.add_argument("--provider", choices=["ollama", "mlx", "anthropic"], default="mlx")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Limitar número de temas")
    parser.add_argument("--rounds", type=int, default=1, help="Quantas rodadas por tema (personas diferentes)")
    parser.add_argument("--fresh", action="store_true", help="Limpar dataset existente e começar do zero")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    default_models = {
        "ollama": "qwen2.5:7b",
        "mlx": "mlx-community/Qwen3-4B-4bit",
        "anthropic": "claude-sonnet-4-20250514",
    }
    model = args.model or default_models[args.provider]

    print(f"🏘️  Gerador de Dataset SFT V2 - Educação Financeira da Periferia")
    print(f"   Provider: {args.provider} | Modelo: {model}")
    print(f"   Output: {SFT_OUTPUT}")
    print(f"   Rounds por tema: {args.rounds}")
    print()

    # Carregar dados de referência
    artigos = load_artigos()
    n_artigos = len(list(ARTIGOS_DIR.glob("*.md")))
    n_trans = len(list(TRANSCRICOES_DIR.glob("*.txt"))) if TRANSCRICOES_DIR.exists() else 0
    print(f"📄 {n_artigos} artigos + {n_trans} transcrições como contexto")
    print(f"🗣️  {len(GIRIAS_CONTEXTO)} gírias com significado correto")
    print(f"👤 {len(PERSONAS)} personas diferentes")

    if args.fresh and SFT_OUTPUT.exists():
        SFT_OUTPUT.unlink()
        print("🗑️  Dataset anterior removido")

    existing = load_existing_dataset()
    print(f"📊 {len(existing)} entradas existentes")

    temas = TEMAS_V2[:args.limit] if args.limit else TEMAS_V2
    total = len(temas) * args.rounds
    print(f"🎯 {len(temas)} temas × {args.rounds} rounds = {total} pares a gerar")
    print()

    if args.dry_run:
        for i, t in enumerate(temas, 1):
            print(f"  {i}. {t['tema']} ({t['contexto_periferia']})")
        return

    generated = 0
    errors = 0

    for round_num in range(args.rounds):
        if args.rounds > 1:
            print(f"\n--- Round {round_num + 1}/{args.rounds} ---\n")

        for i, tema_obj in enumerate(temas, 1):
            idx = round_num * len(temas) + i
            print(f"  [{idx}/{total}] {tema_obj['tema'][:45]}...", end=" ", flush=True)
            try:
                entry = generate_sft_pair(tema_obj, artigos, args.provider, model)
                save_entry(entry)
                generated += 1
                print("✅")
            except Exception as e:
                errors += 1
                print(f"❌ {e}")

    print()
    print(f"=== RESULTADO ===")
    print(f"✅ Gerados: {generated}")
    print(f"❌ Erros: {errors}")
    print(f"📊 Total no dataset: {len(existing) + generated}")


if __name__ == "__main__":
    main()
