import re


STOP = {
    "o", "a", "os", "as", "um", "uma",
    "e", "é", "de", "do", "da", "que",
    "para", "com", "mas", "muito", "foi", "em",
}


def prep(texto, remover_stopwords=True):
    toks = re.findall(
        r"[a-záàâãéêíóôõúç]+",
        texto.lower(),
    )

    if remover_stopwords:
        toks = [
            palavra for palavra in toks
            if palavra not in STOP
        ]

    return " ".join(toks)


def preprocessar_lote(textos, remover_stopwords=True):
    return [
        prep(texto, remover_stopwords=remover_stopwords)
        for texto in textos
    ]
