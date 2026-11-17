import pandas as pd
import re

df = pd.read_csv('data/Avaliacoes.csv')

print(df.head())

print(pd.DataFrame({
    'Quantidade': df['sentimento'].value_counts(),
    'Porcentagem': df['sentimento'].value_counts(normalize=True) * 100
}))

print()
print("Tamanho médio dos textos por classe")
df['quantidade_palavras'] = df['texto'].str.split().str.len()

media_palavras = df.groupby('sentimento')['quantidade_palavras'].mean()

print(media_palavras)

STOP = {
    "o", "a", "os", "as", "um", "uma",
    "e", "é", "de", "do", "da", "que",
    "para", "com", "mas", "muito", "foi", "em"
}

def prep(texto, remover_stopwords = True):
    toks = re.findall(
        r"[a-záàâãéêíóôõúç]+",
        texto.lower()
    )

    if remover_stopwords:
        toks = [
            palavra for palavra in toks
            if palavra not in STOP
        ]

    return " ".join(toks)

texto = "O dia esta lindo hoje, mas ontem foi nublado"
print(prep(texto, False))
print(prep(texto))