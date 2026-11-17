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

