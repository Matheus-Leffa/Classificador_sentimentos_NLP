import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import accuracy_score, classification_report

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

X = df["texto"]
y = df["sentimento"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    stratify=y,
    random_state=42
)

configuracoes = {
    "ConfigA": {
        "stopwords": False,
        "vetorizador": CountVectorizer()
    },
    "ConfigB": {
        "stopwords": False,
        "vetorizador": TfidfVectorizer()
    },
    "ConfigC": {
        "stopwords": True,
        "vetorizador": CountVectorizer()
    },
    "ConfigD": {
        "stopwords": True,
        "vetorizador": TfidfVectorizer()
    }
}

resultados = {}

for nome, config in configuracoes.items():

    preprocessador = FunctionTransformer(
        lambda textos, remover=config["stopwords"]: [
            prep(texto, remover_stopwords=remover)
            for texto in textos
        ],
        validate=False
    )

    pipeline = Pipeline([
        ("preprocessamento", preprocessador),
        ("vetorizador", config["vetorizador"]),
        ("classificador", LogisticRegression(max_iter=1000))
    ])

    # Treinamento
    pipeline.fit(X_train, y_train)

    # Previsão
    previsoes = pipeline.predict(X_test)

    # Avaliação
    acuracia = accuracy_score(y_test, previsoes)

    # Tamanho do vocabulário aprendido no treinamento
    vocabulario = len(
        pipeline.named_steps["vetorizador"].vocabulary_
    )

    resultados[nome] = {
        "Acurácia": acuracia,
        "Vocabulário": vocabulario
    }

    print(f"\n{nome}")
    print(f"Acurácia: {acuracia:.4f}")
    print(f"Vocabulário: {vocabulario}")

    print(classification_report(y_test, previsoes))