import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_validate
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

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

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

def preprocessar_lote(textos, remover_stopwords=True):
    return [
        prep(texto, remover_stopwords=remover_stopwords)
        for texto in textos
    ]

resultados = []

for nome, config in configuracoes.items():

    preprocessador = FunctionTransformer(
        preprocessar_lote,
        kw_args={"remover_stopwords": config["stopwords"]},
        validate=False
    )

    pipeline = Pipeline([
        ("preprocessamento", preprocessador),
        ("vetorizador", config["vetorizador"]),
        ("classificador", LogisticRegression(max_iter=1000))
    ])

    metricas = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring={
            "acc": "accuracy",
            "f1": "f1_macro"
        },
        n_jobs=-1
    )

    pipeline.fit(X_train, y_train)

    vocabulario = len(
        pipeline.named_steps["vetorizador"].vocabulary_
    )

    resultados.append({
        "config": nome,
        "stopwords": "removidas" if config["stopwords"] else "mantidas",
        "vetorizador": config["vetorizador"].__class__.__name__,
        "vocabulario": vocabulario,
        "acc_cv_media": np.mean(metricas["test_acc"]),
        "acc_cv_desvio": np.std(metricas["test_acc"]),
        "f1_cv_media": np.mean(metricas["test_f1"]),
        "f1_cv_desvio": np.std(metricas["test_f1"])
    })

df_resultados = pd.DataFrame(resultados)

print("------------")
print(df_resultados)

df_resultados.to_csv(
    "resultados/comparacao.csv",
    index=False
)

vencedora = df_resultados.iloc[0]

print("Configuração vencedora:", vencedora["config"])
print(f"{vencedora["config"]} possui F1 médio: {vencedora["f1_cv_media"]}")
print(f"{vencedora["config"]} possui Acurácia média: {vencedora["acc_cv_media"]}")

pipeline_config_vencedora = Pipeline([
    ("preprocessamento", FunctionTransformer(
        preprocessar_lote,
        kw_args={"remover_stopwords": False},
        validate=False
    )),
    ("vetorizador", CountVectorizer()),
    ("classificador", LogisticRegression(max_iter=1000))
])

pipeline_config_vencedora.fit(X_train, y_train)
previsoes = pipeline_config_vencedora.predict(X_test)
probabilidades = pipeline_config_vencedora.predict_proba(X_test)

matrizConfusao = confusion_matrix(y_test, previsoes)

disp = ConfusionMatrixDisplay(
    confusion_matrix=matrizConfusao,
    display_labels=pipeline_config_vencedora.classes_
)

disp.plot(cmap="Blues")

plt.title("Matriz de Confusão - Configuração A")
plt.savefig("resultados/matriz_confusao.png", bbox_inches="tight")
#plt.show()

relatorio = classification_report(
    y_test,
    previsoes
)

print(relatorio)

with open(
    "resultados/relatorio_classificacao.txt",
    "w",
    encoding="utf-8"
) as arquivo:
    arquivo.write(relatorio)

erros = pd.DataFrame({
    "texto": X_test.values,
    "real": y_test.values,
    "previsto": previsoes,
    "probabilidade": probabilidades.max(axis=1)
})

erros = erros[erros["real"] != erros["previsto"]]

erros.to_csv(
    "resultados/erros.csv",
    index=False,
    encoding="utf-8-sig"
)

vetorizador = pipeline_config_vencedora.named_steps["vetorizador"]

classificador = pipeline_config_vencedora.named_steps["classificador"]

palavras = vetorizador.get_feature_names_out()

coeficientes = classificador.coef_[0]

df_palavras = pd.DataFrame({
    "palavra": palavras,
    "coeficiente": coeficientes
})

positivas = df_palavras.nlargest(10, "coeficiente")

negativas = df_palavras.nsmallest(10, "coeficiente")

influentes = pd.concat([
    positivas.assign(classe="positivo"),
    negativas.assign(classe="negativo")
])

influentes.to_csv(
    "resultados/palavras_influentes.csv",
    index=False,
    encoding="utf-8-sig"
)