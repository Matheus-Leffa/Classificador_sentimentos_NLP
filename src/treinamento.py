import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)

from src.modelos import CONFIGURACOES, construir_pipeline


def dividir_dados(df):
    X = df["texto"]
    y = df["sentimento"]

    return train_test_split(
        X,
        y,
        test_size=0.25,
        stratify=y,
        random_state=42,
    )


def avaliar_configuracoes_no_teste(X_train, y_train, X_test, y_test):
    for nome, config in CONFIGURACOES.items():
        pipeline = construir_pipeline(config)
        pipeline.fit(X_train, y_train)

        previsoes = pipeline.predict(X_test)
        acuracia = accuracy_score(y_test, previsoes)
        vocabulario = len(
            pipeline.named_steps["vetorizador"].vocabulary_
        )

        print(f"\n{nome}")
        print(f"Acurácia: {acuracia:.4f}")
        print(f"Vocabulário: {vocabulario}")
        print(classification_report(y_test, previsoes))


def comparar_configuracoes(X_train, y_train):
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    resultados = []

    for nome, config in CONFIGURACOES.items():
        pipeline = construir_pipeline(config)

        metricas = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring={
                "acc": "accuracy",
                "f1": "f1_macro",
            },
            n_jobs=-1,
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
            "f1_cv_desvio": np.std(metricas["test_f1"]),
        })

    return pd.DataFrame(resultados)


def identificar_configuracao_vencedora(df_resultados):
    ordenados = df_resultados.sort_values(
        ["f1_cv_media", "acc_cv_media"],
        ascending=[False, False],
        kind="stable",
    )
    return ordenados.iloc[0]


def treinar_configuracao_vencedora(vencedora, X_train, y_train):
    pipeline = construir_pipeline(CONFIGURACOES[vencedora["config"]])
    pipeline.fit(X_train, y_train)
    return pipeline
