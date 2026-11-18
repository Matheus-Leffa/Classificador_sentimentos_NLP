import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)


def gerar_matriz_confusao(pipeline, y_test, previsoes, caminho):
    matriz = confusion_matrix(y_test, previsoes)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=matriz,
        display_labels=pipeline.classes_,
    )
    disp.plot(cmap="Blues")
    plt.title("Matriz de Confusão")
    plt.savefig(caminho, bbox_inches="tight")
    plt.close()


def gerar_relatorio_classificacao(y_test, previsoes, caminho):
    relatorio = classification_report(y_test, previsoes)
    print(relatorio)

    with open(caminho, "w", encoding="utf-8") as arquivo:
        arquivo.write(relatorio)

    return relatorio


def gerar_erros(X_test, y_test, previsoes, probabilidades, caminho):
    erros = pd.DataFrame({
        "texto": X_test.values,
        "real": y_test.values,
        "previsto": previsoes,
        "probabilidade": probabilidades.max(axis=1),
    })
    erros = erros[erros["real"] != erros["previsto"]]
    erros.to_csv(caminho, index=False, encoding="utf-8-sig")
    return erros


def gerar_palavras_influentes(pipeline, caminho):
    vetorizador = pipeline.named_steps["vetorizador"]
    classificador = pipeline.named_steps["classificador"]

    palavras = vetorizador.get_feature_names_out()
    coeficientes = classificador.coef_[0]

    df_palavras = pd.DataFrame({
        "palavra": palavras,
        "coeficiente": coeficientes,
    })

    positivas = df_palavras.nlargest(10, "coeficiente")
    negativas = df_palavras.nsmallest(10, "coeficiente")

    influentes = pd.concat([
        positivas.assign(classe="positivo"),
        negativas.assign(classe="negativo"),
    ])
    influentes.to_csv(caminho, index=False, encoding="utf-8-sig")
    return influentes
