from pathlib import Path

import pandas as pd

from src.analise import (
    exibir_distribuicao_classes,
    exibir_tamanho_medio_por_classe,
    gerar_previsoes_casos_dificeis,
)
from src.avaliacao import (
    gerar_erros,
    gerar_matriz_confusao,
    gerar_palavras_influentes,
    gerar_relatorio_classificacao,
)
from src.preprocessamento import prep
from src.treinamento import (
    avaliar_configuracoes_no_teste,
    comparar_configuracoes,
    dividir_dados,
    identificar_configuracao_vencedora,
    treinar_configuracao_vencedora,
)


PROJETO_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJETO_DIR / "data"
RESULTADOS_DIR = PROJETO_DIR / "resultados"


def carregar_dados():
    return pd.read_csv(DATA_DIR / "Avaliacoes.csv")


def main():
    RESULTADOS_DIR.mkdir(parents=True, exist_ok=True)

    df = carregar_dados()

    print(df.head())
    exibir_distribuicao_classes(df)
    exibir_tamanho_medio_por_classe(df)

    texto_exemplo = "O dia esta lindo hoje, mas ontem foi nublado"
    print(prep(texto_exemplo, False))
    print(prep(texto_exemplo))

    X_train, X_test, y_train, y_test = dividir_dados(df)
    avaliar_configuracoes_no_teste(
        X_train,
        y_train,
        X_test,
        y_test,
    )

    df_resultados = comparar_configuracoes(X_train, y_train)

    print("------------")
    print(df_resultados)
    df_resultados.to_csv(
        RESULTADOS_DIR / "comparacao.csv",
        index=False,
    )

    vencedora = identificar_configuracao_vencedora(df_resultados)

    print("Configuração vencedora:", vencedora["config"])
    print(
        f"{vencedora['config']} possui F1 médio: "
        f"{vencedora['f1_cv_media']}"
    )
    print(
        f"{vencedora['config']} possui Acurácia média: "
        f"{vencedora['acc_cv_media']}"
    )

    pipeline_vencedora = treinar_configuracao_vencedora(
        vencedora,
        X_train,
        y_train,
    )
    previsoes = pipeline_vencedora.predict(X_test)
    probabilidades = pipeline_vencedora.predict_proba(X_test)

    gerar_matriz_confusao(
        pipeline_vencedora,
        y_test,
        previsoes,
        RESULTADOS_DIR / "matriz_confusao.png",
    )
    gerar_relatorio_classificacao(
        y_test,
        previsoes,
        RESULTADOS_DIR / "relatorio_classificacao.txt",
    )
    gerar_erros(
        X_test,
        y_test,
        previsoes,
        probabilidades,
        RESULTADOS_DIR / "erros.csv",
    )
    gerar_palavras_influentes(
        pipeline_vencedora,
        RESULTADOS_DIR / "palavras_influentes.csv",
    )
    gerar_previsoes_casos_dificeis(
        pipeline_vencedora,
        DATA_DIR / "casos_dificeis.csv",
        RESULTADOS_DIR / "casos_dificeis_predicoes.csv",
    )


if __name__ == "__main__":
    main()
