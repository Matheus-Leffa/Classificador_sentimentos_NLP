import pandas as pd


def exibir_distribuicao_classes(df):
    distribuicao = pd.DataFrame({
        "Quantidade": df["sentimento"].value_counts(),
        "Porcentagem": df["sentimento"].value_counts(normalize=True) * 100,
    })
    print(distribuicao)


def exibir_tamanho_medio_por_classe(df):
    df_com_tamanho = df.copy()
    df_com_tamanho["quantidade_palavras"] = (
        df_com_tamanho["texto"].str.split().str.len()
    )
    media_palavras = df_com_tamanho.groupby(
        "sentimento"
    )["quantidade_palavras"].mean()

    print()
    print("Tamanho médio dos textos por classe")
    print(media_palavras)


def gerar_previsoes_casos_dificeis(pipeline, caminho_dados, caminho_saida):
    casos_dificeis = pd.read_csv(caminho_dados)

    previsoes = pipeline.predict(casos_dificeis["texto"])
    probabilidades = pipeline.predict_proba(casos_dificeis["texto"])

    classes = pipeline.named_steps["classificador"].classes_
    indice_positivo = list(classes).index("positivo")
    prob_positivo = probabilidades[:, indice_positivo]

    resultado = pd.DataFrame({
        "texto": casos_dificeis["texto"].values,
        "tipo": casos_dificeis["tipo"].values,
        "esperado": casos_dificeis["sentimento_esperado"].values,
        "previsto": previsoes,
        "probabilidade": prob_positivo,
    })
    resultado.to_csv(caminho_saida, index=False, encoding="utf-8-sig")
    return resultado
