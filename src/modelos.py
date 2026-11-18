from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from src.preprocessamento import preprocessar_lote


CONFIGURACOES = {
    "ConfigA": {
        "stopwords": False,
        "vetorizador": CountVectorizer(),
    },
    "ConfigB": {
        "stopwords": False,
        "vetorizador": TfidfVectorizer(),
    },
    "ConfigC": {
        "stopwords": True,
        "vetorizador": CountVectorizer(),
    },
    "ConfigD": {
        "stopwords": True,
        "vetorizador": TfidfVectorizer(),
    },
}


def construir_pipeline(config):
    preprocessador = FunctionTransformer(
        preprocessar_lote,
        kw_args={"remover_stopwords": config["stopwords"]},
        validate=False,
    )

    vetorizador = config["vetorizador"]

    return Pipeline([
        ("preprocessamento", preprocessador),
        ("vetorizador", vetorizador),
        ("classificador", LogisticRegression(max_iter=1000)),
    ])
