# Classificador de Sentimentos

Aplicação Python para análise de sentimentos em avaliações de texto usando
pré-processamento, vetorização com `CountVectorizer`/`TfidfVectorizer` e
regressão logística.

## Requisitos

- Python 3.10 ou superior
- `pip`

## Instalação

No terminal, a partir da pasta do projeto, crie e ative um ambiente virtual:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Instale as dependências:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Execução

Execute o ponto de entrada da aplicação:

```powershell
python main.py
```

O programa carrega os dados de `data/`, compara as quatro configurações por
validação cruzada estratificada, treina a configuração vencedora e salva os
resultados em `resultados/`. A pasta é criada automaticamente quando
necessário.

Os principais arquivos gerados são:

- `comparacao.csv`: métricas da comparação das configurações;
- `matriz_confusao.png`: matriz de confusão do modelo vencedor;
- `relatorio_classificacao.txt`: relatório da avaliação final;
- `erros.csv`: avaliações classificadas incorretamente;
- `palavras_influentes.csv`: palavras com maiores coeficientes;
- `casos_dificeis_predicoes.csv`: previsões para `data/casos_dificeis.csv`.