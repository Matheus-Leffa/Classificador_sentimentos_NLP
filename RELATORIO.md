# RELATORIO — Análise de Sentimento com PLN

**Nome: Matheus Santos Leffa** 
**Repositório: https://github.com/Matheus-Leffa/Classificador_sentimentos_NLP** 
**Data: 17/11/2026** 

## Tabela-resumo (preencha com os seus resultados)

| Config | Stopwords | Vetorizador | Vocabulário | Acurácia CV (média ± desvio) | F1 CV (média ± desvio) |
|---|---|---|-------------|------------------------------|------------------------|
| A | mantidas | BoW | 420         | 0.8154                       | 0.856                  |
| B | mantidas | TF-IDF | 420         | 0.8615                       | 0.845                  |
| C | removidas | BoW | 406         | 0.8308                       | 0.840                  |
| D | removidas | TF-IDF | 406         | 0.8308                       | 0.851                  |

**Configuração escolhida:** A **Acurácia no teste:** 0.8154 **F1 no teste:** 0.856

---

## 1. Pré-processamento

Criei uma função chamada prep() que transforma todas as letras em minúsculas,
Identifica e extrai apenas palavras compostas por letras de a a z e caracteres acentuados especificados na expressão regular.
Remove pontuações e números.
Permite remover as palavras presentes em STOP(stopwords).
Com as stopwords temos 420 palavras. Sem stopword temos 402 (18 palavras foram removidas).
A palavra "não" foi mantida pois ela pode mudar completamente o sentido de uma frase.

## 2. BoW vs. TF-IDF

As 4 configurações possuem um desvio padrão quase equivalente, o que indica que não há uma config expressivamente superior as demais neste aspecto. 
A ConfigA apresentou o melhor resultado, com F1 médio de 0,8563 e desvio padrão de 0,0926.
A BoW considera a frequência com que as palavras aparecem em um texto, não importando a sua posição nele.
Já a TD-IDF acrescenta pesos às palavras de acordo com a sua frequência e raridade no conjunto inteiro de dados.

## 3. Avaliação no teste

![matriz de Confusão](resultados/matriz_confusao.png)

Classe negativa
Precisão: 86,2% - Recall: 75,8%

F1-score: aproximadamente 80,6%
Representa um equilíbrio razoável entre precisão e recall, mas mostra que ainda há dificuldade em identificar todos os comentários negativos.

Classe positiva
Precisão: 77,8% - Recall: 87,5%

F1-score: aproximadamente 82,4%
O desempenho geral para a classe positiva foi ligeiramente superior.

O erro mais grave é classificar uma reclamação como elogio.
Esse erro ocorreu 8 vezes e significa que a loja pode deixar de perceber problemas reais de clientes, não responder à reclamação e prejudicar sua reputação. 

## 4. Validação cruzada vs. divisão única

Usamos a validação cruzada pois ela avalia cada configuração em diferentes divisões dos dados, reduzindo a influência de uma divisão específica.
Não devemos escolher olhando o teste porque ele deve simular dados novos que o modelo ainda não conhece.
Se escolhermos o teste, estaremos adaptando nossas decisões ao teste, comprometendo a avaliação.
Vazamento de dados acontece quando informações que deveriam estar disponíveis apenas na avaliação acabam influenciando o treinamento.
Neste trabalho, vazamentos foram evitados utilizando Pipeline, que garante que o vetorizador seja ajustado somente com os dados de treinamento de cada divisão da validação cruzada.

## 5. Análise de erros e palavras influentes

"Ficou lindo na sala, combinou perfeitamente com os móveis"
Possível motivo do erro: O modelo pode não ter aprendido a associar "lindo" ou "perfeitamente" à avaliação positiva.
Talvez tenha atribuído pesos demasiados a outras palavras do texto.

"Panela de péssima qualidade, a comida gruda e queima"
Possível motivo do erro: O modelo pode ter associado "comida" como algo positivo devido aos testes. 
Mas não associou a palavra às características negativas.

"Cheguei animado, mas me arrependi, tudo estava sujo"
Possível motivo do erro: o modelo pode ter atribuído importância excessiva à palavra "animado", que transmite uma ideia positiva quando analisada isoladamente.

## 6. Casos difíceis e limitações

Negação: o modelo acertou 2 dos 4 casos. Conseguiu classificar corretamente frases como "Não gostei nada do atendimento" e "Não é ruim, gostei bastante".
Ironia: o modelo acertou 2 dos 3 casos. Identificou corretamente algumas críticas irônicas, como "Maravilha, mais uma vez a entrega atrasou", mas classificou incorretamente "Que ótimo, o produto chegou quebrado" como positivo.
Frases mistas: o modelo errou os 2 casos apresentados. Nas frases que apresentam uma característica positiva e outra negativa, ele classificou o texto inteiro como positivo.

BoW e TF-IDF contabilizam e dão pesos as palavras, respectivamente, mas não compreender o contexto em que as palavras estão inseridas.
Assim, palavras como "ótimo" podem levar a uma avaliação positiva mesmo sendo usadas sarcasticamente.

Melhorias possível:
N-gramas: utilizar conjuntos de duas ou mais palavras para melhorar a compreensão de contexto do modelo.
Por exemplo: "não gostei", "muito bom" e etc...

Similaridade de cosseno: O BoW e o TF-IDF transformam cada texto em um vetor numérico. A similaridade do cosseno compara esses vetores para verificar o quanto eles apontam na mesma direção.

---

## Declaração de uso de IA

Marque uma opção e complete:

- [ ] Não usei IA generativa em nenhuma parte deste trabalho.
- [X] Usei IA generativa **somente** para: consultas de sintaxe, uso de funções desconhecidas (sklearn principalmente) e como fonte de consulta e resolução de dúvidas.(não é permitido gerar código ou relatório).
