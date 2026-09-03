# Heart Failure Prediction — Classificador Naive Bayes do zero

Classificador **Naive Bayes** construído inteiramente do zero (NumPy/SciPy puro, sem
`sklearn.naive_bayes`) para prever a presença de doença cardíaca
(`HeartDisease`) na base
[Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
(Kaggle: `fedesoriano/heart-failure-prediction`).

## Sobre o projeto

Estudo dirigido de Inteligência Artificial: modelagem probabilística Bayesiana de três
características clínicas e implementação manual de um classificador Naive Bayes.

- **Problema:** classificação binária supervisionada ($Y=0$: Normal; $Y=1$: Doença cardíaca).
- **Base:** 918 pacientes, 12 atributos, sem valores ausentes.
- **Características escolhidas:**
  - $X_1$ = `MaxHR` (contínua) → modelada por distribuição **Normal**;
  - $X_2$ = `Age` (contínua) → modelada por distribuição **Normal**;
  - $X_3$ = `ChestPainType` (categórica, 4 categorias) → distribuição **discreta**
    com suavização de **Laplace**.
- **Implementação:** priors, densidade Gaussiana, log-verossimilhança, razão de
  verossimilhanças, Teorema de Bayes, fronteiras de decisão e matriz de confusão,
  todos implementados manualmente em `main.py`.
- **scikit-learn** é usado apenas para `train_test_split` (divisão treino/teste).
- Documento acadêmico completo em [`docs/contexto.md`](docs/contexto.md).

## Instalação

Requer Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Execução

```bash
python main.py
```

O script:

1. baixa a base programaticamente via `kagglehub` (arquivo `heart.csv`);
2. divide os dados em 70% treino / 30% teste (`random_state=42`, estratificado);
3. executa as análises Bayesianas univariadas (Etapas 1–5) para cada característica;
4. treina o Naive Bayes do zero e avalia no conjunto de teste;
5. gera as figuras em `docs/figures/`.

## Resultados obtidos

Divisão: 642 observações de treino e 276 de teste.

| Métrica | Valor |
|---|---|
| Acurácia | 77,17% |
| Precisão | 79,61% |
| Recall | 79,08% |
| F1-Score | 79,34% |

Matriz de confusão no teste (linha = real, coluna = predito):

| | Predito 0 | Predito 1 |
|---|---:|---:|
| **Real 0 (Normal)** | 92 (VN) | 31 (FP) |
| **Real 1 (Doença)** | 32 (FN) | 121 (VP) |

Destaques da análise:

- `MaxHR` — fronteira Bayesiana em **141,1 bpm**: abaixo dela decide-se $Y=1$ (doença).
- `Age` — fronteira em **49,1 anos**: acima dela decide-se $Y=1$.
- `ChestPainType` — a categoria `ASY` (dor assintomática) é a evidência isolada mais
  forte de doença ($\Lambda = 2{,}94$); `ATA`, `NAP` e `TA` favorecem $Y=0$.

## Estrutura do repositório

```text
.
├── main.py                  # Classificador Naive Bayes do zero + análises
├── requirements.txt         # Dependências
├── README.md
└── docs/
    ├── contexto.md          # Documento acadêmico com toda a modelagem matemática
    └── figures/             # Figuras geradas por main.py
```

## Base de dados

- **Nome:** Heart Failure Prediction Dataset
- **Fonte:** <https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction>
- **Arquivo:** `heart.csv` (baixado automaticamente pelo `kagglehub`)
