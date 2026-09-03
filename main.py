#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Estudo Dirigido — Classificador Naive Bayes construído do zero
================================================================
Base: Heart Failure Prediction Dataset (Kaggle: fedesoriano/heart-failure-prediction)
Alvo: HeartDisease (1 = Doença Cardíaca, 0 = Normal)

Características escolhidas (X1, X2, X3):
  X1 = MaxHR         (contínua)  -> modelada por distribuição Normal
  X2 = Age           (contínua)  -> modelada por distribuição Normal
  X3 = ChestPainType (categórica, 4 categorias) -> distribuição discreta
                                                   (multinomial) com
                                                   suavização de Laplace

O classificador é implementado manualmente com NumPy/SciPy (sem usar
sklearn.naive_bayes). O scikit-learn é usado apenas para a divisão
treinamento/teste (train_test_split) e, opcionalmente, para conferir
as métricas. Todas as funções de densidade, prior, log-verossimilhança
e decisão foram escritas do zero a partir das equações do Teorema de
Bayes.

Autor: estudo dirigido de Inteligência Artificial.
Data: setembro de 2026.
"""

from __future__ import annotations

import math
import os
import warnings

import matplotlib

matplotlib.use("Agg")  # backend sem interface gráfica (gera arquivos PNG)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm as scipy_norm  # apenas verificação/plots de apoio
from sklearn.model_selection import train_test_split  # apenas divisão treino/teste

try:
    import kagglehub
    from kagglehub import KaggleDatasetAdapter
except ImportError:  # pragma: no cover
    kagglehub = None
    KaggleDatasetAdapter = None

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# Configurações do experimento
# ----------------------------------------------------------------------
DATASET_HANDLE = "fedesoriano/heart-failure-prediction"
DATASET_FILE = "heart.csv"
TARGET = "HeartDisease"

# As três características do estudo (X1, X2, X3).
CONTINUOUS_FEATURES = ["MaxHR", "Age"]
CATEGORICAL_FEATURES = ["ChestPainType"]
FEATURES = CONTINUOUS_FEATURES + CATEGORICAL_FEATURES

RANDOM_STATE = 42          # semente aleatória da divisão treino/teste
TEST_SIZE = 0.30           # 30% para teste, 70% para treinamento
LAPLACE_ALPHA = 1.0        # suavização de Laplace (alpha = 1)
FIGURES_DIR = os.path.join("docs", "figures")
EPS = 1e-12                # evita divisão por zero em desvios nulos

# Categorias da ChestPainType, em ordem padronizada para exibição.
CHEST_PAIN_ORDER = ["ASY", "ATA", "NAP", "TA"]


# ======================================================================
# 1. Funções matemáticas implementadas do zero (NumPy/SciPy puro)
# ======================================================================
def gaussian_pdf(x: np.ndarray | float, mu: float, sigma: float) -> np.ndarray | float:
    """
    Densidade de probabilidade da distribuição Normal N(mu, sigma^2).

        p(x | mu, sigma) = (1 / sqrt(2*pi*sigma^2)) * exp(-(x - mu)^2 / (2*sigma^2))

    Implementada manualmente a partir da equação, sem usar funções prontas
    de densidade de bibliotecas de aprendizado de máquina.
    """
    sigma = max(float(sigma), EPS)
    coefficient = 1.0 / (math.sqrt(2.0 * math.pi) * sigma)
    exponent = -0.5 * ((np.asarray(x, dtype=float) - mu) / sigma) ** 2
    return coefficient * np.exp(exponent)


def gaussian_logpdf(x: float, mu: float, sigma: float) -> float:
    """
    Logaritmo natural da densidade Normal — usado no classificador para
    trabalhar no domínio logarítmico e evitar underflow numérico.

        log p(x | mu, sigma) = -0.5*log(2*pi) - log(sigma) - (x - mu)^2/(2*sigma^2)
    """
    sigma = max(float(sigma), EPS)
    z = (float(x) - mu) / sigma
    return -0.5 * math.log(2.0 * math.pi) - math.log(sigma) - 0.5 * z * z


def gaussian_decision_boundaries(
    mu0: float, sigma0: float, mu1: float, sigma1: float, pi0: float, pi1: float
) -> list[float]:
    """
    Fronteiras de decisão do classificador Bayesiano univariado com duas
    classes Gaussianas (variâncias possivelmente diferentes).

    Condição de fronteira:  P(Y=0|x) = P(Y=1|x)
        p(x|Y=0)*P(Y=0) = p(x|Y=1)*P(Y=1)

    Substituindo as densidades Normais e aplicando logaritmos, obtém-se a
    equação quadrática  a*x^2 + b*x + c = 0, com:

        a = 0.5*(1/sigma1^2 - 1/sigma0^2)
        b = mu0/sigma0^2 - mu1/sigma1^2
        c = 0.5*(mu1^2/sigma1^2 - mu0^2/sigma0^2)
            + log(sigma1/sigma0) + log(pi0/pi1)

    Retorna as raízes reais (0, 1 ou 2 fronteiras).
    """
    s0, s1 = max(sigma0, EPS), max(sigma1, EPS)
    a = 0.5 * (1.0 / s1**2 - 1.0 / s0**2)
    b = mu0 / s0**2 - mu1 / s1**2
    c = (
        0.5 * (mu1**2 / s1**2 - mu0**2 / s0**2)
        + math.log(s1 / s0)
        + math.log(pi0 / pi1)
    )

    if abs(a) < 1e-15:  # caso linear (variâncias iguais)
        if abs(b) < EPS:
            return []
        return [float(-c / b)]

    discriminant = b**2 - 4.0 * a * c
    if discriminant < 0:
        return []
    sqrt_d = math.sqrt(discriminant)
    roots = [float((-b + sqrt_d) / (2.0 * a)), float((-b - sqrt_d) / (2.0 * a))]
    return sorted(roots)


def log_sum_exp(values: list[float]) -> float:
    """log(exp(v1) + exp(v2) + ...) calculado de forma numericamente estável."""
    m = max(values)
    return m + math.log(sum(math.exp(v - m) for v in values))


# ======================================================================
# 2. Classificador Naive Bayes do zero (contínuas + categóricas)
# ======================================================================
class NaiveBayesFromScratch:
    """
    Classificador Naive Bayes Gaussiano/Discreto implementado do zero.

    Hipótese de independência condicional:

        p(x1, x2, x3 | Y=c) = p(x1|Y=c) * p(x2|Y=c) * p(x3|Y=c)

    Decisão no domínio logarítmico:

        y_hat = argmax_c [ log P(Y=c) + soma_j log p(xj | Y=c) ]
    """

    def __init__(self, continuous_features: list[str], categorical_features: list[str],
                 alpha: float = 1.0):
        self.continuous_features = list(continuous_features)
        self.categorical_features = list(categorical_features)
        self.alpha = float(alpha)

        self.classes_ = None
        self.priors_ = {}
        self.gaussian_params_ = {}
        self.categorical_params_ = {}
        self.categories_ = {}

    # ------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "NaiveBayesFromScratch":
        """Estima todos os parâmetros usando APENAS o conjunto de treinamento."""
        self.classes_ = np.array(sorted(y.unique()))
        n_train = len(y)

        # --- Probabilidades a priori P(Y=c) = freq. relativa no treino ---
        self.priors_ = {int(c): float((y == c).sum()) / n_train for c in self.classes_}

        # --- Parâmetros das contínuas: Normal(mu_c, sigma_c^2) por classe ---
        # Média amostral e desvio padrão MLE (ddof=0), como em Naive Bayes clássico.
        self.gaussian_params_ = {}
        for c in self.classes_:
            self.gaussian_params_[int(c)] = {}
            for feature in self.continuous_features:
                values = X.loc[y == c, feature].astype(float)
                mu = float(values.mean())
                sigma = float(values.std(ddof=0))
                sigma = max(sigma, EPS)
                self.gaussian_params_[int(c)][feature] = (mu, sigma)

        # --- Probabilidades das categóricas com suavização de Laplace ---
        # P(Xj = ak | Y=c) = (contagem(ak, c) + alpha) / (n_c + alpha*K)
        self.categorical_params_ = {}
        self.categories_ = {}
        for feature in self.categorical_features:
            categories = sorted(X[feature].astype(str).unique())
            self.categories_[feature] = categories
            self.categorical_params_[feature] = {}
            k = len(categories)
            for c in self.classes_:
                values = X.loc[y == c, feature].astype(str)
                n_c = len(values)
                denominator = n_c + self.alpha * k
                probs = {
                    cat: (float((values == cat).sum()) + self.alpha) / denominator
                    for cat in categories
                }
                # Probabilidade para categoria nunca vista no treino (efeito Laplace)
                probs["__unseen__"] = self.alpha / denominator
                self.categorical_params_[feature][int(c)] = probs
        return self

    # ------------------------------------------------------------------
    def log_likelihood(self, x: pd.Series) -> dict[int, float]:
        """Retorna log p(x | Y=c) para cada classe c."""
        log_liks = {}
        for c in self.classes_:
            total = 0.0
            for feature in self.continuous_features:
                mu, sigma = self.gaussian_params_[int(c)][feature]
                total += gaussian_logpdf(float(x[feature]), mu, sigma)
            for feature in self.categorical_features:
                probs = self.categorical_params_[feature][int(c)]
                p = probs.get(str(x[feature]), probs["__unseen__"])
                total += math.log(max(p, EPS))
            log_liks[int(c)] = total
        return log_liks

    # ------------------------------------------------------------------
    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """Calcula P(Y=c|x) via Teorema de Bayes no domínio logarítmico."""
        posteriors = []
        for _, row in X.iterrows():
            log_liks = self.log_likelihood(row)
            log_joint = {
                int(c): math.log(self.priors_[int(c)]) + log_liks[int(c)]
                for c in self.classes_
            }
            # normalização estável: p_c = exp(log_joint_c) / soma exp(log_joint_k)
            z = log_sum_exp([log_joint[int(c)] for c in self.classes_])
            posteriors.append(
                {int(c): math.exp(log_joint[int(c)] - z) for c in self.classes_}
            )
        return pd.DataFrame(posteriors, columns=[int(c) for c in self.classes_])

    # ------------------------------------------------------------------
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Regra de decisão MAP: y_hat = argmax_c P(Y=c|x)."""
        proba = self.predict_proba(X)
        return proba.idxmax(axis=1).to_numpy(dtype=int)


# ======================================================================
# 3. Métricas e matriz de confusão (implementadas do zero)
# ======================================================================
def confusion_components(y_true, y_pred, positive: int = 1) -> dict[str, int]:
    """
    Matriz de confusão binária. Layout (linha = real, coluna = predito):

              Predito 0   Predito 1
    Real 0       VN          FP
    Real 1       FN          VP
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    vp = int(((yt == positive) & (yp == positive)).sum())
    vn = int(((yt != positive) & (yp != positive)).sum())
    fp = int(((yt != positive) & (yp == positive)).sum())
    fn = int(((yt == positive) & (yp != positive)).sum())
    return {"VN": vn, "FP": fp, "FN": fn, "VP": vp}


def metrics_from_confusion(parts: dict[str, int]) -> dict[str, float]:
    """Acurácia, Precisão, Recall e F1-Score a partir da matriz de confusão."""
    vn, fp, fn, vp = parts["VN"], parts["FP"], parts["FN"], parts["VP"]
    total = vn + fp + fn + vp
    accuracy = (vp + vn) / total if total else 0.0
    precision = vp / (vp + fp) if (vp + fp) else 0.0
    recall = vp / (vp + fn) if (vp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "Acurácia": accuracy,
        "Precisão": precision,
        "Recall": recall,
        "F1-Score": f1,
    }


# ======================================================================
# 4. Rotinas de análise univariada (Etapas 1 a 5 do estudo dirigido)
# ======================================================================
def analyze_continuous_feature(
    name: str, feature: str, X_train: pd.DataFrame, y_train: pd.Series,
    eval_points: list[float]
) -> None:
    """Executa as 5 etapas Bayesianas para uma característica contínua."""
    print("\n" + "=" * 78)
    print(f"ANÁLISE UNIVARIADA — {name}: {feature} (contínua)")
    print("=" * 78)

    # Etapa 1 — hipótese Normal + parâmetros estimados por classe
    print("\n[Etapa 1] Hipótese: X|Y=c ~ Normal(mu_c, sigma_c^2)")
    params = {}
    for c in sorted(y_train.unique()):
        values = X_train.loc[y_train == c, feature].astype(float)
        mu = float(values.mean())
        sigma = max(float(values.std(ddof=0)), EPS)
        params[int(c)] = (mu, sigma)
        print(f"  Y={c}: mu_{c} = {mu:.4f}, sigma_{c} = {sigma:.4f}  "
              f"(n={int((y_train == c).sum())})")

    pi0 = float((y_train == 0).mean())
    pi1 = float((y_train == 1).mean())
    print(f"  Priors do treino: P(Y=0) = {pi0:.4f}, P(Y=1) = {pi1:.4f}")

    # Etapas 2, 3 e 4 — verossimilhanças, razão e posterior em pontos escolhidos
    print("\n[Etapas 2/3/4] Verossimilhança, razão Lambda(x) e posterior")
    print(f"  {'x':>8} | {'p(x|Y=0)':>12} | {'p(x|Y=1)':>12} | "
          f"{'Lambda(x)':>10} | {'P(Y=1|x)':>10} | Evidência")
    print("  " + "-" * 74)
    mu0, sigma0 = params[0]
    mu1, sigma1 = params[1]
    for x in eval_points:
        p0 = float(gaussian_pdf(x, mu0, sigma0))
        p1 = float(gaussian_pdf(x, mu1, sigma1))
        lam = p1 / p0
        post1 = pi1 * p1 / (pi0 * p0 + pi1 * p1)
        evidence = "Y=1" if lam > 1 else ("Y=0" if lam < 1 else "neutro")
        print(f"  {x:8.1f} | {p0:12.6e} | {p1:12.6e} | {lam:10.4f} | "
              f"{post1:10.4f} | {evidence}")

    # Etapa 5 — fronteira(s) de decisão
    print("\n[Etapa 5] Fronteira(s) de decisão: P(Y=0|x) = P(Y=1|x)")
    roots = gaussian_decision_boundaries(mu0, sigma0, mu1, sigma1, pi0, pi1)
    if roots:
        for r in roots:
            print(f"  x* = {r:.4f}")
    else:
        print("  Nenhuma fronteira real no domínio (uma classe domina).")

    # Regra de decisão por intervalos
    x_min = float(X_train[feature].min())
    x_max = float(X_train[feature].max())
    grid = np.linspace(x_min, x_max, 401)
    d = np.array([
        math.log(pi0) + float(gaussian_logpdf(x, mu0, sigma0))
        - (math.log(pi1) + float(gaussian_logpdf(x, mu1, sigma1)))
        for x in grid
    ])
    intervals = []
    current = "Y=1" if d[0] < 0 else "Y=0"
    start = x_min
    for i in range(1, len(grid)):
        decision = "Y=1" if d[i] < 0 else "Y=0"
        if decision != current:
            intervals.append((start, float(grid[i - 1]), current))
            start = float(grid[i - 1])
            current = decision
    intervals.append((start, x_max, current))
    print("  Regra de decisão induzida (intervalos de x):")
    for a, b, cls in intervals:
        print(f"    x em [{a:.1f}, {b:.1f}] -> decide {cls}")

    # Verificação: densidade implementada vs scipy.stats.norm.pdf
    check_x = float(np.median(eval_points))
    mine = float(gaussian_pdf(check_x, mu0, sigma0))
    ref = float(scipy_norm.pdf(check_x, loc=mu0, scale=sigma0))
    print(f"\n  [Verificação] gaussian_pdf({check_x:.1f}) = {mine:.8e} | "
          f"scipy.norm.pdf = {ref:.8e} | diferença = {abs(mine - ref):.2e}")


def analyze_categorical_feature(
    name: str, feature: str, X_train: pd.DataFrame, y_train: pd.Series
) -> None:
    """Executa as 5 etapas Bayesianas para uma característica categórica."""
    print("\n" + "=" * 78)
    print(f"ANÁLISE UNIVARIADA — {name}: {feature} (categórica)")
    print("=" * 78)

    categories = CHEST_PAIN_ORDER
    pi0 = float((y_train == 0).mean())
    pi1 = float((y_train == 1).mean())
    print(f"\n[Etapa 1] Hipótese: distribuição discreta (multinomial) por classe")
    print(f"  Priors do treino: P(Y=0) = {pi0:.4f}, P(Y=1) = {pi1:.4f}")
    print(f"  Suavização de Laplace com alpha = {LAPLACE_ALPHA}")

    probs = {}
    print("\n[Etapa 2] Probabilidades P(X=ak | Y=c) estimadas no treino:")
    print(f"  {'Categoria':>10} | {'P(ak|Y=0)':>12} | {'P(ak|Y=1)':>12}")
    print("  " + "-" * 46)
    for c in sorted(y_train.unique()):
        values = X_train.loc[y_train == c, feature].astype(str)
        n_c = len(values)
        probs[int(c)] = {
            cat: (float((values == cat).sum()) + LAPLACE_ALPHA)
            / (n_c + LAPLACE_ALPHA * len(categories))
            for cat in categories
        }
    for cat in categories:
        print(f"  {cat:>10} | {probs[0][cat]:12.4f} | {probs[1][cat]:12.4f}")

    # Etapas 3 e 4 — razão de verossimilhanças e posterior por categoria
    print("\n[Etapas 3/4] Razão Lambda(ak) e probabilidade a posteriori por categoria")
    print(f"  {'Categoria':>10} | {'Lambda(ak)':>12} | {'P(Y=1|ak)':>12} | Decisão")
    print("  " + "-" * 56)
    for cat in categories:
        p0, p1 = probs[0][cat], probs[1][cat]
        lam = p1 / p0
        post1 = pi1 * p1 / (pi0 * p0 + pi1 * p1)
        decision = "Y=1" if post1 >= 0.5 else "Y=0"
        print(f"  {cat:>10} | {lam:12.4f} | {post1:12.4f} | {decision}")

    # Etapa 5 — regra de decisão categórica
    print("\n[Etapa 5] Regra de decisão categórica (h3):")
    for cat in categories:
        post1 = pi1 * probs[1][cat] / (pi0 * probs[0][cat] + pi1 * probs[1][cat])
        decision = "Y=1" if post1 >= 0.5 else "Y=0"
        print(f"  X3 = {cat:>4} -> decide {decision}")


# ======================================================================
# 5. Visualizações
# ======================================================================
def plot_continuous_feature(
    feature: str, X_train: pd.DataFrame, y_train: pd.Series, model: NaiveBayesFromScratch
) -> None:
    """Histogramas por classe + densidades ajustadas + fronteira de decisão."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    mu0, sigma0 = model.gaussian_params_[0][feature]
    mu1, sigma1 = model.gaussian_params_[1][feature]
    pi0 = model.priors_[0]
    pi1 = model.priors_[1]

    # (a) comportamento observado por classe
    bins = np.histogram_bin_edges(X_train[feature], bins=20)
    axes[0].hist(X_train.loc[y_train == 0, feature], bins=bins,
                 alpha=0.55, density=True, color="tab:blue", label="Y=0 (Normal)")
    axes[0].hist(X_train.loc[y_train == 1, feature], bins=bins,
                 alpha=0.55, density=True, color="tab:red", label="Y=1 (Doença)")
    axes[0].set_title(f"{feature} — histograma por classe (treino)")
    axes[0].set_xlabel(feature)
    axes[0].set_ylabel("Densidade")
    axes[0].legend()

    # (b) densidades ajustadas + fronteiras + regiões de decisão
    x_min = float(X_train[feature].min()) - 2 * max(sigma0, sigma1)
    x_max = float(X_train[feature].max()) + 2 * max(sigma0, sigma1)
    grid = np.linspace(x_min, x_max, 600)
    p0 = gaussian_pdf(grid, mu0, sigma0)
    p1 = gaussian_pdf(grid, mu1, sigma1)
    post1 = pi1 * p1 / (pi0 * p0 + pi1 * p1)

    axes[1].plot(grid, p0, color="tab:blue", lw=2, label="p(x|Y=0)")
    axes[1].plot(grid, p1, color="tab:red", lw=2, label="p(x|Y=1)")
    axes[1].fill_between(grid, 0, p0, where=(post1 < 0.5), alpha=0.10, color="tab:blue",
                         label="Região decide Y=0")
    axes[1].fill_between(grid, 0, p1, where=(post1 >= 0.5), alpha=0.10, color="tab:red",
                         label="Região decide Y=1")

    roots = gaussian_decision_boundaries(mu0, sigma0, mu1, sigma1, pi0, pi1)
    for r in roots:
        if x_min <= r <= x_max:
            axes[1].axvline(r, color="black", ls="--", lw=1.5,
                            label=f"Fronteira x*={r:.2f}")
    axes[1].set_title(f"{feature} — densidades ajustadas e fronteira Bayesiana")
    axes[1].set_xlabel(feature)
    axes[1].set_ylabel("Densidade")
    axes[1].legend()

    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, f"univariada_{feature.lower()}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [figura] {path}")


def plot_categorical_feature(
    feature: str, X_train: pd.DataFrame, y_train: pd.Series, model: NaiveBayesFromScratch
) -> None:
    """Barras das probabilidades por categoria/classe e razão de verossimilhanças."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    categories = CHEST_PAIN_ORDER
    x_pos = np.arange(len(categories))
    width = 0.38

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    probs0 = [model.categorical_params_[feature][0][c] for c in categories]
    probs1 = [model.categorical_params_[feature][1][c] for c in categories]
    axes[0].bar(x_pos - width / 2, probs0, width, label="Y=0 (Normal)", color="tab:blue")
    axes[0].bar(x_pos + width / 2, probs1, width, label="Y=1 (Doença)", color="tab:red")
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(categories)
    axes[0].set_ylabel("P(X3=ak | Y=c)")
    axes[0].set_title(f"{feature} — probabilidades por classe (Laplace)")
    axes[0].legend()

    lam = [p1 / p0 for p0, p1 in zip(probs0, probs1)]
    colors = ["tab:red" if v >= 1 else "tab:blue" for v in lam]
    axes[1].bar(x_pos, lam, color=colors, alpha=0.8)
    axes[1].axhline(1.0, color="black", ls="--", lw=1.2, label="Lambda = 1")
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(categories)
    axes[1].set_ylabel("Lambda(ak) = P(ak|Y=1)/P(ak|Y=0)")
    axes[1].set_title(f"{feature} — razão de verossimilhanças")
    axes[1].legend()

    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, f"univariada_{feature.lower()}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [figura] {path}")


def plot_confusion_matrix(parts: dict[str, int], path: str | None = None) -> None:
    """Heatmap da matriz de confusão (linha = real, coluna = predito)."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    matrix = np.array([[parts["VN"], parts["FP"]],
                       [parts["FN"], parts["VP"]]], dtype=int)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], ["Predito 0", "Predito 1"])
    ax.set_yticks([0, 1], ["Real 0", "Real 1"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                    fontsize=18, color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    ax.set_title("Matriz de Confusão — conjunto de teste")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    path = path or os.path.join(FIGURES_DIR, "matriz_confusao.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [figura] {path}")


# ======================================================================
# 6. Função principal
# ======================================================================
def main() -> None:
    print("=" * 78)
    print("CLASSIFICADOR NAIVE BAYES DO ZERO — HEART FAILURE PREDICTION")
    print("=" * 78)

    # ------------------------------------------------------------------
    # 0. Carregamento programático da base via KaggleHub
    # ------------------------------------------------------------------
    print("\n[0] Carregando a base de dados via KaggleHub ...")
    if kagglehub is not None:
        try:
            # KaggleHub 1.x exige o nome do arquivo dentro do dataset.
            # (O caminho "" não é mais aceito; o arquivo da base é heart.csv.)
            df = kagglehub.dataset_load(
                KaggleDatasetAdapter.PANDAS,
                DATASET_HANDLE,
                DATASET_FILE,
            )
            print(f"    Base baixada: {DATASET_HANDLE} ({DATASET_FILE})")
            # Guarda uma cópia local para execuções futuras sem rede.
            os.makedirs("data", exist_ok=True)
            df.to_csv(os.path.join("data", DATASET_FILE), index=False)
        except Exception as exc:  # fallback: arquivo local, caso já exista
            print(f"    KaggleHub indisponível ({exc}).")
            local = os.path.join("data", DATASET_FILE)
            df = pd.read_csv(local)
            print(f"    Base carregada localmente: {local}")
    else:
        local = os.path.join("data", DATASET_FILE)
        df = pd.read_csv(local)
        print(f"    KaggleHub não instalado; base local: {local}")

    print(f"    Observações: {df.shape[0]} | Atributos: {df.shape[1]}")
    print(f"    Atributos: {df.columns.tolist()}")

    # ------------------------------------------------------------------
    # 1. Descrição da base e das classes
    # ------------------------------------------------------------------
    print("\n[1] Descrição da base e distribuição das classes")
    counts = df[TARGET].value_counts().sort_index()
    for c, n in counts.items():
        print(f"    Y={c}: {n} observações ({n / len(df):.2%})")

    print("\n    Atributos numéricos: Age, RestingBP, Cholesterol, FastingBS, "
          "MaxHR, Oldpeak")
    print("    Atributos categóricos: Sex, ChestPainType, RestingECG, "
          "ExerciseAngina, ST_Slope")
    print(f"    Características escolhidas: X1={CONTINUOUS_FEATURES[0]} "
          f"(contínua), X2={CONTINUOUS_FEATURES[1]} (contínua), "
          f"X3={CATEGORICAL_FEATURES[0]} (categórica)")

    # ------------------------------------------------------------------
    # 2. Separação treino/teste (apenas o treino estima parâmetros)
    # ------------------------------------------------------------------
    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,  # mantém a proporção das classes nos dois conjuntos
    )
    print(f"\n[2] Separação treino/teste")
    print(f"    Semente aleatória: {RANDOM_STATE} | "
          f"Proporção: {1 - TEST_SIZE:.0%} treino / {TEST_SIZE:.0%} teste")
    print(f"    Treino: {len(X_train)} observações | Teste: {len(X_test)} observações")
    for conjunto_name, y_set in (("Treino", y_train), ("Teste", y_test)):
        n0 = int((y_set == 0).sum())
        n1 = int((y_set == 1).sum())
        print(f"    {conjunto_name}: Y=0 -> {n0} ({n0 / len(y_set):.2%}) | "
              f"Y=1 -> {n1} ({n1 / len(y_set):.2%})")

    # ------------------------------------------------------------------
    # 3. Experimentos univariados (Etapas 1 a 5) para cada característica
    # ------------------------------------------------------------------
    print("\n[3] Experimentos isolados (uma característica por vez)")
    analyze_continuous_feature(
        "X1", CONTINUOUS_FEATURES[0], X_train, y_train,
        eval_points=[90, 110, 130, 140, 150, 160, 170, 180],
    )
    analyze_continuous_feature(
        "X2", CONTINUOUS_FEATURES[1], X_train, y_train,
        eval_points=[35, 40, 45, 50, 55, 60, 65, 70, 75],
    )
    analyze_categorical_feature("X3", CATEGORICAL_FEATURES[0], X_train, y_train)

    # ------------------------------------------------------------------
    # 4. Classificador Naive Bayes multivariado (as 3 características)
    # ------------------------------------------------------------------
    print("\n[4] Classificador Naive Bayes com as 3 características")
    model = NaiveBayesFromScratch(
        continuous_features=CONTINUOUS_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
        alpha=LAPLACE_ALPHA,
    )
    model.fit(X_train, y_train)
    print(f"    Priors estimados no treino: "
          f"P(Y=0) = {model.priors_[0]:.4f}, P(Y=1) = {model.priors_[1]:.4f}")
    for f in CONTINUOUS_FEATURES:
        (m0, s0) = model.gaussian_params_[0][f]
        (m1, s1) = model.gaussian_params_[1][f]
        print(f"    {f}: Y=0 -> N({m0:.2f}, {s0:.2f}^2) | "
              f"Y=1 -> N({m1:.2f}, {s1:.2f}^2)")
    print(f"    {CATEGORICAL_FEATURES[0]} (Laplace alpha={LAPLACE_ALPHA}):")
    for cat in CHEST_PAIN_ORDER:
        p0 = model.categorical_params_[CATEGORICAL_FEATURES[0]][0][cat]
        p1 = model.categorical_params_[CATEGORICAL_FEATURES[0]][1][cat]
        print(f"      P({cat}|Y=0) = {p0:.4f} | P({cat}|Y=1) = {p1:.4f}")

    # Exemplo de decisão logarítmica para 3 observações do teste.
    # Obs.: log P(Y=c|x) = log P(Y=c) + log p(x|Y=c) - log p(x); o termo
    # log p(x) é constante entre as classes e pode ser ignorado na decisão.
    print("\n    Exemplos de decisão no domínio logarítmico (3 casos do teste):")
    for idx in list(X_test.index)[:3]:
        row = X_test.loc[idx]
        log_liks = model.log_likelihood(row)
        log_joint = {c: math.log(model.priors_[c]) + log_liks[c]
                     for c in model.classes_}
        pred = max(log_joint, key=log_joint.get)
        true = int(y_test.loc[idx])
        print(f"      Índice {idx}: verdadeiro Y={true}, "
              f"log[P(Y=0)*p(x|Y=0)]={log_joint[0]:.4f}, "
              f"log[P(Y=1)*p(x|Y=1)]={log_joint[1]:.4f} -> previsto Y={pred}")

    y_pred = model.predict(X_test)

    # ------------------------------------------------------------------
    # 5. Avaliação no conjunto de teste
    # ------------------------------------------------------------------
    print("\n[5] Avaliação no conjunto de teste")
    parts = confusion_components(y_test, y_pred, positive=1)
    print("    Matriz de confusão (linha = real, coluna = predito):")
    print("                    Predito 0   Predito 1")
    print(f"    Real 0 (Normal)     {parts['VN']:>5}      {parts['FP']:>5}")
    print(f"    Real 1 (Doença)     {parts['FN']:>5}      {parts['VP']:>5}")
    print(f"\n    VP (Verdadeiro Positivo) = {parts['VP']} | "
          f"VN (Verdadeiro Negativo) = {parts['VN']}")
    print(f"    FP (Falso Positivo) = {parts['FP']} | "
          f"FN (Falso Negativo) = {parts['FN']}")

    metrics = metrics_from_confusion(parts)
    print("\n    Métricas:")
    for k, v in metrics.items():
        print(f"      {k:<10}: {v:.4f} ({v:.2%})")

    # ------------------------------------------------------------------
    # 6. Figuras
    # ------------------------------------------------------------------
    print("\n[6] Gerando figuras ...")
    for f in CONTINUOUS_FEATURES:
        plot_continuous_feature(f, X_train, y_train, model)
    plot_categorical_feature(CATEGORICAL_FEATURES[0], X_train, y_train, model)
    plot_confusion_matrix(parts)

    print("\nConcluído. Figuras salvas em docs/figures/.")


if __name__ == "__main__":
    main()
