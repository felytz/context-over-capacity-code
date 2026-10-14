"""
arrival_prediction.predictors
================================
Etapa B: implementación de los 4 algoritmos del paper Dublin Bus 46A.

    1. DelayPredictor             — Sec. III-A, ecuación (2)
    2. LinearRegressionPredictor  — Sec. III-B, ecuación (5)
    3. KNNPredictor               — Sec. III-C, ecuación (7), K=10
    4. KernelRegressionPredictor  — Sec. III-D, ecuaciones (8)-(10), b=1

Todos exponen la interfaz:

    .fit(T_train, l_grid)
    .predict_batch(T_obs, l_idx) -> np.ndarray (M_val, n - l_idx)

donde T_obs[:, :l_idx+1] son los time stamps observados de los viajes
de validación. La predicción es una matriz con las predicciones de t_{l+h}
para h = 1, 2, ..., n - l.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist
from typing import Optional


class BasePredictor:
    """Interfaz común. Las subclases sobrescriben fit() y predict_batch()."""

    name: str = "base"

    def fit(self, T_train: np.ndarray, l_grid: np.ndarray) -> "BasePredictor":
        raise NotImplementedError

    def predict_batch(self, T_obs: np.ndarray, l_idx: int) -> np.ndarray:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r}>"


# ─────────────────────────────────────────────────────────────────────────────
# 1) Delay-based — Sec. III-A
# ─────────────────────────────────────────────────────────────────────────────

class DelayPredictor(BasePredictor):
    """t̂_{l,l+h} = t_l + (μ_{l+h} − μ_l)

    Asume que el delay actual permanece constante el resto del viaje.
    Es la línea base contra la que se comparan los demás.
    """

    name = "delay"

    def fit(self, T_train: np.ndarray, l_grid: np.ndarray) -> "DelayPredictor":
        self.mu_     = T_train.mean(axis=0)   # (n+1,)
        self.l_grid_ = l_grid
        return self

    def predict_batch(self, T_obs: np.ndarray, l_idx: int) -> np.ndarray:
        # T_obs : (M_val, l_idx+1)
        # output: (M_val, n - l_idx)
        delta = self.mu_[l_idx + 1:] - self.mu_[l_idx]   # (n - l_idx,)
        return T_obs[:, -1:] + delta[np.newaxis, :]


# ─────────────────────────────────────────────────────────────────────────────
# 2) Linear Regression — Sec. III-B
# ─────────────────────────────────────────────────────────────────────────────

class LinearRegressionPredictor(BasePredictor):
    """t̂_{l,l+h} = μ_{l+h} + σ^T_{l,l+h} · Σ^{-1}_{ll} · (t_l − μ_l)

    Posterior media bajo gaussiana multivariada. Captura dependencias
    lineales globales entre los time stamps en distintas posiciones.

    Regularización (shrinkage RELATIVO, no absoluto)
    ------------------------------------------------
    Σ_ll se estima con M_train viajes para hasta n posiciones observadas;
    con M_train ~ cientos y l_idx ~ 100, y posiciones vecinas casi
    perfectamente correlacionadas, Σ_ll es casi singular. Un ridge
    absoluto (p. ej. 1e-6) es despreciable frente a varianzas en s² de
    orden 1e4–1e6 y NO estabiliza la inversión: los coeficientes explotan
    y las colas de error alcanzan días (observado: RMSE 28,573 s en la
    ruta 1602 ene-2024 con ~360 viajes/fold). Por eso el ridge se aplica
    relativo a la escala de la propia matriz (λ · tr(Σ_ll)/n · I), la
    forma estándar de shrinkage hacia la identidad. El paper no especifica
    el valor de la regularización; esto es un detalle de implementación
    numérica, no un cambio del método.
    """

    name = "linreg"

    def __init__(self, ridge: float = 1e-3) -> None:
        # Fracción de la varianza promedio de Σ_ll que se suma a la diagonal.
        self.ridge = float(ridge)

    def fit(self, T_train: np.ndarray, l_grid: np.ndarray) \
            -> "LinearRegressionPredictor":
        self.mu_     = T_train.mean(axis=0)        # (n+1,)
        self.cov_    = np.cov(T_train.T, ddof=0)   # (n+1, n+1)
        self.l_grid_ = l_grid
        return self

    def predict_batch(self, T_obs: np.ndarray, l_idx: int) -> np.ndarray:
        mu_obs       = self.mu_[:l_idx + 1]                  # (l+1,)
        mu_future    = self.mu_[l_idx + 1:]                  # (n-l,)
        Sigma_ll     = self.cov_[:l_idx + 1, :l_idx + 1]    # (l+1, l+1)
        sigma_future = self.cov_[:l_idx + 1, l_idx + 1:]    # (l+1, n-l)

        # Escala real de la matriz (varianza promedio de la diagonal);
        # max(·, 1.0) evita ridge nulo si Σ_ll es degenerada (todo ceros).
        scale = max(float(np.trace(Sigma_ll)) / (l_idx + 1), 1.0)
        Sigma_reg = Sigma_ll + self.ridge * scale * np.eye(l_idx + 1)
        try:
            X = np.linalg.solve(Sigma_reg, sigma_future)    # (l+1, n-l)
        except np.linalg.LinAlgError:
            X = np.linalg.pinv(Sigma_reg) @ sigma_future

        diff = T_obs - mu_obs[np.newaxis, :]                 # (M_val, l+1)
        return mu_future[np.newaxis, :] + diff @ X           # (M_val, n-l)


# ─────────────────────────────────────────────────────────────────────────────
# 3) K-Nearest Neighbor — Sec. III-C
# ─────────────────────────────────────────────────────────────────────────────

class KNNPredictor(BasePredictor):
    """t̂_{l,l+h} = t_l + (1/K) Σ_k (t^{(m_k)}_{l+h} − t^{(m_k)}_l)

    Los m_k son los K viajes de entrenamiento con menor distancia euclidiana
    en los time stamps observados. El paper reporta K=10 como buena elección.
    """

    name = "knn"

    def __init__(self, K: int = 10) -> None:
        self.K = int(K)

    def fit(self, T_train: np.ndarray, l_grid: np.ndarray) -> "KNNPredictor":
        self.T_train_ = T_train
        self.l_grid_  = l_grid
        return self

    def predict_batch(self, T_obs: np.ndarray, l_idx: int) -> np.ndarray:
        T_train_obs = self.T_train_[:, :l_idx + 1]           # (M_train, l+1)
        delta_train = (self.T_train_[:, l_idx + 1:]
                       - self.T_train_[:, l_idx:l_idx + 1])  # (M_train, n-l)

        K = min(self.K, T_train_obs.shape[0])

        dists  = cdist(T_obs, T_train_obs, metric="euclidean")  # (M_val, M_train)
        nn_idx = np.argpartition(dists, K - 1, axis=1)[:, :K]  # (M_val, K)

        # delta_train[nn_idx] → (M_val, K, n-l)
        avg_delta = delta_train[nn_idx].mean(axis=1)            # (M_val, n-l)
        return T_obs[:, -1:] + avg_delta


# ─────────────────────────────────────────────────────────────────────────────
# 4) Kernel Regression — Sec. III-D (algoritmo ganador del paper)
# ─────────────────────────────────────────────────────────────────────────────

class KernelRegressionPredictor(BasePredictor):
    """Kernel gaussiano estandarizado — algoritmo ganador del paper.

    t̂_{l,l+h} = t_l + Σ_m w_m · Δ^(m)_{l+h}
                        ─────────────────────
                              Σ_m w_m

    con w_m = exp(-||x_obs − x^(m)||² / b), argumentos estandarizados
    por max(1, σ²_i). b=1 según el paper.
    """

    name = "kernel"

    def __init__(self, b: float = 1.0) -> None:
        self.b = float(b)

    def fit(self, T_train: np.ndarray, l_grid: np.ndarray) \
            -> "KernelRegressionPredictor":
        self.T_train_    = T_train
        self.l_grid_     = l_grid
        # Varianzas empíricas con clip a 1 (paper, fórmula 10)
        self.var_clipped_ = np.maximum(1.0, T_train.var(axis=0, ddof=0))
        self.std_         = np.sqrt(self.var_clipped_)
        # Pre-estandarizar train
        self.T_train_std_ = T_train / self.std_[np.newaxis, :]
        return self

    def predict_batch(self, T_obs: np.ndarray, l_idx: int) -> np.ndarray:
        T_obs_std       = T_obs / self.std_[np.newaxis, :l_idx + 1]
        T_train_obs_std = self.T_train_std_[:, :l_idx + 1]

        # Distancias estandarizadas al cuadrado: (M_val, M_train)
        sq_dists = cdist(T_obs_std, T_train_obs_std, metric="sqeuclidean")

        # Estabilizar exp restando el mínimo por fila
        sq_dists -= sq_dists.min(axis=1, keepdims=True)
        weights     = np.exp(-sq_dists / self.b)               # (M_val, M_train)
        weights_sum = np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)

        delta_train    = (self.T_train_[:, l_idx + 1:]
                          - self.T_train_[:, l_idx:l_idx + 1])  # (M_train, n-l)
        weighted_delta = (weights @ delta_train) / weights_sum   # (M_val, n-l)

        return T_obs[:, -1:] + weighted_delta


# ─────────────────────────────────────────────────────────────────────────────
# Factoría
# ─────────────────────────────────────────────────────────────────────────────

def default_predictors(K: int = 10, b: float = 1.0) -> dict[str, BasePredictor]:
    """Devuelve los 4 predictores con la configuración del paper."""
    return {
        "delay":  DelayPredictor(),
        "linreg": LinearRegressionPredictor(),
        "knn":    KNNPredictor(K=K),
        "kernel": KernelRegressionPredictor(b=b),
    }
