"""
xgb_prediction.model
======================
Etapa B: wrapper de XGBoost para la regresión del tiempo restante.

El modelo predice ``dt_rem = t_{l+h} − t_l`` (segundos). La reconstrucción a la
escala de los baselines (``t_pred = t_obs_l + dt_rem_hat``) la hace el harness
de CV, no el modelo.

Objetivo: se reportan DOS variantes, y no es un detalle cosmético.

  * ``xgb_mse`` (`reg:squarederror`) — paridad de pérdida con el LSTM, que se
    entrena con `masked_mse_loss` (`lstm_prediction_tf/trainer.py:162`).
  * ``xgb_mae`` (`reg:absoluteerror`) — alineado con las métricas del reporte
    (MAE y % on-time, ambas L1).

Medido en la ruta 101 / leg T0->T1 / fold 0 (diagnóstico de la fase 1), con
idénticos features e hiperparámetros salvo el objetivo:

    objetivo            MAE val    RMSE val
    reg:squarederror     505.01     1773.84
    reg:absoluteerror    429.32     1743.11     ← gana en AMBAS

La cola es muy pesada (RMSE/MAE ≈ 3.8), así que el predictor óptimo-MSE queda
lejos del óptimo-MAE; entrenar con MSE ni siquiera gana en RMSE porque las
colas desestabilizan el ajuste. Reportar ambas mantiene honesta la comparación
con el LSTM (que solo tiene la variante MSE) y deja ver cuánto del déficit del
LSTM podría ser desalineación de pérdida y no arquitectura.
"""
from __future__ import annotations

import numpy as np
from typing import Any, Dict, Optional

# Hiperparámetros por defecto. Conservadores: el objetivo del estudio es medir
# el efecto de la información disponible, no exprimir el modelo.
# `n_estimators` alto a propósito: el early stopping elige el número real.
DEFAULT_PARAMS: Dict[str, Any] = {
    "objective":        "reg:squarederror",
    "n_estimators":     600,
    "learning_rate":    0.05,
    "max_depth":        6,
    "min_child_weight": 20,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "reg_lambda":       1.0,
    "tree_method":      "hist",
    "n_jobs":           -1,
    "random_state":     42,
}

# Variantes que se reportan por defecto.
VARIANTS: Dict[str, Dict[str, Any]] = {
    "xgb_mse": {"objective": "reg:squarederror"},
    "xgb_mae": {"objective": "reg:absoluteerror"},
}


class XGBRemainingTime:
    """
    Regresor del tiempo restante. Interfaz mínima: `fit` / `predict`.

    Parameters
    ----------
    params : dict | None
        Sobrescribe `DEFAULT_PARAMS`.
    early_stopping_rounds : int | None
        Si se pasa junto con un `eval_set` en `fit`, activa early stopping.
    """

    name = "xgb_a"

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        early_stopping_rounds: Optional[int] = 30,
        name: Optional[str] = None,
    ) -> None:
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.early_stopping_rounds = early_stopping_rounds
        if name:
            self.name = name
        self._model = None

    # ────────────────────────────────────────────────────────────
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: Optional[tuple] = None,
        verbose: bool = False,
    ) -> "XGBRemainingTime":
        from xgboost import XGBRegressor

        kwargs = dict(self.params)
        if eval_set is not None and self.early_stopping_rounds:
            kwargs["early_stopping_rounds"] = self.early_stopping_rounds

        self._model = XGBRegressor(**kwargs)
        fit_kw: Dict[str, Any] = {"verbose": verbose}
        if eval_set is not None:
            fit_kw["eval_set"] = [eval_set]

        self._model.fit(X, y, **fit_kw)
        return self

    # ────────────────────────────────────────────────────────────
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("El modelo no está entrenado (llama a fit primero).")
        return self._model.predict(X)

    # ────────────────────────────────────────────────────────────
    @property
    def best_iteration(self) -> Optional[int]:
        """Árboles elegidos por early stopping (None si no se activó)."""
        if self._model is None:
            return None
        return getattr(self._model, "best_iteration", None)

    # ────────────────────────────────────────────────────────────
    def feature_importance(self, feature_names: list[str]) -> Dict[str, float]:
        """Importancia por ganancia total, normalizada a fracción."""
        if self._model is None:
            raise RuntimeError("El modelo no está entrenado.")
        booster = self._model.get_booster()
        raw = booster.get_score(importance_type="total_gain")
        # XGBoost devuelve claves 'f0', 'f1', … cuando se entrena con ndarray.
        out: Dict[str, float] = {}
        for k, v in raw.items():
            idx = int(k[1:]) if k.startswith("f") and k[1:].isdigit() else None
            key = feature_names[idx] if idx is not None and idx < len(feature_names) else k
            out[key] = float(v)
        total = sum(out.values()) or 1.0
        return {k: v / total for k, v in sorted(out.items(), key=lambda kv: -kv[1])}

    def __repr__(self) -> str:
        return f"<XGBRemainingTime name={self.name!r} trained={self._model is not None}>"
