"""
xgb_prediction.stratified_baselines
=====================================
Baseline `delay` ESTRATIFICADO POR HORA DEL DÍA.

Motivación
----------
En el piloto de Tier B (ruta 101 / T0→T1, q=0.05) los exógenos bajaron el MAE un
11% sobre Tier A, y las importancias apuntaron a **calendario** (hour_sin,
hour_cos, day_of_week) por encima de headway y dwell.

La explicación más simple: el prior histórico de Tier A —y el `DelayPredictor`
original— promedia **todas las horas juntas**. Los features de calendario no
aportan información nueva; solo dejan que el modelo condicione ese prior a la
hora. Si eso es lo que pasa, un `delay` estratificado por hora debería capturar
buena parte del 11% **sin ningún modelo**.

Este predictor existe para probar exactamente eso. Es el baseline honesto que el
estudio no tenía, y si gana buena parte de la brecha, la conclusión no es "hace
falta XGBoost" sino "hace falta un prior condicionado a la hora".

Formulación
-----------
Igual que `DelayPredictor` pero con la media tomada dentro del estrato:

    t̂_{l,l+h} = t_l + (μ^{(b)}_{l+h} − μ^{(b)}_l)

con b = hora del día en el punto de corte. Si el estrato tiene menos de
`min_stratum` viajes de entrenamiento, cae al promedio agrupado — así la
estratificación nunca empeora por falta de datos.

Interfaz
--------
Necesita el estrato por (viaje, corte), así que NO cumple la firma de
`arrival_prediction.BasePredictor`. Se marca con `needs_strata = True` y
`run_matched_cv` le pasa la matriz de horas.
"""
from __future__ import annotations

import numpy as np
from typing import Dict, Optional

# Franjas por defecto: hora del día (24 estratos). Con ~2000 viajes de train
# quedan ~80 por hora, suficiente para una media estable; por debajo de
# `min_stratum` se usa el fallback agrupado.
N_HOUR_BINS = 24


class HourStratifiedDelay:
    """
    `delay` con media histórica condicionada a la hora del día.

    Parameters
    ----------
    min_stratum : int
        Mínimo de viajes de train en un estrato para usar su media. Por debajo,
        se usa la media agrupada (fallback). Evita que estratos vacíos o casi
        vacíos (madrugada) generen predicciones peores que el baseline plano.
    n_bins : int
        Número de franjas horarias (24 = una por hora).
    """

    name = "delay_hour"
    needs_strata = True

    def __init__(self, min_stratum: int = 20, n_bins: int = N_HOUR_BINS) -> None:
        self.min_stratum = int(min_stratum)
        self.n_bins = int(n_bins)
        self._T: Optional[np.ndarray] = None
        self._strata: Optional[np.ndarray] = None
        self._cache: Dict[int, tuple] = {}

    # ────────────────────────────────────────────────────────────
    def fit(
        self,
        T_train: np.ndarray,
        l_grid: np.ndarray,
        strata_train: Optional[np.ndarray] = None,
    ) -> "HourStratifiedDelay":
        """
        Parameters
        ----------
        strata_train : (M_train, n_l) horas en cada corte, NaN si no hay ping.
        """
        self._T = T_train
        self._strata = strata_train
        self._cache = {}
        self.mu_ = T_train.mean(axis=0)
        return self

    # ────────────────────────────────────────────────────────────
    def _profiles(self, l_idx: int, li: int) -> tuple:
        """
        (deltas por estrato, máscara de estratos válidos, delta agrupado) para
        una posición. Se cachea: `predict_batch` se llama una vez por fold y
        posición, pero el cálculo no depende del conjunto de validación.
        """
        if li in self._cache:
            return self._cache[li]

        T = self._T
        # Restante de cada viaje de train desde l_idx: (M_train, n_h)
        rem = T[:, l_idx + 1:] - T[:, l_idx][:, None]
        pooled = rem.mean(axis=0)

        n_h = rem.shape[1]
        deltas = np.zeros((self.n_bins, n_h), dtype=np.float64)
        valid = np.zeros(self.n_bins, dtype=bool)

        if self._strata is not None:
            b_tr = self._strata[:, li]
            ok = ~np.isnan(b_tr)
            b_int = np.where(ok, np.nan_to_num(b_tr), -1).astype(int) % self.n_bins
            for b in range(self.n_bins):
                sel = ok & (b_int == b)
                if int(sel.sum()) >= self.min_stratum:
                    deltas[b] = rem[sel].mean(axis=0)
                    valid[b] = True

        out = (deltas, valid, pooled)
        self._cache[li] = out
        return out

    # ────────────────────────────────────────────────────────────
    def predict_batch(
        self,
        T_obs: np.ndarray,
        l_idx: int,
        li: int = 0,
        strata_val: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Returns
        -------
        (M_val, n_h) tiempo ACUMULADO predicho — misma convención que los
        predictores de `arrival_prediction`.
        """
        deltas, valid, pooled = self._profiles(l_idx, li)
        M_val = T_obs.shape[0]

        if strata_val is None:
            delta = np.tile(pooled, (M_val, 1))
        else:
            b_va = strata_val[:, li]
            ok = ~np.isnan(b_va)
            b_int = np.where(ok, np.nan_to_num(b_va), -1).astype(int) % self.n_bins
            use = ok & valid[b_int]
            delta = np.where(use[:, None], deltas[b_int], pooled[None, :])

        return T_obs[:, -1:] + delta

    def __repr__(self) -> str:
        return (f"<HourStratifiedDelay bins={self.n_bins} "
                f"min_stratum={self.min_stratum}>")
