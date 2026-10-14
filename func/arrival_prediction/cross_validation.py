"""
arrival_prediction.cross_validation
=====================================
Etapa C: validación cruzada con J folds (Algoritmo 1 del paper).
"""
from __future__ import annotations

import numpy as np
import polars as pl
from typing import Dict, Iterable, Optional

from .predictors import BasePredictor


class InsufficientDataError(ValueError):
    """M (viajes interpolados tras dataset_builder) < n_folds.

    np.array_split deja al menos un fold de entrenamiento vacío y los
    predictores revientan más abajo con errores de numpy poco claros
    (p.ej. 'zero-size array to reduction operation minimum'). Se levanta
    aquí, temprano y con mensaje claro, para que los wrappers (step_10,
    step_11) puedan capturarla y saltar el grupo en vez de tumbar el run.
    """


def run_cross_validation(
    T:          np.ndarray,
    l_grid:     np.ndarray,
    predictors: Dict[str, BasePredictor],
    n_folds:    int = 5,
    trip_ids:   Optional[list] = None,
    l_indices:  Optional[Iterable[int]] = None,
    seed:       int = 42,
    verbose:    bool = False,
) -> pl.DataFrame:
    M, n_plus_1 = T.shape
    n = n_plus_1 - 1

    if M < n_folds:
        raise InsufficientDataError(
            f"Solo {M} viaje(s) interpolado(s) (M) para n_folds={n_folds}: "
            f"al menos un fold de entrenamiento quedaría vacío. Baja n_folds "
            f"o usa una ventana de datos con más viajes para este grupo."
        )

    if l_indices is None:
        l_indices = np.linspace(2, n - 2, min(30, n - 4), dtype=int).tolist()
    l_indices = list(l_indices)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(M)
    folds = np.array_split(perm, n_folds)

    total_steps = n_folds * len(predictors) * len(l_indices)
    step = 0
    rows = []

    for fold_id, val_idx in enumerate(folds):
        train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != fold_id])
        T_train = T[train_idx]
        T_val   = T[val_idx]

        for name, pred in predictors.items():
            pred.fit(T_train, l_grid)

            for l_idx in l_indices:
                step += 1
                if l_idx >= n:
                    continue

                T_obs      = T_val[:, :l_idx + 1]
                T_truth    = T_val[:, l_idx + 1:]
                T_pred_all = pred.predict_batch(T_obs, l_idx)

                n_h = min(T_truth.shape[1], T_pred_all.shape[1])
                T_truth    = T_truth[:, :n_h]
                T_pred_all = T_pred_all[:, :n_h]

                err_abs = np.abs(T_pred_all - T_truth)
                err_sq  = (T_pred_all - T_truth) ** 2

                t_obs_l     = T_obs[:, -1]
                horizon_s   = T_truth - t_obs_l[:, np.newaxis]
                horizon_min = horizon_s / 60.0

                h_meters  = l_grid[l_idx + 1: l_idx + 1 + n_h] - l_grid[l_idx]
                h_idx_arr = np.arange(1, n_h + 1)

                for vi in range(T_val.shape[0]):
                    for hi in range(n_h):
                        rows.append({
                            "algorithm":   name,
                            "fold":        fold_id,
                            "trip_idx":    int(val_idx[vi]),
                            # trip_id real (si el caller lo pasa): habilita el
                            # matching 1-a-1 vs LSTM en análisis offline.
                            "trip_id":     (str(trip_ids[val_idx[vi]])
                                            if trip_ids is not None else None),
                            "l_idx":       int(l_idx),
                            "l_meters":    float(l_grid[l_idx]),
                            "h_idx":       int(h_idx_arr[hi]),
                            "h_meters":    float(h_meters[hi]),
                            "t_obs_l":     float(t_obs_l[vi]),
                            "t_true":      float(T_truth[vi, hi]),
                            "t_pred":      float(T_pred_all[vi, hi]),
                            "error_abs":   float(err_abs[vi, hi]),
                            "error_sq":    float(err_sq[vi, hi]),
                            "horizon_min": float(horizon_min[vi, hi]),
                        })

                if verbose and step % 20 == 0:
                    print(f"    [{step:4d}/{total_steps}] "
                          f"{name:7s} fold={fold_id} l={l_idx} ✓")

    df = pl.DataFrame(rows)
    if verbose:
        print(f"\n  Total filas en df_errors: {df.height:,}")
        print("=" * 60)
    return df


def summarize_rmse_by_location(df_errors: pl.DataFrame) -> pl.DataFrame:
    per_fold = (
        df_errors
        .group_by(["algorithm", "fold", "l_idx", "l_meters"])
        .agg(pl.col("error_sq").mean().sqrt().alias("rmse"))
    )
    summary = (
        per_fold
        .group_by(["algorithm", "l_idx", "l_meters"])
        .agg(
            pl.col("rmse").mean().alias("rmse_mean"),
            pl.col("rmse").std(ddof=1).alias("rmse_std"),
            pl.len().alias("n_folds"),
        )
        .sort(["algorithm", "l_idx"])
    )
    return summary


def summarize_error_by_horizon(
    df_errors:    pl.DataFrame,
    horizon_bins: Optional[list] = None,
) -> pl.DataFrame:
    if horizon_bins is None:
        horizon_bins = list(range(5, 55, 5))

    import pandas as pd
    pdf = df_errors.select(["algorithm", "error_abs", "horizon_min"]).to_pandas()
    pdf["horizon_bin_min"] = pd.cut(
        pdf["horizon_min"],
        bins=[0] + list(horizon_bins),
        labels=horizon_bins,
        right=True,
    ).astype(float)
    pdf = pdf.dropna(subset=["horizon_bin_min"])
    return pl.from_pandas(pdf)
