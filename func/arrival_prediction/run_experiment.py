"""
arrival_prediction.run_experiment
====================================
Etapa E: orquestador del pipeline completo.

Función pública: build_arrival_prediction_pipeline()

Replica el experimento del paper sobre tu df_final:
    1. Construye la matriz de trayectorias T (Etapa A)
    2. Corre 5-fold CV con los 4 predictores (Etapas B + C)
    3. (Opcional) Repite con grid 500 m para Figura 6
    4. Genera reporte estático (PDF + PNGs) y dashboard HTML
    5. Persiste df_errors en parquet
"""
from __future__ import annotations

import polars as pl
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List

from .dataset_builder import build_trajectory_matrix
from .predictors      import default_predictors
from .cross_validation import run_cross_validation
from .plots           import build_static_report, build_html_dashboard


def build_arrival_prediction_pipeline(
    df_final:          pl.DataFrame,
    direction:         int   = 0,
    group_col:         str   = "direction",
    group_value:       object = None,
    grid_step_m:       int   = 100,
    n_folds:           int   = 5,
    K:                 int   = 10,
    b:                 float = 1.0,
    n_route_quantile:  float = 0.95,
    min_coverage:      float = 0.99,
    l_indices:         Optional[List[int]] = None,
    horizon_bins_min:  Optional[List[float]] = None,
    include_grid_500:  bool  = True,
    output_dir:        str   = "./arrival_prediction_output/",
    seed:              int   = 42,
    verbose:           bool  = True,
) -> Dict:
    """
    Orquestador completo. Replica el experimento del paper sobre tu df_final.

    Parameters
    ----------
    df_final : pl.DataFrame
        DataFrame enriquecido (salida del paso 07). Debe contener:
        bus_code, trip_id, time, cum_dist_m, direction, trip_valid.
    direction : int
        Dirección a procesar (0 o 1).
    grid_step_m : int
        Tamaño de la grid de interpolación principal (paper: 100 m).
    n_folds : int
        Número de folds de CV (paper: 5).
    K : int
        Vecinos para K-NN (paper: K=10).
    b : float
        Bandwidth del kernel gaussiano (paper: b=1.0).
    n_route_quantile : float
        Cuantil de cum_dist_m para definir n_route. El paper original usa P95,
        pero ese valor retiene solo el ~5% de viajes MÁS LARGOS y sesga la
        muestra 2.0 sd en longitud (ver `docs/PLAN_REPRESENTATIVIDAD.md` §1.1):
        el régimen del estudio pasó a **0.05** (soporte común).
    min_coverage : float
        Fracción de la grid que un viaje debe cubrir. `np.interp` **satura** en
        vez de extrapolar, así que por debajo de 1.0 el tramo no cubierto lleva
        tiempo acumulado constante ⇒ tiempo restante FABRICADO. 1.0 lo elimina
        y hace que `n_pairs` cuadre exacto con el LSTM sobre el mismo frame.
    l_indices : list[int] or None
        Posiciones en la grid donde se hacen predicciones.
        None → 30 puntos uniformes.
    horizon_bins_min : list[float] or None
        Bins de horizonte temporal en minutos para la Figura 5.
        None → [5, 10, 15, ..., 50].
    include_grid_500 : bool
        Si True, repite el CV con grid de 500 m (Figura 6 del paper).
    output_dir : str
        Carpeta de salida.
    seed : int
        Semilla para la reproducibilidad de los folds.
    verbose : bool

    Returns
    -------
    dict con:
        T              : np.ndarray (M, n+1) — matriz de trayectorias
        l_grid         : np.ndarray (n+1,)  — grid en metros
        df_errors      : pl.DataFrame — errores del CV (grid 100m)
        df_errors_500  : pl.DataFrame | None — errores del CV (grid 500m)
        metadata       : dict — diagnóstico del dataset
        paths          : dict — rutas a todos los outputs generados
    """
    # Unidad de análisis: leg (terminal→terminal) o, por compat, direction 0/1.
    gval = group_value if group_value is not None else direction

    if verbose:
        print("█" * 60)
        print(f"  ARRIVAL PREDICTION PIPELINE")
        print(f"  {group_col}={gval} · grid={grid_step_m}m · "
              f"{n_folds}-fold CV · K={K} · b={b}")
        print("█" * 60)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Etapa A: construir matriz de trayectorias ──────────────────────────
    if verbose:
        print("\n  [A] Construyendo matriz de trayectorias …")
    ds = build_trajectory_matrix(
        df_final,
        group_col        = group_col,
        group_value      = gval,
        grid_step_m      = grid_step_m,
        n_route_quantile = n_route_quantile,
        min_coverage     = min_coverage,
        verbose          = verbose,
    )
    T      = ds["T"]
    l_grid = ds["l_grid"]
    metadata = ds["metadata"]

    # ── Etapas B+C: 4 predictores + 5-fold CV (grid principal) ────────────
    if verbose:
        print(f"\n  [B+C] Validación cruzada ({n_folds} folds) …")
    preds = default_predictors(K=K, b=b)
    # trip_ids de la matriz vienen como "bus|trip_id"; se exporta el trip_id
    # crudo (llave de join con los errores del LSTM para el matching 1-a-1).
    _raw_trip_ids = [str(k).split("|", 1)[-1] for k in ds.get("trip_ids", [])] or None
    df_errors = run_cross_validation(
        T, l_grid, preds,
        n_folds   = n_folds,
        trip_ids  = _raw_trip_ids,
        l_indices = l_indices,
        seed      = seed,
        verbose   = verbose,
    )

    # ── Guardar parquet de errores ─────────────────────────────────────────
    parquet_path = out_dir / "df_errors.parquet"
    from _lib_data import save_df
    save_df(df_errors, parquet_path)
    if verbose:
        print(f"  ✔ Parquet errores: {parquet_path.name}")

    # ── (Opcional) CV con grid 500 m ──────────────────────────────────────
    df_errors_500 = None
    if include_grid_500:
        if verbose:
            print("\n  [B+C·500] Repitiendo con grid 500 m (Figura 6) …")
        ds500 = build_trajectory_matrix(
            df_final,
            group_col        = group_col,
            group_value      = gval,
            grid_step_m      = 500,
            n_route_quantile = n_route_quantile,
            min_coverage     = min_coverage,
            verbose          = False,
        )
        _raw_trip_ids_500 = ([str(k).split("|", 1)[-1] for k in ds500.get("trip_ids", [])]
                             or None)
        df_errors_500 = run_cross_validation(
            ds500["T"], ds500["l_grid"], default_predictors(K=K, b=b),
            n_folds   = n_folds,
            trip_ids  = _raw_trip_ids_500,
            l_indices = l_indices,
            seed      = seed,
            verbose   = False,
        )
        p500 = out_dir / "df_errors_500.parquet"
        from _lib_data import save_df
        save_df(df_errors_500, p500)
        if verbose:
            print(f"  ✔ Parquet errores (grid 500m): {p500.name}")

    # ── Etapa D: visualizaciones ───────────────────────────────────────────
    if verbose:
        print("\n  [D] Generando figuras y dashboard …")

    static_paths = build_static_report(
        df_errors        = df_errors,
        T                = T,
        l_grid           = l_grid,
        output_dir       = str(out_dir / "static"),
        df_errors_500    = df_errors_500,
        horizon_bins_min = horizon_bins_min,
        metadata         = metadata,
        direction        = gval,
        grid_step_m      = grid_step_m,
        verbose          = verbose,
    )

    html_path = build_html_dashboard(
        df_errors        = df_errors,
        output_path      = str(out_dir / "dashboard.html"),
        metadata         = metadata,
        horizon_bins_min = horizon_bins_min,
        verbose          = verbose,
    )

    paths = {
        "parquet":   str(parquet_path),
        "dashboard": html_path,
        **static_paths,
    }

    if verbose:
        print(f"\n  ── Resumen RMSE final por algoritmo ──")
        from .cross_validation import summarize_rmse_by_location
        rmse = summarize_rmse_by_location(df_errors)
        by_algo = (
            rmse.group_by("algorithm")
                .agg(pl.col("rmse_mean").mean().alias("avg"))
                .sort("avg")
        )
        for r in by_algo.iter_rows(named=True):
            print(f"    {r['algorithm']:7s} → RMSE prom = {r['avg']:6.1f} s")
        print(f"\n  ✅ Todo en: {out_dir}")
        print("█" * 60 + "\n")

    return {
        "T":             T,
        "l_grid":        l_grid,
        "df_errors":     df_errors,
        "df_errors_500": df_errors_500,
        "metadata":      metadata,
        "paths":         paths,
    }
