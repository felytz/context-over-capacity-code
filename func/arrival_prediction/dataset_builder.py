"""
arrival_prediction.dataset_builder
====================================
Etapa A del pipeline de replicación del paper de Dublin Bus 46A.

Construye la matriz T de trayectorias interpoladas en una grid uniforme de
distancias (l_grid) a partir del df_final enriquecido del proyecto Ruta 101.

Cada fila de T es una trayectoria t^(m) = (t_0, t_1, ..., t_n), donde:
  - l_i = i * grid_step_m  (distancia desde el inicio del viaje, en metros)
  - t_i = tiempo desde el inicio del viaje (en segundos) en la posición l_i
"""
from __future__ import annotations

import numpy as np
import polars as pl
from typing import Dict, Any


def build_trajectory_matrix(
    df_final: pl.DataFrame,
    direction: int = 0,
    grid_step_m: int = 100,
    n_route_quantile: float = 0.95,
    min_coverage: float = 0.99,
    group_col: str = "direction",
    group_value: Any = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Construye la matriz de trayectorias T y la grid l_grid.

    Parameters
    ----------
    df_final : pl.DataFrame
        DataFrame enriquecido con (al menos) las columnas:
        bus_code, trip_id, time, cum_dist_m, trip_valid y la columna de
        agrupación ``group_col`` (por defecto ``direction``).
    direction : int, default=0
        Compatibilidad: valor de agrupación cuando ``group_value`` es None y
        ``group_col == "direction"``.
    group_col : str, default="direction"
        Columna que define la unidad de análisis. El método nuevo de viajes usa
        ``"leg"`` (terminal→terminal) en vez de la vieja ``direction`` 0/1.
    group_value : Any, default=None
        Valor de ``group_col`` a procesar. Si None se usa ``direction``.
    grid_step_m : int, default=100
        Tamaño del paso de la grid de interpolación en metros.
    n_route_quantile : float, default=0.95
        Quantil de max(cum_dist_m) por viaje que define la longitud n_route.
    min_coverage : float, default=0.99
        Fracción mínima de la grid que un viaje debe cubrir para incluirse.
    verbose : bool

    Returns
    -------
    dict con claves:
      - 'T'        : np.ndarray (M, n+1)  — trayectorias interpoladas
      - 'l_grid'   : np.ndarray (n+1,)    — distancias en metros
      - 'trip_ids' : list[str]            — identificador de cada fila
      - 'n_route'  : int                  — n_route en metros
      - 'metadata' : dict                 — diagnóstico
    """
    gval = group_value if group_value is not None else direction
    required = {"bus_code", "trip_id", "time", "cum_dist_m",
                "trip_valid", group_col}
    missing = required - set(df_final.columns)
    if missing:
        raise ValueError(f"Columnas faltantes en df_final: {missing}")

    if verbose:
        print("=" * 60)
        print(f"  DATASET BUILDER — {group_col}={gval}, grid={grid_step_m}m")
        print("=" * 60)

    # ── 1) Filtrar viajes válidos en el grupo elegido (leg/direction) ──────
    df = df_final.filter(
        (pl.col("trip_valid") == True) &   # noqa: E712
        (pl.col(group_col)    == gval)
    ).sort(["bus_code", "trip_id", "time"])

    n_records  = df.height
    n_trips_in = df["trip_id"].n_unique()

    if verbose:
        print(f"  Registros tras filtro trip_valid+{group_col}: {n_records:,}")
        print(f"  Viajes únicos                              : {n_trips_in:,}")

    if n_trips_in == 0:
        raise ValueError(
            f"No hay viajes válidos para {group_col}={gval!r}. "
            f"Verifica que df_final tenga trip_valid==True para ese grupo."
        )

    # ── 2) Tiempo desde el inicio del viaje (segundos) ────────────────────
    # time puede ser datetime[us] o datetime[ms]; cast a Int64 (microsegundos)
    # y normalizar dividiendo por 1_000_000 (us→s) o 1_000 (ms→s).
    time_dtype = str(df["time"].dtype)
    divisor = 1_000_000 if "us" in time_dtype else (1_000 if "ms" in time_dtype else 1)

    df = df.with_columns(
        (
            (pl.col("time").cast(pl.Int64)
             - pl.col("time").cast(pl.Int64).min().over(["bus_code", "trip_id"]))
            / divisor
        ).cast(pl.Float64).alias("t_seconds")
    )

    # ── 3) Determinar n_route a partir del quantil de max(cum_dist_m) ──────
    trip_max = (
        df.group_by(["bus_code", "trip_id"])
          .agg(pl.col("cum_dist_m").max().alias("trip_max_m"))
    )
    max_lengths = trip_max["trip_max_m"].to_numpy()
    n_route_m = float(np.quantile(max_lengths, n_route_quantile))
    n_route_m = int(np.ceil(n_route_m / grid_step_m) * grid_step_m)

    if verbose:
        print(f"  Long. media de viaje     : {np.mean(max_lengths):,.0f} m")
        print(f"  Long. P{int(100*n_route_quantile):2d} (n_route)    : "
              f"{n_route_m:,} m")

    # ── 4) Grid uniforme ───────────────────────────────────────────────────
    l_grid = np.arange(0, n_route_m + 1, grid_step_m, dtype=np.float64)
    n_grid_pts = len(l_grid)
    if verbose:
        print(f"  Puntos de grid           : {n_grid_pts}")

    # ── 5) Interpolar cada viaje en la grid ───────────────────────────────
    rows_T:           list[np.ndarray] = []
    valid_trip_ids:   list[str]        = []
    n_dropped_short      = 0
    n_dropped_nonmono    = 0
    n_dropped_few_points = 0

    df_np = (
        df.select(["bus_code", "trip_id", "cum_dist_m", "t_seconds"])
          .to_numpy()
    )
    bus_arr  = df_np[:, 0]
    trip_arr = df_np[:, 1]
    dist_arr = df_np[:, 2].astype(np.float64)
    time_arr = df_np[:, 3].astype(np.float64)

    # Encontrar fronteras de bloques (bus_code, trip_id)
    keys   = np.array([f"{b}|{t}" for b, t in zip(bus_arr, trip_arr)])
    change = np.r_[True, keys[1:] != keys[:-1]]
    starts = np.where(change)[0]
    ends   = np.r_[starts[1:], len(keys)]

    cov_threshold = min_coverage * n_route_m

    for s, e in zip(starts, ends):
        d_block = dist_arr[s:e]
        t_block = time_arr[s:e]
        trip_key = keys[s]

        if len(d_block) < 5:
            n_dropped_few_points += 1
            continue
        if d_block.max() < cov_threshold:
            n_dropped_short += 1
            continue

        # Garantizar monotonicidad estricta de la distancia
        # (mantener primer registro de cada distancia única)
        _, first_idx = np.unique(d_block, return_index=True)
        d_mono = d_block[first_idx]
        t_mono = t_block[first_idx]

        # Verificar orden creciente (debe serlo tras sort por time)
        if not np.all(np.diff(d_mono) >= 0):
            n_dropped_nonmono += 1
            continue

        # Interpolación lineal en la grid
        row = np.interp(l_grid, d_mono, t_mono)
        rows_T.append(row)
        valid_trip_ids.append(trip_key)

    if verbose:
        print(f"  Viajes interpolados (M)  : {len(rows_T):,}")
        print(f"  Descartados (pocos pts)  : {n_dropped_few_points:,}")
        print(f"  Descartados (cortos)     : {n_dropped_short:,}")
        print(f"  Descartados (no mono)    : {n_dropped_nonmono:,}")

    if len(rows_T) == 0:
        raise ValueError(
            "Ningún viaje pasó los filtros de construcción de la matriz T. "
            "Revisa min_coverage y n_route_quantile."
        )

    T = np.vstack(rows_T)   # (M, n+1)

    metadata = {
        "direction":          direction,
        "group_col":          group_col,
        "group_value":        gval,
        "grid_step_m":        grid_step_m,
        "n_route":            n_route_m,
        "n_trips_input":      n_trips_in,
        "n_trips_output":     len(rows_T),
        "n_grid_pts":         n_grid_pts,
        "dropped_short":      n_dropped_short,
        "dropped_nonmono":    n_dropped_nonmono,
        "dropped_few_points": n_dropped_few_points,
        "median_total_time_s": float(np.median(T[:, -1])),
    }

    if verbose:
        print(f"  Forma de T               : {T.shape}")
        print(f"  Tiempo mediano fin viaje : "
              f"{metadata['median_total_time_s']/60:.1f} min")
        print("=" * 60)

    return {
        "T":        T,
        "l_grid":   l_grid,
        "trip_ids": valid_trip_ids,
        "n_route":  n_route_m,
        "metadata": metadata,
    }
