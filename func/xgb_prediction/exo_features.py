"""
xgb_prediction.exo_features
=============================
Tier B — features EXÓGENOS tomados de `df_full`.

Qué añade sobre Tier A
----------------------
Tier A solo ve la trayectoria (posición y tiempo por celda), que es exactamente
la información que consumen los baselines y el LSTM en modo grid. Tier B añade
lo que el bus "sabe" del entorno en el instante de corte: headway, dwell
acumulado, velocidad reciente y calendario.

Las columnas base ya las calcula el paso 07 (`_lib_features/extended.py`) y
están densas en `df_full` (0% de nulos verificado en la ruta 101).

⚠ REGLA ANTI-FUGA
-----------------
Todo se muestrea en el **último ping con `cum_dist_m <= l_grid[l_idx]`**, nunca
más allá. Los acumulados (dwell, σ de velocidad) se suman solo sobre esa parte
observada del viaje. Nada de `time_to_trip_end` ni `s_next_stop*`: son el target
disfrazado.

Tampoco se usan las columnas `hist_*` de `df_full`: se construyeron con su
propio split de train en el paso 07, que no coincide con estos folds. El prior
histórico lo reconstruye Tier A desde `T_train` (ver `feature_builder.py`).
"""
from __future__ import annotations

import numpy as np
import polars as pl
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Headway sin bus adelante/atrás llega a valores absurdos (p90 = 23,343 s = 6.5 h
# en la ruta 101). Más allá de una hora la señal es la misma —"no hay bus
# cerca"— así que se satura ahí en vez de dejar que el árbol parta sobre basura.
HEADWAY_CAP_S = 3600.0

EXO_FEATURES: List[str] = [
    "hour_sin",            # hora del día, codificación cíclica
    "hour_cos",
    "day_of_week",
    "is_weekend",
    "headway_ahead",       # saturado a HEADWAY_CAP_S
    "headway_behind",
    "dwell_cum_s",         # dwell acumulado en el tramo OBSERVADO
    "dwell_recent_s",      # dwell en los últimos 10 pings observados
    "speed_recent_5min",   # en el ping de corte
    "speed_std_obs",       # σ de la velocidad GPS en el tramo observado
    "segment_count_ahead",
    "n_pings_obs",         # densidad de datos hasta el corte
]

# Grupos para la ablación. La ganancia por splits NO sirve para atribuir el
# aporte de cada grupo: los priors absorben ~91% del gain y aun así los exógenos
# mueven el MAE un 11%. La atribución honesta es añadir un grupo a la vez sobre
# Tier A y medir cuánto MAE recupera cada uno.
EXO_GROUPS: Dict[str, List[str]] = {
    "cal":     ["hour_sin", "hour_cos", "day_of_week", "is_weekend"],
    "headway": ["headway_ahead", "headway_behind"],
    "dwell":   ["dwell_cum_s", "dwell_recent_s"],
    "speed":   ["speed_recent_5min", "speed_std_obs"],
    "route":   ["segment_count_ahead", "n_pings_obs"],
}


def group_cols(groups: Optional[Sequence[str]]) -> Optional[List[int]]:
    """
    Índices de columna (dentro de EXO_FEATURES) de los grupos pedidos.

    None → Tier A (sin exógenos). "all" o la lista completa → Tier B entero.
    """
    if groups is None:
        return None
    if isinstance(groups, str):
        groups = [groups]
    if "all" in groups:
        return list(range(len(EXO_FEATURES)))
    names: List[str] = []
    for g in groups:
        if g not in EXO_GROUPS:
            raise ValueError(
                f"grupo desconocido {g!r}; disponibles: {sorted(EXO_GROUPS)}"
            )
        names.extend(EXO_GROUPS[g])
    return [EXO_FEATURES.index(n) for n in names]


# Columnas que se leen de df_full.
_SRC_COLS = [
    "bus_code", "trip_id", "time", "cum_dist_m",
    "hour_of_day", "day_of_week", "headway_ahead", "headway_behind",
    "dwell_time", "speed_recent_5min", "speed", "segment_count_ahead",
]


def build_exo_matrix(
    df_full:     pl.DataFrame,
    group_col:   str,
    group_value: Any,
    trip_ids:    Sequence[str],
    l_grid:      np.ndarray,
    l_indices:   Sequence[int],
    verbose:     bool = False,
    return_hours: bool = False,
):
    """
    Construye el tensor de features exógenos.

    Parameters
    ----------
    trip_ids : claves "bus_code|trip_id" EN EL ORDEN DE LAS FILAS DE T
        (tal como las devuelve `build_trajectory_matrix`). El orden importa:
        la fila i de la salida corresponde a la fila i de T.

    return_hours : si True devuelve también la matriz (M, n_l) con la HORA
        CRUDA en cada corte, que usa el baseline estratificado por hora.

    Returns
    -------
    np.ndarray (M, n_l, len(EXO_FEATURES)) float32
        Fila = viaje · columna = posición de corte · profundidad = feature.
        Los viajes de `trip_ids` que no aparezcan en df_full quedan en NaN
        (XGBoost trata NaN de forma nativa).
    Si `return_hours`, devuelve la tupla (exo, hours) con hours (M, n_l) float32
    y NaN donde no hay ping observado.
    """
    missing = set(_SRC_COLS) - set(df_full.columns)
    if missing:
        raise ValueError(
            f"df_full no tiene las columnas de Tier B: {sorted(missing)}. "
            f"Requiere el paso 07 completo (compute_all_features)."
        )

    n_l = len(l_indices)
    k   = len(EXO_FEATURES)
    M   = len(trip_ids)
    out = np.full((M, n_l, k), np.nan, dtype=np.float32)
    hours_out = np.full((M, n_l), np.nan, dtype=np.float32)

    df = (df_full
          .filter((pl.col("trip_valid") == True) &          # noqa: E712
                  (pl.col(group_col) == group_value))
          .select(_SRC_COLS)
          .sort(["bus_code", "trip_id", "time"]))

    if df.height == 0:
        return (out, hours_out) if return_hours else out

    key_arr = (df["bus_code"].cast(pl.Utf8) + "|"
               + df["trip_id"].cast(pl.Utf8)).to_numpy()

    dist = df["cum_dist_m"].to_numpy().astype(np.float64)
    hour = df["hour_of_day"].to_numpy().astype(np.float64)
    dow  = df["day_of_week"].to_numpy().astype(np.float64)
    hw_a = df["headway_ahead"].to_numpy().astype(np.float64)
    hw_b = df["headway_behind"].to_numpy().astype(np.float64)
    dwl  = np.nan_to_num(df["dwell_time"].to_numpy().astype(np.float64))
    spr  = df["speed_recent_5min"].to_numpy().astype(np.float64)
    spd  = df["speed"].to_numpy().astype(np.float64)
    sca  = df["segment_count_ahead"].to_numpy().astype(np.float64)

    # Fronteras de bloque por viaje (mismo patrón que build_trajectory_matrix)
    change = np.r_[True, key_arr[1:] != key_arr[:-1]]
    starts = np.where(change)[0]
    ends   = np.r_[starts[1:], len(key_arr)]
    pos_of_key = {key_arr[s]: i for i, s in enumerate(starts)}

    l_meters = np.array([l_grid[l] for l in l_indices], dtype=np.float64)

    n_found = 0
    for row, key in enumerate(trip_ids):
        bi = pos_of_key.get(str(key))
        if bi is None:
            continue
        n_found += 1
        s, e = starts[bi], ends[bi]

        d_b = dist[s:e]
        # Último ping OBSERVADO en cada corte: cum_dist_m <= l_meters.
        # searchsorted 'right' − 1 da ese índice; si el viaje aún no llegó a
        # esa distancia, j = -1 y la fila queda NaN (no se inventa nada).
        j_all = np.searchsorted(d_b, l_meters, side="right") - 1

        dwl_b = dwl[s:e]
        cum_dwell = np.concatenate([[0.0], np.cumsum(dwl_b)])
        spd_b = spd[s:e]
        # Sumas acumuladas para σ en O(1) por corte
        c1 = np.concatenate([[0.0], np.cumsum(spd_b)])
        c2 = np.concatenate([[0.0], np.cumsum(spd_b ** 2)])

        for li, j in enumerate(j_all):
            if j < 0:
                continue
            n_obs = j + 1

            h = hour[s + j]
            hours_out[row, li] = h
            out[row, li, 0] = np.sin(2 * np.pi * h / 24.0)
            out[row, li, 1] = np.cos(2 * np.pi * h / 24.0)
            dw = dow[s + j]
            out[row, li, 2] = dw
            out[row, li, 3] = 1.0 if dw >= 5 else 0.0
            out[row, li, 4] = min(hw_a[s + j], HEADWAY_CAP_S)
            out[row, li, 5] = min(hw_b[s + j], HEADWAY_CAP_S)
            out[row, li, 6] = cum_dwell[n_obs]
            lo = max(0, n_obs - 10)
            out[row, li, 7] = cum_dwell[n_obs] - cum_dwell[lo]
            out[row, li, 8] = spr[s + j]
            mean = c1[n_obs] / n_obs
            var = max(c2[n_obs] / n_obs - mean * mean, 0.0)
            out[row, li, 9] = np.sqrt(var)
            out[row, li, 10] = sca[s + j]
            out[row, li, 11] = float(n_obs)

    if verbose:
        cov = 100.0 * np.mean(~np.isnan(out[:, :, 0]))
        print(f"    Tier B: {n_found}/{M} viajes emparejados · "
              f"cobertura de cortes {cov:.1f}%", flush=True)

    return (out, hours_out) if return_hours else out
