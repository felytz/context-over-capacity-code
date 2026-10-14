"""Features dinámicas (speed/headways/segment_count) + orquestador compute_all_features.

La tabla histórica (hist_p50, hist_p90, relative_progress_vs_median, …) y las
gráficas de diagnóstico se quitaron: ningún modelo las leía. Están en el tag
`features-completas`.

Submódulo de `_lib_features` (antes era una sección del módulo monolítico).
Importar desde el paquete: `from _lib_features import ...`.
"""
from __future__ import annotations

from datetime import timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl

from _lib_geo import _haversine_m

from .enrich import enrich_trip_segments
from .interp import compute_interpolated_crossings
from .segment_features import compute_alternative_features

##############################################################################
#   4. Extended + Orquestador compute_all_features
#   (originalmente: extended_features.py)
##############################################################################

###############################################################################
###                              HELPERS                                    ###
###############################################################################

# _haversine_m: ver import al inicio del módulo (consolidado en lib/_lib_geo.py)


def _validate_required_columns(df: pl.DataFrame, required: set, fn_name: str):
    """Lanza ValueError si faltan columnas requeridas."""
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"[{fn_name}] columnas faltantes: {missing}")


###############################################################################
###             FEATURE 1 — VELOCIDAD MEDIA RECIENTE                        ###
###############################################################################

def compute_speed_recent_window(
    df: pl.DataFrame,
    window_seconds: int = 300,
    output_column: str = "speed_recent_5min",
    verbose: bool = False,
) -> pl.DataFrame:
    """
    Velocidad media espacial del propio bus en los últimos `window_seconds`
    segundos, computada como Δcum_dist_m / Δt sobre los registros del propio
    bus en [time-window, time].

    Resultado en km/h. NaN si no hay registros suficientes en la ventana
    (típicamente al inicio de un viaje o tras un gap > window_seconds).

    REFERENCIAS
    -----------
        Cathey & Dailey (2003), Petersen et al. (2019).
        Quiroga & Bullock (1998) — uso de ventana espacial-temporal en
        lugar de velocidad instantánea por SNR bajo muestreo.

    NOTA
    ----
        - Requiere `cum_dist_m` (de camino1_interpolation).
        - No usa Haversine entre dos pings consecutivos (ruido alto en
          urbano a 60 s).
    """
    _validate_required_columns(
        df, {"bus_code", "trip_id", "time", "cum_dist_m"},
        "compute_speed_recent_window"
    )

    if verbose:
        print(f"  Computing {output_column} (ventana = {window_seconds}s)...")

    df_sorted = df.sort(["bus_code", "trip_id", "time"])

    times = df_sorted["time"].to_list()
    buses = df_sorted["bus_code"].to_list()
    trips = df_sorted["trip_id"].to_list()
    cumd  = df_sorted["cum_dist_m"].cast(pl.Float64).to_numpy()
    n     = len(times)

    speed = np.full(n, np.nan, dtype=np.float64)

    # Loop por bloque (bus_code, trip_id)
    i = 0
    while i < n:
        j = i
        while (j < n
               and buses[j] == buses[i]
               and trips[j] == trips[i]):
            j += 1

        # Dentro del bloque [i, j), para cada r encontrar el k más temprano
        # tal que (times[r] - times[k]) <= window_seconds.
        # Usamos two-pointer porque los tiempos son monótonos crecientes.
        k = i
        for r in range(i, j):
            t_r = times[r]
            t_lower = t_r - timedelta(seconds=window_seconds)
            while k < r and times[k] < t_lower:
                k += 1
            if k == r:
                # Solo el propio registro o ninguno previo en la ventana
                continue
            dt = (t_r - times[k]).total_seconds()
            if dt <= 0:
                continue
            d_m = cumd[r] - cumd[k]
            if d_m < 0:
                # No debería pasar (cum_dist_m monótono por viaje), pero
                # por seguridad numérica
                continue
            # m/s -> km/h
            speed[r] = (d_m / dt) * 3.6

        i = j

    df_out = df_sorted.with_columns(
        pl.Series(output_column, np.round(speed, 2))
    )

    if verbose:
        valid = ~np.isnan(speed)
        if valid.any():
            print(f"    mediana = {np.nanmedian(speed[valid]):.1f} km/h")
            print(f"    cobertura = {valid.sum()/n:.1%}")
    return df_out


###############################################################################
###             FEATURE 2 — HEADWAYS (BUS DELANTERO Y TRASERO)              ###
###############################################################################

_US_POR_S = 1_000_000


def _timedelta_us(seg: np.ndarray) -> np.ndarray:
    """Los microsegundos de `timedelta(seconds=s)` para cada s ≥ 0, redondeados como CPython
    (Modules/_datetimemodule.c, `delta_new` y `accum`): la parte entera, exacta; la fracción
    por 1e6 en double; y lo que sobra, al microsegundo más cercano, y en .5 exacto, al total
    par."""
    frac, ent = np.modf(seg)
    us = ent.astype(np.int64) * _US_POR_S
    frac2, ent2 = np.modf(frac * 1e6)
    us = us + ent2.astype(np.int64)
    return us + ((frac2 > 0.5) | ((frac2 == 0.5) & (us % 2 == 1)))


def _headways_bucle(
    df_dir: pl.DataFrame, max_headway_seconds: float
) -> Tuple[np.ndarray, np.ndarray]:
    """El cálculo original, registro por registro: O(n²) en Python puro. Era el de
    `compute_headways` hasta el 7-oct-2026; queda como referencia de
    `_headways_vectorizado`, que da lo mismo bit a bit."""
    buses_dir = df_dir["bus_code"].to_list()
    times_dir = df_dir["time"].to_list()
    cumd_dir  = df_dir["cum_dist_m"].cast(pl.Float64).to_numpy()

    head_ahead  = np.full(len(times_dir), np.nan, dtype=np.float64)
    head_behind = np.full(len(times_dir), np.nan, dtype=np.float64)

    # Pre-agrupar índices por bus_code dentro de la dirección
    bus_to_indices: Dict[str, List[int]] = {}
    for local_i, b in enumerate(buses_dir):
        bus_to_indices.setdefault(b, []).append(local_i)

    # Para cada registro target, examinar el resto de buses en la dirección
    for local_target, t_target in enumerate(times_dir):
        x_target = cumd_dir[local_target]
        bus_target = buses_dir[local_target]

        t_lower = t_target - timedelta(seconds=max_headway_seconds)
        t_upper = t_target + timedelta(seconds=max_headway_seconds)

        best_ahead_dt  = None  # mayor t_b < t_target
        best_behind_dt = None  # menor t_b > t_target

        for bus_other, idxs in bus_to_indices.items():
            if bus_other == bus_target:
                continue
            # Buscar dos índices consecutivos del mismo bus tales que
            # cum_dist_m del bus_other los flanquea x_target, y al menos
            # uno está dentro de la ventana temporal.
            xs = cumd_dir[idxs]
            ts = [times_dir[k] for k in idxs]

            # Filtrar a la ventana temporal expandida (con cierto margen
            # para captar el cruce)
            # Buscar consecutivos donde xs[a] <= x_target <= xs[a+1]
            # o viceversa (dependiendo del orden monotónico del bus_other)
            for a in range(len(idxs) - 1):
                x_a, x_b = xs[a], xs[a + 1]
                t_a, t_b = ts[a], ts[a + 1]

                # Solo registros dentro de la ventana temporal
                if t_b < t_lower or t_a > t_upper:
                    continue

                # x_target debe estar entre x_a y x_b (cualquier orden)
                lo_x, hi_x = min(x_a, x_b), max(x_a, x_b)
                if not (lo_x <= x_target <= hi_x):
                    continue

                # Interpolación lineal del tiempo en que pasa por x_target
                if abs(x_b - x_a) < 1e-3:
                    t_cross = t_a
                else:
                    alpha = (x_target - x_a) / (x_b - x_a)
                    alpha = max(0.0, min(1.0, alpha))
                    dt_step = (t_b - t_a).total_seconds()
                    t_cross = t_a + timedelta(
                        seconds=alpha * dt_step
                    )

                delta = (t_cross - t_target).total_seconds()
                if delta < 0:
                    # bus_other pasó antes -> candidato para ahead
                    if (best_ahead_dt is None
                            or abs(delta) < best_ahead_dt):
                        best_ahead_dt = abs(delta)
                else:
                    # bus_other pasó después -> candidato para behind
                    if (best_behind_dt is None
                            or delta < best_behind_dt):
                        best_behind_dt = delta

        if best_ahead_dt is not None:
            head_ahead[local_target] = best_ahead_dt
        if best_behind_dt is not None:
            head_behind[local_target] = best_behind_dt

    return head_ahead, head_behind


def _headways_vectorizable(df_dir: pl.DataFrame) -> bool:
    """`_headways_vectorizado` replica el bucle con tiempos ingenuos en ms o µs (los de
    `to_list()` son exactos en enteros de µs) y sin nulos en time ni en bus_code. Lo demás
    va por el bucle."""
    dt = df_dir.schema["time"]
    return (isinstance(dt, pl.Datetime) and dt.time_zone is None
            and dt.time_unit in ("us", "ms")
            and df_dir["time"].null_count() == 0
            and df_dir["bus_code"].null_count() == 0)


def _headways_vectorizado(
    df_dir: pl.DataFrame,
    max_headway_seconds: float,
    bloque: int = 512,
    corto_s: int = 600,
) -> Tuple[np.ndarray, np.ndarray]:
    """Lo mismo que `_headways_bucle`, bit a bit, por bloques de registros con numpy.

    Mismos pares (registros consecutivos de cada bus en el orden de `df_dir`), mismas
    condiciones y mismas operaciones en double: el tiempo va en enteros de µs y el cruce se
    redondea como `timedelta` (`_timedelta_us`). Los pares candidatos de un bloque salen de los
    pares ordenados por su inicio: los cortos (≤ `corto_s`) por búsqueda binaria y los largos
    (huecos: la noche, una pausa), todos los que se solapan con la ventana del bloque.

    `df_dir` llega ordenado por tiempo (`compute_headways`): en cada par t_b ≥ t_a, y el
    desfase del cruce, alfa·(t_b − t_a), nunca es negativo."""
    t = df_dir["time"].dt.epoch("us").to_numpy()
    bus = df_dir["bus_code"].rank("dense").to_numpy()
    x = df_dir["cum_dist_m"].cast(pl.Float64).to_numpy()
    h_us = timedelta(seconds=max_headway_seconds) // timedelta(microseconds=1)
    corto_us = corto_s * _US_POR_S
    m = t.size

    ahead = np.full(m, np.inf, dtype=np.float64)
    behind = np.full(m, np.inf, dtype=np.float64)

    # Pares de registros consecutivos del mismo bus (orden estable: el de df_dir)
    orden = np.argsort(bus, kind="stable")
    mismo = bus[orden[1:]] == bus[orden[:-1]]
    pa, pb = orden[:-1][mismo], orden[1:][mismo]
    if pa.size:
        ta, tb, xa, xb, p_bus = t[pa], t[pb], x[pa], x[pb], bus[pa]
        lo = np.where(xb < xa, xb, xa)   # min(x_a, x_b) de Python, con NaN incluido
        hi = np.where(xb > xa, xb, xa)   # max(x_a, x_b)
        dx = xb - xa
        plano = np.abs(dx) < 1e-3
        dt_s = (tb - ta) / 1e6           # timedelta.total_seconds()

        largo = (tb - ta) > corto_us
        i_l = np.flatnonzero(largo)
        i_c = np.flatnonzero(~largo)
        i_c = i_c[np.argsort(ta[i_c], kind="stable")]
        ta_c = ta[i_c]

        for s in range(0, m, bloque):
            e = min(s + bloque, m)
            T, X, B = t[s:e], x[s:e], bus[s:e]
            t0, t1 = T.min(), T.max()
            cand = i_c[np.searchsorted(ta_c, t0 - h_us - corto_us, side="left"):
                       np.searchsorted(ta_c, t1 + h_us, side="right")]
            if i_l.size:
                cand = np.concatenate([cand, i_l[(ta[i_l] <= t1 + h_us)
                                                 & (tb[i_l] >= t0 - h_us)]])
            if not cand.size:
                continue
            # Las condiciones del bucle, registro × par: otro bus, el par se solapa con
            # [t - h, t + h] y flanquea la posición del registro.
            ok = ((tb[cand][None, :] >= (T - h_us)[:, None])
                  & (ta[cand][None, :] <= (T + h_us)[:, None])
                  & (p_bus[cand][None, :] != B[:, None])
                  & (lo[cand][None, :] <= X[:, None])
                  & (X[:, None] <= hi[cand][None, :]))
            ii, jj = np.nonzero(ok)
            if not ii.size:
                continue
            p = cand[jj]
            with np.errstate(divide="ignore", invalid="ignore"):
                alfa = (X[ii] - xa[p]) / dx[p]
            alfa = np.where(alfa < 1.0, alfa, 1.0)   # min(1.0, alfa)
            alfa = np.where(alfa > 0.0, alfa, 0.0)   # max(0.0, ·)
            cruce = ta[p] + np.where(plano[p], 0, _timedelta_us(alfa * dt_s[p]))
            delta = (cruce - T[ii]) / 1e6            # (t_cross - t_target).total_seconds()
            antes = delta < 0
            np.minimum.at(ahead, s + ii[antes], np.abs(delta[antes]))
            np.minimum.at(behind, s + ii[~antes], delta[~antes])

    ahead[np.isinf(ahead)] = np.nan
    behind[np.isinf(behind)] = np.nan
    return ahead, behind


def compute_headways(
    df: pl.DataFrame,
    max_headway_seconds: float = 1800.0,
    verbose: bool = False,
    bucle: bool = False,
) -> pl.DataFrame:
    """
    Para cada registro (bus_code, time, direction, cum_dist_m), calcula:

        headway_ahead[i]  = time[i] - time(bus_anterior, mismo cum_dist_m,
                                           misma direction)
        headway_behind[i] = time(bus_siguiente, mismo cum_dist_m,
                                 misma direction) - time[i]

    Implementación: para cada bus_b distinto al actual, interpolamos el
    instante en que bus_b pasó por cum_dist_m[i] usando su trayectoria
    (cum_dist_m, time). Esto reusa el patrón de interpolación lineal de
    camino1_interpolation, aplicado a una posición arbitraria sobre la ruta.

    En detalle: se miran los pares de registros consecutivos de bus_b que se
    solapan con [time[i] - max_headway, time[i] + max_headway] y flanquean
    cum_dist_m[i]. headway_ahead es el |Δt| más chico de los cruces de antes,
    y headway_behind el más chico de los de después (o a la vez). La ventana
    se aplica al par, no al cruce: un par que abarca un hueco largo (la noche,
    una pausa) puede dar más de max_headway_seconds.

    Hasta el 7-oct-2026 era un bucle en Python puro, O(n²) por trozo semanal y
    el cuello de botella de la fase de datos. Ahora va por bloques con numpy
    (`_headways_vectorizado`) y da lo mismo bit a bit. El bucle sigue en
    `_headways_bucle` (`bucle=True`), para verificarlo.

    REFERENCIAS
    -----------
        Yu, Chen, Wang & Sun (2016, TR-C) — headway como predictor estructural.
        Bhutani, Pachal & Achar (2022, arXiv:2210.01655) — headway como
        sustituto de schedule adherence en transporte concesionado.

    PARAMS
    ------
        max_headway_seconds : valor centinela. Si no se encuentra un bus
                              en una ventana de [-max_headway, +max_headway]
                              alrededor de time[i], la feature queda NaN.
                              Útil para que el modelo distinga "no hay datos"
                              de "headway grande".

    REQUISITOS
    ----------
        - df debe contener TODOS los buses operando en la ruta. Si ya
          filtraste por bus, headway no se puede calcular.
        - Columnas requeridas: bus_code, trip_id, time, direction,
          cum_dist_m.

    NOTA OPERACIONAL
    ----------------
        En operación nocturna o de early-morning con un solo bus activo,
        ambas features serán NaN. El modelo (XGBoost/LightGBM) maneja NaN
        nativamente. Para LSTM, imputar con max_headway_seconds antes de
        normalizar.
    """
    _validate_required_columns(
        df,
        {"bus_code", "trip_id", "time", "direction", "cum_dist_m"},
        "compute_headways",
    )

    if verbose:
        print("  Computing headway_ahead, headway_behind...")

    df_sorted = df.sort(["direction", "time"])
    n = df_sorted.height

    head_ahead  = np.full(n, np.nan, dtype=np.float64)
    head_behind = np.full(n, np.nan, dtype=np.float64)

    # Procesar cada dirección por separado
    for dir_value in df_sorted["direction"].unique().to_list():
        if dir_value is None:
            continue
        mask = (df_sorted["direction"] == dir_value)
        idx_global = np.where(mask.to_numpy())[0]
        if idx_global.size < 2:
            continue

        df_dir = df_sorted.filter(mask)
        if bucle or not _headways_vectorizable(df_dir):
            ahead, behind = _headways_bucle(df_dir, max_headway_seconds)
        else:
            ahead, behind = _headways_vectorizado(df_dir, max_headway_seconds)
        head_ahead[idx_global] = ahead
        head_behind[idx_global] = behind

    df_out = df_sorted.with_columns([
        pl.Series("headway_ahead",  np.round(head_ahead, 1)),
        pl.Series("headway_behind", np.round(head_behind, 1)),
    ])

    if verbose:
        valid_a = ~np.isnan(head_ahead)
        valid_b = ~np.isnan(head_behind)
        if valid_a.any():
            print(f"    headway_ahead  — mediana: "
                  f"{np.nanmedian(head_ahead[valid_a]):.0f}s, "
                  f"cobertura: {valid_a.sum()/n:.1%}")
        if valid_b.any():
            print(f"    headway_behind — mediana: "
                  f"{np.nanmedian(head_behind[valid_b]):.0f}s, "
                  f"cobertura: {valid_b.sum()/n:.1%}")
    return df_out


###############################################################################
###             FEATURE 3 — SEGMENT_COUNT_AHEAD                             ###
###############################################################################

def compute_segment_count_ahead(
    df: pl.DataFrame,
    df_segments_route: Optional[pl.DataFrame] = None,
    verbose: bool = False,
) -> pl.DataFrame:
    """
    Número de paradas restantes hasta el final del viaje, por dirección.

    Construye automáticamente el orden de paradas por dirección: cada parada
    va en la posición de la mediana de `cum_dist_m` (el avance en el viaje)
    de los registros que la tienen por siguiente parada, de menor a mayor. La
    primera parada del viaje vale N (las N paradas de la dirección) y la
    última, 1.

    ⚠ Hasta el 7-oct-2026 se ordenaban por la media de `m_next_stop`, que son
    los metros al fin del tramo actual: eso ordena por el LARGO del tramo, no
    por la posición, y la feature salía como un código de la parada sin orden
    (correlación de rangos con el avance de −0.13 a +0.04 en el piloto de la
    vuelta). Arreglado con la enmienda E1 del pre-registro
    (docs/PREREGISTRO_VUELTA.md §6). Las paradas sin código no cuentan.

    REFERENCIAS
    -----------
        Petersen et al. (2019, ESWA 120) — variable de horizonte para
        multi-output.
        Bhutani et al. (2022) — `stop_position` y `total_stops` como
        features separadas.
        Rahmani & Koutsopoulos (2013, TR-B 53) — covarianza entre tiempos
        de segmentos sucesivos depende de la posición en la red.

    NOTA
    ----
        Si tu pipeline ya conoce el endpoint del viaje (via end_stops de
        make_dir_trips), considera pasarlo directamente en lugar de
        inferirlo. La inferencia automática asume que cada (route, direction)
        tiene un único orden de paradas.
    """
    _validate_required_columns(
        df,
        {"bus_code", "trip_id", "next_stop", "direction", "cum_dist_m"},
        "compute_segment_count_ahead",
    )

    if verbose:
        print("  Computing segment_count_ahead...")

    df_sorted = df.sort(["direction", "bus_code", "trip_id", "time"])

    # Orden de paradas por dirección: por dónde del viaje suelen ser la siguiente
    # (mediana del avance), con el código para desempatar y que el orden no dependa
    # del de group_by.
    order_by_dir: Dict[int, List[str]] = {}
    for dir_value in df_sorted["direction"].unique().to_list():
        if dir_value is None:
            continue
        df_dir = df_sorted.filter((pl.col("direction") == dir_value)
                                  & pl.col("next_stop").is_not_null())
        order_df = (
            df_dir.group_by("next_stop")
                  .agg(pl.col("cum_dist_m").median().alias("pos_m"))
                  .sort(["pos_m", "next_stop"], nulls_last=True)
        )
        order_by_dir[dir_value] = order_df["next_stop"].to_list()

    # Posición ordinal por (direction, next_stop) → 0-indexed
    position_map: Dict[Tuple[int, str], int] = {}
    total_stops_by_dir: Dict[int, int] = {}
    for dir_value, stops in order_by_dir.items():
        total_stops_by_dir[dir_value] = len(stops)
        for pos, s in enumerate(stops):
            position_map[(dir_value, s)] = pos

    # Asignar segment_count_ahead = total_stops - position
    dirs       = df_sorted["direction"].to_list()
    next_stops = df_sorted["next_stop"].to_list()
    n = len(dirs)

    seg_count_ahead = np.full(n, np.nan, dtype=np.float64)

    for r in range(n):
        d, s = dirs[r], next_stops[r]
        if d is None or s is None:
            continue
        if (d, s) in position_map:
            seg_count_ahead[r] = total_stops_by_dir[d] - position_map[(d, s)]

    df_out = df_sorted.with_columns(
        pl.Series("segment_count_ahead", seg_count_ahead.astype(np.float32))
    )

    if verbose:
        valid = ~np.isnan(seg_count_ahead)
        if valid.any():
            print(f"    rango: [{int(np.nanmin(seg_count_ahead))}, "
                  f"{int(np.nanmax(seg_count_ahead))}]")
            print(f"    cobertura: {valid.sum()/n:.1%}")
    return df_out


###############################################################################
###          ORQUESTADOR — TODAS LAS FEATURES NUEVAS                        ###
###############################################################################

def compute_extended_features(
    df: pl.DataFrame,
    df_segments_route: Optional[pl.DataFrame] = None,
    speed_window_seconds: int = 300,
    headway_max_seconds: float = 1800.0,
    with_headways: bool = True,
    verbose: bool = False,
) -> pl.DataFrame:
    """
    Aplica las features extendidas:
        - speed_recent_5min
        - headway_ahead, headway_behind  (omitidas si with_headways=False)
        - segment_count_ahead

    Asume que el dataframe entrante ya pasó por:
        - camino1_interpolation (provee s_next_stop_interp, cum_dist_m)
        - camino2_alternative_features (provee link_travel_time, etc.)

    Parameters
    ----------
    with_headways : bool
        Si False, omite el cálculo de headway_ahead/headway_behind y añade
        ambas columnas como null para mantener el esquema. Era útil cuando
        el cálculo era O(n²) en Python puro (hasta el 7-oct-2026, el cuello
        de botella del pipeline para meses completos); ahora va vectorizado.
    """
    if verbose:
        print("=" * 60)
        print("  EXTENDED FEATURES")
        print("=" * 60)
        print(f"  Registros entrada : {df.height:,}")

    df = compute_speed_recent_window(
        df,
        window_seconds=speed_window_seconds,
        output_column=f"speed_recent_{speed_window_seconds // 60}min",
        verbose=verbose,
    )

    if with_headways:
        df = compute_headways(
            df,
            max_headway_seconds=headway_max_seconds,
            verbose=verbose,
        )
    else:
        df = df.with_columns([
            pl.lit(None).cast(pl.Float64).alias("headway_ahead"),
            pl.lit(None).cast(pl.Float64).alias("headway_behind"),
        ])
        if verbose:
            print("  [skip] headway_ahead / headway_behind (with_headways=False)")

    df = compute_segment_count_ahead(
        df,
        df_segments_route=df_segments_route,
        verbose=verbose,
    )

    if verbose:
        hw_note = (f"speed_recent_{speed_window_seconds // 60}min, "
                   f"headway_ahead, headway_behind, segment_count_ahead")
        if not with_headways:
            hw_note += "  [headways=null]"
        print(f"\n  Columnas extendidas agregadas: {hw_note}")
        print("=" * 60)

    return df


###############################################################################
###          ORQUESTADOR — PIPELINE COMPLETO (camino1 + camino2 + ext)      ###
###############################################################################

def compute_all_features(
    df_enriched: pl.DataFrame,
    df_segments_route: Optional[pl.DataFrame] = None,
    speed_window_seconds: int = 300,
    headway_max_seconds: float = 1800.0,
    speed_zero_threshold: float = 2.0,
    compute_headways: bool = True,
    verbose: bool = False,
) -> pl.DataFrame:
    """
    Feature engineering desde la salida de enrich_trip_segments.

    Aplica en orden:
        1. compute_interpolated_crossings  (camino 1: cum_dist_m, …)
        2. compute_alternative_features    (camino 2: link_travel_time,
                                            dwell_time, hour_of_day, …)
        3. compute_extended_features       (speed_recent, headways,
                                            segment_count_ahead)

    Devuelve todas las columnas que producen las tres etapas; el paso 7 se
    queda solo con las que leen los modelos (DF_FULL_COLUMNS).
    """
    if verbose:
        print("\n" + "█" * 60)
        print("  PIPELINE COMPLETO DE FEATURE ENGINEERING")
        print("█" * 60)

    # 1. Interpolación
    df = compute_interpolated_crossings(
        df_enriched, df_segments_route, verbose=verbose
    )

    # 2. Features alternativas
    df = compute_alternative_features(
        df,
        df_segments_route=df_segments_route,
        speed_zero_threshold=speed_zero_threshold,
        verbose=verbose,
    )

    # 3. Features extendidas
    df = compute_extended_features(
        df,
        df_segments_route=df_segments_route,
        speed_window_seconds=speed_window_seconds,
        headway_max_seconds=headway_max_seconds,
        with_headways=compute_headways,
        verbose=verbose,
    )

    if verbose:
        print("\n" + "█" * 60)
        print(f"  PIPELINE COMPLETO  registros finales: {df.height:,}")
        print(f"  columnas finales: {len(df.columns)}")
        print("█" * 60 + "\n")

    return df
