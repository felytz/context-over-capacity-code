"""
step_06_make_trips.py
=====================
PASO 6 · Filtro temporal + Segmentación de viajes.

Este paso fusiona los antiguos pasos 04 (temporal_filter) y 05 (make_trips)
en un único paso operativo. Las dos sub-tareas se ejecutan en orden:

    6a. Filtro de consistencia temporal (gap ≤ P90)
    -----------------------------------------------
    Wrapper sobre `_lib_temporal.run_temporal_consistency`. Detecta huecos
    temporales entre pings consecutivos del mismo bus y produce:

        - df_gaps             : un registro por hueco detectado.
        - df_global           : distribución global (cantidad vs duración).
        - df_quality          : clasifica camiones en alta/media/baja calidad.
        - df_filter_temporal  : df_rec_clean filtrado por gap_filter (P90).

    Además guarda `gap_distribution.png` en `PATHS.reports_dir`.

    Nota técnica: `df_filter_temporal` se computa con fines DIAGNÓSTICOS
    (auditoría y reporte). El input efectivo de make_dir_trips es
    `df_route_direction` — make_dir_trips aplica su propio criterio de
    gap interno (TripParams.max_gap_min=7) sobre la dirección snapped.

    6b. Segmentación make_dir_trips
    -------------------------------
    Aplica `_lib_map_match.make_dir_trips` con los TripParams validados
    en el PDF. El algoritmo segmenta por:

        1. Zona terminal     — paso por turn_hex ± turn_zone_kring (v2).
        2. Gap temporal      — silencio > max_gap_min minutos.
        3. Outlier velocidad — gap_dist / gap_time > V_max (Schuessler &
                               Axhausen 2009 · default V_max=180 km/h).
        4. Cambio de bus_code.

    Cada viaje recibe un `trip_id = "{bus_code}_D{dir}_T{N}"` y un
    `trip_valid ∈ {True, False}`. Las razones de invalidación quedan en
    `trip_reject` (rejection log del PDF):

        - too_few_points          ← min_points
        - too_short_dist          ← min_dist_m
        - exceeds_polyline_len    ← acum_dist > rama_len
        - below_min_completeness  ← acum_dist < min_completeness * rama_len

v2 (mayo 2026): migrado a turn_hex (hexágonos H3) + filtro de velocidad
implícita (Schuessler & Axhausen 2009). Las terminales ya no son paradas
GTFS sino entidades geométricas detectadas por densidad.

References
----------
- Lin & Zeng (1999):           umbral temporal P90.
- Newson & Krumm (2009):       map-matching subyacente al df_route_direction.
- Cathey & Dailey (2003):      end-stop + cooldown (concepto, rebrand turn_hex).
- Schuessler & Axhausen (2009): outlier de velocidad implícita.

Usage
-----
    from step_06_make_trips import run_make_trips_step

    df_filter, df_trips = run_make_trips_step(
        df_rec_clean, df_route_direction, df_stops,
        make_map=True, verbose=True,
    )
"""

from __future__ import annotations

# Ejecutable directo: registra func/ en el path para hallar config (que a su vez registra lib/, steps/ y audits/).
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from typing import Optional, Tuple

import polars as pl

from config import (
    PATHS,
    TURN_HEX_MODE,
    TURN_HEX_MANUAL,
    TURN_HEX_RESOLUTION,
    TURN_HEX_MIN_HEX_COUNT,
    TURN_HEX_MIN_DIST_FRACTION,
    TURN_HEX_MAX_SPEED,
    TURN_HEX_OUTLIER_METHOD,
    TURN_HEX_OUTLIER_PERCENTILE,
    TURN_HEX_OUTLIER_ABSOLUTE,
    TURN_HEX_SELECTION_METRIC,
    TURN_ZONE_KRING,
    TRIP_PARAMS,
    GAP_FILTER_PCTL,
    MAX_GAP_DISPLAY,
)


# ──────────────────────────────────────────────────────────────────
# 6a · FILTRO TEMPORAL (gap ≤ P90)
# ──────────────────────────────────────────────────────────────────

def run_temporal_filter(
    df_rec_clean: pl.DataFrame,
    pct_cutoff_visual: int = 90,
    verbose: bool = False,
) -> Tuple[Optional[pl.DataFrame], Optional[pl.DataFrame],
           Optional[pl.DataFrame], pl.DataFrame]:
    """
    Aplica el filtro temporal y guarda el reporte gráfico.

    Parameters
    ----------
    df_rec_clean : pl.DataFrame
        Registros limpios del PASO 1.
    pct_cutoff_visual : int
        Percentil mostrado como línea de corte en la gráfica (default 90).

    Returns
    -------
    (df_gaps, df_global, df_quality, df_filter_temporal)
        Las primeras tres pueden ser None si se carga del cache.
        df_filter_temporal es el que se usa después.
    """
    PATHS.ensure_dirs()
    cache_filter = PATHS.df_filter_temporal
    cache_png    = PATHS.gap_distribution_png

    if cache_filter.is_file() and cache_png.is_file():
        df_filter = pl.read_parquet(cache_filter)
        if verbose:
            print(f"  ✔ df_filter_temporal cargado de cache: "
                  f"{cache_filter.name} ({df_filter.height:,} filas)")
            print(f"  ✔ gap_distribution.png ya existe.")
        return None, None, None, df_filter

    from _lib_temporal import run_temporal_consistency
    df_gaps, df_global, df_quality, df_filter = run_temporal_consistency(
        df_rec      = df_rec_clean,
        pct_cutoff  = pct_cutoff_visual,
        max_gap_display = MAX_GAP_DISPLAY,
        gap_filter  = GAP_FILTER_PCTL,
        save_path   = cache_png,
        show_plot   = False,   # batch: solo guarda el PNG; no plt.show() (backend Agg)
        verbose     = verbose,
    )

    from _lib_data import save_df
    save_df(df_filter, cache_filter)
    if verbose:
        print(f"  ✔ df_filter_temporal guardado: "
              f"{cache_filter} ({df_filter.height:,} filas)")

    return df_gaps, df_global, df_quality, df_filter


# ──────────────────────────────────────────────────────────────────
# 6b · SEGMENTACIÓN DE VIAJES (make_dir_trips)
# ──────────────────────────────────────────────────────────────────

def make_trips(
    df_route_direction: pl.DataFrame,
    df_stops: pl.DataFrame,
    df_rec_clean: pl.DataFrame | None = None,
    df_segments: pl.DataFrame | None = None,
    turn_hex_mode: str | None = None,
    turn_hex_manual: list[str] | None = None,
    verbose: bool = False,
) -> pl.DataFrame:
    """
    Segmenta los registros en viajes individuales aplicando TripParams.

    Parameters
    ----------
    df_route_direction : pl.DataFrame
        Salida del PASO 4 (con dirección y snap).
    df_stops : pl.DataFrame
        Paradas de la ruta. Solo se usa para visualización de referencia.
    df_rec_clean : pl.DataFrame, optional
        Registros limpios. Solo necesario si el cache de turn_hex
        (``PATHS.turn_hex_json``) no existe y hay que detectar de cero.
    df_segments : pl.DataFrame, optional
        Polilínea segmentada. Idem para detección de cero.
    turn_hex_mode : {"auto", "manual"} | None
        Override del modo. None → ``config.TURN_HEX_MODE``.
    turn_hex_manual : list[str] | None
        Override de la lista manual de códigos H3.
    verbose : bool

    Returns
    -------
    pl.DataFrame
        df_route_direction enriquecido con:
          - trip_id, trip_seq, trip_valid, trip_reject
          - trip_n_pts, trip_dist_m, trip_completeness
          - gap_time_min, gap_dist_m, cut_reason
    """
    if verbose:
        print(f"  Parámetros TripParams:")
        for k, v in TRIP_PARAMS.__dict__.items():
            print(f"    {k:<25} = {v}")
        print()

    PATHS.ensure_dirs()
    cache = PATHS.df_trips

    if cache.is_file():
        df = pl.read_parquet(cache)
        if verbose:
            print(f"  ✔ df_trips cargado de cache: "
                  f"{cache.name} ({df.height:,} filas)")
        return df

    # ── Resolver turn_hex (cache si existe, sino detectar) ──
    if df_segments is None and PATHS.df_segments.is_file():
        df_segments = pl.read_parquet(PATHS.df_segments)
    if df_rec_clean is None and PATHS.df_rec_clean.is_file():
        df_rec_clean = pl.read_parquet(PATHS.df_rec_clean)

    mode      = turn_hex_mode   or TURN_HEX_MODE
    manual_h3 = turn_hex_manual or TURN_HEX_MANUAL

    from _lib_turn_hex import resolve_turn_hex
    th_result = resolve_turn_hex(
        df_rec_clean               = df_rec_clean,
        df_segments_route          = df_segments,
        mode                       = mode,
        manual_h3                  = manual_h3 if mode == "manual" else None,
        cache_path                 = PATHS.turn_hex_json,
        resolution                 = TURN_HEX_RESOLUTION,
        min_hex_count              = TURN_HEX_MIN_HEX_COUNT,
        min_dist_fraction          = TURN_HEX_MIN_DIST_FRACTION,
        max_speed                  = TURN_HEX_MAX_SPEED,
        outlier_method             = TURN_HEX_OUTLIER_METHOD,
        outlier_percentile         = TURN_HEX_OUTLIER_PERCENTILE,
        outlier_absolute_threshold = TURN_HEX_OUTLIER_ABSOLUTE,
        selection_metric           = TURN_HEX_SELECTION_METRIC,
            turn_zone_kring            = TURN_ZONE_KRING,
        verbose                    = verbose,
    )
    df_end_stop    = th_result["df_end_stop"]
    end_stop_zones = th_result["zones"]

    from _lib_map_match import make_dir_trips
    df = make_dir_trips(
        df_route_direction,
        df_stop_route         = df_stops,
        end_stops             = list(end_stop_zones.keys()),
        end_stop_zones        = end_stop_zones,
        neighbor_window       = TURN_ZONE_KRING,              # k-ring H3 real
        end_stop_radius_m     = TRIP_PARAMS.end_stop_radius_m,
        end_stop_cooldown_min = TRIP_PARAMS.end_stop_cooldown_min,
        max_gap_min           = TRIP_PARAMS.max_gap_min,
        max_implied_speed_kmh = TRIP_PARAMS.max_implied_speed_kmh,  # Schuessler & Axhausen 2009
        min_hour              = TRIP_PARAMS.min_hour,
        max_hour              = TRIP_PARAMS.max_hour,
        min_points            = TRIP_PARAMS.min_points,
        min_dist_m            = TRIP_PARAMS.min_dist_m,
        min_completeness      = TRIP_PARAMS.min_completeness,
        geojson_path          = str(PATHS.geojson_path),
        df_end_stop           = df_end_stop,
        verbose               = verbose,
    )
    from _lib_data import save_df
    save_df(df, cache)

    if verbose:
        print(f"  ✔ df_trips guardado: {cache} ({df.height:,} filas)")
    return df


# ──────────────────────────────────────────────────────────────────
# Mapa de validación por bus (opcional)
# ──────────────────────────────────────────────────────────────────

def make_bus_trips_map(
    df_trips: pl.DataFrame,
    df_stops: pl.DataFrame,
    bus_code: str | None = None,
    max_days: int = 30,
    verbose: bool = False,
) -> Optional[Path]:
    """
    Genera un mapa Folium con los viajes de UN bus durante `max_days` días.
    Si `bus_code=None`, escoge el bus con más registros válidos.
    """
    if bus_code is None:
        valid = df_trips.filter(pl.col("trip_valid") == True)
        if valid.height == 0:
            if verbose:
                print("  ⚠ No hay trip_valid=True; saltando mapa.")
            return None
        bus_code = valid.select(pl.col("bus_code").mode()).item(0, 0)

    out = PATHS.maps_dir / f"visualize_bus_trips_{bus_code}_{PATHS.date_range}.html"
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.is_file():
        if verbose:
            print(f"  ✔ bus_trips_map existe: {out.name}")
        return out

    from _lib_turn_hex import resolve_turn_hex
    th_result = resolve_turn_hex(
        df_rec_clean = None,        # solo carga del cache
        cache_path   = PATHS.turn_hex_json,
        verbose      = verbose,
    )
    df_end_stop = th_result["df_end_stop"]

    from _lib_map_match import visualize_bus_trips
    m = visualize_bus_trips(
        df_trips,
        PATHS.geojson_path,
        bus_code    = bus_code,
        max_days    = max_days,
        df_end_stop = df_end_stop,
        verbose     = verbose,
    )
    m.save(str(out))
    if verbose:
        print(f"  ✔ bus_trips_map guardado: {out}")
    return out


# ──────────────────────────────────────────────────────────────────
# Función pública
# ──────────────────────────────────────────────────────────────────

def run_make_trips_step(
    df_rec_clean: pl.DataFrame,
    df_route_direction: pl.DataFrame,
    df_stops: pl.DataFrame,
    make_map: bool = True,
    verbose: bool = False,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Ejecuta PASO 6 completo: filtro temporal (6a) + make_dir_trips (6b).

    Parameters
    ----------
    df_rec_clean : pl.DataFrame
        Registros limpios del PASO 1 (input para el filtro temporal).
    df_route_direction : pl.DataFrame
        Salida del PASO 4 (input para make_dir_trips).
    df_stops : pl.DataFrame
        Paradas de la ruta.
    make_map : bool
        Si True, genera el mapa de validación de un bus.

    Returns
    -------
    df_filter_temporal : pl.DataFrame   (diagnóstico, auditado en paso 09)
    df_trips           : pl.DataFrame   (input para PASO 7)
    """
    if verbose:
        print("=" * 60)
        print("  PASO 6 · Temporal filter + make_dir_trips")
        print("=" * 60)

    # 6a · Filtro temporal (diagnóstico)
    if verbose:
        print("\n  [6a] Filtro de consistencia temporal …")
    _, _, _, df_filter_temporal = run_temporal_filter(
        df_rec_clean, verbose=verbose
    )

    # 6b · Segmentación de viajes
    if verbose:
        print("\n  [6b] Segmentación make_dir_trips …")
    df_trips = make_trips(
        df_route_direction,
        df_stops,
        df_rec_clean = df_rec_clean,
        verbose      = verbose,
    )

    # Mapa opcional
    if make_map:
        if verbose:
            print("\n  [6b·map] Mapa de validación de un bus …")
        make_bus_trips_map(df_trips, df_stops, verbose=verbose)

    if verbose:
        print("\n  → Trips listos para PASO 7 (enrich + features)\n")

    return df_filter_temporal, df_trips


if __name__ == "__main__":
    # Smoke test: requiere que los pasos 1 y 4 ya estén cacheados.
    import polars as _pl
    from step_01_load_data import load_data
    from step_04_direction import compute_direction

    _df_clean, _df_stops, _ = load_data(verbose=True)
    _df_dir = compute_direction(_df_clean, _df_stops, verbose=True)
    run_make_trips_step(_df_clean, _df_dir, _df_stops,
                        make_map=False, verbose=True)
