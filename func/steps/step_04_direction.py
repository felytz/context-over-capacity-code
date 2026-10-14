"""
step_04_direction.py
====================
PASO 4 · Asignación de dirección por map-matching.

Aplica la función `map_match_direction.map_match_direction` (HMM-style,
Newson & Krumm 2009) sobre los registros limpios. La salida añade la columna
`direction ∈ {0=A, 1=B}` y posiciones snapped (`snap_lat`, `snap_lon`)
proyectadas sobre la polilínea oficial.

Sub-pasos:
    4.1  df_route_direction  — registros con dirección y snap.
    4.2  direction_map       — visualización Folium en ventanas de 30 min.

v2 (mayo 2026): migrado de `resolve_end_stops` a `resolve_turn_hex`. Las
terminales ahora son turn_hex (hexágonos H3 detectados por densidad), no
stop_codes GTFS. La API pública sigue siendo la misma; las constantes
END_STOPS_* del config se siguen leyendo como alias retro-compatibles.

References
----------
Newson, P. & Krumm, J. (2009). "Hidden Markov Map Matching Through Noise
and Sparseness." ACM SIGSPATIAL.

Usage
-----
    from step_04_direction import compute_direction
    df_route_direction = compute_direction(df_rec_clean, df_stops, verbose=True)
"""

from __future__ import annotations

# Ejecutable directo: registra func/ en el path para hallar config (que a su vez registra lib/, steps/ y audits/).
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from typing import Optional, List

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
    TURN_HEX_MAX_PINGS_PER_BUS_PER_HEX,
    TURN_ZONE_KRING,
)


# ──────────────────────────────────────────────────────────────────
# 4.1 · MAP MATCHING DE DIRECCIÓN
# ──────────────────────────────────────────────────────────────────

def compute_direction(
    df_rec_clean: pl.DataFrame,
    df_stops: pl.DataFrame,
    df_segments: Optional[pl.DataFrame] = None,
    turn_hex_mode: Optional[str] = None,
    turn_hex_manual: Optional[List[str]] = None,
    verbose: bool = False,
) -> pl.DataFrame:
    """
    Calcula `df_route_direction` con dirección y snap por registro.

    Si el parquet cacheado existe, se carga; en caso contrario se ejecuta
    `map_match_direction` y se guarda.

    Parameters
    ----------
    df_rec_clean : pl.DataFrame
        Salida del PASO 1 (records limpios).
    df_stops : pl.DataFrame
        Paradas de la ruta (referencia visual; ya no obligatorio para
        detección — los turn_hex se detectan a partir de los registros).
    df_segments : pl.DataFrame, optional
        Polilínea segmentada. Necesario para el modo auto (longitud).
        Si es None se intenta leer de ``PATHS.df_segments``.
    turn_hex_mode : {"auto", "manual"} | None
        Override del modo configurado. None → usa ``config.TURN_HEX_MODE``.
    turn_hex_manual : list[str] | None
        Override de la lista manual de **códigos H3** (no stop_codes).
        None → usa ``config.TURN_HEX_MANUAL``.
    verbose : bool

    Returns
    -------
    pl.DataFrame
        df_rec_clean enriquecido con `direction`, `snap_lat`, `snap_lon`
        y métricas auxiliares de map-matching.
    """
    if verbose:
        print("=" * 60)
        print("  PASO 4 · Map-match de dirección")
        print("=" * 60)

    PATHS.ensure_dirs()
    cache = PATHS.df_route_direction

    if cache.is_file():
        df = pl.read_parquet(cache)
        if verbose:
            print(f"  ✔ df_route_direction cargado de cache: "
                  f"{cache.name} ({df.height:,} filas)")
        return df

    # ── Resolver turn_hex (auto-H3 o manual) ──
    if df_segments is None and PATHS.df_segments.is_file():
        df_segments = pl.read_parquet(PATHS.df_segments)

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
        max_pings_per_bus_per_hex  = TURN_HEX_MAX_PINGS_PER_BUS_PER_HEX,
        turn_zone_kring            = TURN_ZONE_KRING,
        verbose                    = verbose,
    )

    # Para map_match_direction necesitamos un df_end_stop con columnas legacy
    # (code, latitude, longitude). resolve_turn_hex ya devuelve este alias.
    df_end_stop = th_result["df_end_stop"]

    if df_end_stop.height != 2:
        raise ValueError(
            f"Esperaba 2 turn_hex, encontré {df_end_stop.height}. "
            f"Modo: {mode}. Revisa la detección o la config."
        )

    from _lib_map_match import map_match_direction
    df = map_match_direction(
        df_rec_clean,
        PATHS.geojson_path,
        df_end_stop,
        verbose=verbose,
    )
    from _lib_data import save_df
    save_df(df, cache)
    if verbose:
        print(f"  ✔ df_route_direction guardado: "
              f"{cache} ({df.height:,} filas)")

    return df


# ──────────────────────────────────────────────────────────────────
# 4.2 · MAPA DE DIRECCIÓN (ventanas de 30 min)
# ──────────────────────────────────────────────────────────────────

def make_direction_map(
    df_route_direction: pl.DataFrame,
    df_stops: pl.DataFrame,
    verbose: bool = False,
) -> Path:
    """Visualización Folium del df con dirección en ventanas de 30 min."""
    out = PATHS.direction_map
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.is_file():
        if verbose:
            print(f"  ✔ direction_map existe: {out.name}")
        return out

    # Cargar turn_hex desde cache (debe existir tras compute_direction)
    from _lib_turn_hex import resolve_turn_hex
    th_result = resolve_turn_hex(
        df_rec_clean = None,           # solo carga del cache
        cache_path   = PATHS.turn_hex_json,
        verbose      = verbose,
    )
    df_end_stop = th_result["df_end_stop"]

    from _lib_map_match import visualize_time_windows_simple
    m = visualize_time_windows_simple(
        df_route_direction,
        PATHS.geojson_path,
        window_minutes = 30,
        df_end_stop    = df_end_stop,
    )
    m.save(str(out))
    if verbose:
        print(f"  ✔ direction_map guardado: {out}")
    return out


# ──────────────────────────────────────────────────────────────────
# Función pública
# ──────────────────────────────────────────────────────────────────

def run_direction_step(
    df_rec_clean: pl.DataFrame,
    df_stops: pl.DataFrame,
    make_map: bool = True,
    verbose: bool = False,
) -> pl.DataFrame:
    """
    Ejecuta PASO 4 completo: dirección + (opcional) mapa de validación.
    """
    df_route_direction = compute_direction(df_rec_clean, df_stops, verbose=verbose)

    if make_map:
        make_direction_map(df_route_direction, df_stops, verbose=verbose)

    if verbose:
        print("  → Dirección lista para PASO 5 (sensitivity) / PASO 6 (make_trips)\n")

    return df_route_direction
