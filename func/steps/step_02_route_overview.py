"""
step_02_route_overview.py
=========================
PASO 2 · Mapas exploratorios de control.

Genera dos mapas Folium para verificar visualmente la calidad geométrica
del input antes de cualquier procesamiento:

    2.1  route_overview   — paradas + polilíneas + nube de registros
                           (sin filtro de dirección).
    2.2  stopped_records  — densidad H3 de registros con speed≤0
                           (ayuda a identificar paradas terminales reales).
                           Si se proveen turn_hexes (resueltos por step_04
                           o por _lib_turn_hex.resolve_turn_hex), se dibujan
                           con resaltado especial: cluster contiguo en
                           color suave, centro con borde grueso y popup
                           con métricas detalladas.

Si el HTML ya existe en `PATHS.maps_dir`, se omite la regeneración.

Usage
-----
    from step_02_route_overview import build_overview_maps
    build_overview_maps(df_rec_clean, df_stops, df_segments,
                        turn_hexes=resolved_turn_hexes, verbose=True)
"""

from __future__ import annotations

# Ejecutable directo: registra func/ en el path para hallar config (que a su vez registra lib/, steps/ y audits/).
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from typing import Any, Dict, List, Optional

import polars as pl

from config import PATHS, INT_ROUTE_STOP, TURN_HEX_MAX_PINGS_PER_BUS_PER_HEX


# ──────────────────────────────────────────────────────────────────
# 2.1 · OVERVIEW (paradas + segmentos + registros)
# ──────────────────────────────────────────────────────────────────

def make_overview_map(
    df_rec_clean: pl.DataFrame,
    df_stops: pl.DataFrame,
    df_segments: pl.DataFrame,
    verbose: bool = False,
) -> Path:
    """
    Crea el mapa `route_overview_<date_range>.html`.
    """
    out = PATHS.overview_map
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.is_file():
        if verbose:
            print(f"  ✔ overview_map existe: {out.name}")
        return out

    from _lib_plot import plot_route_overview
    m = plot_route_overview(
        df_stop_route      = df_stops,
        df_segments_route  = df_segments,
        df_rec_route_clean = df_rec_clean,
        route_prefix       = INT_ROUTE_STOP,
    )
    m.save(str(out))
    if verbose:
        print(f"  ✔ overview_map guardado: {out}")
    return out


# ──────────────────────────────────────────────────────────────────
# 2.2 · STOPPED RECORDS (densidad H3 + turn_hex resaltados)
# ──────────────────────────────────────────────────────────────────

def make_stopped_map(
    df_rec_clean: pl.DataFrame,
    df_stops: pl.DataFrame,
    df_segments: pl.DataFrame,
    turn_hexes: Optional[List[Dict[str, Any]]] = None,
    verbose: bool = False,
) -> Path:
    """
    Crea el mapa `stopped_rec_<date_range>.html`.

    Filtra registros con velocidad ≤ 0 y los dibuja como hexágonos H3 (res=12),
    con un mínimo de 5 registros por hexágono. Útil para detectar dónde se
    estacionan los buses (terminales, semáforos, depósitos).

    Si se proveen ``turn_hexes`` (lista de dicts con keys label, h3, lat, lon,
    stopped_count, cluster_h3s, cluster_total_count), se dibujan como una capa
    adicional con prioridad visual sobre la detección automática interna.
    Esto permite que el mapa sea consistente con los turn_hex usados
    operacionalmente en el resto del pipeline (paso 04 en adelante).

    Parameters
    ----------
    turn_hexes : list[dict] | None
        Lista de turn_hexes resueltos. Típicamente proviene de
        ``_lib_turn_hex.resolve_turn_hex(...)['turn_hexes']`` o del cache
        ``PATHS.turn_hex_json``.
    """
    out = PATHS.stopped_map
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.is_file():
        if verbose:
            print(f"  ✔ stopped_map existe: {out.name}")
        return out

    from _lib_plot import plot_speed_density
    # ── Coherencia visual v3: ──
    # - Si hay turn_hexes resueltos (cache válido), los marcadores son
    #   correctos y queremos que la densidad de fondo cuente la misma
    #   historia, por eso pasamos el cap bus-aware.
    # - Si NO hay turn_hexes (cache no existe aún), desactivamos la
    #   detección legacy de end-stops; mostrar marcadores legacy basados
    #   en conteo crudo es exactamente lo que confunde al lector cuando
    #   el detector real ya está corregido.
    show_legacy_end_stops = turn_hexes is not None and len(turn_hexes) > 0 \
                             and False  # rama legacy desactivada incluso con turn_hexes
    # Nota: la rama legacy (show_end_stops=True sin turn_hexes) está
    # deprecada en v3. Cuando hay turn_hexes, plot_speed_density ya los
    # dibuja como capa v2 superior. Cuando no los hay, mostrar legacy
    # confundiría al lector (criterio crudo vs v3 bus-aware).
    if verbose and turn_hexes is None:
        print("  ⚠ no hay turn_hexes resueltos (cache ausente). "
              "El mapa NO mostrará marcadores de terminal. "
              "Corre step_04 primero para regenerar el cache.")

    m = plot_speed_density(
        df_rec                    = df_rec_clean,
        df_stops                  = df_stops,
        df_segments_route         = df_segments,
        max_speed                 = 0.0,
        min_hex_count             = 5,
        resolution                = 12,
        show_stops                = True,
        show_end_stops            = show_legacy_end_stops,
        turn_hexes                = turn_hexes,
        max_pings_per_bus_per_hex = TURN_HEX_MAX_PINGS_PER_BUS_PER_HEX,
        verbose                   = verbose,
    )
    m.save(str(out))
    if verbose:
        print(f"  ✔ stopped_map guardado: {out}")
        if turn_hexes:
            print(f"     ⭐ con capa turn_hex (v2): "
                  f"{[t['label'] for t in turn_hexes]}")
    return out


# ──────────────────────────────────────────────────────────────────
# Función pública
# ──────────────────────────────────────────────────────────────────

def build_overview_maps(
    df_rec_clean: pl.DataFrame,
    df_stops: pl.DataFrame,
    df_segments: pl.DataFrame,
    turn_hexes: Optional[List[Dict[str, Any]]] = None,
    verbose: bool = False,
) -> dict:
    """
    Genera ambos mapas y devuelve sus rutas.

    Parameters
    ----------
    turn_hexes : list[dict] | None
        Si se provee, se pasa a `make_stopped_map` para resaltado consistente.
        Si no, intenta leer del cache `PATHS.turn_hex_json` automáticamente.
    """
    if verbose:
        print("=" * 60)
        print(f"  PASO 2 · Mapas exploratorios · Ruta {INT_ROUTE_STOP}")
        print("=" * 60)

    # ── Resolver turn_hexes: cache → detector en-línea → None ────────
    # Orden de prioridad:
    #   1. Si se pasaron turn_hexes explícitamente, usarlos.
    #   2. Si existe el cache turn_hex.json (generado por step_04), cargarlo.
    #   3. Si no hay cache, correr el detector directamente con los datos
    #      disponibles (df_rec_clean + df_segments). Esto permite que step_02
    #      corra en orden natural (antes de step_04) y muestre ya las
    #      terminales correctas con el método v3 bus-aware.
    if turn_hexes is None:
        if PATHS.turn_hex_json.is_file():
            import json
            try:
                with open(PATHS.turn_hex_json, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                turn_hexes = cached.get("turn_hexes")
                if verbose and turn_hexes:
                    print(f"  ✔ turn_hexes cargados del cache: "
                          f"{[t['label'] for t in turn_hexes]}")
            except Exception as exc:
                if verbose:
                    print(f"  ⚠ no pude leer {PATHS.turn_hex_json.name}: {exc}")
        else:
            # Cache ausente → detectar en-línea con los datos disponibles
            if verbose:
                print("  ℹ turn_hex.json no existe, ejecutando detector "
                      "en-línea (v3 bus-aware)…")
            try:
                from _lib_turn_hex import resolve_turn_hex
                from config import (
                    TURN_HEX_MODE, TURN_HEX_MANUAL,
                    TURN_HEX_RESOLUTION, TURN_HEX_MIN_HEX_COUNT,
                    TURN_HEX_MIN_DIST_FRACTION, TURN_HEX_MAX_SPEED,
                    TURN_HEX_OUTLIER_METHOD, TURN_HEX_OUTLIER_PERCENTILE,
                    TURN_HEX_OUTLIER_ABSOLUTE, TURN_HEX_SELECTION_METRIC,
                    TURN_HEX_MAX_PINGS_PER_BUS_PER_HEX, TURN_ZONE_KRING,
                )
                th_result = resolve_turn_hex(
                    df_rec_clean               = df_rec_clean,
                    df_segments_route          = df_segments,
                    mode                       = TURN_HEX_MODE,
                    manual_h3                  = TURN_HEX_MANUAL
                                                 if TURN_HEX_MODE == "manual"
                                                 else None,
                    cache_path                 = None,   # no persistir desde step_02
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
                turn_hexes = th_result["raw"]["turn_hexes"]
                if verbose and turn_hexes:
                    print(f"  ✔ turn_hexes detectados en-línea: "
                          f"{[t['label'] for t in turn_hexes]}")
            except Exception as exc:
                if verbose:
                    print(f"  ⚠ detección en-línea falló: {exc}. "
                          f"El mapa no mostrará marcadores de terminal.")

    paths = {
        "overview": make_overview_map(df_rec_clean, df_stops, df_segments, verbose),
        "stopped":  make_stopped_map(df_rec_clean, df_stops, df_segments,
                                     turn_hexes=turn_hexes, verbose=verbose),
    }

    if verbose:
        print("  → Mapas listos para PASO 3 (direction)\n")

    return paths
