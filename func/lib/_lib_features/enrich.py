"""Enriquecimiento por proyección sobre segmentos (next_stop, m/s_next_stop).

Submódulo de `_lib_features` (antes era una sección del módulo monolítico).
Importar desde el paquete: `from _lib_features import ...`.
"""
from __future__ import annotations

import ast
import json
import math
import warnings
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

from _lib_geo import _haversine_m, _parse_polyline_struct

##############################################################################
#   1. enrich_trip_segments — proyección sobre segmentos
#   (originalmente: _lib_enrich.py)
##############################################################################

###############
### Helpers ###
###############

from _lib_geo import _parse_polyline_struct  # consolidado en lib/_lib_geo.py


def _cum_dists(pts: List[Tuple[float, float]]) -> np.ndarray:
    """
    Distancias acumuladas a lo largo de una polilínea (metros).

    Parameters
    ----------
        pts : list of (lat, lon)

    Returns
    -------
        cum (np.ndarray): shape=(len(pts),), cum[0]=0.
    """
    cum = [0.0]
    for i in range(1, len(pts)):
        la1, lo1 = pts[i-1]
        la2, lo2 = pts[i]
        R = 6_371_000.0
        phi1, phi2 = math.radians(la1), math.radians(la2)
        dphi = math.radians(la2 - la1)
        dlam = math.radians(lo2 - lo1)
        a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
        cum.append(cum[-1] + R * 2 * math.asin(math.sqrt(max(0.0, a))))
    return np.array(cum)


def _project_point(p_lat: float, p_lon: float,
                   pts: List[Tuple[float, float]],
                   cum: np.ndarray) -> Tuple[float, float, float]:
    """
    Proyecta (p_lat, p_lon) sobre la polilínea, retorna
    (distancia_perp_m, posicion_sobre_poly_m, metros_restantes_m).

    Parameters
    ----------
        p_lat, p_lon : float             Punto a proyectar.
        pts          : list of (lat,lon) Polilínea del segmento.
        cum          : np.ndarray        Distancias acumuladas del segmento.

    Returns
    -------
        perp_m    (float): Distancia perpendicular al segmento en metros.
        pos_m     (float): Posición sobre el segmento desde el inicio (metros).
        remain_m  (float): Metros restantes hasta el fin del segmento.
    """
    best_perp = math.inf
    best_pos  = 0.0
    total_len = float(cum[-1]) if len(cum) > 0 else 0.0

    for i in range(len(pts) - 1):
        la1, lo1 = pts[i]
        la2, lo2 = pts[i+1]
        seg_len = cum[i+1] - cum[i]
        if seg_len < 1e-9:
            continue

        cos_mid = math.cos(math.radians((la1 + la2) / 2.0))
        dx_s = (lo2 - lo1) * cos_mid
        dy_s = la2 - la1
        seg_sq = dx_s**2 + dy_s**2
        if seg_sq < 1e-20:
            continue

        dx_p = (p_lon - lo1) * cos_mid
        dy_p = p_lat - la1
        t = max(0.0, min(1.0, (dx_p*dx_s + dy_p*dy_s) / seg_sq))

        proj_lat = la1 + t*(la2 - la1)
        proj_lon = lo1 + t*(lo2 - lo1)
        perp = math.sqrt(
            ((p_lon - proj_lon) * cos_mid * 111_320)**2
            + ((p_lat - proj_lat) * 110_540)**2
        )
        if perp < best_perp:
            best_perp = perp
            best_pos  = cum[i] + t * seg_len

    remain = max(0.0, total_len - best_pos)
    return best_perp, best_pos, remain


#──────────────────────────────────────────────────────────────────────────#
# ── 1. Preparación de geometría de segmentos ── #
#──────────────────────────────────────────────────────────────────────────#

def _prepare_segments(
    df_segments_route: pl.DataFrame,
    verbose: bool = False,
) -> List[Dict]:
    """
    Parsea df_segments_route a una lista de dicts con geometría pre-procesada.

    Cada elemento contiene:
        stop     (str)                  : código de la siguiente parada
        pts      (list of (lat, lon))   : polilínea parseada
        cum      (np.ndarray)           : distancias acumuladas
        length_m (float)                : longitud total del segmento

    Parameters
    ----------
        df_segments_route : pl.DataFrame  Con columnas stop y polyline.
        verbose           : bool

    Returns
    -------
        segs (list of dict)
    """
    # Detectar columna de código de parada
    stop_col = next(
        (c for c in ["stop", "codigo", "code", "parada"] if c in df_segments_route.columns),
        None
    )
    if stop_col is None:
        raise ValueError(
            "df_segments_route debe tener columna 'stop'. "
            f"Columnas disponibles: {df_segments_route.columns}"
        )

    poly_col = next(
        (c for c in ["polyline", "geometry", "coords", "puntos"] if c in df_segments_route.columns),
        None
    )
    if poly_col is None:
        raise ValueError(
            "df_segments_route debe tener columna 'polyline'. "
            f"Columnas disponibles: {df_segments_route.columns}"
        )

    rows = df_segments_route.select([stop_col, poly_col]).iter_rows(named=True)
    segs = []
    n_empty = 0

    for row in rows:
        stop = str(row[stop_col])
        pts  = _parse_polyline_struct(row[poly_col])
        if len(pts) < 2:
            n_empty += 1
            continue
        cum    = _cum_dists(pts)
        segs.append({
            "stop":     stop,
            "pts":      pts,
            "cum":      cum,
            "length_m": float(cum[-1]),
        })

    if verbose:
        print(f"  Segmentos cargados : {len(segs)}  "
              f"(descartados sin geometría: {n_empty})")

    return segs


#──────────────────────────────────────────────────────────────────────────#
# ── 2. Función pública: enrich_trip_segments ── #
#──────────────────────────────────────────────────────────────────────────#

def enrich_trip_segments(
    df_trips_route: pl.DataFrame,
    df_segments_route: pl.DataFrame,
    route_prefix: Optional[str] = None,
    verbose: bool = False,
) -> pl.DataFrame:
    """
    Enriquece df_trips_route con columnas de parada calculadas desde la
    geometría de df_segments_route.

    Columnas generadas:
        next_stop   (str)   : Código de la siguiente parada (campo 'stop' del
                              segmento más cercano al snap del registro).
                              Si route_prefix es provisto (ej. "101"), el código
                              se devuelve con prefijo completo ("101-FQAE").
        m_next_stop (float) : Metros restantes hasta el fin del segmento actual
                              (proxy de distancia a la siguiente parada).
        s_next_stop (float) : Segundos reales hasta que next_stop cambia en los
                              datos del mismo bus+viaje. NaN si el cambio no se
                              observa dentro del viaje.

    Parameters
    ----------
        df_trips_route    : pl.DataFrame   Salida del filtro sobre df_trips.
                                           Columnas requeridas: bus_code, trip_id,
                                           time, latitude, longitude.
        df_segments_route : pl.DataFrame   Geometría de segmentos.
                                           Columnas requeridas: stop, polyline.
        route_prefix      : str | None     Prefijo de ruta para el código de parada
                                           (ej. "101" → next_stop = "101-FQAE").
                                           None → devuelve el código tal cual.
        verbose           : bool           Imprimir progreso. Default False.

    Returns
    -------
        df_out (pl.DataFrame): df_trips_route con columnas adicionales:
                               next_stop, m_next_stop, s_next_stop.

    Examples
    --------
        df_enriched = enrich_trip_segments(
            df_trips_route,
            df_segments_route,
            route_prefix="101",
            verbose=True,
        )
    """

    #──────────────────────────────────────────────────────────#
    # ── 2.1 Validar columnas ── #
    #──────────────────────────#

    required = {"bus_code", "trip_id", "time", "latitude", "longitude"}
    missing  = required - set(df_trips_route.columns)
    if missing:
        raise ValueError(
            f"Columnas faltantes en df_trips_route: {missing}"
        )

    if verbose:
        print("=" * 60)
        print("  ENRICH TRIP SEGMENTS")
        print("=" * 60)
        print(f"\n  Registros entrada : {df_trips_route.height:,}")
        print(f"  Buses únicos      : {df_trips_route['bus_code'].n_unique()}")
        print(f"  Viajes únicos     : {df_trips_route['trip_id'].n_unique()}")
        print(f"  route_prefix      : {route_prefix!r}")

    #──────────────────────────────────────────────────────────#
    # ── 2.2 Preparar geometría de segmentos ── #
    #──────────────────────────────────────────#

    segs = _prepare_segments(df_segments_route, verbose=verbose)
    n_segs = len(segs)

    if n_segs == 0:
        raise ValueError("df_segments_route no contiene segmentos con geometría válida.")

    # Extraer arrays numpy para vectorizar la proyección
    # (hacemos la búsqueda del segmento más cercano por registro)
    # Pre-parsear todo en listas para acceso rápido en el loop
    seg_stops   = [s["stop"]     for s in segs]
    seg_pts     = [s["pts"]      for s in segs]
    seg_cums    = [s["cum"]      for s in segs]

    #──────────────────────────────────────────────────────────#
    # ── 2.3 Proyectar cada registro sobre los segmentos ── #
    #──────────────────────────────────────────────────────────#

    df_sorted = df_trips_route.sort(["bus_code", "trip_id", "time"])

    lats     = df_sorted["latitude"].cast(pl.Float64).to_numpy()
    lons     = df_sorted["longitude"].cast(pl.Float64).to_numpy()
    n        = len(lats)

    next_stop_list  = [""] * n
    m_next_list     = [0.0] * n

    if verbose:
        print(f"\n  [1/2] Proyectando {n:,} puntos sobre {n_segs} segmentos...")

    for i in range(n):
        p_lat = lats[i]
        p_lon = lons[i]

        best_perp   = math.inf
        best_stop   = ""
        best_remain = 0.0

        for s_idx in range(n_segs):
            perp, _, remain = _project_point(
                p_lat, p_lon, seg_pts[s_idx], seg_cums[s_idx]
            )
            if perp < best_perp:
                best_perp   = perp
                best_stop   = seg_stops[s_idx]
                best_remain = remain

        # Aplicar prefijo si se provee
        if route_prefix:
            best_stop = f"{route_prefix}-{best_stop}"

        next_stop_list[i] = best_stop
        m_next_list[i]    = round(best_remain, 2)

    if verbose:
        # Distribución de next_stop
        from collections import Counter
        cnt = Counter(next_stop_list)
        top5 = cnt.most_common(5)
        print(f"  Top-5 next_stops asignados: {top5}")
        import numpy as _np
        m_arr = _np.array(m_next_list)
        print(f"  m_next_stop — mediana: {_np.median(m_arr):.1f}m  "
              f"P95: {_np.percentile(m_arr, 95):.1f}m  "
              f"max: {_np.max(m_arr):.1f}m")

    #──────────────────────────────────────────────────────────#
    # ── 2.4 Calcular s_next_stop mirando al futuro ── #
    #──────────────────────────────────────────────────────────#

    if verbose:
        print(f"\n  [2/2] Calculando s_next_stop (tiempo real a siguiente parada)...")

    times_list  = df_sorted["time"].to_list()
    bus_list    = df_sorted["bus_code"].to_list()
    trip_list   = df_sorted["trip_id"].to_list()

    s_next_list = [float("nan")] * n

    # Por cada bus+viaje, recorrer hacia adelante para encontrar
    # cuándo cambia next_stop (= el bus llegó a la siguiente parada)
    i = 0
    while i < n:
        # Encontrar el bloque del mismo bus+viaje
        j = i
        while j < n and bus_list[j] == bus_list[i] and trip_list[j] == trip_list[i]:
            j += 1
        # Bloque [i, j)
        for r in range(i, j):
            current_stop = next_stop_list[r]
            # Buscar el primer registro adelante donde next_stop cambia
            found = False
            for f in range(r + 1, j):
                if next_stop_list[f] != current_stop:
                    # f es el primer registro en la nueva parada
                    dt = (times_list[f] - times_list[r]).total_seconds()
                    s_next_list[r] = round(dt, 1)
                    found = True
                    break
            # Si no se encontró cambio (fin de viaje), queda NaN
        i = j

    if verbose:
        import numpy as _np
        s_arr = _np.array([x for x in s_next_list if not math.isnan(x)])
        n_nan = sum(1 for x in s_next_list if math.isnan(x))
        if len(s_arr) > 0:
            print(f"  s_next_stop — mediana: {_np.median(s_arr):.1f}s  "
                  f"P95: {_np.percentile(s_arr, 95):.1f}s  "
                  f"max: {_np.max(s_arr):.1f}s")
        print(f"  NaN (sin cambio observable): {n_nan:,} "
              f"({n_nan/n*100:.1f}%) — últimos registros de cada parada")

    #──────────────────────────────────────────────────────────#
    # ── 2.5 Ensamblar DataFrame final ── #
    #──────────────────────────────────────────────────────────#

    df_out = df_sorted.with_columns([
        pl.Series("next_stop",   next_stop_list),
        pl.Series("m_next_stop", m_next_list),
        pl.Series("s_next_stop", s_next_list),
    ])

    if verbose:
        print(f"\n  Columnas del resultado: {df_out.columns}")
        print(f"  Registros salida      : {df_out.height:,}")
        print("=" * 60)

    return df_out

