"""
_lib_map_match.py
=================
Map-matching HMM de dirección y segmentación de viajes.

Este archivo es parte interna del pipeline (módulo `_lib_*`); los pasos
`step_*.py` lo importan, pero NO debe usarse directamente desde notebooks.

Origen
------
Copia exacta de `map_match_direction.py` del proyecto BUSONORA original. Mantenerlo
sincronizado si la versión upstream cambia.
"""

#################
### Libraries ###
#################

import json
import math
import warnings
import folium
import numpy as np
import polars as pl
from pathlib import Path
from typing import Optional, List, Tuple, Dict
from datetime import datetime

###############
### Helpers ###
###############

from _lib_geo import _haversine_m  # consolidado en lib/_lib_geo.py


def _cumulative_distances(coords: List[Tuple[float, float]]) -> np.ndarray:
    """
    Calcula distancias acumuladas a lo largo de una polilínea.

    Parameters
    ----------
        coords : list of (lat, lon)  Puntos de la polilínea.

    Returns
    -------
        cum (np.ndarray): Array de distancias acumuladas en metros (shape = len(coords)).
    """
    cum = [0.0]
    for i in range(1, len(coords)):
        d = _haversine_m(coords[i - 1][0], coords[i - 1][1], coords[i][0], coords[i][1])
        cum.append(cum[-1] + d)
    return np.array(cum)


def _project_to_polyline(
    p_lat: float,
    p_lon: float,
    coords: List[Tuple[float, float]],
    cum: np.ndarray,
) -> Tuple[float, float, float, float]:
    """
    Proyecta un punto sobre la polilínea más cercana (distancia perpendicular mínima).

    Parameters
    ----------
        p_lat, p_lon : float              Coordenadas del punto a proyectar.
        coords       : list of (lat, lon) Polilínea.
        cum          : np.ndarray         Distancias acumuladas de la polilínea.

    Returns
    -------
        snap_lat (float): Latitud del punto proyectado.
        snap_lon (float): Longitud del punto proyectado.
        dist_m   (float): Distancia perpendicular en metros.
        pos_m    (float): Posición a lo largo de la polilínea en metros.
    """
    best_perp = np.inf
    best_lat = coords[0][0]
    best_lon = coords[0][1]
    best_pos = 0.0

    for i in range(len(coords) - 1):
        lat1, lon1 = coords[i]
        lat2, lon2 = coords[i + 1]
        seg_len = cum[i + 1] - cum[i]
        if seg_len < 1e-9:
            continue

        cos_mid = math.cos(math.radians((lat1 + lat2) / 2.0))
        dx_s = (lon2 - lon1) * cos_mid
        dy_s = lat2 - lat1
        seg_sq = dx_s ** 2 + dy_s ** 2
        if seg_sq < 1e-20:
            continue

        dx_p = (p_lon - lon1) * cos_mid
        dy_p = p_lat - lat1
        t = max(0.0, min(1.0, (dx_p * dx_s + dy_p * dy_s) / seg_sq))

        proj_lat = lat1 + t * (lat2 - lat1)
        proj_lon = lon1 + t * (lon2 - lon1)

        perp = math.sqrt(
            ((p_lon - proj_lon) * cos_mid * 111_320) ** 2
            + ((p_lat - proj_lat) * 110_540) ** 2
        )

        if perp < best_perp:
            best_perp = perp
            best_lat = proj_lat
            best_lon = proj_lon
            best_pos = cum[i] + t * seg_len

    return best_lat, best_lon, best_perp, best_pos


def _split_circular_by_end_stops(
    coords: List[Tuple[float, float]],
    end_stop_A: Tuple[float, float],
    end_stop_B: Tuple[float, float],
    verbose: bool = False,
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """
    Divide una polilínea circular en dos ramas usando las coordenadas de
    dos paradas terminales (end_stops) snapeadas sobre la polilínea.

    Estrategia:
      1. Snapear end_stop_A y end_stop_B a la polilínea → posiciones pos_A, pos_B
         (distancia acumulada desde el inicio).
      2. Garantizar pos_A < pos_B reordenando si es necesario.
      3. Rama A: subcadena de puntos desde pos_A hasta pos_B (arco corto).
      4. Rama B: subcadena de puntos desde pos_B hasta el final + desde el inicio
         hasta pos_A (arco largo / complementario), recorrido en orden natural
         (no invertido), para mantener la dirección real del bus en ese sentido.

    La dirección del bus en la Rama B es entonces la continuación natural del
    recorrido circular: saliendo de end_stop_B, sigue la polilínea hasta cerrar
    el círculo en end_stop_A.

    Parameters
    ----------
        coords       : list of (lat, lon)   Polilínea circular completa.
        end_stop_A   : (lat, lon)            Coordenadas de la parada terminal A.
        end_stop_B   : (lat, lon)            Coordenadas de la parada terminal B.
        verbose      : bool                  Imprimir información del corte. Default False.

    Returns
    -------
        branch_A (list of (lat, lon)): Puntos de la Rama A (end_stop_A → end_stop_B).
        branch_B (list of (lat, lon)): Puntos de la Rama B (end_stop_B → end_stop_A,
                                        siguiendo el arco complementario).
    """
    cum = _cumulative_distances(coords)
    total_len = cum[-1]

    # Snapear cada end_stop a la polilínea para obtener su posición acumulada
    _, _, dist_snap_A, pos_A = _project_to_polyline(
        end_stop_A[0], end_stop_A[1], coords, cum
    )
    _, _, dist_snap_B, pos_B = _project_to_polyline(
        end_stop_B[0], end_stop_B[1], coords, cum
    )

    if verbose:
        print(f"  End-stop A → pos={pos_A/1000:.3f} km  snap_dist={dist_snap_A:.1f}m")
        print(f"  End-stop B → pos={pos_B/1000:.3f} km  snap_dist={dist_snap_B:.1f}m")
        print(f"  Longitud total polilínea: {total_len/1000:.3f} km")

    # Garantizar pos_A < pos_B
    if pos_A > pos_B:
        pos_A, pos_B = pos_B, pos_A
        end_stop_A, end_stop_B = end_stop_B, end_stop_A
        if verbose:
            print(f"  ⟳ Reordenadas: pos_A={pos_A/1000:.3f} km < pos_B={pos_B/1000:.3f} km")

    # Encontrar índices de corte en la polilínea
    # idx_A: primer índice con cum >= pos_A
    # idx_B: primer índice con cum >= pos_B
    idx_A = int(np.searchsorted(cum, pos_A, side="left"))
    idx_B = int(np.searchsorted(cum, pos_B, side="left"))
    idx_A = max(0, min(idx_A, len(coords) - 1))
    idx_B = max(0, min(idx_B, len(coords) - 1))

    # Insertar el punto snapeado exacto para no perder precisión en el corte
    snap_A_lat = end_stop_A[0] + (coords[idx_A][0] - end_stop_A[0]) * 0  # usar el snap real
    snap_B_lat = end_stop_B[0] + (coords[idx_B][0] - end_stop_B[0]) * 0

    # Snap exacto: interpolar en el segmento donde cayó el corte
    def _snap_exact(pos_target: float) -> Tuple[float, float]:
        idx = int(np.searchsorted(cum, pos_target, side="left"))
        idx = max(1, min(idx, len(coords) - 1))
        seg_len = cum[idx] - cum[idx - 1]
        if seg_len < 1e-9:
            return coords[idx]
        t = (pos_target - cum[idx - 1]) / seg_len
        lat = coords[idx - 1][0] + t * (coords[idx][0] - coords[idx - 1][0])
        lon = coords[idx - 1][1] + t * (coords[idx][1] - coords[idx - 1][1])
        return (lat, lon)

    pt_A = _snap_exact(pos_A)
    pt_B = _snap_exact(pos_B)

    # Rama A: pt_A → ... → pt_B  (arco entre los dos end_stops, sentido original)
    branch_A: List[Tuple[float, float]] = [pt_A]
    for i in range(len(coords)):
        if cum[i] > pos_A and cum[i] < pos_B:
            branch_A.append(coords[i])
    branch_A.append(pt_B)

    # Rama B: pt_B → ... → fin + inicio ... → pt_A  (arco complementario, sentido natural)
    branch_B: List[Tuple[float, float]] = [pt_B]
    for i in range(len(coords)):
        if cum[i] > pos_B:
            branch_B.append(coords[i])
    for i in range(len(coords)):
        if cum[i] < pos_A:
            branch_B.append(coords[i])
    branch_B.append(pt_A)

    if verbose:
        len_A = _cumulative_distances(branch_A)[-1] / 1000
        len_B = _cumulative_distances(branch_B)[-1] / 1000
        print(f"  Rama A: {len(branch_A)} puntos  {len_A:.3f} km  "
              f"(idx_A={idx_A} → idx_B={idx_B})")
        print(f"  Rama B: {len(branch_B)} puntos  {len_B:.3f} km  "
              f"(idx_B={idx_B} → idx_A={idx_A} via wrap)")

    return branch_A, branch_B


def _load_geojson_branches(
    geojson_path: str,
    df_end_stop: Optional["pl.DataFrame"] = None,
    verbose: bool = False,
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """
    Carga las dos ramas (Dir A y Dir B) desde un archivo de polilínea.

    Soporta tres formatos de entrada (auto-detectados):
      * **polylines.json anidado (nuevo)**:
        ``{operator_id: {route_id: {polyline: {type: LineString, coordinates: …}}}}``.
        Se selecciona la ruta usando ``config.INT_ROUTE_STOP``.
      * **GeoJSON FeatureCollection** (legacy): se toman todas las features
        ``LineString`` como ramas separadas.
      * **GeoJSON LineString único** (legacy): se trata como ruta circular.

    Estrategia de ramas (igual en los 3 formatos):
      - Si hay 2 o más LineStrings → rama A = línea[0], rama B = línea[1].
      - Si hay 1 LineString y se proveen 2 end_stops → divide con
        ``_split_circular_by_end_stops``.
      - Si hay 1 LineString y no hay end_stops → fallback: rama B = inverso.

    Parameters
    ----------
        geojson_path : str                  Ruta al archivo (.json o .geojson).
        df_end_stop  : pl.DataFrame | None  DataFrame con columnas: code/codigo,
                                            latitude, longitude. 2 filas (terminales).
        verbose      : bool                 Imprimir información de carga.

    Returns
    -------
        branch_A, branch_B : list[(lat, lon)]
    """
    with open(geojson_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    def _extract_coords(geom: dict) -> List[Tuple[float, float]]:
        # GeoJSON usa [lon, lat] → convertir a (lat, lon)
        return [(c[1], c[0]) for c in geom["coordinates"]]

    lines: List[List[Tuple[float, float]]] = []
    top_type = data.get("type") if isinstance(data, dict) else None

    if top_type == "FeatureCollection":
        # FeatureCollection. Dos sub-formatos:
        #   (v3 plano) una feature por ruta, con properties.{id|line|route}
        #              == route_id. Hay que FILTRAR por ruta o se mezclan
        #              polilíneas de rutas distintas (p.ej. rama B terminaría
        #              siendo la línea de OTRA ruta).
        #   (legacy)   las 2 ramas de la misma ruta como LineStrings sueltos,
        #              sin properties de ruta. Se toman todas.
        from config import INT_ROUTE_STOP
        route_key = str(INT_ROUTE_STOP)

        def _feat_route(feat: dict) -> Optional[str]:
            props = feat.get("properties", {}) or {}
            for k in ("id", "line", "route", "route_id"):
                v = props.get(k)
                if v is not None:
                    return str(v)
            return None

        feats = data.get("features", []) or []
        matched = [
            f for f in feats
            if f.get("geometry", {}).get("type") == "LineString"
            and _feat_route(f) == route_key
        ]

        if matched:
            for feat in matched:
                lines.append(_extract_coords(feat["geometry"]))
            if verbose:
                print(f"  FeatureCollection (v3 plano): ruta {route_key} → "
                      f"{len(lines)} LineString(s)")
        else:
            # Legacy: sin properties de ruta → todas las LineStrings.
            for feat in feats:
                geom = feat.get("geometry", {})
                if geom.get("type") == "LineString":
                    lines.append(_extract_coords(geom))
            if verbose:
                print(f"  FeatureCollection (legacy): {len(lines)} LineString(s) "
                      f"(sin filtro de ruta)")

    elif top_type == "LineString":
        # Formato legacy: LineString único
        lines.append(_extract_coords(data))

    else:
        # ── Formato NUEVO: {operator: {route_id: {polyline: LineString}}} ──
        from config import INT_ROUTE_STOP
        route_key = str(INT_ROUTE_STOP)

        found = False
        for operator_id, routes_dict in (data or {}).items():
            if not isinstance(routes_dict, dict):
                continue
            entry = routes_dict.get(route_key)
            if isinstance(entry, dict) and isinstance(entry.get("polyline"), dict):
                geom = entry["polyline"]
                if geom.get("type") == "LineString":
                    lines.append(_extract_coords(geom))
                    found = True
                    if verbose:
                        print(f"  polylines.json: operador={operator_id}, "
                              f"ruta={route_key}")
                    break  # una ruta puede aparecer en un solo operador

        if not found and verbose:
            print(f"  ⚠ Ruta {route_key} no encontrada en {geojson_path}")

    if len(lines) == 0:
        raise ValueError(f"No se encontró ningún LineString en {geojson_path}")

    if len(lines) >= 2:
        branch_A = lines[0]
        branch_B = lines[1]
        if verbose:
            print(f"  Polilínea con {len(lines)} LineStrings → "
                  f"rama A={len(branch_A)} pts, rama B={len(branch_B)} pts")
        return branch_A, branch_B

    # ── 1 LineString: ruta circular ──
    circle = lines[0]
    if verbose:
        print(f"  Polilínea con 1 LineString circular ({len(circle)} pts)")

    if df_end_stop is None or df_end_stop.height < 2:
        # Fallback legacy: rama B = inverso
        if verbose:
            print(f"  Sin df_end_stop → rama B = inverso (fallback)")
        return circle, list(reversed(circle))

    # Tomar las primeras 2 filas de df_end_stop como los dos terminales
    rows = df_end_stop.head(2).iter_rows(named=True)
    r0 = next(rows)
    r1 = next(rows)

    # Detectar columnas de coordenadas con nombres flexibles
    def _get_coord(row: dict, lat_keys=("latitude", "latitud", "lat"),
                   lon_keys=("longitude", "longitud", "lon")) -> Tuple[float, float]:
        lat = next((row[k] for k in lat_keys if k in row), None)
        lon = next((row[k] for k in lon_keys if k in row), None)
        if lat is None or lon is None:
            raise ValueError(f"No se encontraron columnas lat/lon en df_end_stop. "
                             f"Columnas disponibles: {list(row.keys())}")
        return (float(lat), float(lon))

    end_A = _get_coord(r0)
    end_B = _get_coord(r1)
    cod_A = r0.get("code", r0.get("codigo", "end_A"))
    cod_B = r1.get("code", r1.get("codigo", "end_B"))

    if verbose:
        print(f"  End-stop A: {cod_A}  ({end_A[0]:.6f}, {end_A[1]:.6f})")
        print(f"  End-stop B: {cod_B}  ({end_B[0]:.6f}, {end_B[1]:.6f})")

    branch_A, branch_B = _split_circular_by_end_stops(
        circle, end_A, end_B, verbose=verbose
    )
    return branch_A, branch_B


#──────────────────────────────────────────────────────#
# ── 1. Evidencia geométrica (D-S masa primaria) ── #
#──────────────────────────────────────────────────────#

def _geometric_evidence(
    lats: np.ndarray,
    lons: np.ndarray,
    branch_A: List[Tuple[float, float]],
    branch_B: List[Tuple[float, float]],
    cum_A: np.ndarray,
    cum_B: np.ndarray,
    sigma: float = 30.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcula snap + masa geométrica para cada punto GPS.

    La masa sigue la lógica del paper (ecuación 9):
      m(Ai) es proporcional a la confiabilidad del ramal, definida como
      función decreciente de la distancia perpendicular.

    Aquí usamos: m_A = exp(-dist_A / sigma) normalizado sobre {A, B}.

    Parameters
    ----------
        lats, lons : np.ndarray              Arrays de coordenadas GPS.
        branch_A / B : list of (lat, lon)    Polilíneas de cada rama.
        cum_A / B    : np.ndarray            Distancias acumuladas de cada rama.
        sigma        : float                 Parámetro de escala (metros). Default 30.

    Returns
    -------
        snap_lat_A, snap_lon_A : np.ndarray  Proyecciones sobre rama A.
        snap_lat_B, snap_lon_B : np.ndarray  Proyecciones sobre rama B.
        dist_A_m, dist_B_m    : np.ndarray  Distancias perpendicularesen metros.
        m_geo                 : np.ndarray  Masa geométrica en [0,1].
                                             Valores > 0.5 favorecen Dir A.
    """
    n = len(lats)
    snap_lat_A = np.zeros(n)
    snap_lon_A = np.zeros(n)
    snap_lat_B = np.zeros(n)
    snap_lon_B = np.zeros(n)
    dist_A_m   = np.zeros(n)
    dist_B_m   = np.zeros(n)

    for i in range(n):
        la, loa, da, _ = _project_to_polyline(lats[i], lons[i], branch_A, cum_A)
        lb, lob, db, _ = _project_to_polyline(lats[i], lons[i], branch_B, cum_B)
        snap_lat_A[i] = la
        snap_lon_A[i] = loa
        snap_lat_B[i] = lb
        snap_lon_B[i] = lob
        dist_A_m[i]   = da
        dist_B_m[i]   = db

    # Masa tipo Gaussiana: más certeza cuanto más lejos está del ramal opuesto
    w_A = np.exp(-dist_A_m / sigma)
    w_B = np.exp(-dist_B_m / sigma)
    total = w_A + w_B + 1e-12
    m_geo = w_A / total   # en [0,1]; >0.5 = evidencia a favor de A

    return snap_lat_A, snap_lon_A, snap_lat_B, snap_lon_B, dist_A_m, dist_B_m, m_geo


#──────────────────────────────────────────────────────#
# ── 2. Evidencia temporal (ventana de mayoría) ── #
#──────────────────────────────────────────────────────#

def _temporal_evidence(
    directions_geo: np.ndarray,
    bus_codes: np.ndarray,
    window: int = 7,
    weight: float = 0.25,
) -> np.ndarray:
    """
    Calcula la masa temporal basada en la ventana de mayoría de registros vecinos.

    Equivale al suavizado contextual del paper: si los vecinos del mismo bus
    apuntan mayoritariamente a Dir X, se refuerza esa asignación.

    Parameters
    ----------
        directions_geo : np.ndarray  Dirección geométrica cruda (0=A, 1=B).
        bus_codes      : np.ndarray  Código de bus para cada registro.
        window         : int         Half-window de registros vecinos. Default 7.
        weight         : float       Peso máximo de la evidencia temporal. Default 0.25.

    Returns
    -------
        m_temporal (np.ndarray): Masa temporal en [-weight, +weight].
                                  Positivo = refuerza Dir A, negativo = refuerza Dir B.
    """
    n = len(directions_geo)
    m_temporal = np.zeros(n)

    # Agrupar índices por bus
    bus_to_idxs: Dict[str, List[int]] = {}
    for i, bc in enumerate(bus_codes):
        bus_to_idxs.setdefault(bc, []).append(i)

    for bc, idxs in bus_to_idxs.items():
        for pos, gi in enumerate(idxs):
            lo = max(0, pos - window)
            hi = min(len(idxs) - 1, pos + window)
            neighbor_dirs = directions_geo[[idxs[k] for k in range(lo, hi + 1)]]
            count_A = int(np.sum(neighbor_dirs == 0))
            count_B = len(neighbor_dirs) - count_A
            # ratio ∈ [-1, 1]: positivo = mayoría A
            ratio = (count_A - count_B) / max(len(neighbor_dirs), 1)
            m_temporal[gi] = ratio * weight

    return m_temporal


#──────────────────────────────────────────────────────#
# ── 3. Combinación D-S (ecuación 10 del paper) ── #
#──────────────────────────────────────────────────────#

def _combine_dempster(m_geo: np.ndarray, m_temporal: np.ndarray) -> np.ndarray:
    """
    Combina las masas geométrica y temporal con la regla de Dempster simplificada.

    m_geo ∈ [0,1] representa la plausibilidad de Dir A.
    m_temporal ∈ [-w, +w] es un ajuste aditivo.

    La masa combinada se clampea a [0,1] para mantener la interpretación
    como probabilidad de Dir A.

    Parameters
    ----------
        m_geo      : np.ndarray  Masa geométrica.
        m_temporal : np.ndarray  Corrección temporal.

    Returns
    -------
        m_combined (np.ndarray): Masa combinada en [0,1].
    """
    m_combined = np.clip(m_geo + m_temporal, 0.0, 1.0)
    return m_combined


#──────────────────────────────────────────────────────────────────────────#
# ── 4. Función pública: map_match_direction ── #
#──────────────────────────────────────────────────────────────────────────#

def map_match_direction(
    df_test: pl.DataFrame,
    geojson_path: str,
    df_end_stop: pl.DataFrame,
    sigma: float = 30.0,
    temporal_window: int = 7,
    temporal_weight: float = 0.25,
    threshold: float = 0.5,
    verbose: bool = False,
) -> pl.DataFrame:
    """
    Aplica Map-Matching con D-S Theory of Evidence para identificar la dirección
    (Dir A / Dir B) de cada registro GPS.

    Parameters
    ----------
        df_test          : pl.DataFrame   Registros GPS a procesar.
                                          Columnas requeridas: bus_code, time, latitude, longitude.
        geojson_path     : str            Ruta al archivo .geojson de la ruta.
                                          Acepta: 1 LineString (bidireccional) o 2 LineStrings.
        df_end_stop      : pl.DataFrame   DataFrame con las paradas terminales.
                                          Columnas requeridas: codigo, latitude, longitude.
        sigma            : float          Escala de la función gaussiana para masa geométrica
                                          (en metros). Default 30.
                                          Valores menores → asignación más estricta por distancia.
        temporal_window  : int            Half-window de registros vecinos para evidencia temporal.
                                          Default 7.
        temporal_weight  : float          Peso máximo de la corrección temporal [0, 0.5].
                                          Default 0.25.
        threshold        : float          Umbral de m_combined para asignar Dir A.
                                          Default 0.5 (por encima = Dir A, por debajo = Dir B).
        verbose          : bool           Imprimir progreso detallado. Default False.

    Returns
    -------
        df_result (pl.DataFrame): df_test con columnas adicionales:
            snap_lat_A, snap_lon_A  : proyección sobre rama A
            snap_lat_B, snap_lon_B  : proyección sobre rama B
            dist_A_m, dist_B_m      : distancias perpendicularesen metros
            m_geo                   : masa geométrica
            m_temporal              : corrección temporal
            m_combined              : masa D-S combinada
            direction               : 0=Dir A / 1=Dir B
            snap_lat, snap_lon      : coordenadas del snap final
            snap_dist_m             : distancia al snap final

    Examples
    --------
        df_result = map_match_direction(
            df_test,
            "101.geojson",
            df_end_stop,
            verbose=True,
        )
    """

    #──────────────────────────────────────────────────────────#
    # ── 4.1 Carga y preparación de la polilínea ── #
    #──────────────────────────────────────────────────────────#

    if verbose:
        print("=" * 60)
        print("  MAP-MATCH DIRECTION — D-S Theory of Evidence")
        print("=" * 60)
        print(f"\n  Archivo GeoJSON : {geojson_path}")
        print(f"  Registros GPS   : {df_test.height:,}")
        print(f"  Buses únicos    : {df_test['bus_code'].n_unique()}")
        print(f"  Parámetros      : sigma={sigma}m, window={temporal_window}, "
              f"weight={temporal_weight}, threshold={threshold}")

    branch_A, branch_B = _load_geojson_branches(geojson_path, df_end_stop=df_end_stop, verbose=verbose)
    cum_A = _cumulative_distances(branch_A)
    cum_B = _cumulative_distances(branch_B)

    if verbose:
        print(f"\n  Longitud rama A : {cum_A[-1]/1000:.2f} km")
        print(f"  Longitud rama B : {cum_B[-1]/1000:.2f} km")

    #──────────────────────────────────────────────────────────#
    # ── 4.2 Ordenar por bus y tiempo ── #
    #──────────────────────────────────────────────────────────#

    df_sorted = df_test.sort(["bus_code", "time"])
    lats      = df_sorted["latitude"].to_numpy().astype(float)
    lons      = df_sorted["longitude"].to_numpy().astype(float)
    bus_codes = df_sorted["bus_code"].to_numpy().astype(str)

    #──────────────────────────────────────────────────────────#
    # ── 4.3 Evidencia geométrica ── #
    #──────────────────────────────────────────────────────────#

    if verbose:
        print(f"\n  [1/3] Calculando evidencia geométrica ({len(lats):,} puntos)...")

    snap_lat_A, snap_lon_A, snap_lat_B, snap_lon_B, dist_A, dist_B, m_geo = (
        _geometric_evidence(lats, lons, branch_A, branch_B, cum_A, cum_B, sigma=sigma)
    )

    dir_geo = (m_geo < threshold).astype(np.int8)  # 0=A, 1=B

    if verbose:
        n_A = int(np.sum(dir_geo == 0))
        n_B = int(np.sum(dir_geo == 1))
        print(f"     Geometría → Dir A: {n_A:,} ({n_A/len(dir_geo)*100:.1f}%)  "
              f"Dir B: {n_B:,} ({n_B/len(dir_geo)*100:.1f}%)")
        print(f"     Dist perpendicular A — mediana: {np.median(dist_A):.1f}m  "
              f"P95: {np.percentile(dist_A,95):.1f}m")
        print(f"     Dist perpendicular B — mediana: {np.median(dist_B):.1f}m  "
              f"P95: {np.percentile(dist_B,95):.1f}m")

    #──────────────────────────────────────────────────────────#
    # ── 4.4 Evidencia temporal ── #
    #──────────────────────────────────────────────────────────#

    if verbose:
        print(f"\n  [2/3] Calculando evidencia temporal (window={temporal_window})...")

    m_temporal = _temporal_evidence(dir_geo, bus_codes, temporal_window, temporal_weight)

    #──────────────────────────────────────────────────────────#
    # ── 4.5 Combinación D-S y asignación final ── #
    #──────────────────────────────────────────────────────────#

    if verbose:
        print(f"\n  [3/3] Combinando evidencias (D-S)...")

    m_combined = _combine_dempster(m_geo, m_temporal)
    direction  = (m_combined < threshold).astype(np.int8)

    # Coordenadas del snap final según dirección asignada
    snap_lat = np.where(direction == 0, snap_lat_A, snap_lat_B)
    snap_lon = np.where(direction == 0, snap_lon_A, snap_lon_B)
    snap_dist = np.where(direction == 0, dist_A, dist_B)

    if verbose:
        n_A_final = int(np.sum(direction == 0))
        n_B_final = int(np.sum(direction == 1))
        n_cambios  = int(np.sum(dir_geo != direction))
        print(f"\n  Resultado final:")
        print(f"     Dir A : {n_A_final:,} ({n_A_final/len(direction)*100:.1f}%)")
        print(f"     Dir B : {n_B_final:,} ({n_B_final/len(direction)*100:.1f}%)")
        print(f"     Cambios por evidencia temporal: {n_cambios:,} "
              f"({n_cambios/len(direction)*100:.1f}%)")
        print(f"     Snap dist final — mediana: {np.median(snap_dist):.1f}m  "
              f"P95: {np.percentile(snap_dist,95):.1f}m  "
              f"Max: {np.max(snap_dist):.1f}m")
        n_gt50  = int(np.sum(snap_dist > 50))
        n_gt100 = int(np.sum(snap_dist > 100))
        print(f"     > 50m : {n_gt50:,} ({n_gt50/len(snap_dist)*100:.1f}%)   "
              f"> 100m: {n_gt100:,} ({n_gt100/len(snap_dist)*100:.1f}%)")

    #──────────────────────────────────────────────────────────#
    # ── 4.6 Ensamblar DataFrame resultado ── #
    #──────────────────────────────────────────────────────────#

    df_result = df_sorted.with_columns([
        pl.Series("snap_lat_A",  snap_lat_A),
        pl.Series("snap_lon_A",  snap_lon_A),
        pl.Series("snap_lat_B",  snap_lat_B),
        pl.Series("snap_lon_B",  snap_lon_B),
        pl.Series("dist_A_m",    dist_A.round(2)),
        pl.Series("dist_B_m",    dist_B.round(2)),
        pl.Series("m_geo",       np.round(m_geo, 4)),
        pl.Series("m_temporal",  np.round(m_temporal, 4)),
        pl.Series("m_combined",  np.round(m_combined, 4)),
        pl.Series("direction",   direction.astype(np.int32)),
        pl.Series("snap_lat",    snap_lat),
        pl.Series("snap_lon",    snap_lon),
        pl.Series("snap_dist_m", snap_dist.round(2)),
    ])

    if verbose:
        print(f"\n  ✅ df_result: {df_result.height:,} filas × {df_result.width} columnas")
        print("=" * 60)

    return df_result


#──────────────────────────────────────────────────────────────────────────#
# ── 5. Función pública: visualize_time_windows_simple ── #
#──────────────────────────────────────────────────────────────────────────#

def visualize_time_windows_simple(
    df_result: pl.DataFrame,
    geojson_path: str,
    bus_code: Optional[str] = None,
    window_minutes: int = 30,
    df_end_stop: Optional["pl.DataFrame"] = None,
    zoom_start: int = 13,
    verbose: bool = False,
) -> folium.Map:
    """
    Visualiza ventanas temporales de registros GPS snapeados sobre la ruta,
    coloreadas por dirección y agrupadas en ventanas de tiempo.

    Cada ventana de `window_minutes` minutos recibe un color de opacidad
    diferente para distinguir el avance temporal. La dirección (A/B) se
    representa con colores distintos (azul = Dir A, rojo = Dir B).

    Parameters
    ----------
        df_result        : pl.DataFrame        Resultado de map_match_direction.
                                               Columnas: snap_lat, snap_lon, direction,
                                               time, bus_code, m_combined.
        geojson_path     : str                 Ruta al .geojson de la ruta (para mostrar
                                               la polilínea de fondo).
        bus_code         : str | None          Código de bus a visualizar. Si None,
                                               usa el primer bus del DataFrame.
        window_minutes   : int                 Tamaño de ventana temporal en minutos.
                                               Default 30.
        df_end_stop      : pl.DataFrame | None DataFrame con end_stops (codigo, latitude,
                                               longitude). Necesario si el GeoJSON es
                                               circular (1 LineString) para dividir las
                                               ramas correctamente en el fondo del mapa.
        zoom_start       : int                 Zoom inicial del mapa Folium. Default 13.
        verbose          : bool                Imprimir resumen del mapa. Default False.

    Returns
    -------
        m (folium.Map): Mapa Folium interactivo listo para mostrar en Jupyter.

    Examples
    --------
        m = visualize_time_windows_simple(
            df_result,
            "101.geojson",
            bus_code=bus_codes[0],
            window_minutes=30,
            df_end_stop=df_end_stop,
            verbose=True,
        )
        m
    """

    #──────────────────────────────────────────────────────────#
    # ── 5.1 Filtrar bus ── #
    #──────────────────────#

    if bus_code is None:
        bus_code = df_result["bus_code"][0]

    df_bus = df_result.filter(pl.col("bus_code") == bus_code).sort("time")

    if df_bus.height == 0:
        raise ValueError(f"No se encontraron registros para bus_code='{bus_code}'")

    if verbose:
        print(f"  Bus: {bus_code}  |  Registros: {df_bus.height:,}  |  "
              f"Ventana: {window_minutes} min")

    #──────────────────────────────────────────────────────────#
    # ── 5.2 Crear ventanas temporales ── #
    #──────────────────────────────────────#

    times = df_bus["time"].to_list()
    t_min = min(times)
    t_max = max(times)

    from datetime import timedelta
    window_td = timedelta(minutes=window_minutes)
    windows = []
    t_cur = t_min
    while t_cur <= t_max:
        windows.append((t_cur, t_cur + window_td))
        t_cur += window_td

    n_windows = len(windows)
    if verbose:
        print(f"  Ventanas temporales: {n_windows}  "
              f"({t_min.strftime('%H:%M')} – {t_max.strftime('%H:%M')})")

    #──────────────────────────────────────────────────────────#
    # ── 5.3 Construir mapa base ── #
    #──────────────────────────────#

    lats_all = df_bus["snap_lat"].to_list()
    lons_all = df_bus["snap_lon"].to_list()
    center = (float(np.mean(lats_all)), float(np.mean(lons_all)))

    m = folium.Map(location=center, zoom_start=zoom_start, tiles="CartoDB positron")

    # Polilínea de ruta de fondo (ambas ramas)
    try:
        branch_A, branch_B = _load_geojson_branches(geojson_path, df_end_stop=df_end_stop, verbose=False)
        folium.PolyLine(
            [(lat, lon) for lat, lon in branch_A],
            color="#9b59b6", weight=4, opacity=0.35,
            tooltip="Ruta — Rama A",
        ).add_to(m)
        folium.PolyLine(
            [(lat, lon) for lat, lon in branch_B],
            color="#e67e22", weight=4, opacity=0.35,
            tooltip="Ruta — Rama B",
        ).add_to(m)
    except Exception:
        pass

    #──────────────────────────────────────────────────────────#
    # ── 5.4 Dibujar puntos por ventana temporal ── #
    #──────────────────────────────────────────────#

    COLOR_A = "#2980b9"   # azul Dir A
    COLOR_B = "#c0392b"   # rojo Dir B

    for w_idx, (t_start, t_end) in enumerate(windows):
        opacity = 0.3 + 0.7 * (w_idx / max(n_windows - 1, 1))
        layer_name = (f"Ventana {w_idx+1}: "
                      f"{t_start.strftime('%H:%M')}–{t_end.strftime('%H:%M')}")
        layer = folium.FeatureGroup(name=layer_name, show=(w_idx < 4))

        df_win = df_bus.filter(
            (pl.col("time") >= t_start) & (pl.col("time") < t_end)
        )

        if df_win.height == 0:
            continue

        for row in df_win.iter_rows(named=True):
            color     = COLOR_A if row["direction"] == 0 else COLOR_B
            dir_label = "A" if row["direction"] == 0 else "B"
            tooltip   = (
                f"Bus: {bus_code}<br>"
                f"Dir: {dir_label}<br>"
                f"m_combined: {row['m_combined']:.3f}<br>"
                f"snap_dist: {row['snap_dist_m']:.1f}m<br>"
                f"Hora: {row['time'].strftime('%H:%M:%S')}"
            )

            orig_lat = row["latitude"]
            orig_lon = row["longitude"]
            snap_lat_pt = row["snap_lat"]
            snap_lon_pt = row["snap_lon"]

            # Línea punteada: punto GPS original → snap sobre la ruta
            folium.PolyLine(
                [(orig_lat, orig_lon), (snap_lat_pt, snap_lon_pt)],
                color="#888888",
                weight=1,
                opacity=opacity * 0.7,
                dash_array="4 6",
                tooltip=tooltip,
            ).add_to(layer)

            # Punto original GPS (círculo pequeño hueco, gris)
            folium.CircleMarker(
                location=(orig_lat, orig_lon),
                radius=3,
                color="#888888",
                fill=True,
                fill_color="#ffffff",
                fill_opacity=opacity * 0.6,
                opacity=opacity * 0.7,
                tooltip=tooltip,
            ).add_to(layer)

            # Punto snapeado sobre la ruta (círculo sólido con color de dirección)
            folium.CircleMarker(
                location=(snap_lat_pt, snap_lon_pt),
                radius=4,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=opacity,
                opacity=opacity,
                tooltip=tooltip,
            ).add_to(layer)

        # Trazar línea de trayectoria dentro de la ventana
        if df_win.height >= 2:
            traj_coords = list(zip(df_win["snap_lat"].to_list(), df_win["snap_lon"].to_list()))
            # Segmentar por dirección para colorear diferente
            dirs = df_win["direction"].to_list()
            seg_start = 0
            for k in range(1, len(dirs) + 1):
                if k == len(dirs) or dirs[k] != dirs[seg_start]:
                    if k - seg_start >= 2:
                        seg_color = COLOR_A if dirs[seg_start] == 0 else COLOR_B
                        folium.PolyLine(
                            traj_coords[seg_start:k],
                            color=seg_color,
                            weight=2,
                            opacity=opacity * 0.8,
                        ).add_to(layer)
                    seg_start = k

        layer.add_to(m)

    #──────────────────────────────────────────────────────────#
    # ── 5.5 Leyenda y controles ── #
    #──────────────────────────────#

    legend_html = f"""
    <div style="position:fixed; bottom:30px; left:30px; z-index:9999;
                background:white; padding:10px 14px; border-radius:8px;
                border:1px solid #ccc; font-size:12px; font-family:monospace;">
      <b>Bus {bus_code}</b><br>
      <span style="color:{COLOR_A};">●</span> Dir A — snap sobre rama A<br>
      <span style="color:{COLOR_B};">●</span> Dir B — snap sobre rama B<br>
      <span style="color:#888888;">○</span> Punto GPS original<br>
      <span style="color:#888888;">┄</span> Desplazamiento al snap<br>
      <span style="color:#9b59b6;">━</span> Ruta rama A (fondo)<br>
      <span style="color:#e67e22;">━</span> Ruta rama B (fondo)<br>
      <small>Opacidad ↑ = más reciente en la ventana</small>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl(collapsed=False).add_to(m)

    if verbose:
        n_dir_A = int(df_bus["direction"].to_list().count(0))
        n_dir_B = df_bus.height - n_dir_A
        print(f"  Dir A: {n_dir_A:,}  Dir B: {n_dir_B:,}")
        print(f"  ✅ Mapa generado con {n_windows} ventanas de {window_minutes} min")

    return m


#──────────────────────────────────────────────────────────────────────────#
# ── 6. Métricas de efectividad ── #
#──────────────────────────────────────────────────────────────────────────#

def evaluate_direction_accuracy(
    df_result: pl.DataFrame,
    col_true: str = "direction_true",
    col_pred: str = "direction",
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Evalúa la exactitud de las predicciones de dirección contra etiquetas reales.

    Útil para medir la efectividad del model cuando se tiene un subconjunto
    de datos con la dirección correcta ya conocida (p.ej. del pipeline de
    snap_and_enrich.py).

    Parameters
    ----------
        df_result  : pl.DataFrame   DataFrame con columnas de predicción y verdad.
        col_true   : str            Columna con la dirección verdadera. Default 'direction_true'.
        col_pred   : str            Columna con la dirección predicha. Default 'direction'.
        verbose    : bool           Imprimir reporte completo. Default True.

    Returns
    -------
        df_metrics (pl.DataFrame): DataFrame con métricas por bus y globales.
                                   Columnas: bus_code, n_total, n_correct, accuracy.
    """

    #──────────────────────────────────────────────────────────#
    # ── 6.1 Validar columnas ── #
    #──────────────────────────#

    for col in [col_true, col_pred]:
        if col not in df_result.columns:
            raise ValueError(f"Columna '{col}' no encontrada en df_result")

    df_eval = df_result.with_columns(
        (pl.col(col_true) == pl.col(col_pred)).alias("correct")
    )

    #──────────────────────────────────────────────────────────#
    # ── 6.2 Métricas globales ── #
    #──────────────────────────#

    n_total   = df_eval.height
    n_correct = int(df_eval["correct"].sum())
    accuracy  = n_correct / n_total if n_total > 0 else 0.0

    #──────────────────────────────────────────────────────────#
    # ── 6.3 Métricas por bus ── #
    #──────────────────────────#

    df_by_bus = (
        df_eval
        .group_by("bus_code")
        .agg([
            pl.len().alias("n_total"),
            pl.col("correct").sum().alias("n_correct"),
        ])
        .with_columns(
            (pl.col("n_correct") / pl.col("n_total")).alias("accuracy")
        )
        .sort("accuracy")
    )

    #──────────────────────────────────────────────────────────#
    # ── 6.4 Reporte ── #
    #──────────────────#

    if verbose:
        print("=" * 60)
        print("  EVALUACIÓN DE EFECTIVIDAD — Map-Match Direction")
        print("=" * 60)
        print(f"\n  Total registros : {n_total:,}")
        print(f"  Correctos       : {n_correct:,}")
        print(f"  Exactitud global: {accuracy*100:.2f}%")

        # Matriz de confusión simple
        tp = int(df_eval.filter((pl.col(col_pred) == 0) & (pl.col(col_true) == 0)).height)
        tn = int(df_eval.filter((pl.col(col_pred) == 1) & (pl.col(col_true) == 1)).height)
        fp = int(df_eval.filter((pl.col(col_pred) == 0) & (pl.col(col_true) == 1)).height)
        fn = int(df_eval.filter((pl.col(col_pred) == 1) & (pl.col(col_true) == 0)).height)

        print(f"\n  Matriz de confusión (pred/real):")
        print(f"             Real A   Real B")
        print(f"  Pred A  :  {tp:>7,}  {fp:>7,}")
        print(f"  Pred B  :  {fn:>7,}  {tn:>7,}")

        prec_A = tp / max(tp + fp, 1)
        rec_A  = tp / max(tp + fn, 1)
        prec_B = tn / max(tn + fn, 1)
        rec_B  = tn / max(tn + fp, 1)

        print(f"\n  Dir A — Precisión: {prec_A*100:.1f}%  Recall: {rec_A*100:.1f}%")
        print(f"  Dir B — Precisión: {prec_B*100:.1f}%  Recall: {rec_B*100:.1f}%")

        print(f"\n  {'Bus':<20} {'N':>6} {'Correctos':>10} {'Accuracy':>10}")
        print(f"  {'─'*20} {'─'*6} {'─'*10} {'─'*10}")
        for row in df_by_bus.iter_rows(named=True):
            marker = "✅" if row["accuracy"] >= 0.90 else ("⚠️ " if row["accuracy"] >= 0.70 else "❌")
            print(f"  {marker} {str(row['bus_code']):<18} {row['n_total']:>6,} "
                  f"{row['n_correct']:>10,} {row['accuracy']*100:>9.1f}%")
        print("=" * 60)

    return df_by_bus


#──────────────────────────────────────────────────────────────────────────#
# ── 7. Función pública: make_dir_trips ── #
#──────────────────────────────────────────────────────────────────────────#

def _build_end_stop_zones(
    df_stop_route: "pl.DataFrame",
    end_stops: List[str],
    neighbor_window: int = 2,
    verbose: bool = False,
) -> Dict[str, List[Tuple[float, float]]]:
    """
    Construye las zonas de terminal: para cada end_stop, retorna las coordenadas
    de las paradas vecinas (±neighbor_window posiciones en stop_order) que forman
    la zona de corte de viaje.

    Parameters
    ----------
        df_stop_route    : pl.DataFrame   Paradas de la ruta en orden.
                                          Columnas: codigo, latitude, longitude,
                                          stop_order (int).
        end_stops        : list[str]      Códigos de las 2 paradas terminales.
        neighbor_window  : int            Paradas vecinas a incluir en cada lado.
                                          Default 2.
        verbose          : bool           Imprimir resumen. Default False.

    Returns
    -------
        zones (dict): {end_stop_code: [(lat, lon), ...]}
                      Lista de coordenadas de todas las paradas en la zona.
    """
    # Detectar columna de código
    code_col = next(
        (c for c in ["code", "codigo", "codigo_original"] if c in df_stop_route.columns),
        None
    )
    if code_col is None:
        raise ValueError(
            f"df_stop_route debe tener columna 'code', 'codigo' o 'codigo_original'. "
            f"Columnas disponibles: {df_stop_route.columns}"
        )

    order_col = next(
        (c for c in ["stop_order", "orden", "order"] if c in df_stop_route.columns),
        None
    )

    # Si no hay stop_order, usar el índice de fila como orden
    if order_col is None:
        df_stop_route = df_stop_route.with_row_index("_row_order")
        order_col = "_row_order"

    zones: Dict[str, List[Tuple[float, float]]] = {}

    for es in end_stops:
        # Buscar la fila del end_stop
        row_es = df_stop_route.filter(pl.col(code_col) == es)
        if row_es.height == 0:
            if verbose:
                print(f"  ⚠️  End-stop '{es}' no encontrado en df_stop_route")
            zones[es] = []
            continue

        es_order = int(row_es[order_col][0])

        # Paradas vecinas en un rango de ±neighbor_window
        lo = es_order - neighbor_window
        hi = es_order + neighbor_window
        df_zone = df_stop_route.filter(
            (pl.col(order_col) >= lo) & (pl.col(order_col) <= hi)
        )

        # Detectar columnas lat/lon
        lat_col = next((c for c in ["latitude","latitud","lat"] if c in df_zone.columns), None)
        lon_col = next((c for c in ["longitude","longitud","lon"] if c in df_zone.columns), None)
        if lat_col is None or lon_col is None:
            raise ValueError("df_stop_route debe tener columnas de latitud y longitud.")

        coords = list(zip(
            df_zone[lat_col].cast(pl.Float64).to_list(),
            df_zone[lon_col].cast(pl.Float64).to_list(),
        ))
        zones[es] = coords

        if verbose:
            print(f"  End-stop '{es}' → stop_order={es_order}  "
                  f"zona: orders [{lo}..{hi}]  ({len(coords)} paradas)")

    return zones


def _point_in_zone(
    lat: float,
    lon: float,
    zone_coords: List[Tuple[float, float]],
    radius_m: float = 80.0,
) -> bool:
    """
    Devuelve True si el punto (lat, lon) está dentro del radio de cualquier
    parada de la zona terminal.

    Parameters
    ----------
        lat, lon    : float               Coordenadas del punto a evaluar.
        zone_coords : list of (lat, lon)  Paradas de la zona terminal.
        radius_m    : float               Radio de detección en metros. Default 80.

    Returns
    -------
        bool: True si está dentro de la zona.
    """
    for z_lat, z_lon in zone_coords:
        if _haversine_m(lat, lon, z_lat, z_lon) <= radius_m:
            return True
    return False


def make_dir_trips(
    df_result: pl.DataFrame,
    df_stop_route: Optional["pl.DataFrame"] = None,
    end_stops: Optional[List[str]] = None,
    neighbor_window: int = 2,
    end_stop_radius_m: float = 80.0,
    end_stop_cooldown_min: float = 3.0,
    max_gap_min: float = 15.0,
    max_implied_speed_kmh: float = 180.0,    # Schuessler & Axhausen (2009) — 50 m/s
    min_hour: Optional[float] = None,
    max_hour: Optional[float] = None,
    min_points: int = 5,
    min_dist_m: float = 200.0,
    min_completeness: float = 0.50,
    geojson_path: Optional[str] = None,
    df_end_stop: Optional["pl.DataFrame"] = None,
    end_stop_zones: Optional[Dict[str, List[Tuple[float, float]]]] = None,
    # ─── DEPRECATED ────────────────────────────────────────────────────
    max_gap_dist_m: Optional[float] = None,  # ⚠️ DEPRECATED: usar max_implied_speed_kmh
    # ───────────────────────────────────────────────────────────────────
    verbose: bool = False,
) -> pl.DataFrame:
    """
    Segmenta los registros de map_match_direction en viajes individuales por bus,
    usando el paso por zonas terminales (end_stops ± neighbor_window paradas) como
    criterio principal de corte, complementado con filtros de gap y horario.

    Lógica de corte de viaje (por orden de prioridad):
      1. Zona terminal  — el snap entra en el radio de cualquier parada dentro
                          de la zona de un end_stop. Se aplica un cooldown para
                          evitar cortes repetidos mientras el bus está parado en
                          la terminal. Este es el corte "semántico" correcto.
      2. Gap temporal   — silencio > max_gap_min minutos entre registros del bus.
      3. Outlier de velocidad — velocidad implícita entre snaps consecutivos
                          (gap_dist / gap_time) > max_implied_speed_kmh. Detecta
                          saltos físicamente imposibles para un bus urbano,
                          siguiendo Schuessler & Axhausen (2009) con V_max=50 m/s
                          (=180 km/h).
      4. Cambio de bus_code (siempre).

    El corte por dirección (Dir A↔B) se elimina intencionalmente: la detección
    de dirección puede ser ruidosa cerca de las terminales, y el corte geográfico
    de zona terminal es más robusto.

    Cambios respecto a versiones anteriores
    ---------------------------------------
    El filtro espacial absoluto ``max_gap_dist_m`` (P95 automático de Lin & Zeng
    1999) fue reemplazado por ``max_implied_speed_kmh`` (Schuessler & Axhausen
    2009). Razón: en sistemas AVL con muestreo regular (~1 min), un gap espacial
    de 500–800 m es **movimiento legítimo** del bus, no un error GPS. El umbral
    espacial absoluto fragmentaba ~60% de los trips reales. El umbral de velocidad
    implícita captura los saltos físicamente imposibles sin penalizar movimiento
    real. El parámetro ``max_gap_dist_m`` se mantiene como alias deprecated.

    Parameters
    ----------
        df_result            : pl.DataFrame        Salida de map_match_direction.
                                                   Columnas requeridas: bus_code,
                                                   time, snap_lat, snap_lon,
                                                   direction.
        df_stop_route        : pl.DataFrame | None Paradas de la ruta en orden.
                                                   Columnas: codigo (o
                                                   codigo_original), latitude,
                                                   longitude, stop_order.
                                                   Requerido si se proveen
                                                   end_stops.
        end_stops            : list[str] | None    Códigos de las 2 paradas
                                                   terminales. Si None, se
                                                   segmenta solo por gap/hora.
        neighbor_window      : int                 Paradas vecinas a incluir en
                                                   cada lado del end_stop para
                                                   formar la zona de corte.
                                                   Default 2.
        end_stop_radius_m    : float               Radio (metros) alrededor de
                                                   cada parada de la zona para
                                                   detectar paso del bus.
                                                   Default 80.
        end_stop_cooldown_min: float               Minutos mínimos entre cortes
                                                   consecutivos en la misma zona
                                                   terminal. Evita fragmentar el
                                                   viaje mientras el bus está
                                                   parado en la terminal.
                                                   Default 3.
        max_gap_min          : float               Gap temporal máximo (minutos).
                                                   Default 15.
        max_implied_speed_kmh: float               Velocidad implícita máxima
                                                   entre snaps consecutivos
                                                   (km/h). Pings que excedan
                                                   este umbral se consideran
                                                   outliers GPS físicamente
                                                   imposibles y disparan corte
                                                   del viaje. Default 180.0
                                                   (= 50 m/s, Schuessler &
                                                   Axhausen 2009).
        max_gap_dist_m       : float | None        ⚠️ DEPRECATED. Si se provee,
                                                   se emite un DeprecationWarning
                                                   y se convierte aproximadamente
                                                   a un equivalente en velocidad
                                                   (asume gap temporal = 1 min).
                                                   Usar ``max_implied_speed_kmh``.
        min_hour             : float | None        Hora mínima del día decimal
                                                   (ej. 5.5 = 05:30).
        max_hour             : float | None        Hora máxima del día decimal.
        min_points           : int                 Registros mínimos por viaje
                                                   para trip_valid=True.
                                                   Default 5.
        min_dist_m           : float               Distancia acumulada mínima
                                                   absoluta del viaje (metros)
                                                   para trip_valid=True. Default 200.
        min_completeness     : float               Fracción mínima de la longitud
                                                   de la polilínea que debe cubrir
                                                   el viaje para ser válido.
                                                   Ej. 0.50 = el viaje debe recorrer
                                                   al menos el 50% de la rama.
                                                   Solo aplica si se provee
                                                   geojson_path. Default 0.50.
        geojson_path         : str | None          Ruta al .geojson de la ruta.
                                                   Necesario para los filtros de
                                                   longitud máxima y completitud
                                                   mínima por dirección.
        df_end_stop          : pl.DataFrame | None End stops para dividir la ruta
                                                   circular en ramas A y B (igual
                                                   que en map_match_direction).
        verbose              : bool                Imprimir reporte. Default False.

    Returns
    -------
        df_trips (pl.DataFrame): df_result filtrado con columnas adicionales:
            trip_id        (str)   : "{bus_code}_D{dir}_T{N}"
            trip_seq       (int)   : secuencial global del viaje dentro del bus.
            trip_valid     (bool)  : cumple min_points y min_dist_m.
            trip_n_pts     (int)   : registros en el viaje.
            trip_dist_m    (float) : distancia acumulada del viaje (suma de
                                     haversines entre snaps consecutivos), en
                                     metros. Refleja la distancia real recorrida.
            gap_time_min   (float) : gap temporal al registro anterior (NaN=primero).
            gap_dist_m     (float) : gap de distancia al snap anterior (NaN=primero).
            cut_reason     (str)   : razón del corte que inició este viaje.
                                     Valores: "start", "end_stop_zone",
                                     "gap_time", "speed_outlier", "bus_change".
            trip_reject    (str)   : razón de invalidación del viaje, o ""
                                     si trip_valid=True. Posibles valores:
                                     "too_few_points", "too_short_dist",
                                     "exceeds_polyline_len",
                                     "below_min_completeness".
            trip_completeness(float): fracción de la polilínea cubierta
                                     (acum_dist_m / poly_len). NaN si no
                                     se proveyó geojson_path.

    Examples
    --------
        df_trips = make_dir_trips(
            df_result,
            df_stop_route         = df_stop_101,
            end_stops             = ["101-5UJZ", "101-4YHR"],
            neighbor_window       = 2,
            end_stop_radius_m     = 80,
            end_stop_cooldown_min = 3,
            max_gap_min           = 7,
            max_implied_speed_kmh = 180.0,   # Schuessler & Axhausen (2009)
            min_hour              = 6.5,
            max_hour              = 20.0,
            geojson_path          = "101.geojson",
            df_end_stop           = df_end_stop,
            min_completeness      = 0.75,
            verbose               = True,
        )

    References
    ----------
        Schuessler, N., & Axhausen, K. W. (2009). Processing raw data from
            global positioning systems without additional information.
            Transportation Research Record, 2105(1), 28–36.
            https://doi.org/10.3141/2105-04
    """

    #──────────────────────────────────────────────────────────#
    # ── 7.1 Validar columnas requeridas ── #
    #──────────────────────────────────────#

    required = {"bus_code", "time", "snap_lat", "snap_lon", "direction"}
    missing = required - set(df_result.columns)
    if missing:
        raise ValueError(
            f"Columnas faltantes en df_result: {missing}. "
            f"Asegúrate de pasar el output de map_match_direction()."
        )

    if verbose:
        print("=" * 60)
        print("  MAKE DIR TRIPS — Segmentación de viajes")
        print("=" * 60)
        print(f"\n  Registros entrada    : {df_result.height:,}")
        print(f"  Buses únicos         : {df_result['bus_code'].n_unique()}")
        print(f"  Límites:")
        print(f"    end_stops          : {end_stops}")
        print(f"    neighbor_window    : ±{neighbor_window} paradas")
        print(f"    end_stop_radius_m  : {end_stop_radius_m} m")
        print(f"    end_stop_cooldown  : {end_stop_cooldown_min} min")
        print(f"    max_gap_min        : {max_gap_min} min")
        print(f"    max_implied_speed  : {max_implied_speed_kmh} km/h "
              f"(Schuessler & Axhausen 2009)")
        print(f"    rango horario      : "
              f"{min_hour if min_hour is not None else '—'} – "
              f"{max_hour if max_hour is not None else '—'}")
        print(f"    min_points         : {min_points}")
        print(f"    min_dist_m         : {min_dist_m} m")

    # ── Compatibilidad retro: max_gap_dist_m ──────────────────────────
    # En versiones anteriores el filtro espacial era un umbral absoluto
    # (P95 de gap_dist). Esa lógica fragmentaba trips legítimos en sistemas
    # AVL con muestreo regular. El umbral correcto es velocidad implícita
    # (Schuessler & Axhausen 2009). Si el caller pasa max_gap_dist_m, emitimos
    # warning y lo ignoramos — el cómputo ahora usa max_implied_speed_kmh.
    if max_gap_dist_m is not None:
        warnings.warn(
            "El parámetro `max_gap_dist_m` está deprecado. El filtro espacial "
            "absoluto (Lin & Zeng 1999) fragmentaba trips legítimos en datos "
            "AVL con muestreo regular. Use `max_implied_speed_kmh` "
            "(Schuessler & Axhausen 2009) en su lugar. Default: 180.0 km/h.",
            DeprecationWarning,
            stacklevel=2,
        )
        if verbose:
            print(f"    ⚠ max_gap_dist_m={max_gap_dist_m} ignorado (deprecated)")

    #──────────────────────────────────────────────────────────#
    # ── 7.2 Construir zonas terminales ── #
    #──────────────────────────────────────#

    end_zones: Dict[str, List[Tuple[float, float]]] = {}
    # Prioridad: end_stop_zones explícitas > derivar de (end_stops + df_stop_route)
    if end_stop_zones is not None and len(end_stop_zones) >= 1:
        end_zones = end_stop_zones
        use_end_stops = True
        if verbose:
            n_total = sum(len(v) for v in end_zones.values())
            print(f"  Zonas terminales (provistas): {len(end_zones)} zonas, "
                  f"{n_total} coords totales")
            for k, v in end_zones.items():
                print(f"    · {k}: {len(v)} coords")
        all_zone_coords: Dict[str, List[Tuple[float, float]]] = end_zones
    else:
        use_end_stops = (
            end_stops is not None
            and len(end_stops) >= 1
            and df_stop_route is not None
        )

        if use_end_stops:
            end_zones = _build_end_stop_zones(
                df_stop_route, end_stops,
                neighbor_window=neighbor_window,
                verbose=verbose,
            )
            # Lista plana de todas las coords de zona (para check rápido)
            all_zone_coords = end_zones
        else:
            all_zone_coords = {}
            if verbose and end_stops is not None:
                print("  ⚠️  end_stops provisto pero df_stop_route=None → "
                      "segmentación solo por gap/hora")

    #──────────────────────────────────────────────────────────#
    # ── 7.3 Filtro horario ── #
    #──────────────────────────#

    df_work = df_result.sort(["bus_code", "time"])
    df_work = df_work.with_columns(
        (pl.col("time").dt.hour() + pl.col("time").dt.minute() / 60.0).alias("_hour_dec")
    )

    n_before = df_work.height
    if min_hour is not None:
        df_work = df_work.filter(pl.col("_hour_dec") >= min_hour)
    if max_hour is not None:
        df_work = df_work.filter(pl.col("_hour_dec") <= max_hour)

    if verbose and (min_hour is not None or max_hour is not None):
        print(f"\n  Filtro horario: {n_before:,} → {df_work.height:,} "
              f"({n_before - df_work.height:,} removidos)")

    if df_work.height == 0:
        if verbose:
            print("  ⚠️  Sin registros tras filtro horario.")
        return pl.DataFrame()

    #──────────────────────────────────────────────────────────#
    # ── 7.4 Calcular gaps ── #
    #──────────────────────────#

    bus_list  = df_work["bus_code"].to_list()
    time_list = df_work["time"].to_list()
    slat_list = df_work["snap_lat"].to_numpy().astype(float)
    slon_list = df_work["snap_lon"].to_numpy().astype(float)
    dir_list  = df_work["direction"].to_list()

    n = len(bus_list)
    gap_time = [float("nan")] * n
    gap_dist = [float("nan")] * n

    for i in range(1, n):
        if bus_list[i] == bus_list[i - 1]:
            dt = (time_list[i] - time_list[i - 1]).total_seconds() / 60.0
            dd = _haversine_m(
                slat_list[i - 1], slon_list[i - 1],
                slat_list[i],     slon_list[i],
            )
            gap_time[i] = round(dt, 3)
            gap_dist[i] = round(dd, 2)

    #──────────────────────────────────────────────────────────#
    # ── 7.5 Filtro de outliers por velocidad implícita ── #
    #──────────────────────────────────────────────────#
    # Schuessler & Axhausen (2009) — V_max = 50 m/s ≈ 180 km/h por defecto.
    # El filtro espacial absoluto anterior (Lin & Zeng 1999, P95 automático)
    # fragmentaba ~60% de los trips legítimos en sistemas AVL con muestreo
    # regular. Ahora se cortan solo pings con velocidad físicamente imposible
    # (preserva movimiento legítimo del bus en avenidas rápidas).

    # Convertir a m/min para evitar divisiones repetidas en el loop
    max_implied_speed_m_per_min = max_implied_speed_kmh * 1000.0 / 60.0

    if verbose:
        n_outlier_total = sum(
            1 for i in range(n)
            if not math.isnan(gap_dist[i]) and not math.isnan(gap_time[i])
            and gap_time[i] > 0
            and (gap_dist[i] / gap_time[i]) > max_implied_speed_m_per_min
        )
        print(f"\n  max_implied_speed_kmh : {max_implied_speed_kmh:.1f} km/h "
              f"(= {max_implied_speed_m_per_min:.1f} m/min)")
        print(f"  Outliers de velocidad : {n_outlier_total:,} pings "
              f"({100*n_outlier_total/max(n,1):.4f}%)")

    #──────────────────────────────────────────────────────────#
    # ── 7.6 Segmentar: zona terminal + gap/bus ── #
    #──────────────────────────────────────────────#

    trip_raw    = [0] * n
    cut_reasons = ["start"] * n
    current_trip = 0

    # Cooldown: último tiempo de corte por zona terminal para cada bus
    last_end_stop_cut: Dict[str, Optional[object]] = {}  # bus → datetime | None

    for i in range(n):
        if i == 0:
            current_trip += 1
            trip_raw[i] = current_trip
            cut_reasons[i] = "start"
            continue

        cut        = False
        cut_reason = ""

        # 1. Cambio de bus
        if bus_list[i] != bus_list[i - 1]:
            cut = True
            cut_reason = "bus_change"

        # 2. Gap temporal
        elif not math.isnan(gap_time[i]) and gap_time[i] > max_gap_min:
            cut = True
            cut_reason = "gap_time"

        # 3. Outlier de velocidad — Schuessler & Axhausen (2009).
        # Solo corta si la velocidad implícita es físicamente imposible para
        # un bus (default V_max=180 km/h = 50 m/s). Movimiento legítimo del
        # bus en avenidas rápidas (~50 km/h) NO dispara este corte.
        elif (not math.isnan(gap_dist[i])
              and not math.isnan(gap_time[i])
              and gap_time[i] > 0
              and (gap_dist[i] / gap_time[i]) > max_implied_speed_m_per_min):
            cut = True
            cut_reason = "speed_outlier"

        # 4. Zona terminal (solo si hay end_stops configurados)
        elif use_end_stops:
            s_lat = slat_list[i]
            s_lon = slon_list[i]
            bus   = bus_list[i]
            t_now = time_list[i]

            # Revisar si el punto está en alguna zona terminal
            in_zone = False
            for es_code, zone_coords in all_zone_coords.items():
                if _point_in_zone(s_lat, s_lon, zone_coords, end_stop_radius_m):
                    in_zone = True
                    break

            if in_zone:
                # Aplicar cooldown: no cortar si ya cortamos hace menos de cooldown
                last_cut = last_end_stop_cut.get(bus)
                if last_cut is None:
                    # Primer corte por zona para este bus
                    cut = True
                    cut_reason = "end_stop_zone"
                    last_end_stop_cut[bus] = t_now
                else:
                    elapsed = (t_now - last_cut).total_seconds() / 60.0
                    if elapsed >= end_stop_cooldown_min:
                        cut = True
                        cut_reason = "end_stop_zone"
                        last_end_stop_cut[bus] = t_now
                    # else: dentro del cooldown, no cortamos (bus todavía en terminal)

        if cut:
            current_trip += 1
            cut_reasons[i] = cut_reason
        else:
            cut_reasons[i] = ""

        trip_raw[i] = current_trip

    #──────────────────────────────────────────────────────────#
    # ── 7.7 Longitudes de polilínea por dirección ── #
    #──────────────────────────────────────────────────#

    # Si se provee geojson_path + df_end_stop, calcular la longitud real
    # de cada rama para usarla como techo máximo de trip_dist_m.
    poly_len: Dict[int, float] = {}   # {0: metros_rama_A, 1: metros_rama_B}

    if geojson_path is not None:
        try:
            _br_A, _br_B = _load_geojson_branches(
                geojson_path, df_end_stop=df_end_stop, verbose=False
            )
            poly_len[0] = float(_cumulative_distances(_br_A)[-1])
            poly_len[1] = float(_cumulative_distances(_br_B)[-1])
            if verbose:
                print(f"\n  Longitud polilínea rama A : {poly_len[0]/1000:.3f} km")
                print(f"  Longitud polilínea rama B : {poly_len[1]/1000:.3f} km")
                print(f"  → exceeds_polyline_len : acum_dist_m > poly_len")
                print(f"  → below_min_completeness: acum_dist_m < {min_completeness*100:.0f}% de poly_len")
        except Exception as _e:
            if verbose:
                print(f"  ⚠️  No se pudo cargar geojson para límite de longitud: {_e}")

    #──────────────────────────────────────────────────────────#
    # ── 7.8 Construir trip_id y métricas por viaje ── #
    #──────────────────────────────────────────────────────────#

    from collections import defaultdict as _defaultdict
    trip_to_idxs: dict = _defaultdict(list)
    for i, t in enumerate(trip_raw):
        trip_to_idxs[t].append(i)

    trip_id_list      = [""] * n
    trip_seq_list     = [0]  * n
    trip_valid_list   = [False] * n
    trip_reject_list  = [""] * n
    trip_npts_list    = [0]  * n
    trip_dist_list    = [0.0] * n

    trip_completeness_list = [float('nan')] * n

    bus_seq: Dict[str, int]               = {}
    bus_dir_seq: Dict[Tuple, int]         = {}

    for t_idx in sorted(trip_to_idxs.keys()):
        idxs  = trip_to_idxs[t_idx]
        i0    = idxs[0]
        bus   = bus_list[i0]
        n_pts = len(idxs)

        # Dirección mayoritaria del viaje (más robusto que solo el primer punto)
        dir_counts = [0, 0]
        for ix in idxs:
            dir_counts[int(dir_list[ix])] += 1
        d = 0 if dir_counts[0] >= dir_counts[1] else 1

        # ── Distancia acumulada del viaje (suma de haversines consecutivos) ──
        # Se usa en vez de haversine primer↔último porque la ruta puede doblar:
        # un viaje circular tendría dist_haversine ≈ 0 aunque recorrió 15 km.
        # Esta es la distancia real recorrida sobre los snaps del viaje.
        acum_dist_m = 0.0
        for k in range(1, len(idxs)):
            acum_dist_m += _haversine_m(
                slat_list[idxs[k-1]], slon_list[idxs[k-1]],
                slat_list[idxs[k]],   slon_list[idxs[k]],
            )

        # dist_endpoint_m: haversine primer↔último (se guarda para referencia)
        i_last = idxs[-1]
        dist_endpoint_m = _haversine_m(
            slat_list[i0], slon_list[i0],
            slat_list[i_last], slon_list[i_last],
        )

        bus_seq[bus]        = bus_seq.get(bus, 0) + 1
        key_bd              = (bus, d)
        bus_dir_seq[key_bd] = bus_dir_seq.get(key_bd, 0) + 1

        dir_label = "A" if d == 0 else "B"
        tid   = f"{bus}_D{dir_label}_T{bus_dir_seq[key_bd]}"

        # ── Completitud del viaje respecto a la polilínea ──
        # completeness = acum_dist_m / longitud_rama  ∈ [0, ∞)
        # > 1 significa que el viaje fue más largo que la rama (ruido/loop extra)
        branch_len   = poly_len.get(d, None) if poly_len else None
        completeness = acum_dist_m / branch_len if branch_len else float("nan")

        # ── Filtros de validez ──
        # F1: mínimo de puntos
        # F2: distancia acumulada mínima absoluta (min_dist_m)
        # F3: no superar la longitud de la polilínea (exceeds_polyline_len)
        # F4: cubrir al menos min_completeness de la polilínea (below_min_completeness)
        reject = ""
        if n_pts < min_points:
            reject = "too_few_points"
        elif acum_dist_m < min_dist_m:
            reject = "too_short_dist"
        elif branch_len and acum_dist_m > branch_len:
            reject = "exceeds_polyline_len"
        elif branch_len and completeness < min_completeness:
            reject = "below_min_completeness"
        valid = (reject == "")

        for ix in idxs:
            trip_id_list[ix]           = tid
            trip_seq_list[ix]          = bus_seq[bus]
            trip_valid_list[ix]        = valid
            trip_reject_list[ix]       = reject
            trip_npts_list[ix]         = n_pts
            trip_dist_list[ix]         = round(acum_dist_m, 2)
            trip_completeness_list[ix] = round(completeness, 4) if not math.isnan(completeness) else float("nan")

    #──────────────────────────────────────────────────────────#
    # ── 7.9 Ensamblar DataFrame final ── #
    #──────────────────────────────────────#

    df_trips = df_work.drop("_hour_dec").with_columns([
        pl.Series("trip_id",      trip_id_list),
        pl.Series("trip_seq",     trip_seq_list,    dtype=pl.Int32),
        pl.Series("trip_valid",   trip_valid_list),
        pl.Series("trip_reject",  trip_reject_list),
        pl.Series("trip_n_pts",   trip_npts_list,   dtype=pl.Int32),
        pl.Series("trip_dist_m",        trip_dist_list),
        pl.Series("trip_completeness", trip_completeness_list),
        pl.Series("gap_time_min", gap_time),
        pl.Series("gap_dist_m",   gap_dist),
        pl.Series("cut_reason",   cut_reasons),
    ])

    #──────────────────────────────────────────────────────────#
    # ── 7.10 Reporte verbose ── #
    #──────────────────────────#

    if verbose:
        n_trips_total = len(trip_to_idxs)
        unique_trips: Dict[str, Tuple] = {}
        for i in range(n):
            tid = trip_id_list[i]
            if tid not in unique_trips:
                unique_trips[tid] = (dir_list[i], trip_valid_list[i])

        n_valid   = sum(1 for _, (_, v) in unique_trips.items() if v)
        n_invalid = n_trips_total - n_valid
        n_A_v = sum(1 for _, (d, v) in unique_trips.items() if d == 0 and v)
        n_B_v = sum(1 for _, (d, v) in unique_trips.items() if d == 1 and v)
        n_A_i = sum(1 for _, (d, v) in unique_trips.items() if d == 0 and not v)
        n_B_i = sum(1 for _, (d, v) in unique_trips.items() if d == 1 and not v)

        n_cut_es   = sum(1 for r in cut_reasons if r == "end_stop_zone")
        n_cut_gt   = sum(1 for r in cut_reasons if r == "gap_time")
        n_cut_so   = sum(1 for r in cut_reasons if r == "speed_outlier")
        n_cut_bus  = sum(1 for r in cut_reasons if r == "bus_change")

        n_rej_few  = sum(1 for r in trip_reject_list if r == "too_few_points")
        n_rej_sh   = sum(1 for r in trip_reject_list if r == "too_short_dist")
        n_rej_poly = sum(1 for r in trip_reject_list if r == "exceeds_polyline_len")
        # Contar por viaje único, no por registro
        seen_rej: dict = {}
        for i in range(n):
            tid = trip_id_list[i]
            if tid not in seen_rej:
                seen_rej[tid] = trip_reject_list[i]
        n_rej_few_u  = sum(1 for r in seen_rej.values() if r == "too_few_points")
        n_rej_sh_u   = sum(1 for r in seen_rej.values() if r == "too_short_dist")
        n_rej_poly_u = sum(1 for r in seen_rej.values() if r == "exceeds_polyline_len")
        n_rej_comp_u = sum(1 for r in seen_rej.values() if r == "below_min_completeness")

        # Completitud promedio de viajes válidos por dirección
        valid_comp_A = [trip_completeness_list[i] for i in range(n)
                        if trip_valid_list[i] and dir_list[i] == 0
                        and not math.isnan(trip_completeness_list[i])]
        valid_comp_B = [trip_completeness_list[i] for i in range(n)
                        if trip_valid_list[i] and dir_list[i] == 1
                        and not math.isnan(trip_completeness_list[i])]
        avg_comp_A = sum(valid_comp_A)/len(valid_comp_A) if valid_comp_A else float("nan")
        avg_comp_B = sum(valid_comp_B)/len(valid_comp_B) if valid_comp_B else float("nan")

        print(f"\n  Viajes generados  : {n_trips_total:,}")
        print(f"    Válidos         : {n_valid:,}  "
              f"(Dir A: {n_A_v}  Dir B: {n_B_v})")
        print(f"    Inválidos       : {n_invalid:,}  "
              f"(Dir A: {n_A_i}  Dir B: {n_B_i})")
        print(f"      → muy pocos puntos    : {n_rej_few_u:,} viajes")
        print(f"      → dist. muy corta     : {n_rej_sh_u:,} viajes")
        print(f"      → excede polilínea    : {n_rej_poly_u:,} viajes")
        print(f"      → completitud < {min_completeness*100:.0f}%   : {n_rej_comp_u:,} viajes")
        if not math.isnan(avg_comp_A):
            print(f"\n  Completitud media válidos Dir A: {avg_comp_A*100:.1f}%")
        if not math.isnan(avg_comp_B):
            print(f"  Completitud media válidos Dir B: {avg_comp_B*100:.1f}%")
        print(f"\n  Cortes por:")
        print(f"    Zona terminal   : {n_cut_es:,}  ← principal")
        print(f"    Gap temporal    : {n_cut_gt:,}")
        print(f"    Speed outlier   : {n_cut_so:,}  (Schuessler & Axhausen 2009)")
        print(f"    Cambio de bus   : {n_cut_bus:,}")
        print(f"\n  Registros en df_trips: {df_trips.height:,}")
        print("=" * 60)

    return df_trips


#──────────────────────────────────────────────────────────────────────────#
# ── 8. Función pública: visualize_bus_trips ── #
#──────────────────────────────────────────────────────────────────────────#

def visualize_bus_trips(
    df_trips: pl.DataFrame,
    geojson_path: str,
    bus_code: Optional[str] = None,
    max_days: int = 30,
    df_end_stop: Optional["pl.DataFrame"] = None,
    zoom_start: int = 13,
    verbose: bool = False,
) -> folium.Map:
    """
    Visualiza los viajes segmentados de un bus en los primeros N días válidos.
    El slider navega viaje por viaje dentro del día activo.

    Controles del mapa:
      - Slider      : avanza viaje por viaje en el día. La etiqueta muestra
                      trip_id, dirección, rango horario y número de puntos.
      - Checkboxes  : un checkbox por día válido; por defecto solo el primer
                      día activo está marcado.
      - Leyenda     : lista todos los viajes del día activo, resaltando el que
                      está visible en el slider.
      - Puntos GPS  : círculo hueco gris (original) + línea punteada + círculo
                      sólido coloreado (snap, color por trip_id).
      - Fondo       : polilínea de la Rama A (violeta) y Rama B (naranja).

    Parameters
    ----------
        df_trips     : pl.DataFrame        Salida de make_dir_trips().
        geojson_path : str                 Ruta al .geojson de la ruta.
        bus_code     : str | None          Bus a visualizar. None → el de más
                                           registros.
        max_days     : int                 Máximo de días válidos. Default 30.
        df_end_stop  : pl.DataFrame | None End stops para dividir ruta circular.
        zoom_start   : int                 Zoom inicial. Default 13.
        verbose      : bool                Default False.

    Returns
    -------
        m (folium.Map): Mapa Folium con panel JS embebido.

    Examples
    --------
        m = visualize_bus_trips(
            df_trips,
            "101.geojson",
            df_end_stop=df_end_stop,
            verbose=True,
        )
        m
    """

    #──────────────────────────────────────────────────────────#
    # ── 8.1 Selección de bus ── #
    #──────────────────────────#

    if bus_code is None:
        bus_counts = (
            df_trips
            .group_by("bus_code")
            .agg(pl.len().alias("n"))
            .sort("n", descending=True)
        )
        bus_code = bus_counts["bus_code"][0]
        if verbose:
            print("  Bus auto-seleccionado:", bus_code,
                  "(%d registros)" % bus_counts["n"][0])

    df_bus = df_trips.filter(pl.col("bus_code") == bus_code).sort("time")
    if df_bus.height == 0:
        raise ValueError("No se encontraron registros para bus_code='%s'" % bus_code)

    #──────────────────────────────────────────────────────────#
    # ── 8.2 Seleccionar primeros max_days días válidos ── #
    #──────────────────────────────────────────────────────────#

    df_bus = df_bus.with_columns(pl.col("time").dt.date().alias("_date"))

    valid_days = (
        df_bus.filter(pl.col("trip_valid"))
        .select("_date").unique().sort("_date")
        ["_date"].to_list()[:max_days]
    )
    if not valid_days:
        raise ValueError(
            "El bus '%s' no tiene días con viajes válidos." % bus_code
        )

    df_bus = df_bus.filter(pl.col("_date").is_in(valid_days))

    if verbose:
        print("  Bus            :", bus_code)
        print("  Días válidos   :", len(valid_days),
              " (primero: %s  último: %s)" % (valid_days[0], valid_days[-1]))

    #──────────────────────────────────────────────────────────#
    # ── 8.3 Paleta de colores por trip_id ── #
    #──────────────────────────────────────────#

    PALETTE = [
        "#e74c3c","#3498db","#2ecc71","#f39c12","#9b59b6",
        "#1abc9c","#e67e22","#2980b9","#27ae60","#8e44ad",
        "#d35400","#16a085","#c0392b","#2471a3","#1e8449",
    ]
    COLOR_INVALID = "#cccccc"
    COLOR_A_ROUTE = "#9b59b6"
    COLOR_B_ROUTE = "#e67e22"

    trip_ids_valid = sorted(
        df_bus.filter(pl.col("trip_valid"))["trip_id"].unique().to_list()
    )
    trip_color = {tid: PALETTE[i % len(PALETTE)] for i, tid in enumerate(trip_ids_valid)}

    #──────────────────────────────────────────────────────────#
    # ── 8.4 Preparar datos: DATA[date][trip_idx] ── #
    #──────────────────────────────────────────────────────────#

    times_all  = df_bus["time"].to_list()
    slats_all  = df_bus["snap_lat"].to_list()
    slons_all  = df_bus["snap_lon"].to_list()
    olats_all  = df_bus["latitude"].to_list()
    olons_all  = df_bus["longitude"].to_list()
    dirs_all   = df_bus["direction"].to_list()
    tids_all   = df_bus["trip_id"].to_list()
    valids_all = df_bus["trip_valid"].to_list()
    sdist_all  = df_bus["snap_dist_m"].to_list()
    dates_all  = df_bus["_date"].to_list()
    mcomb_all  = (df_bus["m_combined"].to_list()
                  if "m_combined" in df_bus.columns
                  else [None] * df_bus.height)

    from collections import OrderedDict as _OD
    import json as _json

    day_trip_pts:   dict = {}
    day_trip_dir:   dict = {}
    day_trip_valid: dict = {}

    for i in range(len(times_all)):
        ds  = str(dates_all[i])
        tid = str(tids_all[i])
        if ds not in day_trip_pts:
            day_trip_pts[ds]   = _OD()
            day_trip_dir[ds]   = {}
            day_trip_valid[ds] = {}
        if tid not in day_trip_pts[ds]:
            day_trip_pts[ds][tid]   = []
            day_trip_dir[ds][tid]   = int(dirs_all[i])
            day_trip_valid[ds][tid] = bool(valids_all[i])
        mc = mcomb_all[i]
        day_trip_pts[ds][tid].append({
            "slat":  float(slats_all[i]),
            "slon":  float(slons_all[i]),
            "olat":  float(olats_all[i]),
            "olon":  float(olons_all[i]),
            "dir":   int(dirs_all[i]),
            "tid":   tid,
            "valid": bool(valids_all[i]),
            "sdist": float(sdist_all[i]) if sdist_all[i] is not None else 0.0,
            "time":  times_all[i].strftime("%H:%M:%S"),
            "mc":    round(float(mc), 3) if mc is not None else None,
        })

    data_js: dict         = {}
    trip_labels_js: dict  = {}

    for ds, trips_od in day_trip_pts.items():
        data_js[ds]        = {}
        trip_labels_js[ds] = []
        for t_idx, (tid, pts) in enumerate(trips_od.items()):
            data_js[ds][str(t_idx)] = pts
            d = day_trip_dir[ds][tid]
            v = day_trip_valid[ds][tid]
            trip_labels_js[ds].append({
                "idx":     t_idx,
                "tid":     tid,
                "valid":   v,
                "dir":     "A" if d == 0 else "B",
                "t_start": pts[0]["time"],
                "t_end":   pts[-1]["time"],
                "n_pts":   len(pts),
            })

    max_trips_in_day = max((len(v) for v in data_js.values()), default=1)
    days_ordered     = [str(d) for d in valid_days]
    default_day      = days_ordered[0]

    colors_js = dict(trip_color)
    colors_js["__invalid__"] = COLOR_INVALID

    data_json        = _json.dumps(data_js)
    colors_json      = _json.dumps(colors_js)
    days_json        = _json.dumps(days_ordered)
    trip_labels_json = _json.dumps(trip_labels_js)

    #──────────────────────────────────────────────────────────#
    # ── 8.5 Construir mapa base + polilínea de ruta ── #
    #──────────────────────────────────────────────────────────#

    center_lat = float(np.mean(slats_all))
    center_lon = float(np.mean(slons_all))
    m = folium.Map(location=(center_lat, center_lon),
                   zoom_start=zoom_start, tiles="CartoDB positron")

    try:
        branch_A, branch_B = _load_geojson_branches(
            geojson_path, df_end_stop=df_end_stop, verbose=False
        )
        folium.PolyLine(
            [(lat, lon) for lat, lon in branch_A],
            color=COLOR_A_ROUTE, weight=4, opacity=0.30, tooltip="Rama A",
        ).add_to(m)
        folium.PolyLine(
            [(lat, lon) for lat, lon in branch_B],
            color=COLOR_B_ROUTE, weight=4, opacity=0.30, tooltip="Rama B",
        ).add_to(m)
    except Exception:
        pass

    #──────────────────────────────────────────────────────────#
    # ── 8.6 Panel HTML + JS ── #
    # Note: CSS is built as a plain string to avoid Python 3.12
    # f-string parser rejecting CSS unit literals like "10px".
    #──────────────────────────────────────────────────────────#

    css = (
        "<style>"
        "#vbt-panel{"
            "position:fixed;top:10px;right:10px;z-index:9999;"
            "background:rgba(15,17,23,0.93);color:#e0e0e0;"
            "font-family:monospace;font-size:12px;"
            "padding:12px 16px;border-radius:10px;"
            "border:1px solid rgba(255,255,255,0.12);"
            "max-height:92vh;overflow-y:auto;width:255px;"
            "box-shadow:0 4px 24px rgba(0,0,0,0.5);}"
        "#vbt-panel h3{margin:0 0 8px 0;font-size:13px;color:#6c9bff;"
            "white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}"
        ".sec-label{font-size:10px;color:#888;text-transform:uppercase;"
            "letter-spacing:.06em;margin:10px 0 4px 0;display:block;}"
        "#vbt-slider{width:100%;accent-color:#6c9bff;cursor:pointer;}"
        "#vbt-trip-label{text-align:center;font-size:11px;font-weight:bold;"
            "color:#f5f5f5;margin:4px 0 2px 0;line-height:1.4;}"
        "#vbt-trip-sub{text-align:center;font-size:10px;color:#888;margin-bottom:2px;}"
        ".vbt-day-row{display:flex;align-items:center;gap:6px;margin:3px 0;"
            "cursor:pointer;padding:2px 4px;border-radius:4px;"
            "transition:background .12s;}"
        ".vbt-day-row:hover{background:rgba(255,255,255,.06);}"
        ".vbt-day-row input{cursor:pointer;accent-color:#6c9bff;}"
        ".vbt-day-row label{cursor:pointer;font-size:11px;color:#ccc;}"
        "#vbt-legend{margin-top:6px;max-height:160px;overflow-y:auto;}"
        ".vbt-leg-row{display:flex;align-items:center;gap:5px;"
            "margin:2px 0;font-size:11px;}"
        ".vbt-swatch{width:10px;height:10px;border-radius:50%;flex-shrink:0;}"
        "#vbt-stats{margin-top:8px;font-size:10px;color:#888;line-height:1.6;}"
        "</style>"
    )

    html_panel = (
        "<div id='vbt-panel'>"
        "<h3>\U0001f68c " + str(bus_code) + "</h3>"
        "<span class='sec-label'>Viaje en el d\u00eda</span>"
        "<input id='vbt-slider' type='range' min='0'"
            " max='" + str(max_trips_in_day - 1) + "' value='0' step='1'>"
        "<div id='vbt-trip-label'>\u2014</div>"
        "<div id='vbt-trip-sub'>\u2014</div>"
        "<span class='sec-label'>D\u00edas v\u00e1lidos</span>"
        "<div id='vbt-days'></div>"
        "<span class='sec-label'>Leyenda viajes del d\u00eda</span>"
        "<div id='vbt-legend'></div>"
        "<div id='vbt-stats'></div>"
        "</div>"
    )

    # JS uses {{ }} for literal JS braces (we're inside a regular Python string,
    # NOT an f-string, so we write literal { } directly)
    js_tpl = """
<script>
(function() {
  var DATA        = %s;
  var COLORS      = %s;
  var DAYS        = %s;
  var TRIP_LABELS = %s;
  var DEFAULT_DAY = %s;

  var activeDays = {};
  DAYS.forEach(function(d) { activeDays[d] = (d === DEFAULT_DAY); });
  var currentIdx = 0;

  function getMap() {
    for (var k in window)
      if (window[k] && window[k]._leaflet_id !== undefined && window[k].addLayer)
        return window[k];
    return null;
  }

  var layer = null;

  function updateSliderRange() {
    var maxT = 0;
    DAYS.forEach(function(d) {
      if (!activeDays[d]) return;
      var lbs = TRIP_LABELS[d];
      if (lbs && lbs.length > maxT) maxT = lbs.length;
    });
    if (maxT === 0) maxT = 1;
    var sl = document.getElementById('vbt-slider');
    sl.max = maxT - 1;
    if (currentIdx >= maxT) currentIdx = 0;
    sl.value = currentIdx;
  }

  function render() {
    var lmap = getMap();
    if (!lmap) { setTimeout(render, 300); return; }
    if (layer) { lmap.removeLayer(layer); }
    layer = L.layerGroup();

    var nPts = 0;
    var firstMeta = null;

    DAYS.forEach(function(day) {
      if (!activeDays[day]) return;
      var dayData = DATA[day];
      var labels  = TRIP_LABELS[day];
      if (!dayData || !labels) return;
      var meta = labels[currentIdx];
      if (!meta) return;
      if (!firstMeta) firstMeta = meta;
      var pts = dayData[String(currentIdx)];
      if (!pts) return;

      pts.forEach(function(p) {
        nPts++;
        var color = p.valid ? (COLORS[p.tid] || '#aaa') : COLORS['__invalid__'];
        var tip = 'Bus: ' + p.tid
          + '<br>Hora: ' + p.time
          + '<br>Dir: ' + (p.dir === 0 ? 'A' : 'B')
          + (p.mc !== null ? '<br>m_c: ' + p.mc : '')
          + '<br>snap: ' + p.sdist + 'm'
          + (p.valid ? '' : '<br><i>inv\u00e1lido</i>');

        L.polyline([[p.olat,p.olon],[p.slat,p.slon]],
          {color:'#666',weight:1,opacity:.55,dashArray:'4 6'})
          .bindTooltip(tip).addTo(layer);
        L.circleMarker([p.olat,p.olon],
          {radius:3,color:'#888',fillColor:'#fff',fillOpacity:.5,opacity:.6,weight:1})
          .bindTooltip(tip).addTo(layer);
        L.circleMarker([p.slat,p.slon],
          {radius: p.valid?5:3, color:color, fillColor:color,
           fillOpacity: p.valid?.85:.35, opacity: p.valid?.95:.45,
           weight: p.valid?1.5:1})
          .bindTooltip(tip).addTo(layer);
        if (p.valid) {}
      });
    });

    layer.addTo(lmap);

    var lbl = document.getElementById('vbt-trip-label');
    var sub = document.getElementById('vbt-trip-sub');
    if (firstMeta) {
      lbl.textContent = (firstMeta.valid?'\u2705':'\u26a0\ufe0f') + ' ' + firstMeta.tid;
      sub.textContent = 'Dir ' + firstMeta.dir
        + '  ' + firstMeta.t_start + ' \u2013 ' + firstMeta.t_end
        + '  (' + firstMeta.n_pts + ' pts)';
    } else {
      lbl.textContent = '\u2014 sin viaje \u2014';
      sub.textContent = '';
    }

    var legDiv = document.getElementById('vbt-legend');
    legDiv.innerHTML = '';
    var seen = {};
    DAYS.forEach(function(day) {
      if (!activeDays[day]) return;
      var lbs = TRIP_LABELS[day]; if (!lbs) return;
      lbs.forEach(function(meta) {
        if (seen[meta.tid]) return; seen[meta.tid]=true;
        var row = document.createElement('div');
        row.className = 'vbt-leg-row';
        if (meta.idx === currentIdx) {
          row.style.background = 'rgba(108,155,255,.12)';
          row.style.borderRadius = '3px';
        }
        var sw = document.createElement('div');
        sw.className = 'vbt-swatch';
        sw.style.background = meta.valid?(COLORS[meta.tid]||'#aaa'):COLORS['__invalid__'];
        var sp = document.createElement('span');
        sp.style.color = meta.idx===currentIdx?'#f5f5f5':'#aaa';
        sp.textContent = (meta.valid?'':'\\u26a0 ')+meta.tid
          +' ('+meta.t_start.slice(0,5)+')';
        row.appendChild(sw); row.appendChild(sp);
        legDiv.appendChild(row);
      });
    });
    if (!Object.keys(seen).length)
      legDiv.innerHTML = '<span style="color:#555;font-size:10px">Sin viajes</span>';

    document.getElementById('vbt-stats').innerHTML =
      'Viaje '+(currentIdx+1)+' de '
      +(parseInt(document.getElementById('vbt-slider').max)+1)
      +'<br>Puntos: <b>'+nPts+'</b>';
  }

  document.getElementById('vbt-slider').addEventListener('input', function() {
    currentIdx = parseInt(this.value); render();
  });

  var daysDiv = document.getElementById('vbt-days');
  DAYS.forEach(function(day) {
    var row = document.createElement('div'); row.className='vbt-day-row';
    var cb  = document.createElement('input'); cb.type='checkbox';
    cb.id='vbt-d-'+day; cb.checked=activeDays[day];
    cb.addEventListener('change', function() {
      activeDays[day]=cb.checked; updateSliderRange(); render();
    });
    var lbl = document.createElement('label');
    lbl.htmlFor='vbt-d-'+day; lbl.textContent=day;
    row.appendChild(cb); row.appendChild(lbl);
    daysDiv.appendChild(row);
  });

  setTimeout(function() { updateSliderRange(); render(); }, 600);
})();
</script>
"""
    # Inject Python data into JS placeholders
    js_filled = js_tpl % (
        data_json,
        colors_json,
        days_json,
        trip_labels_json,
        _json.dumps(default_day),
    )

    panel_html = css + html_panel + js_filled
    m.get_root().html.add_child(folium.Element(panel_html))

    #──────────────────────────────────────────────────────────#
    # ── 8.7 Leyenda fija de ruta ── #
    #──────────────────────────────#

    legend_html = (
        "<div style='position:fixed;bottom:20px;left:20px;z-index:9998;"
        "background:rgba(15,17,23,.88);padding:8px 12px;"
        "border-radius:8px;border:1px solid rgba(255,255,255,.10);"
        "font-size:11px;font-family:monospace;color:#ccc;'>"
        "<span style='color:" + COLOR_A_ROUTE + ";'>\u2501</span> Rama A &nbsp;"
        "<span style='color:" + COLOR_B_ROUTE + ";'>\u2501</span> Rama B<br>"
        "<span style='color:#888;'>\u25cb</span> GPS original &nbsp;"
        "<span style='color:#888;'>\u2504</span> \u2192 snap<br>"
        "<span style='color:#ccc;'>\u25cf</span> v\u00e1lido &nbsp;"
        "<span style='color:#888;'>\u25cf</span> inv\u00e1lido"
        "</div>"
    )
    m.get_root().html.add_child(folium.Element(legend_html))

    if verbose:
        n_valid = df_bus.filter(pl.col("trip_valid"))["trip_id"].n_unique()
        n_total = df_bus["trip_id"].n_unique()
        print("  Viajes totales         :", n_total)
        print("  Viajes válidos         :", n_valid)
        print("  Max viajes en un día   :", max_trips_in_day)
        print("  Colores asignados      :", len(trip_color))
        print("  Mapa generado OK")

    return m