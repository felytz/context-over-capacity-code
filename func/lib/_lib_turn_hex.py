"""
_lib_turn_hex.py
=================
Detección y resolución de **turn_hex** — los dos hexágonos H3 que marcan las
zonas de retorno (terminales) de la ruta.

Este módulo reemplaza conceptualmente a ``_lib_end_stops`` con un cambio de
paradigma: las terminales ya no son paradas físicas codificadas (stop_codes
GTFS), sino entidades puramente geométricas detectadas por densidad de
registros parados en hexágonos H3.

Glosario
--------
turn_hex
    Hexágono H3 (resolución 12 por defecto) que representa el centro de una
    zona de retorno. Es el hex más denso de su cluster contiguo.

turn_cluster
    Componente conexa (vía k-ring=1) de hex parados que contiene al turn_hex
    como su miembro más denso. Caracteriza el "cluster operacional" completo
    de la terminal.

turn_zone
    turn_hex + sus vecinos k-ring=N que también contienen polilínea (filtrado
    por intersección con la geometría de la ruta). Define la zona efectiva
    para el corte de viajes en ``make_dir_trips``. Reemplaza la antigua
    ``neighbor_window`` por paradas.

Algoritmo de detección automática
---------------------------------
    1. Agregar registros con speed ≤ max_speed por celda H3 (default res=12).
    2. Filtrar hex con conteo ≥ min_hex_count.
    3. Construir clusters: componentes conexas (k-ring=1) sobre los hex parados.
    4. Por cada cluster, identificar su hex "centro" (el más denso).
    5. turn_hex_A := hex con mayor conteo individual GLOBALMENTE.
    6. Calcular distancia haversine (absoluta) desde turn_hex_A a todos los
       demás centros candidatos.
    7. Filtrar candidatos a distancia ≥ (1/4) * polyline_length.
    8. turn_hex_B := el más denso de los sobrevivientes.

Modo manual
-----------
Lista explícita de **dos códigos H3** (strings, no stop_codes GTFS):

    TURN_HEX_MANUAL = ["8c4...", "8c4..."]

Formato del cache (JSON)
------------------------
::

    {
      "mode":              "auto" | "manual",
      "resolution":        12,
      "polyline_length_m": 16545.3,
      "min_separation_m":  4136.3,
      "min_hex_count":     5,
      "max_speed":         0.0,
      "turn_hexes": [
        {
          "label":               "turn_hex_A",
          "h3":                  "8c4...",
          "lat":                 29.0989,
          "lon":                 -110.9227,
          "stopped_count":       1245,
          "cluster_h3s":         ["8c4...", "8c4...", ...],
          "cluster_total_count": 3421
        },
        { "label": "turn_hex_B", ... }
      ]
    }

Compatibilidad retro
--------------------
Para no romper el resto del pipeline mientras se migra, ``resolve_turn_hex``
también devuelve las claves legacy ``df_end_stop`` y ``zones`` (basadas en
los turn_hex y sus k-ring vecinos con polilínea). El módulo ``_lib_end_stops``
debería convertirse en un shim deprecated que llama aquí.
"""

from __future__ import annotations

import json
import math
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

import polars as pl


__all__ = [
    "resolve_turn_hex",
    "TurnHexResult",
    "compute_turn_zone_kring",
]


# ──────────────────────────────────────────────────────────────────────
# Tipos
# ──────────────────────────────────────────────────────────────────────

TurnHexResult = Dict[str, Any]
"""
Diccionario con:
  - "turn_hexes":   List[dict] con los 2 turn_hex resueltos (A y B).
  - "df_turn_hex":  pl.DataFrame (label, h3, lat, lon, stopped_count).
  - "df_end_stop":  pl.DataFrame legacy (code, latitude, longitude) — alias
                    de df_turn_hex con code=label. Para compatibilidad con
                    código que aún espera el nombre viejo.
  - "zones":        Dict[label, List[(lat, lon)]] con los centros de los
                    hex de la turn_zone (vecinos k-ring filtrados por
                    polilínea). Alias del concepto legacy.
  - "mode":         "auto" | "manual"
  - "raw":          dict crudo cacheable.
"""


# ──────────────────────────────────────────────────────────────────────
# Helpers H3 (lazy import para no exigir h3 si no se usa)
# ──────────────────────────────────────────────────────────────────────

def _h3_module():
    """Importa h3 con un mensaje útil si falta."""
    try:
        import h3
        return h3
    except ImportError as e:
        raise ImportError(
            "_lib_turn_hex requiere el paquete `h3` (pip install h3)."
        ) from e


from _lib_geo import _haversine_m  # consolidado en lib/_lib_geo.py


def _parse_polyline(raw) -> List[List[float]]:
    """
    Convierte el campo polyline a lista de [lat, lon] válidos.

    Maneja múltiples formatos:
      - Lista de [lat, lon] / (lat, lon)
      - Lista de dicts {"lat": ..., "lon": ...} / {"latitude": ..., "longitude": ...}
      - Lista de structs Polars (tienen __getitem__ con key string)
      - Objetos con atributos latitude/longitude
      - JSON strings
      - Python repr strings (fallback)
    """
    import ast

    if raw is None:
        return []

    # Si es string, intentar deserializar primero
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return []
        try:
            raw = json.loads(raw)
        except Exception:
            try:
                raw = ast.literal_eval(raw)
            except Exception:
                return []

    # Normalizar a list si es algún iterable Polars
    if not isinstance(raw, list):
        try:
            raw = list(raw)
        except Exception:
            return []

    if len(raw) == 0:
        return []

    points: List[List[float]] = []
    for item in raw:
        # 1. Lista o tupla [lat, lon]
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            try:
                points.append([float(item[0]), float(item[1])])
                continue
            except (TypeError, ValueError, KeyError):
                pass

        # 2. Dict Python {"lat":..., "latitude":..., etc.}
        if isinstance(item, dict):
            lat_key = next((k for k in ("lat", "latitude", "latitud")
                            if k in item), None)
            lon_key = next((k for k in ("lon", "lng", "longitude", "longitud")
                            if k in item), None)
            if lat_key and lon_key:
                try:
                    points.append([float(item[lat_key]), float(item[lon_key])])
                    continue
                except (TypeError, ValueError):
                    pass

        # 3. Struct Polars / objeto con acceso por key string
        #    (Polars structs son subscriptables: item["latitude"])
        for lat_key in ("lat", "latitude", "latitud"):
            for lon_key in ("lon", "lng", "longitude", "longitud"):
                try:
                    lat_v = item[lat_key]
                    lon_v = item[lon_key]
                    points.append([float(lat_v), float(lon_v)])
                    break
                except (KeyError, TypeError, IndexError):
                    continue
            else:
                continue
            break
        else:
            # 4. Objeto con atributos (último recurso)
            lat_v = lon_v = None
            for lat_attr in ("lat", "latitude", "latitud"):
                if hasattr(item, lat_attr):
                    lat_v = getattr(item, lat_attr)
                    break
            for lon_attr in ("lon", "lng", "longitude", "longitud"):
                if hasattr(item, lon_attr):
                    lon_v = getattr(item, lon_attr)
                    break
            if lat_v is not None and lon_v is not None:
                try:
                    points.append([float(lat_v), float(lon_v)])
                except (TypeError, ValueError):
                    pass

    return points


def _polyline_length_m(df_segments: pl.DataFrame) -> float:
    """Longitud total de la polilínea sumando haversine de tramos."""
    poly_col = next(
        (c for c in ("polyline", "geometry", "coords") if c in df_segments.columns),
        None,
    )
    if poly_col is None:
        return 0.0

    total = 0.0
    for val in df_segments[poly_col].to_list():
        pts = _parse_polyline(val)
        for i in range(1, len(pts)):
            total += _haversine_m(pts[i - 1][0], pts[i - 1][1],
                                  pts[i][0],     pts[i][1])
    return total


def _polyline_hex_set(
    df_segments: pl.DataFrame,
    resolution: int,
) -> Set[str]:
    """
    Devuelve el conjunto de índices H3 (a la resolución dada) que la polilínea
    atraviesa. Usa el centro de cada vértice; a resolución 12 (~9 m) y
    polilíneas urbanas el muestreo por vértices es suficientemente denso.
    """
    h3 = _h3_module()
    poly_col = next(
        (c for c in ("polyline", "geometry", "coords") if c in df_segments.columns),
        None,
    )
    if poly_col is None:
        return set()

    hex_set: Set[str] = set()
    for val in df_segments[poly_col].to_list():
        pts = _parse_polyline(val)
        for p in pts:
            hex_set.add(h3.latlng_to_cell(p[0], p[1], resolution))
    return hex_set


# ──────────────────────────────────────────────────────────────────────
# Conteo de hex parados
# ──────────────────────────────────────────────────────────────────────

def _h3_counts_from_stopped(
    df_rec: pl.DataFrame,
    resolution: int,
    max_speed: float,
    min_hex_count: int,
    max_pings_per_bus_per_hex: Optional[int] = None,
) -> pl.DataFrame:
    """
    Filtra registros con speed ≤ max_speed, los agrupa por celda H3 y devuelve
    un DataFrame ordenado descendente por conteo.

    El filtro ``min_hex_count`` aquí es un piso ABSOLUTO duro (default=1 para
    no filtrar nada). El filtro de outliers ("qué es realmente una terminal")
    se aplica DESPUÉS en ``_filter_outlier_hex`` con un método configurable.

    Parameters
    ----------
    max_pings_per_bus_per_hex : int | None
        Si es ``None`` (default), conteo crudo: cada ping parado cuenta como 1.
        Si es un entero K > 0, cada par ``(bus_code, h3_index)`` aporta como
        mucho K pings al ``stopped_count``. Esto es la métrica *bus-aware*
        que mitiga el sesgo de unidades con telemetría estacionaria
        (un bus dormido reportando miles de pings en el mismo hex).

        Caso especial: ``K=1`` equivale a contar buses únicos por hex
        (``n_unique(bus_code)``). En la práctica, K=1 suele descartar
        información operacional legítima (los buses esperan turno y reportan
        múltiples pings por pasada); valores K∈[5, 15] preservan esa señal
        mientras acotan unidades dormidas.

        Requiere que ``df_rec`` tenga columna ``bus_code``. Si no la tiene,
        se ignora el parámetro y se usa el conteo crudo (con un warning).

    Notes
    -----
    Referencia metodológica: Cathey & Dailey (2003) "A Prescription for
    Transit Arrival/Departure Prediction Using AVL Data" y la línea
    subsiguiente recomiendan ponderar por unidad vehicular en lugar de
    pings crudos al estimar densidad de paradas. La validación empírica
    de este cambio para la Ruta 101 está en
    ``audit/turn_hex/turn_hex_robustness_report.html``.
    """
    h3 = _h3_module()

    speed_col = next(
        (c for c in ("speed", "velocidad", "vel") if c in df_rec.columns),
        None,
    )
    if speed_col is None:
        raise ValueError(
            "df_rec_clean no tiene columna de velocidad. Esperaba "
            "alguna de: speed, velocidad, vel."
        )

    lat_col = "latitude" if "latitude" in df_rec.columns else "lat"
    lon_col = "longitude" if "longitude" in df_rec.columns else "lon"

    # Validar bus_code si se pidió cap
    use_cap = max_pings_per_bus_per_hex is not None
    if use_cap and "bus_code" not in df_rec.columns:
        import warnings
        warnings.warn(
            "max_pings_per_bus_per_hex se solicitó pero df_rec no tiene "
            "columna 'bus_code'. Cayendo al conteo crudo.",
            stacklevel=2,
        )
        use_cap = False
    if use_cap and max_pings_per_bus_per_hex < 1:
        raise ValueError(
            f"max_pings_per_bus_per_hex debe ser ≥ 1 o None, "
            f"recibí {max_pings_per_bus_per_hex}."
        )

    df_stopped = df_rec.filter(
        (pl.col(speed_col) <= max_speed)
        & pl.col(lat_col).is_not_null()
        & pl.col(lon_col).is_not_null()
    )
    if df_stopped.height == 0:
        raise ValueError(
            f"No hay registros parados (speed ≤ {max_speed}) en el dataset."
        )

    lats = df_stopped[lat_col].to_numpy()
    lons = df_stopped[lon_col].to_numpy()
    indices = [h3.latlng_to_cell(float(la), float(lo), resolution)
               for la, lo in zip(lats, lons)]

    if not use_cap:
        # ── Conteo crudo (comportamiento original) ──
        df_counts = (
            pl.DataFrame({"h3_index": indices})
            .group_by("h3_index")
            .agg(pl.len().alias("stopped_count"))
            .sort("stopped_count", descending=True)
        )
    else:
        # ── Conteo bus-aware con cap K por (bus, hex) ──
        # Para cada (bus, hex) limitamos contribución a K pings. Esto
        # impide que un bus con telemetría estacionaria domine la densidad
        # de un hex y, con ella, la elección de turn_hex.
        cap = int(max_pings_per_bus_per_hex)
        df_with_hex = df_stopped.with_columns(
            pl.Series("h3_index", indices)
        )
        df_counts = (
            df_with_hex
            # Contribución de cada bus a cada hex, capeada
            .group_by(["bus_code", "h3_index"])
            .agg(pl.len().alias("bus_hex_count"))
            .with_columns(
                pl.min_horizontal(pl.col("bus_hex_count"), pl.lit(cap))
                .alias("bus_hex_capped")
            )
            # Suma sobre buses para tener stopped_count por hex
            .group_by("h3_index")
            .agg(pl.col("bus_hex_capped").sum().alias("stopped_count"))
            .sort("stopped_count", descending=True)
        )

    if min_hex_count > 1:
        df_counts = df_counts.filter(pl.col("stopped_count") >= min_hex_count)
    return df_counts


def _compute_outlier_threshold(
    counts: List[int],
    method: Literal["percentile", "tukey", "absolute"],
    percentile: float,
    absolute_threshold: int,
    verbose: bool,
) -> int:
    """
    Calcula el umbral mínimo de conteo para considerar un hex "outlier"
    (candidato real a terminal). Tres métodos:

    - ``"percentile"`` : hex con conteo ≥ P_q (default q=95).
    - ``"tukey"``      : hex con conteo ≥ Q3 + 1.5 · IQR.
    - ``"absolute"``   : hex con conteo ≥ ``absolute_threshold``.

    Returns
    -------
    int : umbral inclusivo de stopped_count.
    """
    import numpy as np

    arr = np.asarray(counts, dtype=float)
    if arr.size == 0:
        return 0

    if method == "percentile":
        threshold = float(np.percentile(arr, percentile))
    elif method == "tukey":
        q1 = float(np.percentile(arr, 25))
        q3 = float(np.percentile(arr, 75))
        iqr = q3 - q1
        threshold = q3 + 1.5 * iqr
    elif method == "absolute":
        threshold = float(absolute_threshold)
    else:
        raise ValueError(
            f"outlier_method debe ser 'percentile', 'tukey' o 'absolute'; "
            f"recibí {method!r}"
        )

    threshold_int = max(1, int(math.ceil(threshold)))

    if verbose:
        print(f"  [turn_hex] Umbral de outliers ({method}"
              f"{f', p={percentile}' if method == 'percentile' else ''}): "
              f"stopped_count ≥ {threshold_int} "
              f"(de un rango {int(arr.min())}–{int(arr.max())})")

    return threshold_int


# ──────────────────────────────────────────────────────────────────────
# Clustering por componentes conexas (k-ring=1)
# ──────────────────────────────────────────────────────────────────────

def _cluster_hex_components(
    h3_indices: List[str],
    counts: Dict[str, int],
) -> List[Dict[str, Any]]:
    """
    Construye clusters por componentes conexas vía vecinos k-ring=1.

    Returns
    -------
    List of dict con keys:
        - h3s          : lista de hex en el cluster
        - center_h3    : hex con mayor conteo dentro del cluster
        - center_count : conteo del centro
        - total_count  : suma de conteos del cluster completo
    """
    h3 = _h3_module()
    hex_set = set(h3_indices)
    visited: Set[str] = set()
    clusters: List[Dict[str, Any]] = []

    for start in h3_indices:
        if start in visited:
            continue

        # BFS sobre vecinos k-ring=1 limitados al conjunto de hex parados
        component: Set[str] = set()
        q = deque([start])
        while q:
            cur = q.popleft()
            if cur in visited:
                continue
            visited.add(cur)
            component.add(cur)
            for nbr in h3.grid_disk(cur, 1):
                if nbr != cur and nbr in hex_set and nbr not in visited:
                    q.append(nbr)

        comp_list = list(component)
        # Centro del cluster: el hex con mayor conteo
        comp_list.sort(key=lambda h: counts.get(h, 0), reverse=True)
        center = comp_list[0]
        clusters.append({
            "h3s":          comp_list,
            "center_h3":    center,
            "center_count": int(counts.get(center, 0)),
            "total_count":  int(sum(counts.get(h, 0) for h in comp_list)),
        })

    # Ordenar por conteo del centro (descendente)
    clusters.sort(key=lambda c: c["center_count"], reverse=True)
    return clusters


# ──────────────────────────────────────────────────────────────────────
# Algoritmo de detección automática
# ──────────────────────────────────────────────────────────────────────

def _detect_auto(
    df_rec_clean: pl.DataFrame,
    df_segments_route: pl.DataFrame,
    max_speed: float,
    resolution: int,
    min_hex_count: int,
    min_dist_fraction: float,
    outlier_method: Literal["percentile", "tukey", "absolute"],
    outlier_percentile: float,
    outlier_absolute_threshold: int,
    selection_metric: Literal["center_count", "cluster_total_count"],
    verbose: bool,
    max_pings_per_bus_per_hex: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], float, float, float, Dict[str, Any]]:
    """
    Detección automática de los dos turn_hex.

    Returns
    -------
    (turn_hexes, polyline_length_m, branch_length_m, min_separation_m, diagnostics)
        turn_hexes         : lista de 2 dicts.
        polyline_length_m  : longitud total de la polilínea (ida+vuelta).
        branch_length_m    : longitud estimada de una rama (= polyline_length / 2).
        min_separation_m   : distancia mínima exigida entre los dos turn_hex
                             (haversine absoluta), calculada sobre branch_length_m.
        diagnostics        : dict con info del filtro de outliers + clustering.
    """
    h3 = _h3_module()

    # 1) Longitud total de la polilínea
    poly_len = _polyline_length_m(df_segments_route)
    if poly_len <= 0:
        raise ValueError(
            "No pude calcular la longitud de la polilínea. Revisa "
            "df_segments_route (columna polyline/geometry/coords)."
        )
    # branch_length: la polilínea raw contiene un ciclo completo (ida+vuelta).
    # Una rama ≈ mitad del ciclo. Esto es una APROXIMACIÓN porque las dos
    # ramas no son necesariamente iguales, pero es bastante mejor que usar
    # el ciclo completo (que producía el bug del 67% rechazo por completeness).
    branch_len = poly_len / 2.0
    min_sep = min_dist_fraction * branch_len

    if verbose:
        print(f"  [turn_hex/auto] Polilínea total (ciclo): {poly_len:,.1f} m "
              f"({poly_len/1000:.2f} km)")
        print(f"  [turn_hex/auto] Branch length estimado:  {branch_len:,.1f} m "
              f"({branch_len/1000:.2f} km) — ciclo / 2")
        print(f"  [turn_hex/auto] Separación mínima:       "
              f"{min_sep:,.1f} m ({min_dist_fraction:.0%} del branch_length)")
        if max_pings_per_bus_per_hex is not None:
            print(f"  [turn_hex/auto] Density bus-aware:        "
                  f"cap = {max_pings_per_bus_per_hex} pings/bus/hex")
        else:
            print(f"  [turn_hex/auto] Density:                  "
                  f"raw count (no cap)")

    # 2) Conteos por hex (con piso absoluto suave para no llevar todo al filtro)
    #    Si el método bus-aware no produce suficientes hex válidos, hacer
    #    fallback al conteo crudo (baseline) con un warning.
    def _counts_with_fallback(cap):
        df = _h3_counts_from_stopped(
            df_rec=df_rec_clean,
            resolution=resolution,
            max_speed=max_speed,
            min_hex_count=min_hex_count,
            max_pings_per_bus_per_hex=cap,
        )
        if df.height < 2:
            raise ValueError(
                f"Solo {df.height} hex con ≥{min_hex_count} registros parados. "
                "Reduce min_hex_count o cambia la resolución."
            )
        return df

    if max_pings_per_bus_per_hex is not None:
        try:
            df_counts = _counts_with_fallback(max_pings_per_bus_per_hex)
        except ValueError as exc_cap:
            import warnings
            warnings.warn(
                f"[turn_hex] Método bus-aware (cap={max_pings_per_bus_per_hex}) "
                f"no encontró hexágonos válidos: {exc_cap}. "
                f"Haciendo fallback al conteo crudo (baseline).",
                stacklevel=3,
            )
            if verbose:
                print(f"  [turn_hex/auto] ⚠ bus-aware sin resultados, "
                      f"fallback a conteo crudo.")
            df_counts = _counts_with_fallback(None)
    else:
        df_counts = _counts_with_fallback(None)

    # 3) Filtrar a outliers (los hex que REALMENTE destacan)
    #    Si el cap dejó tan pocos hex que no hay ≥2 outliers, fallback a crudo.
    def _apply_outlier_filter(df_c):
        threshold = _compute_outlier_threshold(
            counts             = list(dict(zip(
                df_c["h3_index"].to_list(),
                df_c["stopped_count"].to_list(),
            )).values()),
            method             = outlier_method,
            percentile         = outlier_percentile,
            absolute_threshold = outlier_absolute_threshold,
            verbose            = False,  # verbose se imprime abajo si hay fallback
        )
        counts_map = dict(zip(
            df_c["h3_index"].to_list(),
            df_c["stopped_count"].to_list(),
        ))
        return {h: c for h, c in counts_map.items() if c >= threshold}, threshold

    all_counts: Dict[str, int] = dict(zip(
        df_counts["h3_index"].to_list(),
        df_counts["stopped_count"].to_list(),
    ))
    if verbose:
        print(f"  [turn_hex/auto] Hex con ≥{min_hex_count} parados: "
              f"{len(all_counts):,}")

    # Recomputar threshold con verbose
    threshold = _compute_outlier_threshold(
        counts             = list(all_counts.values()),
        method             = outlier_method,
        percentile         = outlier_percentile,
        absolute_threshold = outlier_absolute_threshold,
        verbose            = verbose,
    )
    outlier_hex = {h: c for h, c in all_counts.items() if c >= threshold}

    if len(outlier_hex) < 2 and max_pings_per_bus_per_hex is not None:
        import warnings
        warnings.warn(
            f"[turn_hex] Método bus-aware (cap={max_pings_per_bus_per_hex}) "
            f"produjo solo {len(outlier_hex)} hex outliers (threshold={threshold}). "
            f"Haciendo fallback al conteo crudo (baseline).",
            stacklevel=3,
        )
        if verbose:
            print(f"  [turn_hex/auto] ⚠ bus-aware: {len(outlier_hex)} outliers "
                  f"insuficientes, fallback a conteo crudo.")
        df_counts_fallback = _counts_with_fallback(None)
        all_counts = dict(zip(
            df_counts_fallback["h3_index"].to_list(),
            df_counts_fallback["stopped_count"].to_list(),
        ))
        threshold = _compute_outlier_threshold(
            counts             = list(all_counts.values()),
            method             = outlier_method,
            percentile         = outlier_percentile,
            absolute_threshold = outlier_absolute_threshold,
            verbose            = verbose,
        )
        outlier_hex = {h: c for h, c in all_counts.items() if c >= threshold}

    if len(outlier_hex) < 2:
        raise ValueError(
            f"Solo {len(outlier_hex)} hex pasan el filtro de outliers "
            f"(threshold={threshold}). Relaja `outlier_percentile` o cambia "
            f"`outlier_method`."
        )

    if verbose:
        print(f"  [turn_hex/auto] Hex outliers tras filtro: "
              f"{len(outlier_hex):,} (de {len(all_counts):,} candidatos)")

    # 4) Clustering: SOLO sobre outliers
    clusters = _cluster_hex_components(list(outlier_hex.keys()), outlier_hex)
    if verbose:
        sizes = [len(c["h3s"]) for c in clusters]
        print(f"  [turn_hex/auto] Clusters de outliers contiguos: {len(clusters)} "
              f"(tamaños: max={max(sizes)}, "
              f"min={min(sizes)}, median={sorted(sizes)[len(sizes)//2]})")

    # 5) Re-ordenar clusters según la métrica de selección
    if selection_metric == "cluster_total_count":
        clusters.sort(key=lambda c: c["total_count"], reverse=True)
    else:  # "center_count" — orden ya aplicado en _cluster_hex_components
        clusters.sort(key=lambda c: c["center_count"], reverse=True)

    # 6) turn_hex_A := el mejor cluster según la métrica
    cluster_a = clusters[0]
    lat_a, lon_a = h3.cell_to_latlng(cluster_a["center_h3"])

    if verbose:
        print(f"  [turn_hex/auto] turn_hex_A (selección por {selection_metric}):")
        print(f"      center_h3      = {cluster_a['center_h3']}")
        print(f"      center_count   = {cluster_a['center_count']:,}")
        print(f"      cluster_total  = {cluster_a['total_count']:,}")
        print(f"      cluster_size   = {len(cluster_a['h3s'])} hex")

    # 7) Filtrar candidatos B lejanos en haversine absoluto
    def _metric_of(c: Dict[str, Any]) -> int:
        return c["total_count"] if selection_metric == "cluster_total_count" \
               else c["center_count"]

    def _find_candidate_b(clusters_list, lat_ref, lon_ref):
        """Devuelve (cluster_b, dist_b) o lanza ValueError."""
        candidates: List[Tuple[Dict, float]] = []
        for c in clusters_list[1:]:
            lat_c, lon_c = h3.cell_to_latlng(c["center_h3"])
            d = _haversine_m(lat_ref, lon_ref, lat_c, lon_c)
            if d >= min_sep:
                candidates.append((c, d))
        if not candidates:
            nearest = clusters_list[1] if len(clusters_list) >= 2 else None
            nearest_d = None
            if nearest is not None:
                ln, lo = h3.cell_to_latlng(nearest["center_h3"])
                nearest_d = _haversine_m(lat_ref, lon_ref, ln, lo)
            raise ValueError(
                f"No hay cluster a ≥ {min_sep:,.0f} m del turn_hex_A. "
                f"Mejor candidato cercano: "
                f"{nearest_d:,.0f} m (metric={_metric_of(nearest) if nearest else 'NA'}). "
                "Baja `min_dist_fraction` o revisa la geografía de la ruta."
            )
        return max(candidates, key=lambda x: _metric_of(x[0]))

    # Intentar con los clusters actuales; si falla y se usó cap, reintentar
    # con conteo crudo (fallback completo al baseline).
    try:
        cluster_b, dist_b = _find_candidate_b(clusters, lat_a, lon_a)
    except ValueError as exc_b:
        if max_pings_per_bus_per_hex is None:
            raise  # ya era crudo — no hay más fallback posible
        import warnings
        warnings.warn(
            f"[turn_hex] Método bus-aware (cap={max_pings_per_bus_per_hex}) "
            f"no pudo separar dos terminales: {exc_b}. "
            f"Haciendo fallback completo al conteo crudo (baseline).",
            stacklevel=3,
        )
        if verbose:
            print(f"  [turn_hex/auto] ⚠ bus-aware: {exc_b}")
            print(f"  [turn_hex/auto] ⚠ Fallback completo a conteo crudo.")

        # Reconstruir desde cero con cap=None
        df_counts_fb = _counts_with_fallback(None)
        all_counts_fb: Dict[str, int] = dict(zip(
            df_counts_fb["h3_index"].to_list(),
            df_counts_fb["stopped_count"].to_list(),
        ))
        threshold_fb = _compute_outlier_threshold(
            counts             = list(all_counts_fb.values()),
            method             = outlier_method,
            percentile         = outlier_percentile,
            absolute_threshold = outlier_absolute_threshold,
            verbose            = False,
        )
        outlier_hex_fb = {h: c for h, c in all_counts_fb.items()
                          if c >= threshold_fb}
        if len(outlier_hex_fb) < 2:
            raise ValueError(
                f"Fallback a conteo crudo también falló: solo "
                f"{len(outlier_hex_fb)} hex outliers. "
                f"Revisa la geografía de la ruta o baja min_dist_fraction."
            )
        clusters_fb = _cluster_hex_components(
            list(outlier_hex_fb.keys()), outlier_hex_fb
        )
        if selection_metric == "cluster_total_count":
            clusters_fb.sort(key=lambda c: c["total_count"], reverse=True)
        else:
            clusters_fb.sort(key=lambda c: c["center_count"], reverse=True)

        cluster_a  = clusters_fb[0]
        lat_a, lon_a = h3.cell_to_latlng(cluster_a["center_h3"])
        # Usar _find_candidate_b con cap=None (fallback final — no hay más reintentos)
        cluster_b, dist_b = _find_candidate_b(clusters_fb, lat_a, lon_a)
        # Actualizar all_counts para que el resto del flujo use los del fallback
        all_counts = all_counts_fb
    lat_b, lon_b = h3.cell_to_latlng(cluster_b["center_h3"])

    if verbose:
        print(f"  [turn_hex/auto] turn_hex_B:")
        print(f"      center_h3      = {cluster_b['center_h3']}")
        print(f"      center_count   = {cluster_b['center_count']:,}")
        print(f"      cluster_total  = {cluster_b['total_count']:,}")
        print(f"      cluster_size   = {len(cluster_b['h3s'])} hex")
        print(f"      dist_a_to_b    = {dist_b:,.1f} m "
              f"({100*dist_b/branch_len:.1f}% del branch_length)")

    turn_hexes = [
        {
            "label":               "turn_hex_A",
            "h3":                  cluster_a["center_h3"],
            "lat":                 float(lat_a),
            "lon":                 float(lon_a),
            "stopped_count":       cluster_a["center_count"],
            "cluster_h3s":         list(cluster_a["h3s"]),
            "cluster_total_count": cluster_a["total_count"],
        },
        {
            "label":               "turn_hex_B",
            "h3":                  cluster_b["center_h3"],
            "lat":                 float(lat_b),
            "lon":                 float(lon_b),
            "stopped_count":       cluster_b["center_count"],
            "cluster_h3s":         list(cluster_b["h3s"]),
            "cluster_total_count": cluster_b["total_count"],
        },
    ]

    diagnostics = {
        "n_hex_total":             len(all_counts),
        "outlier_method":          outlier_method,
        "outlier_threshold":       threshold,
        "outlier_percentile":      outlier_percentile if outlier_method == "percentile" else None,
        "n_hex_outliers":          len(outlier_hex),
        "n_clusters":              len(clusters),
        "selection_metric":        selection_metric,
    }

    return turn_hexes, poly_len, branch_len, min_sep, diagnostics


# ──────────────────────────────────────────────────────────────────────
# Modo manual (lista de 2 códigos H3)
# ──────────────────────────────────────────────────────────────────────

def _detect_manual(
    manual_h3: List[str],
    df_rec_clean: Optional[pl.DataFrame],
    df_segments_route: Optional[pl.DataFrame],
    max_speed: float,
    resolution: int,
    min_hex_count: int,
    outlier_method: Literal["percentile", "tukey", "absolute"],
    outlier_percentile: float,
    outlier_absolute_threshold: int,
    verbose: bool,
    max_pings_per_bus_per_hex: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], float, float]:
    """
    Modo manual: el usuario provee 2 códigos H3 directamente.
    Si hay df_rec_clean, computa el conteo del centro y el cluster contiguo
    de hex outliers para diagnóstico (mismo criterio que el modo auto).

    Returns
    -------
    (turn_hexes, polyline_length_m, branch_length_m)
    """
    h3 = _h3_module()

    if len(manual_h3) != 2:
        raise ValueError(
            f"Modo manual requiere exactamente 2 códigos H3, recibí {len(manual_h3)}."
        )

    for code in manual_h3:
        if not isinstance(code, str) or not h3.is_valid_cell(code):
            raise ValueError(f"Código H3 inválido: {code!r}")

    res_actual = h3.get_resolution(manual_h3[0])
    if res_actual != resolution and verbose:
        print(f"  [turn_hex/manual] ⚠ resolución del input H3 ({res_actual}) "
              f"≠ resolución configurada ({resolution}). Usando {res_actual}.")
        resolution = res_actual

    # Conteo + outlier filter (opcional si hay registros)
    all_counts: Dict[str, int] = {}
    outlier_hex: Set[str] = set()
    if df_rec_clean is not None:
        df_counts = _h3_counts_from_stopped(
            df_rec=df_rec_clean,
            resolution=resolution,
            max_speed=max_speed,
            min_hex_count=min_hex_count,
            max_pings_per_bus_per_hex=max_pings_per_bus_per_hex,
        )
        all_counts = dict(zip(
            df_counts["h3_index"].to_list(),
            df_counts["stopped_count"].to_list(),
        ))
        if all_counts:
            threshold = _compute_outlier_threshold(
                counts             = list(all_counts.values()),
                method             = outlier_method,
                percentile         = outlier_percentile,
                absolute_threshold = outlier_absolute_threshold,
                verbose            = verbose,
            )
            outlier_hex = {h for h, c in all_counts.items() if c >= threshold}

    poly_len = (_polyline_length_m(df_segments_route)
                if df_segments_route is not None else 0.0)
    branch_len = poly_len / 2.0 if poly_len > 0 else 0.0

    turn_hexes: List[Dict[str, Any]] = []
    for i, code in enumerate(manual_h3):
        lat, lon = h3.cell_to_latlng(code)
        # Cluster contiguo via BFS sobre outliers
        cluster_h3s: List[str] = [code]
        cluster_total = all_counts.get(code, 0)
        if outlier_hex:
            visited = {code}
            q = deque([code])
            while q:
                cur = q.popleft()
                for nbr in h3.grid_disk(cur, 1):
                    if nbr == cur or nbr in visited:
                        continue
                    if nbr in outlier_hex:
                        visited.add(nbr)
                        q.append(nbr)
                        cluster_h3s.append(nbr)
                        cluster_total += all_counts.get(nbr, 0)

        turn_hexes.append({
            "label":               f"turn_hex_{chr(ord('A') + i)}",
            "h3":                  code,
            "lat":                 float(lat),
            "lon":                 float(lon),
            "stopped_count":       int(all_counts.get(code, 0)),
            "cluster_h3s":         cluster_h3s,
            "cluster_total_count": int(cluster_total),
        })

    if verbose:
        for th in turn_hexes:
            print(f"  [turn_hex/manual] {th['label']}: {th['h3']}  "
                  f"count={th['stopped_count']:,}  "
                  f"cluster={len(th['cluster_h3s'])} hex  "
                  f"cluster_total={th['cluster_total_count']:,}")
        if branch_len > 0:
            d = _haversine_m(
                turn_hexes[0]["lat"], turn_hexes[0]["lon"],
                turn_hexes[1]["lat"], turn_hexes[1]["lon"],
            )
            print(f"  [turn_hex/manual] dist A↔B = {d:,.1f} m "
                  f"({100 * d / branch_len:.1f}% del branch_length de "
                  f"{branch_len:,.0f} m)")

    return turn_hexes, poly_len, branch_len


# ──────────────────────────────────────────────────────────────────────
# Cálculo de turn_zone vía k-ring filtrado por polilínea
# ──────────────────────────────────────────────────────────────────────

def compute_turn_zone_kring(
    turn_hex_h3: str,
    kring: int,
    polyline_hex_set: Optional[Set[str]] = None,
    df_segments_route: Optional[pl.DataFrame] = None,
    resolution: Optional[int] = None,
) -> List[str]:
    """
    Devuelve la turn_zone: hex en k-ring=N del turn_hex que ADEMÁS contienen
    polilínea de la ruta.

    Si ``polyline_hex_set`` se provee, se usa directamente (más rápido). En su
    defecto, se calcula a partir de ``df_segments_route`` y ``resolution``.

    Parameters
    ----------
    turn_hex_h3 : str
        Código H3 del centro de la zona.
    kring : int
        Número de anillos vecinos a explorar (0 = solo el centro, 1 = +6 hex
        adyacentes, 2 = +18 hex, etc).
    polyline_hex_set : set[str], optional
        Conjunto pre-calculado de hex H3 que la polilínea atraviesa.
    df_segments_route : pl.DataFrame, optional
        Polilínea segmentada (alternativa a polyline_hex_set).
    resolution : int, optional
        Resolución H3 a usar si se calcula desde df_segments_route.

    Returns
    -------
    List[str]
        Hex de la turn_zone, incluyendo siempre al menos `turn_hex_h3`.
    """
    h3 = _h3_module()

    if polyline_hex_set is None:
        if df_segments_route is None or resolution is None:
            # Sin filtro de polilínea: devuelve k-ring puro.
            return list(h3.grid_disk(turn_hex_h3, kring))
        polyline_hex_set = _polyline_hex_set(df_segments_route, resolution)

    candidate_ring = set(h3.grid_disk(turn_hex_h3, kring))
    # Filtrar: solo hex que tocan la polilínea (o el centro mismo, que siempre
    # debe quedar incluso si la polilínea no lo cubre por muestreo escaso).
    filtered = {h for h in candidate_ring if h in polyline_hex_set}
    filtered.add(turn_hex_h3)  # garantía
    return list(filtered)


# ──────────────────────────────────────────────────────────────────────
# Empaquetado de resultado (raw + legacy aliases)
# ──────────────────────────────────────────────────────────────────────

def _result_from_raw(raw: Dict[str, Any]) -> TurnHexResult:
    """
    Construye el dict de retorno a partir del dict cacheable.
    Incluye claves legacy ``df_end_stop`` y ``zones`` para compatibilidad.

    Las ``zones`` se construyen usando ``compute_turn_zone_kring`` con el
    parámetro ``_turn_zone_kring`` inyectado en el raw (no se serializa a
    JSON — se pasa en memoria desde ``resolve_turn_hex``). Esto garantiza
    que las zonas de corte de make_dir_trips usan el k-ring H3 real, no
    simplemente los centroides del cluster de detección.
    """
    h3 = _h3_module()
    resolution      = int(raw.get("resolution", 12))
    kring           = int(raw.get("_turn_zone_kring", 1))
    df_segs         = raw.get("_df_segments_route")   # puede ser None

    # Pre-calcular el set de hex de la polilínea (una sola vez para todas las zonas)
    poly_hex_set: Optional[Set[str]] = None
    if df_segs is not None:
        try:
            poly_hex_set = _polyline_hex_set(df_segs, resolution)
        except Exception:
            poly_hex_set = None

    rows_new    = []
    rows_legacy = []
    zones: Dict[str, List[Tuple[float, float]]] = {}

    for th in raw["turn_hexes"]:
        rows_new.append({
            "label":         th["label"],
            "h3":            th["h3"],
            "lat":           float(th["lat"]),
            "lon":           float(th["lon"]),
            "stopped_count": int(th.get("stopped_count", 0)),
        })
        rows_legacy.append({
            "code":      th["label"],
            "latitude":  float(th["lat"]),
            "longitude": float(th["lon"]),
        })

        # Construir la zona con k-ring H3 filtrado por polilínea
        try:
            zone_hex_codes = compute_turn_zone_kring(
                turn_hex_h3      = th["h3"],
                kring            = kring,
                polyline_hex_set = poly_hex_set,
            )
        except Exception:
            # Fallback: solo el hex centro si H3 no está disponible
            zone_hex_codes = [th["h3"]]

        # Convertir hex → coordenadas de centroide
        zone_coords: List[Tuple[float, float]] = []
        for hx in zone_hex_codes:
            try:
                la, lo = h3.cell_to_latlng(hx)
                zone_coords.append((float(la), float(lo)))
            except Exception:
                continue
        zones[th["label"]] = zone_coords

    df_turn_hex = pl.DataFrame(rows_new)
    df_end_stop = pl.DataFrame(rows_legacy)

    # Limpiar claves privadas antes de exponer raw al caller
    raw_public = {k: v for k, v in raw.items()
                  if not k.startswith("_")}

    return {
        "turn_hexes":  raw["turn_hexes"],
        "df_turn_hex": df_turn_hex,
        "df_end_stop": df_end_stop,
        "zones":       zones,
        "mode":        raw["mode"],
        "raw":         raw_public,
    }


# ──────────────────────────────────────────────────────────────────────
# Función pública
# ──────────────────────────────────────────────────────────────────────

def resolve_turn_hex(
    df_rec_clean: Optional[pl.DataFrame] = None,
    df_segments_route: Optional[pl.DataFrame] = None,
    *,
    mode: Literal["auto", "manual"] = "auto",
    manual_h3: Optional[List[str]] = None,
    cache_path: Optional[Path] = None,
    force: bool = False,
    # Parámetros modo auto
    resolution: int = 12,
    min_hex_count: int = 1,
    min_dist_fraction: float = 1.0 / 4.0,
    max_speed: float = 0.0,
    # Filtro de outliers
    outlier_method: Literal["percentile", "tukey", "absolute"] = "percentile",
    outlier_percentile: float = 95.0,
    outlier_absolute_threshold: int = 50,
    # Métrica de selección
    selection_metric: Literal["center_count", "cluster_total_count"] = "center_count",
    # Turn zone: k-ring H3 para construir la zona de corte de make_dir_trips
    turn_zone_kring: int = 1,
    # Density bus-aware (cap por (bus, hex))
    max_pings_per_bus_per_hex: Optional[int] = None,
    verbose: bool = False,
) -> TurnHexResult:
    """
    Resuelve los dos turn_hex de la ruta.

    Parameters
    ----------
    df_rec_clean : pl.DataFrame, optional
        Registros limpios con columna de velocidad. Requerido para auto.
    df_segments_route : pl.DataFrame, optional
        Polilínea segmentada. Requerido para auto (longitud y branch_length).
    mode : {"auto", "manual"}
        Default ``"auto"``.
    manual_h3 : list[str] | None
        Dos códigos H3 si ``mode="manual"``.
    cache_path : Path | None
        Si existe y ``force=False``, carga del cache.
    force : bool
        Forzar recálculo.
    resolution : int
        Resolución H3. Default 12 (~9 m de diámetro).
    min_hex_count : int
        Piso absoluto mínimo (default 1). El filtro real de outliers es
        ``outlier_method``. Ponlo en >1 si necesitas un piso duro.
    min_dist_fraction : float
        Fracción del ``branch_length`` (= polyline_total / 2) que debe haber
        en haversine absoluto entre los dos turn_hex. Default 1/4.
    max_speed : float
        Umbral de "parado" (default 0).
    outlier_method : {"percentile", "tukey", "absolute"}
        Cómo decidir qué hex son outliers (terminales candidatas):
          - "percentile": stopped_count ≥ P_q (default q=95).
          - "tukey": stopped_count ≥ Q3 + 1.5·IQR.
          - "absolute": stopped_count ≥ outlier_absolute_threshold.
    outlier_percentile : float
        Percentil para outlier_method="percentile". Default 95.0.
    outlier_absolute_threshold : int
        Umbral fijo para outlier_method="absolute". Default 50.
    selection_metric : {"center_count", "cluster_total_count"}
        Qué métrica usar para elegir el "mejor" cluster:
          - "center_count": densidad del hex pico individual (default).
          - "cluster_total_count": suma de conteos de todo el cluster.
    turn_zone_kring : int
        Número de anillos H3 vecinos del turn_hex que forman la zona de corte
        de viajes en ``make_dir_trips``. Los hex del k-ring se filtran por
        polilínea (solo los que la ruta atraviesa). 0 = solo el hex centro.
        Default 1 (corresponde a ``config.TURN_ZONE_KRING``).
        Este parámetro se guarda en el cache JSON y se propaga al resultado
        ``zones`` usado por ``make_dir_trips``.
    verbose : bool

    Returns
    -------
    TurnHexResult

    Raises
    ------
    ValueError
        Si los argumentos no son consistentes o no hay datos suficientes.
    """
    # ── 1. Cache ──
    if cache_path is not None and not force and Path(cache_path).is_file():
        with open(cache_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        # Validar que sea formato turn_hex (no end_stops legacy)
        if "turn_hexes" not in raw:
            if verbose:
                print(f"  [turn_hex] ⚠ cache en formato legacy end_stops, "
                      f"ignorando: {cache_path}")
        else:
            # Inyectar turn_zone_kring y df_segments para que _result_from_raw
            # pueda construir las zonas H3 correctas aunque el cache sea viejo.
            raw["_turn_zone_kring"]     = turn_zone_kring
            raw["_df_segments_route"]   = df_segments_route
            if verbose:
                print(f"  [turn_hex] ✔ cache cargado: {cache_path} "
                      f"(mode={raw.get('mode')}, "
                      f"n={len(raw['turn_hexes'])}, "
                      f"kring={turn_zone_kring})")
            return _result_from_raw(raw)

    # ── 2. Detectar ──
    if mode == "auto":
        if df_rec_clean is None or df_segments_route is None:
            raise ValueError(
                "mode='auto' requiere df_rec_clean y df_segments_route."
            )
        if verbose:
            print(f"  [turn_hex] Detección automática "
                  f"(H3 res={resolution}, outlier={outlier_method}, "
                  f"selection={selection_metric}, "
                  f"min_dist={min_dist_fraction:.0%} de branch_length)")
        turn_hexes, poly_len, branch_len, min_sep, diag = _detect_auto(
            df_rec_clean               = df_rec_clean,
            df_segments_route          = df_segments_route,
            max_speed                  = max_speed,
            resolution                 = resolution,
            min_hex_count              = min_hex_count,
            min_dist_fraction          = min_dist_fraction,
            outlier_method             = outlier_method,
            outlier_percentile         = outlier_percentile,
            outlier_absolute_threshold = outlier_absolute_threshold,
            selection_metric           = selection_metric,
            max_pings_per_bus_per_hex  = max_pings_per_bus_per_hex,
            verbose                    = verbose,
        )
        raw = {
            "mode":              "auto",
            "resolution":        resolution,
            "min_hex_count":     min_hex_count,
            "max_speed":         max_speed,
            "min_dist_fraction": min_dist_fraction,
            "outlier_method":    outlier_method,
            "outlier_percentile": outlier_percentile if outlier_method == "percentile" else None,
            "outlier_absolute_threshold": outlier_absolute_threshold if outlier_method == "absolute" else None,
            "selection_metric":  selection_metric,
            "max_pings_per_bus_per_hex": max_pings_per_bus_per_hex,
            "polyline_length_m": poly_len,
            "branch_length_m":   branch_len,
            "min_separation_m":  min_sep,
            "diagnostics":       diag,
            "turn_hexes":        turn_hexes,
            # No se serializa en JSON (se inyecta en memoria para _result_from_raw)
            "_turn_zone_kring":   turn_zone_kring,
            "_df_segments_route": df_segments_route,
        }

    elif mode == "manual":
        if manual_h3 is None or len(manual_h3) != 2:
            raise ValueError(
                "mode='manual' requiere `manual_h3` con exactamente 2 códigos H3."
            )
        if verbose:
            print(f"  [turn_hex] Modo manual: {manual_h3}")
        turn_hexes, poly_len, branch_len = _detect_manual(
            manual_h3                  = manual_h3,
            df_rec_clean               = df_rec_clean,
            df_segments_route          = df_segments_route,
            max_speed                  = max_speed,
            resolution                 = resolution,
            min_hex_count              = min_hex_count,
            outlier_method             = outlier_method,
            outlier_percentile         = outlier_percentile,
            outlier_absolute_threshold = outlier_absolute_threshold,
            max_pings_per_bus_per_hex  = max_pings_per_bus_per_hex,
            verbose                    = verbose,
        )
        raw = {
            "mode":              "manual",
            "resolution":        resolution,
            "min_hex_count":     min_hex_count,
            "max_speed":         max_speed,
            "outlier_method":    outlier_method,
            "outlier_percentile": outlier_percentile if outlier_method == "percentile" else None,
            "outlier_absolute_threshold": outlier_absolute_threshold if outlier_method == "absolute" else None,
            "max_pings_per_bus_per_hex": max_pings_per_bus_per_hex,
            "polyline_length_m": poly_len,
            "branch_length_m":   branch_len,
            "manual_h3":         manual_h3,
            "turn_hexes":        turn_hexes,
            "_turn_zone_kring":   turn_zone_kring,
            "_df_segments_route": df_segments_route,
        }

    else:
        raise ValueError(f"mode debe ser 'auto' o 'manual', recibí: {mode!r}")

    # ── 3. Cache ──
    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        # Excluir claves privadas (prefijo _) que no son serializables a JSON
        raw_serializable = {k: v for k, v in raw.items()
                            if not k.startswith("_")}
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(raw_serializable, f, indent=2, ensure_ascii=False)
        if verbose:
            print(f"  [turn_hex] ✔ cache guardado: {cache_path}")

    return _result_from_raw(raw)