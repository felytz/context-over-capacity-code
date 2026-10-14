"""Densidad H3 por velocidad + detección de end-stops (plot_speed_density).

Submódulo de `_lib_plot` (antes era una sección del módulo monolítico).
Importar desde el paquete: `from _lib_plot import ...`.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl
import folium
from folium import LayerControl
import h3

from _lib_geo import _haversine_m, _parse_polyline_struct

from .overview import _parse_polyline

##############################################################################
#   2. plot_speed_density — densidad H3 por velocidad
#   (originalmente: _lib_plot_speed.py)
##############################################################################

import polars as pl
import numpy as np
import h3
import folium
from typing import Optional, List, Tuple, Dict


#──────────────────────────────────────────────────────────────────────────────#
# ── 1. Funciones de gradiente de colores ── #
#──────────────────────────────────────────────────────────────────────────────#

# Viridis colormap (sampled at 256 points, stored as RGB tuples)
# Rango: azul oscuro (0) → verde azulado → verde → amarillo (1)
_VIRIDIS = [
    (68, 1, 84), (68, 2, 86), (69, 4, 87), (69, 5, 89), (70, 7, 90),
    (70, 8, 92), (70, 10, 93), (70, 11, 94), (71, 13, 96), (71, 14, 97),
    (71, 16, 99), (71, 17, 100), (71, 19, 101), (72, 20, 103), (72, 22, 104),
    (72, 23, 105), (72, 24, 106), (72, 26, 108), (72, 27, 109), (72, 28, 110),
    (72, 29, 111), (72, 31, 112), (72, 32, 113), (72, 33, 115), (72, 35, 116),
    (72, 36, 117), (72, 37, 118), (72, 38, 119), (72, 40, 120), (72, 41, 121),
    (71, 42, 122), (71, 44, 122), (71, 45, 123), (71, 46, 124), (71, 47, 125),
    (70, 48, 126), (70, 50, 126), (70, 51, 127), (69, 52, 128), (69, 53, 129),
    (69, 55, 129), (68, 56, 130), (68, 57, 131), (68, 58, 131), (67, 60, 132),
    (67, 61, 132), (66, 62, 133), (66, 63, 133), (66, 64, 134), (65, 66, 134),
    (65, 67, 135), (64, 68, 135), (64, 69, 136), (63, 71, 136), (63, 72, 137),
    (62, 73, 137), (62, 74, 137), (62, 75, 138), (61, 76, 138), (61, 78, 138),
    (60, 79, 139), (60, 80, 139), (59, 81, 139), (59, 82, 139), (58, 83, 140),
    (58, 84, 140), (57, 85, 140), (57, 86, 140), (56, 88, 140), (56, 89, 141),
    (55, 90, 141), (55, 91, 141), (54, 92, 141), (54, 93, 141), (53, 94, 141),
    (53, 95, 141), (52, 96, 141), (52, 97, 141), (51, 98, 141), (51, 99, 141),
    (50, 100, 142), (50, 101, 142), (49, 102, 142), (49, 103, 142), (49, 104, 142),
    (48, 105, 142), (48, 106, 142), (47, 107, 142), (47, 108, 142), (46, 109, 142),
    (46, 110, 142), (46, 111, 142), (45, 112, 142), (45, 113, 142), (44, 113, 142),
    (44, 114, 142), (44, 115, 142), (43, 116, 142), (43, 117, 142), (42, 118, 142),
    (42, 119, 142), (42, 120, 142), (41, 121, 142), (41, 122, 142), (40, 122, 142),
    (40, 123, 142), (40, 124, 142), (39, 125, 142), (39, 126, 142), (38, 127, 142),
    (38, 128, 142), (38, 129, 142), (37, 130, 142), (37, 131, 141), (36, 132, 141),
    (36, 132, 141), (35, 133, 141), (35, 134, 141), (35, 135, 141), (34, 136, 141),
    (34, 137, 141), (33, 138, 141), (33, 139, 140), (32, 139, 140), (32, 140, 140),
    (31, 141, 140), (31, 142, 140), (31, 143, 139), (30, 144, 139), (30, 145, 139),
    (29, 146, 139), (29, 146, 138), (28, 147, 138), (28, 148, 138), (27, 149, 137),
    (27, 150, 137), (26, 151, 137), (26, 152, 136), (25, 152, 136), (25, 153, 135),
    (24, 154, 135), (24, 155, 134), (24, 156, 134), (23, 157, 133), (23, 158, 133),
    (22, 158, 132), (22, 159, 132), (22, 160, 131), (21, 161, 131), (21, 162, 130),
    (20, 163, 129), (20, 163, 129), (20, 164, 128), (19, 165, 127), (19, 166, 127),
    (19, 167, 126), (18, 168, 125), (18, 168, 125), (18, 169, 124), (18, 170, 123),
    (17, 171, 122), (17, 172, 122), (17, 173, 121), (17, 173, 120), (17, 174, 119),
    (16, 175, 118), (16, 176, 117), (16, 177, 117), (16, 177, 116), (16, 178, 115),
    (16, 179, 114), (16, 180, 113), (16, 181, 112), (16, 181, 111), (16, 182, 110),
    (17, 183, 109), (17, 184, 108), (17, 184, 107), (17, 185, 106), (18, 186, 105),
    (18, 187, 104), (18, 188, 103), (19, 188, 102), (19, 189, 101), (20, 190, 100),
    (20, 191, 99), (21, 191, 98), (22, 192, 96), (22, 193, 95), (23, 194, 94),
    (24, 194, 93), (25, 195, 92), (26, 196, 90), (27, 196, 89), (28, 197, 88),
    (29, 198, 87), (30, 198, 85), (31, 199, 84), (32, 200, 83), (34, 200, 81),
    (35, 201, 80), (36, 202, 79), (38, 202, 77), (39, 203, 76), (41, 203, 74),
    (42, 204, 73), (44, 205, 71), (46, 205, 70), (47, 206, 68), (49, 206, 67),
    (51, 207, 65), (53, 207, 64), (55, 208, 62), (57, 208, 60), (59, 209, 59),
    (61, 209, 57), (63, 210, 55), (65, 210, 54), (67, 211, 52), (70, 211, 50),
    (72, 212, 49), (74, 212, 47), (77, 213, 45), (79, 213, 43), (82, 213, 42),
    (84, 214, 40), (87, 214, 38), (89, 215, 36), (92, 215, 35), (94, 215, 33),
    (97, 216, 31), (100, 216, 29), (102, 216, 28), (105, 217, 26), (108, 217, 24),
    (111, 217, 23), (113, 218, 21), (116, 218, 20), (119, 218, 18), (122, 218, 17),
    (125, 219, 15), (128, 219, 14), (130, 219, 13), (133, 219, 12), (136, 219, 11),
    (139, 220, 10), (142, 220, 10), (145, 220, 9), (148, 220, 9), (151, 220, 9),
    (153, 220, 9), (156, 221, 10), (159, 221, 10), (162, 221, 11), (165, 221, 12),
    (168, 221, 13), (170, 221, 14), (173, 221, 16), (176, 221, 17), (179, 221, 19),
    (181, 222, 20), (184, 222, 22), (187, 222, 24), (189, 222, 26), (192, 222, 28),
    (195, 222, 30), (197, 222, 32), (200, 222, 34), (202, 222, 36), (205, 222, 38),
    (207, 222, 40), (210, 222, 43), (212, 222, 45), (215, 222, 47), (217, 222, 50),
    (219, 222, 52), (222, 222, 55), (224, 222, 57), (226, 222, 60), (228, 222, 62),
    (230, 222, 65), (232, 222, 68), (234, 222, 70), (236, 222, 73), (238, 222, 76),
    (240, 223, 79), (242, 223, 81), (243, 223, 84), (245, 223, 87), (246, 224, 90),
    (248, 224, 93), (249, 224, 96), (250, 225, 99), (251, 225, 102), (252, 226, 105),
    (253, 226, 108), (254, 227, 111), (254, 227, 114), (255, 228, 117), (255, 229, 120),
    (255, 229, 124), (255, 230, 127), (255, 231, 130), (255, 231, 133), (255, 232, 137),
    (255, 233, 140), (255, 234, 143), (255, 234, 147), (255, 235, 150), (255, 236, 154),
]

# Gradiente para outliers: continúa desde amarillo hacia magenta/rosa intenso
# Rango: amarillo viridis (0) → naranja → rojo → magenta (1)
_OUTLIER_GRADIENT = [
    (253, 231, 37), (254, 226, 37), (254, 220, 38), (254, 214, 39), (254, 208, 41),
    (254, 202, 43), (254, 195, 46), (254, 189, 49), (253, 182, 52), (253, 176, 56),
    (252, 169, 60), (251, 163, 64), (250, 156, 68), (249, 150, 73), (248, 143, 77),
    (246, 137, 82), (245, 130, 87), (243, 124, 91), (241, 118, 96), (239, 112, 100),
    (236, 106, 105), (234, 100, 109), (231, 94, 113), (228, 89, 117), (225, 83, 121),
    (222, 78, 125), (218, 73, 128), (215, 68, 131), (211, 64, 134), (207, 60, 137),
    (203, 56, 139), (199, 52, 141), (194, 49, 143), (190, 46, 145), (185, 44, 146),
    (181, 42, 147), (176, 40, 148), (171, 39, 149), (166, 38, 149), (161, 38, 149),
    (156, 38, 149), (151, 39, 149), (146, 39, 149), (141, 40, 148), (136, 41, 147),
    (131, 43, 146), (126, 44, 145), (122, 46, 144), (117, 47, 142), (113, 49, 140),
]


def _interpolate_color(value: float, gradient: list) -> str:
    """
    Interpola un color en un gradiente dado un valor normalizado [0, 1].

    Parameters
    ----------
    value : float
        Valor normalizado entre 0 y 1.
    gradient : list
        Lista de tuplas RGB.

    Returns
    -------
    str
        Color hexadecimal.
    """
    value = max(0.0, min(1.0, value))
    n = len(gradient) - 1
    idx = value * n
    lower_idx = int(idx)
    upper_idx = min(lower_idx + 1, n)
    frac = idx - lower_idx

    r1, g1, b1 = gradient[lower_idx]
    r2, g2, b2 = gradient[upper_idx]

    r = int(r1 + frac * (r2 - r1))
    g = int(g1 + frac * (g2 - g1))
    b = int(b1 + frac * (b2 - b1))

    return f"#{r:02x}{g:02x}{b:02x}"


def _get_color_for_value(
    value: float,
    min_val: float,
    max_val: float,
    outlier_threshold: float,
    outlier_max: float,
) -> Tuple[str, bool]:
    """
    Obtiene el color para un valor usando gradiente viridis o outlier.

    Parameters
    ----------
    value : float
        Valor a colorear.
    min_val : float
        Valor mínimo del rango normal (Q1-Q4).
    max_val : float
        Valor máximo del rango normal (outlier_threshold).
    outlier_threshold : float
        Umbral de outliers (Q3 + 1.5*IQR).
    outlier_max : float
        Valor máximo observado (para normalizar outliers).

    Returns
    -------
    Tuple[str, bool]
        (color_hex, is_outlier)
    """
    if value > outlier_threshold:
        # Outlier: usar gradiente outlier
        if outlier_max > outlier_threshold:
            norm = (value - outlier_threshold) / (outlier_max - outlier_threshold)
        else:
            norm = 1.0
        return _interpolate_color(norm, _OUTLIER_GRADIENT), True
    else:
        # Normal: usar viridis
        if max_val > min_val:
            norm = (value - min_val) / (max_val - min_val)
        else:
            norm = 0.5
        return _interpolate_color(norm, _VIRIDIS), False


#──────────────────────────────────────────────────────────────────────────────#
# ── 2. Funciones auxiliares ── #
#──────────────────────────────────────────────────────────────────────────────#

def _lat_lon_to_h3(lat: float, lon: float, resolution: int) -> str:
    """
    Convierte coordenadas lat/lon a índice H3.

    Parameters
    ----------
    lat : float
        Latitud.
    lon : float
        Longitud.
    resolution : int
        Resolución H3 (0-15). 12 ≈ 9m, 13 ≈ 3.4m de diámetro.

    Returns
    -------
    str
        Índice H3 hexadecimal.
    """
    return h3.latlng_to_cell(lat, lon, resolution)


def _h3_to_polygon(h3_index: str) -> List[Tuple[float, float]]:
    """
    Obtiene los vértices del hexágono H3 como lista de (lat, lon).

    Parameters
    ----------
    h3_index : str
        Índice H3.

    Returns
    -------
    List[Tuple[float, float]]
        Lista de coordenadas (lat, lon) de los vértices.
    """
    boundary = h3.cell_to_boundary(h3_index)
    # h3 retorna (lat, lon), folium espera (lat, lon) también
    return list(boundary)


def _calculate_quartiles_and_outliers(
    values: np.ndarray,
) -> Tuple[float, float, float, float, float]:
    """
    Calcula cuartiles y umbral de outliers (Q3 + 1.5*IQR).

    Parameters
    ----------
    values : np.ndarray
        Array de valores numéricos.

    Returns
    -------
    Tuple[float, float, float, float, float]
        (Q1, Q2, Q3, Q4_max, outlier_threshold)
    """
    q1 = np.percentile(values, 25)
    q2 = np.percentile(values, 50)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1
    outlier_threshold = q3 + 1.5 * iqr
    q4_max = np.max(values)

    return q1, q2, q3, q4_max, outlier_threshold


#──────────────────────────────────────────────────────────────────────────────#
# ── 3. Función principal ── #
#──────────────────────────────────────────────────────────────────────────────#

##############################################################################
#   Detección automática de end-stops por densidad H3
##############################################################################

def _polyline_total_length_m(df_segments: pl.DataFrame) -> float:
    """
    Suma la longitud haversine de todas las polilíneas en df_segments.
    Devuelve la longitud total en metros.
    """
    import math

    def haversine(a, b):
        R = 6_371_000.0
        la1, lo1 = math.radians(a[0]), math.radians(a[1])
        la2, lo2 = math.radians(b[0]), math.radians(b[1])
        dlat = la2 - la1
        dlon = lo2 - lo1
        h = math.sin(dlat / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin(dlon / 2) ** 2
        return 2 * R * math.asin(math.sqrt(max(0.0, h)))

    poly_col = next((c for c in ["polyline", "geometry", "coords"] if c in df_segments.columns), None)
    if poly_col is None:
        return 0.0

    total = 0.0
    for val in df_segments[poly_col].to_list():
        pts = _parse_polyline(val)
        for i in range(1, len(pts)):
            total += haversine(pts[i - 1], pts[i])
    return total


def _h3_center_latlon(h3_index: str):
    """Devuelve (lat, lon) del centroide de un hexágono H3."""
    lat, lon = h3.cell_to_latlng(h3_index)
    return lat, lon


from _lib_geo import _haversine_m  # consolidado en lib/_lib_geo.py


def _find_end_stop_candidates(
    df_h3_counts: pl.DataFrame,
    df_segments: pl.DataFrame,
    min_distance_fraction: float = 1 / 5,
    verbose: bool = False,
):
    """
    Detecta los dos end-stops candidatos a partir de la densidad H3.

    Lógica:
    -------
    1. Para cada hexágono, calcula la suma de su conteo + el de sus vecinos H3
       presentes en df_h3_counts.  El hexágono con mayor suma acumulada es el
       centro del primer end-stop (end_stop_1).

    2. El segundo end-stop se busca igual, pero limitando los candidatos a
       hexágonos cuya distancia haversine al centro del primero sea ≥
       `min_distance_fraction` × longitud_total_polilínea.

    Parámetros
    ----------
    df_h3_counts           : DataFrame con columnas h3_index, count.
    df_segments            : df_segments_route para calcular longitud total.
    min_distance_fraction  : fracción de la polilínea total como distancia mínima
                             entre los dos end-stops. Default 1/3.

    Devuelve
    --------
    Tupla (end_stop_1, end_stop_2) donde cada elemento es un dict:
        {
          "center_h3"    : str,         # hexágono central
          "center_lat"   : float,
          "center_lon"   : float,
          "neighbor_h3s" : list[str],   # vecinos presentes en df_h3_counts
          "cluster_sum"  : int,         # conteo total del cluster
        }
    end_stop_2 puede ser None si no se encuentra candidato suficientemente lejano.
    """
    # ── índice de conteos para lookup rápido ──
    count_map = dict(zip(
        df_h3_counts["h3_index"].to_list(),
        df_h3_counts["count"].to_list(),
    ))
    all_h3 = set(count_map.keys())

    # ── función: suma del cluster (centro + vecinos presentes) ──
    def cluster_sum(center_h3):
        neighbors = set(h3.grid_disk(center_h3, 1)) - {center_h3}
        present_neighbors = neighbors & all_h3
        total = count_map.get(center_h3, 0) + sum(count_map[n] for n in present_neighbors)
        return total, present_neighbors

    # ── calcular suma de cluster para todos los hexágonos ──
    cluster_sums = {idx: cluster_sum(idx) for idx in all_h3}

    # ── end_stop_1: mayor cluster_sum global ──
    best_h3_1 = max(cluster_sums, key=lambda k: cluster_sums[k][0])
    sum1, neighbors1 = cluster_sums[best_h3_1]
    lat1, lon1 = _h3_center_latlon(best_h3_1)

    end_stop_1 = {
        "center_h3":    best_h3_1,
        "center_lat":   lat1,
        "center_lon":   lon1,
        "neighbor_h3s": list(neighbors1),
        "cluster_sum":  sum1,
    }

    if verbose:
        print(f"  [EndStops] End-stop 1: {best_h3_1[:10]}… sum={sum1:,} "
              f"vecinos={len(neighbors1)} ({lat1:.5f}, {lon1:.5f})")

    # ── longitud total de la polilínea para umbral de distancia ──
    total_len_m = _polyline_total_length_m(df_segments)
    min_dist_m  = total_len_m * min_distance_fraction

    if verbose:
        print(f"  [EndStops] Longitud polilínea total: {total_len_m:,.0f} m")
        print(f"  [EndStops] Distancia mínima end-stop 2: {min_dist_m:,.0f} m (1/{round(1/min_distance_fraction)} de la ruta)")

    # ── end_stop_2: mayor cluster_sum entre candidatos suficientemente lejanos ──
    end_stop_2 = None
    best_sum2  = -1
    for idx, (csum, nbrs) in cluster_sums.items():
        if idx == best_h3_1:
            continue
        clat, clon = _h3_center_latlon(idx)
        dist = _haversine_m(lat1, lon1, clat, clon)
        if dist >= min_dist_m and csum > best_sum2:
            best_sum2 = csum
            end_stop_2 = {
                "center_h3":    idx,
                "center_lat":   clat,
                "center_lon":   clon,
                "neighbor_h3s": list(nbrs),
                "cluster_sum":  csum,
                "dist_to_1_m":  dist,
                "fallback":     False,
            }

    # ── fallback: si no hay candidato suficientemente lejano, usar el segundo
    #    mayor cluster_sum global y marcar como fallback ──
    if end_stop_2 is None:
        best_sum2_fb = -1
        for idx, (csum, nbrs) in cluster_sums.items():
            if idx == best_h3_1:
                continue
            if csum > best_sum2_fb:
                best_sum2_fb = csum
                clat, clon   = _h3_center_latlon(idx)
                dist_fb      = _haversine_m(lat1, lon1, clat, clon)
                end_stop_2   = {
                    "center_h3":    idx,
                    "center_lat":   clat,
                    "center_lon":   clon,
                    "neighbor_h3s": list(nbrs),
                    "cluster_sum":  csum,
                    "dist_to_1_m":  dist_fb,
                    "fallback":     True,
                }
        if end_stop_2 is not None:
            import warnings
            warnings.warn(
                f"[EndStops] ⚠  No se encontró end-stop 2 a ≥{min_dist_m:,.0f} m del primero. "
                f"Se usa el segundo cluster más denso como fallback "
                f"(dist={end_stop_2['dist_to_1_m']:,.0f} m). "
                f"Los dos end-stops podrían estar muy juntos.",
                UserWarning,
                stacklevel=2,
            )
            if verbose:
                print(
                    f"  [EndStops] ⚠  FALLBACK — end-stop 2 muy cercano al 1: "
                    f"dist={end_stop_2['dist_to_1_m']:,.0f} m "
                    f"(mínimo esperado {min_dist_m:,.0f} m). "
                    f"Podrían ser el mismo extremo de la ruta."
                )

    if verbose:
        if end_stop_2:
            tag = " [FALLBACK]" if end_stop_2.get("fallback") else ""
            print(f"  [EndStops] End-stop 2{tag}: {end_stop_2['center_h3'][:10]}… "
                  f"sum={end_stop_2['cluster_sum']:,} "
                  f"dist={end_stop_2['dist_to_1_m']:,.0f} m "
                  f"({end_stop_2['center_lat']:.5f}, {end_stop_2['center_lon']:.5f})")
        else:
            print("  [EndStops] ⚠  No se encontró ningún end-stop 2")

    return end_stop_1, end_stop_2


def _add_end_stops_layer(
    m: folium.Map,
    end_stop_1: dict,
    end_stop_2,
    resolution: int = 12,
    verbose: bool = False,
) -> None:
    """
    Agrega al mapa un FeatureGroup (toggle) con los dos end-stops detectados.

    Colores:
    --------
    - End-stop 1 centro   : #e63946  (rojo brillante)
    - End-stop 1 vecinos  : #f4a1a8  (rosa tenue)
    - End-stop 2 centro   : #2a9d8f  (verde azulado brillante)
    - End-stop 2 vecinos  : #a8d8d3  (verde azulado tenue)

    El toggle está oculto por defecto (show=False).
    """
    COLOR_1_CENTER   = "#e63946"
    COLOR_1_NEIGHBOR = "#f4a1a8"
    COLOR_2_CENTER   = "#2a9d8f"
    COLOR_2_NEIGHBOR = "#a8d8d3"

    layer = folium.FeatureGroup(name="🔴🟢 End-stops detectados", show=True)

    def _draw_hex(h3_idx, color, fill_opacity, label, popup_extra=""):
        vertices = _h3_to_polygon(h3_idx)
        clat, clon = _h3_center_latlon(h3_idx)
        folium.Polygon(
            locations=vertices,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=fill_opacity,
            weight=2,
            tooltip=label,
            popup=folium.Popup(
                f"<b>{label}</b><br>"
                f"H3: <code>{h3_idx[:12]}…</code><br>"
                f"({clat:.5f}, {clon:.5f}){popup_extra}",
                max_width=240,
            ),
        ).add_to(layer)

    # ── End-stop 1 ──
    _draw_hex(
        end_stop_1["center_h3"], COLOR_1_CENTER, 0.85,
        "End-stop 1 (centro)",
        f"<br>Cluster sum: {end_stop_1['cluster_sum']:,}",
    )
    for nbr in end_stop_1["neighbor_h3s"]:
        _draw_hex(nbr, COLOR_1_NEIGHBOR, 0.40, "End-stop 1 (vecino)")

    # ── End-stop 2 ──
    if end_stop_2:
        dist_km  = end_stop_2.get("dist_to_1_m", 0) / 1000
        fallback = end_stop_2.get("fallback", False)
        warn_str = "<br><span style='color:#e07b00'>⚠ Fallback: podrían estar muy juntos</span>" if fallback else ""
        _draw_hex(
            end_stop_2["center_h3"], COLOR_2_CENTER, 0.85,
            "End-stop 2 (centro)" + (" ⚠" if fallback else ""),
            f"<br>Cluster sum: {end_stop_2['cluster_sum']:,}"
            f"<br>Distancia al 1: {dist_km:.2f} km"
            f"{warn_str}",
        )
        for nbr in end_stop_2["neighbor_h3s"]:
            _draw_hex(nbr, COLOR_2_NEIGHBOR, 0.40, "End-stop 2 (vecino)")

    layer.add_to(m)

    if verbose:
        n_drawn = 1 + len(end_stop_1["neighbor_h3s"])
        if end_stop_2:
            n_drawn += 1 + len(end_stop_2["neighbor_h3s"])
        print(f"  [EndStops] ✔  {n_drawn} hexágonos en la capa end-stops")


def _find_stop_cluster_candidates(
    df_h3_counts: pl.DataFrame,
    excluded_h3s: set,
    top_n: int = 10,
    verbose: bool = False,
) -> list:
    """
    Detecta los ``top_n`` clusters de mayor densidad H3, excluyendo los
    hexágonos ya asignados a los end-stops (centros y vecinos).

    Algoritmo greedy: en cada iteración selecciona el hexágono con mayor
    cluster_sum (centro + vecinos presentes), lo añade a la lista y excluye
    su huella (centro + vecinos) antes de la siguiente iteración.

    Parámetros
    ----------
    df_h3_counts : DataFrame con columnas h3_index, count.
    excluded_h3s : Conjunto de h3_index a ignorar (end-stop centros + vecinos).
    top_n        : Número máximo de clusters a devolver (default 10).

    Devuelve
    --------
    Lista de dicts (puede tener < top_n elementos si no hay suficientes
    hexágonos disponibles). Cada dict tiene las claves:
        center_h3, center_lat, center_lon, neighbor_h3s,
        cluster_sum, rank  (1=mayor … top_n=menor)
    """
    count_map = dict(zip(
        df_h3_counts["h3_index"].to_list(),
        df_h3_counts["count"].to_list(),
    ))

    available_h3 = set(count_map.keys()) - excluded_h3s

    if not available_h3:
        if verbose:
            print("  [StopCluster] ⚠  No quedan hexágonos disponibles tras excluir end-stops")
        return []

    def _cluster_sum_of(center, pool):
        neighbors = set(h3.grid_disk(center, 1)) - {center}
        present   = neighbors & pool
        total     = count_map.get(center, 0) + sum(count_map[n] for n in present)
        return total, present

    clusters = []
    pool     = set(available_h3)

    for rank in range(1, top_n + 1):
        if not pool:
            break

        # Calcular cluster_sum de todos los candidatos restantes
        sums = {c: _cluster_sum_of(c, pool) for c in pool}
        best = max(sums, key=lambda k: sums[k][0])
        total, nbrs = sums[best]
        lat, lon    = _h3_center_latlon(best)

        clusters.append({
            "center_h3":    best,
            "center_lat":   lat,
            "center_lon":   lon,
            "neighbor_h3s": list(nbrs),
            "cluster_sum":  total,
            "rank":         rank,
        })

        if verbose:
            print(f"  [StopCluster] Cluster {rank:2d}: {best[:10]}… "
                  f"sum={total:,} vecinos={len(nbrs)} ({lat:.5f}, {lon:.5f})")

        # Retirar la huella de este cluster del pool
        pool -= {best} | set(nbrs)

    if verbose:
        print(f"  [StopCluster] → {len(clusters)} clusters detectados")

    return clusters


def _stop_cluster_color(rank: int, total_ranks: int) -> tuple[str, str]:
    """
    Devuelve (color_centro, color_vecino) para un cluster dado su rank.

    Escala amarillo→rojo en 10 pasos (rank 1 = más registros = rojo más intenso):
      rank 1  → #d62728  (rojo intenso)
      rank 10 → #ffdd57  (amarillo)
    Los vecinos usan la misma tonalidad con 50 % de opacidad (hex más claro).
    """
    # Interpolación en espacio RGB: rojo (#d62728) → amarillo (#ffdd57)
    t = (rank - 1) / max(total_ranks - 1, 1)   # 0 (rank=1, rojo) … 1 (rank=N, amarillo)

    r1, g1, b1 = 0xd6, 0x27, 0x28   # rojo
    r2, g2, b2 = 0xff, 0xdd, 0x57   # amarillo

    r = int(r1 + (r2 - r1) * t)
    g = int(g1 + (g2 - g1) * t)
    b = int(b1 + (b2 - b1) * t)

    center_hex = f"#{r:02x}{g:02x}{b:02x}"

    # Vecinos: mezcla 50 % con blanco (#ffffff)
    rn = int(r + (0xff - r) * 0.55)
    gn = int(g + (0xff - g) * 0.55)
    bn = int(b + (0xff - b) * 0.55)
    neighbor_hex = f"#{rn:02x}{gn:02x}{bn:02x}"

    return center_hex, neighbor_hex


def _add_stop_cluster_layer(
    m: folium.Map,
    clusters: list,
    verbose: bool = False,
) -> None:
    """
    Agrega al mapa un FeatureGroup (toggle) con hasta 10 stop-clusters,
    cada uno coloreado según su ranking por escala amarillo → rojo:

        rank 1 (más registros)  → rojo intenso  (#d62728)
        rank 10 (menos)         → amarillo       (#ffdd57)

    Los vecinos se dibujan con la misma tonalidad suavizada.
    El toggle está visible por defecto (show=True).
    """
    if not clusters:
        return

    layer = folium.FeatureGroup(name="🌡️ Stop clusters (top 10)", show=True)
    n_total = len(clusters)

    def _draw_hex(h3_idx, color, fill_opacity, label, popup_extra=""):
        vertices = _h3_to_polygon(h3_idx)
        clat, clon = _h3_center_latlon(h3_idx)
        folium.Polygon(
            locations=vertices,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=fill_opacity,
            weight=1.5,
            tooltip=label,
            popup=folium.Popup(
                f"<b>{label}</b><br>"
                f"H3: <code>{h3_idx[:12]}…</code><br>"
                f"({clat:.5f}, {clon:.5f}){popup_extra}",
                max_width=260,
            ),
        ).add_to(layer)

    n_drawn = 0
    for cl in clusters:
        rank       = cl["rank"]
        cc, cn     = _stop_cluster_color(rank, n_total)
        label_c    = f"Stop cluster #{rank} (centro)"
        label_n    = f"Stop cluster #{rank} (vecino)"
        popup_info = (
            f"<br>Rank: <b>{rank}</b> de {n_total}"
            f"<br>Cluster sum: {cl['cluster_sum']:,}"
        )
        _draw_hex(cl["center_h3"], cc, 0.85, label_c, popup_info)
        for nbr in cl["neighbor_h3s"]:
            _draw_hex(nbr, cn, 0.38, label_n)
        n_drawn += 1 + len(cl["neighbor_h3s"])

    # ── Leyenda de escala de color (HTML flotante) ──
    if n_total > 1:
        gradient_stops = []
        for i in range(n_total):
            cc, _ = _stop_cluster_color(i + 1, n_total)
            pct   = int(i / max(n_total - 1, 1) * 100)
            gradient_stops.append(f"{cc} {pct}%")
        gradient_css = ", ".join(gradient_stops)

        legend_html = f"""
        <div style="
            position: fixed; bottom: 40px; left: 60px; z-index: 1000;
            background: rgba(255,255,255,0.92); padding: 10px 14px;
            border-radius: 6px; border: 1px solid #ccc;
            font-family: sans-serif; font-size: 12px; box-shadow: 0 2px 6px rgba(0,0,0,.25);
        ">
          <b>Stop clusters (registros parados)</b><br>
          <div style="display:flex; align-items:center; margin-top:6px; gap:6px;">
            <span style="font-size:11px; color:#555">más</span>
            <div style="
                width:130px; height:14px;
                background: linear-gradient(to right, {gradient_css});
                border-radius:3px; border:1px solid #bbb;
            "></div>
            <span style="font-size:11px; color:#555">menos</span>
          </div>
          <div style="display:flex; justify-content:space-between; width:130px;
                      margin-left:34px; font-size:10px; color:#888; margin-top:2px;">
            <span>#{clusters[0]['cluster_sum']:,}</span>
            <span>#{clusters[-1]['cluster_sum']:,}</span>
          </div>
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))

    layer.add_to(m)

    if verbose:
        print(f"  [StopCluster] ✔  {n_drawn} hexágonos en {n_total} clusters (escala rojo→amarillo)")


def plot_speed_density(
    df_rec: pl.DataFrame,
    df_stops: Optional[pl.DataFrame] = None,
    df_segments_route: Optional[pl.DataFrame] = None,
    speed_col: str = "speed",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    time_col: str = "time",
    max_speed: float = 0.0,
    min_hex_count: int = 1,
    resolution: int = 12,
    show_stops: bool = True,
    show_end_stops: bool = True,
    stop_lat_col: str = "latitude",
    stop_lon_col: str = "longitude",
    stop_code_col: str = "codigo",
    hour_range: Optional[Tuple[int, int]] = None,
    zoom_start: int = 13,
    hex_opacity: float = 0.7,
    stop_radius: int = 1,
    # ─── v2: turn_hex resueltos externamente ──────────────────────────
    turn_hexes: Optional[List[Dict[str, Any]]] = None,
    # ──────────────────────────────────────────────────────────────────
    # ─── v3: density bus-aware (cap por (bus, hex)) ───────────────────
    # Si es None, conteo crudo (comportamiento original v2). Si es un
    # entero K, cada par (bus_code, h3_index) aporta como mucho K pings
    # a la densidad. Mantiene la coherencia visual con el detector de
    # turn_hex del paso 4 cuando éste usa la misma métrica.
    # Requiere columna ``bus_code`` en df_rec; si no, se ignora con warning.
    max_pings_per_bus_per_hex: Optional[int] = None,
    bus_code_col: str = "bus_code",
    # ──────────────────────────────────────────────────────────────────
    verbose: bool = False,
) -> folium.Map:
    """
    Visualiza densidad de vehículos detenidos/lentos usando hexágonos H3.

    Genera un mapa de calor donde cada hexágono H3 representa la cantidad
    de registros GPS con velocidad ≤ max_speed. Los colores usan gradiente
    viridis para valores normales y gradiente cálido para outliers.

    Si se proporciona `df_segments_route` y `show_end_stops=True`, agrega
    un toggle "End-stops detectados" que muestra los dos clusters H3 con
    mayor densidad de paradas, separados al menos 1/3 de la longitud total
    de la ruta. El primer end-stop se dibuja en rojo, el segundo en verde.

    Parameters
    ----------
    df_rec : pl.DataFrame
        DataFrame de registros GPS.
    df_stops : pl.DataFrame, optional
        Paradas de la ruta para referencia visual.
    df_segments_route : pl.DataFrame, optional
        Segmentos de la ruta (con columna polyline). Necesario para la
        detección de end-stops y el cálculo de distancia mínima.
    speed_col : str
        Columna de velocidad. Default: ``"speed"``.
    lat_col : str
        Columna de latitud. Default: ``"latitude"``.
    lon_col : str
        Columna de longitud. Default: ``"longitude"``.
    time_col : str
        Columna de tiempo. Default: ``"time"``.
    max_speed : float
        Velocidad máxima para "detenido". Default: ``0.0``.
    min_hex_count : int
        Registros mínimos por hexágono para mostrarlo. Default: ``1``.
    resolution : int
        Resolución H3 (10-15). Default: ``12``.
    show_stops : bool
        Mostrar paradas de la ruta. Default: ``True``.
    show_end_stops : bool
        Mostrar toggle de end-stops detectados (requiere df_segments_route).
        Default: ``True``.
    stop_lat_col : str
        Columna latitud en df_stops. Default: ``"latitude"``.
    stop_lon_col : str
        Columna longitud en df_stops. Default: ``"longitude"``.
    stop_code_col : str
        Columna código en df_stops. Default: ``"codigo"``.
    hour_range : Tuple[int, int], optional
        Filtro de horas (inicio, fin). Default: ``None``.
    zoom_start : int
        Zoom inicial. Default: ``13``.
    hex_opacity : float
        Opacidad de hexágonos (0-1). Default: ``0.7``.
    stop_radius : int
        Radio de marcadores de paradas. Default: ``8``.
    verbose : bool
        Imprimir estadísticas. Default: ``False``.

    Returns
    -------
    folium.Map
    """

    #──────────────────────────────────────────────────────────────────────────#
    # ── 3.1 Validaciones ── #
    #──────────────────────────────────────────────────────────────────────────#
    required_cols = [speed_col, lat_col, lon_col]
    for col in required_cols:
        if col not in df_rec.columns:
            raise ValueError(f"Columna '{col}' no encontrada en df_rec.")

    if show_stops and df_stops is not None:
        for col in [stop_lat_col, stop_lon_col]:
            if col not in df_stops.columns:
                raise ValueError(f"Columna '{col}' no encontrada en df_stops.")

    if resolution < 10 or resolution > 15:
        raise ValueError(f"Resolución H3 debe estar entre 10-15, recibido: {resolution}")

    #──────────────────────────────────────────────────────────────────────────#
    # ── 3.2 Filtrar registros por velocidad (≤ max_speed) ── #
    #──────────────────────────────────────────────────────────────────────────#
    df_stopped = df_rec.filter(pl.col(speed_col) <= max_speed)

    n_total = df_rec.height
    n_stopped = df_stopped.height

    if verbose:
        speed_label = "detenidos" if max_speed == 0 else f"velocidad ≤ {max_speed}"
        print(f"[plot_speed_density] Registros totales   : {n_total}")
        print(f"[plot_speed_density] Registros {speed_label} : {n_stopped} ({n_stopped/n_total*100:.1f}%)")

    #──────────────────────────────────────────────────────────────────────────#
    # ── 3.3 Filtrar por rango de horas (opcional) ── #
    #──────────────────────────────────────────────────────────────────────────#
    if hour_range is not None:
        if time_col not in df_rec.columns:
            raise ValueError(f"Columna '{time_col}' requerida para filtro por hora.")

        hour_start, hour_end = hour_range
        df_stopped = df_stopped.with_columns(
            pl.col(time_col).dt.hour().alias("_hour")
        ).filter(
            (pl.col("_hour") >= hour_start) & (pl.col("_hour") < hour_end)
        ).drop("_hour")

        n_filtered = df_stopped.height
        if verbose:
            print(f"[plot_speed_density] Filtro hora {hour_start}:00-{hour_end}:00 : {n_filtered} registros")

    if df_stopped.height == 0:
        raise ValueError("No hay registros detenidos después de aplicar filtros.")

    #──────────────────────────────────────────────────────────────────────────#
    # ── 3.4 Cast a Float64 si es necesario ── #
    #──────────────────────────────────────────────────────────────────────────#
    cast_exprs = []
    for col_name in [lat_col, lon_col]:
        if df_stopped.schema[col_name] != pl.Float64:
            cast_exprs.append(pl.col(col_name).cast(pl.Float64))
    if cast_exprs:
        df_stopped = df_stopped.with_columns(cast_exprs)

    #──────────────────────────────────────────────────────────────────────────#
    # ── 3.5 Convertir a índices H3 y agregar ── #
    #──────────────────────────────────────────────────────────────────────────#
    lats = df_stopped[lat_col].to_numpy()
    lons = df_stopped[lon_col].to_numpy()

    # Convertir a H3
    h3_indices = [_lat_lon_to_h3(lat, lon, resolution) for lat, lon in zip(lats, lons)]

    # ── v3: validar si se puede aplicar el cap bus-aware ─────────────
    use_cap = max_pings_per_bus_per_hex is not None
    if use_cap and bus_code_col not in df_stopped.columns:
        import warnings
        warnings.warn(
            f"plot_speed_density: max_pings_per_bus_per_hex={max_pings_per_bus_per_hex} "
            f"se solicitó pero df_rec no tiene columna '{bus_code_col}'. "
            f"Cayendo al conteo crudo.",
            stacklevel=2,
        )
        use_cap = False
    if use_cap and max_pings_per_bus_per_hex < 1:
        raise ValueError(
            f"max_pings_per_bus_per_hex debe ser ≥ 1 o None, "
            f"recibí {max_pings_per_bus_per_hex}."
        )

    if not use_cap:
        # Conteo crudo (comportamiento original v2)
        df_h3 = pl.DataFrame({"h3_index": h3_indices})
        df_h3_counts = (
            df_h3
            .group_by("h3_index")
            .agg(pl.len().alias("count"))
            .sort("count", descending=True)
        )
    else:
        # ── v3: density bus-aware con cap K por (bus, hex) ───────────
        # Misma lógica que _lib_turn_hex._h3_counts_from_stopped, para
        # garantizar coherencia visual entre la densidad de fondo y el
        # detector de turn_hex.
        cap = int(max_pings_per_bus_per_hex)
        df_with_hex = df_stopped.with_columns(pl.Series("h3_index", h3_indices))
        df_h3_counts = (
            df_with_hex
            .group_by([bus_code_col, "h3_index"])
            .agg(pl.len().alias("bus_hex_count"))
            .with_columns(
                pl.min_horizontal(pl.col("bus_hex_count"), pl.lit(cap))
                .alias("bus_hex_capped")
            )
            .group_by("h3_index")
            .agg(pl.col("bus_hex_capped").sum().alias("count"))
            .sort("count", descending=True)
        )
        if verbose:
            print(f"[plot_speed_density] Density bus-aware     : "
                  f"cap = {cap} pings/bus/hex")

    n_hexagons_raw = df_h3_counts.height

    # Filtrar hexágonos con conteo mínimo
    if min_hex_count > 1:
        df_h3_counts = df_h3_counts.filter(pl.col("count") >= min_hex_count)

    n_hexagons = df_h3_counts.height
    n_filtered_out = n_hexagons_raw - n_hexagons

    if verbose:
        print(f"[plot_speed_density] Resolución H3       : {resolution}")
        print(f"[plot_speed_density] Hexágonos totales   : {n_hexagons_raw}")
        if min_hex_count > 1:
            print(f"[plot_speed_density] Filtro min_hex_count: ≥{min_hex_count} registros")
            print(f"[plot_speed_density] Hexágonos filtrados : {n_filtered_out} eliminados")
        print(f"[plot_speed_density] Hexágonos mostrados : {n_hexagons}")

    if n_hexagons == 0:
        raise ValueError(f"No hay hexágonos con ≥{min_hex_count} registros.")

    #──────────────────────────────────────────────────────────────────────────#
    # ── 3.6 Calcular estadísticas y preparar colores ── #
    #──────────────────────────────────────────────────────────────────────────#
    counts = df_h3_counts["count"].to_numpy()
    min_count = float(np.min(counts))
    max_count = float(np.max(counts))
    q1, q2, q3, _, outlier_threshold = _calculate_quartiles_and_outliers(counts)

    # Asignar color a cada hexágono usando gradiente continuo
    colors_and_outliers = [
        _get_color_for_value(c, min_count, outlier_threshold, outlier_threshold, max_count)
        for c in counts
    ]
    colors = [c[0] for c in colors_and_outliers]
    is_outlier = [c[1] for c in colors_and_outliers]

    # Asignar cuartil para toggles
    def _assign_quartile(val):
        if val > outlier_threshold:
            return "Outliers"
        elif val > q3:
            return "Q4"
        elif val > q2:
            return "Q3"
        elif val > q1:
            return "Q2"
        else:
            return "Q1"

    quartiles = [_assign_quartile(c) for c in counts]

    df_h3_counts = df_h3_counts.with_columns([
        pl.Series("color", colors),
        pl.Series("is_outlier", is_outlier),
        pl.Series("quartile", quartiles),
    ])

    # Contar por cuartil
    quartile_counts = {}
    for q in ["Q1", "Q2", "Q3", "Q4", "Outliers"]:
        quartile_counts[q] = df_h3_counts.filter(pl.col("quartile") == q).height

    if verbose:
        print(f"[plot_speed_density] ── Estadísticas de densidad ──")
        print(f"[plot_speed_density]   Mínimo               : {min_count:.0f} registros")
        print(f"[plot_speed_density]   Q1 (P25)             : {q1:.0f}")
        print(f"[plot_speed_density]   Q2 (P50/mediana)     : {q2:.0f}")
        print(f"[plot_speed_density]   Q3 (P75)             : {q3:.0f}")
        print(f"[plot_speed_density]   Umbral outliers      : {outlier_threshold:.0f} (Q3+1.5*IQR)")
        print(f"[plot_speed_density]   Máximo               : {max_count:.0f}")
        print(f"[plot_speed_density] ── Hexágonos por cuartil ──")
        for q in ["Q1", "Q2", "Q3", "Q4", "Outliers"]:
            print(f"[plot_speed_density]   {q}: {quartile_counts[q]} hexágonos")

    #──────────────────────────────────────────────────────────────────────────#
    # ── 3.7 Crear mapa base ── #
    #──────────────────────────────────────────────────────────────────────────#
    center_lat = float(np.mean(lats))
    center_lon = float(np.mean(lons))

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_start,
        tiles="CartoDB positron",
    )

    if verbose:
        print(f"[plot_speed_density] Centro del mapa     : ({center_lat:.5f}, {center_lon:.5f})")

    #──────────────────────────────────────────────────────────────────────────#
    # ── 3.8 Agregar hexágonos al mapa (por cuartil) ── #
    #──────────────────────────────────────────────────────────────────────────#
    # Crear FeatureGroup para cada cuartil
    quartile_groups = {
        "Q1": folium.FeatureGroup(name=f"Q1 (≤{q1:.0f})", show=False),
        "Q2": folium.FeatureGroup(name=f"Q2 ({q1:.0f}-{q2:.0f})", show=False),
        "Q3": folium.FeatureGroup(name=f"Q3 ({q2:.0f}-{q3:.0f})", show=False),
        "Q4": folium.FeatureGroup(name=f"Q4 ({q3:.0f}-{outlier_threshold:.0f})", show=False),
        "Outliers": folium.FeatureGroup(name=f"Outliers (>{outlier_threshold:.0f})", show=False),
    }

    for row in df_h3_counts.iter_rows(named=True):
        h3_index = row["h3_index"]
        count = row["count"]
        color = row["color"]
        quartile = row["quartile"]

        # Obtener vértices del hexágono
        vertices = _h3_to_polygon(h3_index)

        # Crear polígono
        polygon = folium.Polygon(
            locations=vertices,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=hex_opacity,
            weight=1,
            popup=folium.Popup(
                f"<b>Conteo:</b> {count}<br>"
                f"<b>Cuartil:</b> {quartile}<br>"
                f"<b>H3:</b> {h3_index[:12]}...",
                max_width=200,
            ),
            tooltip=f"{count} registros ({quartile})",
        )

        # Agregar al grupo correspondiente
        polygon.add_to(quartile_groups[quartile])

    # Agregar todos los grupos al mapa
    for q in ["Q1", "Q2", "Q3", "Q4", "Outliers"]:
        quartile_groups[q].add_to(m)

    if verbose:
        print(f"[plot_speed_density] ✓ {n_hexagons} hexágonos graficados")

    #──────────────────────────────────────────────────────────────────────────#
    # ── 3.8b End-stops detectados (toggle, LEGACY) ── #
    #──────────────────────────────────────────────────────────────────────────#
    # NOTA v2: si el caller pasa `turn_hexes` (rebrand v2), saltamos la
    # detección legacy. La capa 3.8c (turn_hex resueltos) ya cubre el caso
    # con el algoritmo correcto (resolve_turn_hex). Ejecutar ambas duplica
    # marcadores en el mapa y emite logs confusos ("FALLBACK end-stop 2
    # muy cercano al 1") del detector viejo que no usa branch_length.
    if turn_hexes:
        if verbose:
            print("[plot_speed_density] Saltando detección legacy de end-stops "
                  "(turn_hexes v2 provistos).")
    elif show_end_stops and df_segments_route is not None:
        if verbose:
            print("[plot_speed_density] Detectando end-stops (legacy)…")
        try:
            es1, es2 = _find_end_stop_candidates(
                df_h3_counts, df_segments_route, verbose=verbose
            )
            _add_end_stops_layer(m, es1, es2, resolution=resolution, verbose=verbose)

            # ── Stop clusters: misma lógica sin restricción de distancia,
            #    excluyendo todos los hexágonos ya asignados a end-stops ──
            excluded = set()
            excluded.add(es1["center_h3"])
            excluded.update(es1["neighbor_h3s"])
            if es2:
                excluded.add(es2["center_h3"])
                excluded.update(es2["neighbor_h3s"])

            if verbose:
                print("[plot_speed_density] Detectando stop clusters (top 10)…")
            stop_clusters = _find_stop_cluster_candidates(
                df_h3_counts, excluded, top_n=10, verbose=verbose
            )
            _add_stop_cluster_layer(m, stop_clusters, verbose=verbose)

        except Exception as exc:
            if verbose:
                print(f"[plot_speed_density] ⚠  End-stops omitidos: {exc}")
    elif show_end_stops and df_segments_route is None and verbose:
        print("[plot_speed_density] ⚠  show_end_stops=True pero df_segments_route no fue provisto — omitido")

    #──────────────────────────────────────────────────────────────────────────#
    # ── 3.8c Turn_hex resueltos externamente (v2, prioridad alta) ── #
    #──────────────────────────────────────────────────────────────────────────#
    # Si el caller pasa `turn_hexes` (e.g. desde resolve_turn_hex), se dibuja
    # esta capa con prioridad visual sobre la detección automática interna.
    # Marcamos los dos turn_hex como hexágonos con borde grueso y popup
    # detallado. Los clusters contiguos se sombrean en color suave.
    if turn_hexes:
        try:
            import h3 as _h3mod
            layer_th = folium.FeatureGroup(name="⭐ Turn_hex (v2)", show=True)
            color_pairs = [
                ("#a8d4ff", "#0055cc"),   # turn_hex_A: azul claro + azul fuerte
                ("#ffcc99", "#cc6600"),   # turn_hex_B: naranja claro + naranja fuerte
            ]
            for th, (cluster_color, center_color) in zip(turn_hexes, color_pairs):
                # 1) Cluster contiguo en color suave
                for hh in th.get("cluster_h3s", [th["h3"]]):
                    if hh == th["h3"]:
                        continue   # el centro se dibuja después con borde grueso
                    try:
                        boundary = list(_h3mod.cell_to_boundary(hh))
                        folium.Polygon(
                            locations=boundary,
                            color=cluster_color,
                            weight=1,
                            fill=True,
                            fill_color=cluster_color,
                            fill_opacity=0.45,
                            tooltip=f"{th['label']} cluster · H3={hh}",
                        ).add_to(layer_th)
                    except Exception:
                        continue
                # 2) Centro del cluster con borde grueso y popup detallado
                try:
                    boundary = list(_h3mod.cell_to_boundary(th["h3"]))
                    folium.Polygon(
                        locations=boundary,
                        color=center_color,
                        weight=4,
                        fill=True,
                        fill_color=center_color,
                        fill_opacity=0.55,
                        popup=folium.Popup(
                            f"<b>{th['label']}</b><br>"
                            f"H3: {th['h3']}<br>"
                            f"Lat/Lon: ({th['lat']:.5f}, {th['lon']:.5f})<br>"
                            f"Conteo del hex: {th.get('stopped_count', 0):,}<br>"
                            f"Hex en cluster: {len(th.get('cluster_h3s', []))}<br>"
                            f"Conteo cluster total: {th.get('cluster_total_count', 0):,}",
                            max_width=300,
                        ),
                        tooltip=(
                            f"{th['label']} CENTRO · "
                            f"count={th.get('stopped_count', 0):,} · "
                            f"cluster_total={th.get('cluster_total_count', 0):,}"
                        ),
                    ).add_to(layer_th)
                    # Etiqueta visual encima del hex centro
                    folium.map.Marker(
                        location=[th["lat"], th["lon"]],
                        icon=folium.DivIcon(html=(
                            f"<div style='font-size:13px;font-weight:bold;"
                            f"color:{center_color};text-shadow:1px 1px 1px white;"
                            f"white-space:nowrap;'>{th['label']}</div>"
                        )),
                    ).add_to(layer_th)
                except Exception as exc:
                    if verbose:
                        print(f"[plot_speed_density] ⚠ no pude dibujar "
                              f"{th.get('label', '?')}: {exc}")
                    continue
            layer_th.add_to(m)
            if verbose:
                print(f"[plot_speed_density] ✓ {len(turn_hexes)} turn_hex dibujados (v2)")
        except ImportError:
            if verbose:
                print("[plot_speed_density] ⚠ h3 no instalado; "
                      "turn_hexes omitido.")
        except Exception as exc:
            if verbose:
                print(f"[plot_speed_density] ⚠ turn_hexes omitido: {exc}")

    #──────────────────────────────────────────────────────────────────────────#
    # ── 3.9 Agregar paradas como referencia (opcional) ── #
    #──────────────────────────────────────────────────────────────────────────#
    if show_stops and df_stops is not None:
        stops_group = folium.FeatureGroup(name="Paradas", show=True)

        # Usar DivIcon en lugar de CircleMarker para que el tamaño sea fijo
        # en píxeles y no escale con el zoom del mapa.
        _STOP_PX   = stop_radius * 2          # diámetro en píxeles
        _stop_icon_html = (
            f'<div style="'
            f'width:{_STOP_PX}px;height:{_STOP_PX}px;'
            f'border-radius:50%;'
            f'background:#ffffff;'
            f'border:2px solid #333333;'
            f'box-sizing:border-box;'
            f'opacity:0.9;'
            f'"></div>'
        )

        for row in df_stops.iter_rows(named=True):
            lat  = row[stop_lat_col]
            lon  = row[stop_lon_col]
            code = row.get(stop_code_col, "")

            icon = folium.DivIcon(
                html=_stop_icon_html,
                icon_size=(_STOP_PX, _STOP_PX),
                icon_anchor=(_STOP_PX // 2, _STOP_PX // 2),
            )
            folium.Marker(
                location=[lat, lon],
                icon=icon,
                popup=folium.Popup(f"<b>Parada:</b> {code}", max_width=150),
                tooltip=code,
            ).add_to(stops_group)

        stops_group.add_to(m)

        if verbose:
            print(f"[plot_speed_density] ✓ {df_stops.height} paradas graficadas (tamaño fijo {_STOP_PX}px)")

    #──────────────────────────────────────────────────────────────────────────#
    # ── 3.10 Agregar leyenda con gradientes ── #
    #──────────────────────────────────────────────────────────────────────────#
    # Colores de muestra para la leyenda (viridis)
    viridis_start = _interpolate_color(0.0, _VIRIDIS)
    viridis_mid = _interpolate_color(0.5, _VIRIDIS)
    viridis_end = _interpolate_color(1.0, _VIRIDIS)
    
    # Colores de muestra para outliers
    outlier_start = _interpolate_color(0.0, _OUTLIER_GRADIENT)
    outlier_end = _interpolate_color(1.0, _OUTLIER_GRADIENT)

    legend_html = f"""
    <div style="
        position: fixed;
        bottom: 50px;
        left: 50px;
        z-index: 1000;
        background-color: white;
        padding: 12px 15px;
        border-radius: 5px;
        border: 2px solid #ccc;
        font-family: Arial, sans-serif;
        font-size: 12px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.3);
        min-width: 180px;
    ">
        <b style="font-size: 13px;">Densidad de paradas</b><br><br>
        
        <div style="margin-bottom: 8px;">
            <span style="font-size: 11px; color: #666;">Normal ({min_count:.0f} - {outlier_threshold:.0f})</span><br>
            <div style="
                height: 12px;
                width: 100%;
                background: linear-gradient(to right, {viridis_start}, {viridis_mid}, {viridis_end});
                border-radius: 2px;
                margin-top: 3px;
            "></div>
            <div style="display: flex; justify-content: space-between; font-size: 10px; color: #888;">
                <span>{min_count:.0f}</span>
                <span>{outlier_threshold:.0f}</span>
            </div>
        </div>
        
        <div>
            <span style="font-size: 11px; color: #666;">Outliers (>{outlier_threshold:.0f})</span><br>
            <div style="
                height: 12px;
                width: 100%;
                background: linear-gradient(to right, {outlier_start}, {outlier_end});
                border-radius: 2px;
                margin-top: 3px;
            "></div>
            <div style="display: flex; justify-content: space-between; font-size: 10px; color: #888;">
                <span>{outlier_threshold:.0f}</span>
                <span>{max_count:.0f}</span>
            </div>
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # Agregar control de capas
    folium.LayerControl().add_to(m)

    if verbose:
        print(f"[plot_speed_density] Mapa listo.")

    return m


#──────────────────────────────────────────────────────────────────────────────#
# ── 4. Función de diagnóstico rápido ── #
#──────────────────────────────────────────────────────────────────────────────#

def diagnose_stopped_records(
    df_rec: pl.DataFrame,
    speed_col: str = "speed",
    time_col: str = "time",
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Diagnostica registros de vehículos detenidos por hora del día.

    Parameters
    ----------
    df_rec : pl.DataFrame
        DataFrame de registros GPS.
    speed_col : str, optional
        Columna de velocidad. Default: ``"speed"``.
    time_col : str, optional
        Columna de tiempo. Default: ``"time"``.
    verbose : bool, optional
        Imprimir diagnóstico. Default: ``True``.

    Returns
    -------
    pl.DataFrame
        Conteo de registros detenidos por hora.

    Examples
    --------
    >>> df_by_hour = diagnose_stopped_records(df_rec_101_resolved)
    """
    df_stopped = df_rec.filter(pl.col(speed_col) == 0)

    if time_col not in df_rec.columns:
        if verbose:
            n_stopped = df_stopped.height
            n_total = df_rec.height
            print(f"[diagnose_stopped_records] Total: {n_total}, Detenidos: {n_stopped} ({n_stopped/n_total*100:.1f}%)")
        return pl.DataFrame()

    df_by_hour = (
        df_stopped
        .with_columns(pl.col(time_col).dt.hour().alias("hour"))
        .group_by("hour")
        .agg(pl.len().alias("n_stopped"))
        .sort("hour")
    )

    if verbose:
        n_stopped = df_stopped.height
        n_total = df_rec.height
        print(f"[diagnose_stopped_records] ══════════════════════════════════════")
        print(f"[diagnose_stopped_records] Total: {n_total}, Detenidos: {n_stopped} ({n_stopped/n_total*100:.1f}%)")
        print(f"[diagnose_stopped_records] ── Distribución por hora ──")

        max_count = df_by_hour["n_stopped"].max()
        for row in df_by_hour.iter_rows(named=True):
            hour = row["hour"]
            count = row["n_stopped"]
            bar_len = int(count / max_count * 30)
            bar = "█" * bar_len
            print(f"  {hour:02d}:00  {bar} {count}")

    return df_by_hour