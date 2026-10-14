"""Mapa de overview: paradas + polilíneas + records (plot_route_overview).

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

##############################################################################
#   1. plot_route_overview — mapa de paradas + polilíneas + records
#   (originalmente: _lib_plot_overview.py)
##############################################################################

import json
import polars as pl
import numpy as np
import folium
from folium import LayerControl
from typing import Optional


#──────────────────────────────────────────────────────────────────────────────#
# ── 1. Helpers internos ── #
#──────────────────────────────────────────────────────────────────────────────#

def _parse_polyline(raw) -> list:
    """
    Convierte el campo polyline a lista de [lat, lon] válidos.

    Maneja múltiples formatos:
      - Lista de [lat, lon] / (lat, lon)
      - Lista de dicts {"lat": ..., "lon": ...} / {"latitude": ..., "longitude": ...}
      - Lista de structs Polars (subscriptables por key string)
      - Objetos con atributos latitude/longitude
      - JSON strings
      - Python repr strings (fallback)
    """
    import ast

    if raw is None:
        return []

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

    if not isinstance(raw, list):
        try:
            raw = list(raw)
        except Exception:
            return []

    if len(raw) == 0:
        return []

    points = []
    for item in raw:
        # 1. Lista o tupla [lat, lon]
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            try:
                points.append([float(item[0]), float(item[1])])
                continue
            except (TypeError, ValueError, KeyError):
                pass

        # 2. Dict Python
        if isinstance(item, dict):
            lat_key = next((k for k in ["lat", "latitude", "latitud"] if k in item), None)
            lon_key = next((k for k in ["lon", "lng", "longitude", "longitud"] if k in item), None)
            if lat_key and lon_key:
                try:
                    points.append([float(item[lat_key]), float(item[lon_key])])
                    continue
                except (TypeError, ValueError):
                    pass

        # 3. Struct Polars / objeto subscriptable por key string
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


def _centroid(lats: list, lons: list) -> tuple:
    """
    Calcula el centroide de un conjunto de coordenadas.

    Parameters
    ----------
        lats (list): Lista de latitudes.
        lons (list): Lista de longitudes.

    Returns
    -------
        center (tuple): (lat_media, lon_media)
    """
    return (float(np.mean(lats)), float(np.mean(lons)))


def _detect_datetime_col(df: pl.DataFrame) -> Optional[str]:
    """
    Detecta la columna de tiempo en un DataFrame GPS.

    Parameters
    ----------
        df (pl.DataFrame): DataFrame GPS.

    Returns
    -------
        col (str | None): Nombre de la columna de tiempo encontrada, o None.
    """
    for candidate in ["datetime", "timestamp", "fecha_hora", "time", "fecha"]:
        if candidate in df.columns:
            return candidate
    # buscar por dtype
    for col in df.columns:
        if df[col].dtype in (pl.Datetime, pl.Date):
            return col
    return None


def _get_days_sorted(df: pl.DataFrame, time_col: str) -> list:
    """
    Retorna la lista de fechas únicas ordenadas presentes en la columna de tiempo.

    Parameters
    ----------
        df       (pl.DataFrame): DataFrame GPS.
        time_col (str)         : Nombre de la columna temporal.

    Returns
    -------
        days (list): Lista de valores de fecha únicos ordenados.
    """
    col = df[time_col]
    if col.dtype == pl.Datetime:
        dates = df.with_columns(pl.col(time_col).dt.date().alias("_date"))["_date"]
    elif col.dtype == pl.Date:
        dates = col.alias("_date")
    else:
        # intentar cast
        try:
            dates = df.with_columns(
                pl.col(time_col).cast(pl.Datetime).dt.date().alias("_date")
            )["_date"]
        except Exception:
            return []
    return sorted(dates.unique().to_list())


#──────────────────────────────────────────────────────────────────────────────#
# ── 2. Capa de paradas ── #
#──────────────────────────────────────────────────────────────────────────────#

def _add_stop_layer(
    m: folium.Map,
    df_stop: pl.DataFrame,
    verbose: bool = False,
) -> None:
    """
    Agrega capa de paradas al mapa Folium.

    Parameters
    ----------
        m        (folium.Map)  : Mapa al que se agrega la capa.
        df_stop  (pl.DataFrame): DataFrame con columnas latitud, longitud,
                                 codigo_parada y opcionalmente nombre.
        verbose  (bool)        : Si True, imprime resumen de la capa.

    Returns
    -------
        None  (modifica el mapa in-place)
    """
    # ── detectar columnas de coordenadas ──
    lat_col = next((c for c in ["latitud", "latitude", "lat"] if c in df_stop.columns), None)
    lon_col = next((c for c in ["longitud", "longitude", "lon"] if c in df_stop.columns), None)
    cod_col = next((c for c in ["codigo_parada", "codigo", "stop_code", "code"] if c in df_stop.columns), None)
    nom_col = next((c for c in ["nombre", "name", "stop_name"] if c in df_stop.columns), None)

    if lat_col is None or lon_col is None:
        if verbose:
            print("  [Paradas] ⚠  No se encontraron columnas de coordenadas — capa omitida")
        return

    layer = folium.FeatureGroup(name="Paradas", overlay=False, control=False)
    rows = df_stop.to_dicts()

    for row in rows:
        lat = row.get(lat_col)
        lon = row.get(lon_col)
        if lat is None or lon is None:
            continue

        cod  = str(row.get(cod_col, "?")) if cod_col else "?"
        nom  = str(row.get(nom_col, "")) if nom_col else ""
        tip  = cod
        pop  = f"<b>{cod}</b>"
        if nom:
            pop += f"<br>{nom}"
        pop += f"<br>({lat:.5f}, {lon:.5f})"

        folium.CircleMarker(
            location=[lat, lon],
            radius=5,
            color="#e63946",
            fill=True,
            fill_color="#e63946",
            fill_opacity=0.85,
            tooltip=tip,
            popup=folium.Popup(pop, max_width=220),
        ).add_to(layer)

    layer.add_to(m)

    if verbose:
        print(f"  [Paradas] ✔  {len(rows)} paradas agregadas")


#──────────────────────────────────────────────────────────────────────────────#
# ── 3. Capa de segmentos ── #
#──────────────────────────────────────────────────────────────────────────────#

def _extract_polylines_from_df(df_seg: pl.DataFrame, poly_col: str) -> list:
    """
    Extrae polilíneas de una columna Polars que puede ser:
      - list[struct[{lat, lon}]]  → el caso de df_segments_route
      - list[list[float]]         → [[lat, lon], ...]
      - str (JSON)                → parseable con json.loads

    Convierte siempre a lista de [[lat, lon], ...] por segmento.

    Parameters
    ----------
        df_seg   (pl.DataFrame): DataFrame de segmentos.
        poly_col (str)         : Nombre de la columna de geometría.

    Returns
    -------
        all_polylines (list): Lista de N elementos (uno por fila),
                              cada uno es list[[lat, lon]].
    """
    col_dtype = df_seg[poly_col].dtype
    all_polylines = []

    # ── Caso: list[struct[...]] — dtype Polars nativo ──
    # Unnest para obtener coordenadas tabulares, luego reagrupar por fila
    if str(col_dtype).startswith("List(Struct"):
        # explode → unnest → reagrupar por índice de fila
        df_exp = (
            df_seg
            .with_row_index("_row_idx")
            .select(["_row_idx", poly_col])
            .explode(poly_col)
            .unnest(poly_col)
        )
        # detectar nombres de campo lat/lon dentro del struct
        lat_f = next((c for c in ["latitude", "latitud", "lat", "y"] if c in df_exp.columns), None)
        lon_f = next((c for c in ["longitude", "longitud", "lon", "x"] if c in df_exp.columns), None)

        if lat_f and lon_f:
            grouped = (
                df_exp
                .group_by("_row_idx", maintain_order=True)
                .agg([
                    pl.col(lat_f).alias("lats"),
                    pl.col(lon_f).alias("lons"),
                ])
                .sort("_row_idx")
            )
            for row in grouped.to_dicts():
                lats = row["lats"]
                lons = row["lons"]
                points = [[float(la), float(lo)] for la, lo in zip(lats, lons)
                          if la is not None and lo is not None]
                all_polylines.append(points)
        else:
            # struct con campos desconocidos → fallback vacío por fila
            all_polylines = [[] for _ in range(len(df_seg))]

    # ── Caso: cualquier otro tipo → usar _parse_polyline fila a fila ──
    else:
        for val in df_seg[poly_col].to_list():
            all_polylines.append(_parse_polyline(val))

    return all_polylines


def _arrow_bearing(p_from: list, p_to: list) -> float:
    """
    Calcula el ángulo (bearing) en grados desde p_from hacia p_to,
    medido en sentido horario desde el Norte (0° = Norte, 90° = Este).

    Parameters
    ----------
        p_from (list): [lat, lon] del punto origen.
        p_to   (list): [lat, lon] del punto destino.

    Returns
    -------
        bearing (float): Ángulo en grados [0, 360).
    """
    import math
    lat1 = math.radians(p_from[0])
    lat2 = math.radians(p_to[0])
    dlon = math.radians(p_to[1] - p_from[1])
    x    = math.sin(dlon) * math.cos(lat2)
    y    = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    bearing = (math.degrees(math.atan2(x, y)) + 360) % 360
    return bearing


def _midpoint(points: list) -> list:
    """
    Retorna el punto medio de una polilínea (punto más cercano al 50% de la longitud).

    Parameters
    ----------
        points (list): Lista de [lat, lon].

    Returns
    -------
        mid (list): [lat, lon] del punto medio aproximado.
    """
    import math

    def haversine(a, b):
        R = 6_371_000
        la1, lo1 = math.radians(a[0]), math.radians(a[1])
        la2, lo2 = math.radians(b[0]), math.radians(b[1])
        dlat = la2 - la1
        dlon = lo2 - lo1
        h = math.sin(dlat / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin(dlon / 2) ** 2
        return 2 * R * math.asin(math.sqrt(h))

    # acumular longitud total
    dists = [0.0]
    for i in range(1, len(points)):
        dists.append(dists[-1] + haversine(points[i - 1], points[i]))

    half = dists[-1] / 2.0
    # encontrar segmento que contiene la mitad
    for i in range(1, len(dists)):
        if dists[i] >= half:
            # interpolar dentro del segmento
            seg_len = dists[i] - dists[i - 1]
            t = (half - dists[i - 1]) / seg_len if seg_len > 0 else 0
            lat = points[i - 1][0] + t * (points[i][0] - points[i - 1][0])
            lon = points[i - 1][1] + t * (points[i][1] - points[i - 1][1])
            return [lat, lon]

    return points[len(points) // 2]


def _arrow_marker(location: list, bearing: float, color: str, tooltip: str) -> folium.Marker:
    """
    Crea un marcador Folium con un triángulo SVG rotado según el bearing.

    Parameters
    ----------
        location (list) : [lat, lon] donde colocar la flecha.
        bearing  (float): Ángulo de rotación en grados (0 = Norte).
        color    (str)  : Color hexadecimal del triángulo.
        tooltip  (str)  : Texto del tooltip al hacer hover.

    Returns
    -------
        marker (folium.Marker): Marcador listo para agregar al mapa.
    """
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 16 16">'
        f'<polygon points="8,1 15,15 8,11 1,15" '
        f'fill="{color}" stroke="white" stroke-width="1" '
        f'transform="rotate({bearing:.1f}, 8, 8)"/>'
        f'</svg>'
    )
    icon = folium.DivIcon(
        html=svg,
        icon_size=(16, 16),
        icon_anchor=(8, 8),
    )
    return folium.Marker(location=location, icon=icon, tooltip=tooltip)


def _parse_stop_pair(stop_val: str) -> tuple:
    """
    Parsea el campo 'stop' con formato 'COD_INI-COD_FIN' en dos códigos.

    Ejemplos:
      'E5TM-FQAE'  → ('E5TM', 'FQAE')
      'FQAE-PEPM'  → ('FQAE', 'PEPM')

    Parameters
    ----------
        stop_val (str): Valor de la columna stop.

    Returns
    -------
        (cod_ini, cod_fin) (tuple): Par de códigos de parada. ('?', '?') si falla.
    """
    if not isinstance(stop_val, str) or '-' not in stop_val:
        return ('?', '?')
    parts = stop_val.split('-', 1)
    return (parts[0].strip(), parts[1].strip())


def _segment_color_palette(n: int) -> list:
    """
    Genera una paleta de N colores visualmente distinguibles en formato hex,
    usando espacio HSL con hue equiespaciado y saturación/luminosidad fijos.

    Parameters
    ----------
        n (int): Número de colores a generar.

    Returns
    -------
        colors (list[str]): Lista de N strings hex, p.ej. ['#e63946', '#2a9d8f', ...].
    """
    import colorsys
    colors = []
    for i in range(n):
        hue        = i / max(n, 1)          # hue equiespaciado [0, 1)
        r, g, b    = colorsys.hls_to_rgb(hue, 0.45, 0.72)   # L=0.45, S=0.72
        colors.append(f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}")
    return colors


def _add_segment_layer(
    m: folium.Map,
    df_seg: pl.DataFrame,
    route_prefix: Optional[str] = None,
    verbose: bool = False,
) -> None:
    """
    Agrega capa de segmentos al mapa: polilíneas individuales con color
    distinto por segmento, flecha de dirección en el punto medio, y popup
    con el valor del campo 'stop' (siguiente parada del tramo).

    Cada segmento recibe un color único de una paleta HSL equiespaciada,
    lo que permite distinguir visualmente todos los tramos de la ruta.

    El campo 'stop' (p.ej. 'COD_INI-COD_FIN') se usa para:
      - Tooltip de la flecha : PREFIX-COD_INI → PREFIX-COD_FIN
      - Popup de la polilínea: muestra el valor completo de 'stop'
                               (la siguiente parada del tramo)

    Soporta columna polyline en formato:
      - list[struct[{lat, lon}]]  (Polars nativo de df_segments_route)
      - list[[lat, lon]]
      - JSON string

    Parameters
    ----------
        m            (folium.Map)   : Mapa al que se agrega la capa.
        df_seg       (pl.DataFrame) : DataFrame con columnas:
                                        - stop      : str — siguiente parada
                                                      del tramo (o 'COD_INI-COD_FIN')
                                        - polyline  : geometría del tramo
                                      Opcionalmente latitud_ini/longitud_ini/
                                      latitud_fin/longitud_fin como fallback.
        route_prefix (str | None)   : Prefijo de ruta para tooltips de flechas
                                      (p.ej. '101'). Si None, omite el prefijo.
        verbose      (bool)         : Si True, imprime resumen de la capa.

    Returns
    -------
        None  (modifica el mapa in-place)

    Examples
    --------
        # Con prefijo de ruta — cada segmento en color distinto con popup de stop
        _add_segment_layer(m, df_seg_101, route_prefix="101", verbose=True)
    """
    poly_col = next((c for c in ["polyline", "geometry", "coords"] if c in df_seg.columns), None)
    stop_col = next((c for c in ["stop", "tramo", "segment"] if c in df_seg.columns), None)

    # columnas fallback de inicio/fin
    lat_ini_col = next((c for c in ["latitud_ini", "lat_ini", "start_lat"] if c in df_seg.columns), None)
    lon_ini_col = next((c for c in ["longitud_ini", "lon_ini", "start_lon"] if c in df_seg.columns), None)
    lat_fin_col = next((c for c in ["latitud_fin", "lat_fin", "end_lat"] if c in df_seg.columns), None)
    lon_fin_col = next((c for c in ["longitud_fin", "lon_fin", "end_lon"] if c in df_seg.columns), None)

    layer_lines  = folium.FeatureGroup(name="Segmentos", overlay=False, control=False)
    layer_arrows = folium.FeatureGroup(name="Dirección tramos", overlay=False, control=False)

    n_rows = len(df_seg)
    drawn  = 0

    # ── extraer polylines como listas Python [[lat,lon],...] ──
    if poly_col:
        all_polylines = _extract_polylines_from_df(df_seg, poly_col)
    else:
        all_polylines = [[] for _ in range(n_rows)]

    # ── leer stop_col y fallback coords como listas planas ──
    stop_vals     = df_seg[stop_col].to_list() if stop_col else [None] * n_rows
    rows_fallback = None
    if lat_ini_col and lon_ini_col and lat_fin_col and lon_fin_col:
        rows_fallback = df_seg.select(
            [lat_ini_col, lon_ini_col, lat_fin_col, lon_fin_col]
        ).to_dicts()

    # ── paleta de colores distintos: uno por segmento ──
    palette = _segment_color_palette(n_rows)

    if verbose:
        print(f"  [Segmentos] Paleta generada: {n_rows} colores distintos")

    for i, points in enumerate(all_polylines):
        seg_color = palette[i % len(palette)]

        # ── fallback línea recta ──
        if len(points) < 2 and rows_fallback is not None:
            row = rows_fallback[i]
            p0  = row.get(lat_ini_col), row.get(lon_ini_col)
            p1  = row.get(lat_fin_col), row.get(lon_fin_col)
            if all(v is not None for v in [*p0, *p1]):
                points = [list(p0), list(p1)]

        if len(points) < 2:
            continue

        # ── construir etiquetas del tramo ──
        raw_stop = stop_vals[i]
        if raw_stop is not None:
            stop_str     = str(raw_stop)
            cod_ini, cod_fin = _parse_stop_pair(stop_str)
            if route_prefix:
                arrow_tip = f"{route_prefix}-{cod_ini} → {route_prefix}-{cod_fin}"
            else:
                arrow_tip = f"{cod_ini} → {cod_fin}"
            # popup: muestra el valor de stop (siguiente parada del tramo)
            popup_html = (
                f"<b>Siguiente parada</b><br>"
                f"<code style='font-size:13px'>{stop_str}</code><br>"
                f"<span style='color:#555;font-size:11px'>Tramo {i + 1} de {n_rows}</span>"
            )
        else:
            arrow_tip  = f"Tramo {i + 1}"
            popup_html = f"<b>Tramo {i + 1}</b><br><span style='color:#888'>sin código</span>"

        # ── polilínea del tramo con color individual y popup ──
        folium.PolyLine(
            locations=points,
            color=seg_color,
            weight=4,
            opacity=0.85,
            popup=folium.Popup(popup_html, max_width=240),
            tooltip=arrow_tip,
        ).add_to(layer_lines)

        # ── flecha de dirección en el punto medio ──
        mid    = _midpoint(points)
        q1_idx = max(0, len(points) // 4)
        q3_idx = min(len(points) - 1, 3 * len(points) // 4)
        bearing = _arrow_bearing(points[q1_idx], points[q3_idx])

        # flecha con el mismo color del segmento para consistencia visual
        _arrow_marker(mid, bearing, seg_color, arrow_tip).add_to(layer_arrows)
        drawn += 1

    layer_lines.add_to(m)
    layer_arrows.add_to(m)

    if verbose:
        print(f"  [Segmentos] ✔  {drawn}/{n_rows} tramos dibujados"
              f" (color distinto + popup stop)")


#──────────────────────────────────────────────────────────────────────────────#
# ── 4. Capa de registros GPS por mes ── #
#──────────────────────────────────────────────────────────────────────────────#

# Nombres de mes en español para las etiquetas de checkbox
_MONTH_NAMES_ES = {
    1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril",
    5: "Mayo", 6: "Junio", 7: "Julio", 8: "Agosto",
    9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre",
}

# Colores por dirección
_COLOR_DIR_A   = "#1d6fa4"   # azul
_COLOR_DIR_B   = "#f4a261"   # naranja
_COLOR_DIR_UNK = "#8d99ae"   # gris


def _sample_month_boundary_records(
    df_rec: pl.DataFrame,
    time_col: str,
    n_boundary_days: int = 1,
    boundary_hours: float = 0.5,
) -> pl.DataFrame:
    """
    Para cada mes presente en df_rec, selecciona registros de las primeras
    y últimas `boundary_hours` horas de los primeros y últimos `n_boundary_days`
    días del mes.

    Esto produce una muestra representativa del comportamiento de inicio/fin
    de mes sin cargar todos los registros, útil para el mapa overview.

    Parameters
    ----------
        df_rec           : DataFrame GPS con columna datetime.
        time_col         : Nombre de la columna temporal (Datetime).
        n_boundary_days  : Cuántos días del inicio y fin del mes incluir (default 2).
        boundary_hours   : Ventana horaria en horas desde el inicio/fin del día (default 2).

    Returns
    -------
        pl.DataFrame con columna "_month" (int) añadida.
    """
    import datetime as dt

    # Asegurar que time_col sea Datetime
    if df_rec[time_col].dtype != pl.Datetime:
        df_rec = df_rec.with_columns(
            pl.col(time_col).cast(pl.Datetime)
        )

    df_work = df_rec.with_columns([
        pl.col(time_col).dt.year().alias("_year"),
        pl.col(time_col).dt.month().alias("_month"),
        pl.col(time_col).dt.date().alias("_date"),
        pl.col(time_col).dt.hour().cast(pl.Float64).alias("_hour"),
    ])

    months = sorted(df_work.select(["_year", "_month"]).unique().to_dicts(),
                    key=lambda r: (r["_year"], r["_month"]))

    parts = []
    for ym in months:
        year, month = ym["_year"], ym["_month"]
        df_m = df_work.filter(
            (pl.col("_year") == year) & (pl.col("_month") == month)
        )
        days_sorted = sorted(df_m["_date"].unique().to_list())
        if not days_sorted:
            continue

        # Primeros y últimos n_boundary_days días
        boundary_days = set(days_sorted[:n_boundary_days] + days_sorted[-n_boundary_days:])

        df_boundary = df_m.filter(pl.col("_date").is_in(list(boundary_days)))

        # Para cada día de frontera: primeras 2h y últimas 2h
        # Primeras 2h: hora < boundary_hours
        # Últimas 2h: hora >= 24 - boundary_hours
        df_sel = df_boundary.filter(
            (pl.col("_hour") < boundary_hours) |
            (pl.col("_hour") >= (24.0 - boundary_hours))
        )

        # Si quedó vacío (puede pasar si el servicio no opera en esas horas),
        # usar todos los registros de los días de frontera
        if df_sel.is_empty():
            df_sel = df_boundary

        parts.append(df_sel)

    if not parts:
        return df_rec.with_columns(pl.lit(0).alias("_month"))

    return pl.concat(parts).drop(["_year", "_date", "_hour"])


def _add_gps_layers_by_month(
    m: folium.Map,
    df_rec: pl.DataFrame,
    n_boundary_days: int = 1,
    boundary_hours: float = 0.5,
    verbose: bool = False,
) -> None:
    """
    Agrega una capa GPS por mes al mapa Folium.

    Cada mes se convierte en un FeatureGroup independiente (checkbox).
    Solo el primer mes está visible por defecto.

    La muestra por mes incluye las primeras y últimas `boundary_hours` horas
    de los primeros y últimos `n_boundary_days` días del mes, coloreando
    los puntos por dirección (Dir A azul, Dir B naranja, sin dir gris).

    Parameters
    ----------
        m                : Mapa Folium al que se agregan las capas.
        df_rec           : DataFrame GPS limpio.
        n_boundary_days  : Días de frontera por mes (default 2).
        boundary_hours   : Horas de ventana al inicio/fin del día (default 2).
        verbose          : Imprimir resumen.
    """
    lat_col = next((c for c in ["latitude", "latitud", "lat"] if c in df_rec.columns), None)
    lon_col = next((c for c in ["longitude", "longitud", "lon"] if c in df_rec.columns), None)
    dir_col = next((c for c in ["direction", "direccion", "dir"] if c in df_rec.columns), None)
    time_col = _detect_datetime_col(df_rec)

    if lat_col is None or lon_col is None:
        if verbose:
            print("  [GPS] ⚠  No se encontraron columnas de coordenadas — capa omitida")
        return

    if time_col is None:
        if verbose:
            print("  [GPS] ⚠  No se encontró columna temporal — capa omitida")
        return

    # Muestrear registros por mes (fronteras)
    df_sampled = _sample_month_boundary_records(
        df_rec, time_col,
        n_boundary_days=n_boundary_days,
        boundary_hours=boundary_hours,
    )

    # Obtener meses únicos ordenados
    months_present = sorted(df_sampled["_month"].unique().to_list())

    if verbose:
        print(f"  [GPS] Meses detectados: {months_present}")

    for i, month in enumerate(months_present):
        month_name = _MONTH_NAMES_ES.get(month, f"Mes {month}")
        show_by_default = (i == 0)  # solo el primer mes visible al abrir

        df_month = df_sampled.filter(pl.col("_month") == month)

        # Decimar si hay demasiados puntos
        max_pts = 20_000
        n_total = len(df_month)
        if n_total > max_pts:
            step = max(1, n_total // max_pts)
            df_month = df_month[::step]
            if verbose:
                print(f"  [GPS] {month_name}: decimando {n_total:,} → {len(df_month):,} pts")

        layer = folium.FeatureGroup(
            name=f"📅 GPS {month_name}",
            show=show_by_default,
        )

        rows = df_month.to_dicts()
        n_a = n_b = n_unk = 0
        for row in rows:
            lat = row.get(lat_col)
            lon = row.get(lon_col)
            if lat is None or lon is None:
                continue

            direction = row.get(dir_col) if dir_col else None
            if direction == 0:
                color = _COLOR_DIR_A
                n_a += 1
            elif direction == 1:
                color = _COLOR_DIR_B
                n_b += 1
            else:
                color = _COLOR_DIR_UNK
                n_unk += 1

            folium.CircleMarker(
                location=[lat, lon],
                radius=2,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.6,
                weight=0,
            ).add_to(layer)

        layer.add_to(m)

        if verbose:
            print(
                f"  [GPS] {month_name}: {len(rows):,} pts → "
                f"Dir A: {n_a:,} | Dir B: {n_b:,} | Sin dir: {n_unk:,} "
                f"({'visible' if show_by_default else 'oculto'})"
            )


#──────────────────────────────────────────────────────────────────────────────#
# ── 5. Función principal ── #
#──────────────────────────────────────────────────────────────────────────────#

def plot_route_overview(
    df_stop_route: pl.DataFrame,
    df_segments_route: pl.DataFrame,
    df_rec_route_clean: pl.DataFrame,
    route_prefix: Optional[str] = None,
    show_stops: bool = True,
    zoom_start: int = 13,
    n_boundary_days: int = 1,
    boundary_hours: float = 0.5,
    verbose: bool = False,
) -> folium.Map:
    """
    Genera un mapa Folium interactivo con:
      - Segmentos de ruta con dirección (siempre visibles, sin checkbox).
      - Paradas (siempre visibles).
      - Un FeatureGroup (checkbox) por cada mes presente en los datos GPS.
        Solo el primer mes está activo al abrir el mapa.

    Los puntos GPS de cada mes corresponden a las primeras y últimas
    `boundary_hours` horas de los primeros y últimos `n_boundary_days`
    días del mes, coloreados por dirección (azul Dir A, naranja Dir B,
    gris sin dirección).

    Parameters
    ----------
        df_stop_route      (pl.DataFrame): Paradas de la ruta.
        df_segments_route  (pl.DataFrame): Segmentos con polyline y stop.
        df_rec_route_clean (pl.DataFrame): Registros GPS limpios con datetime.
        route_prefix (str | None)        : Prefijo para tooltips de flechas (p.ej. '101').
        show_stops   (bool)              : Mostrar capa de paradas. Default: True.
        zoom_start   (int)               : Zoom inicial. Default: 13.
        n_boundary_days (int)            : Días de frontera por mes. Default: 2.
        boundary_hours  (float)          : Ventana horaria inicio/fin día en horas. Default: 2.0.
        verbose      (bool)              : Imprimir progreso. Default: False.

    Returns
    -------
        m (folium.Map): Mapa interactivo listo para Jupyter o guardar.

    Examples
    --------
        m = plot_route_overview(
            df_stop_route      = df_stop_101,
            df_segments_route  = df_seg_101,
            df_rec_route_clean = df_rec_101_clean,
            route_prefix       = "101",
            verbose            = True,
        )
        m
    """
    #──────────────────────────────────────────────────────────────────────────#
    # ── 5.1 Calcular centroide del mapa ── #
    #──────────────────────────────────────#

    if verbose:
        print("─" * 55)
        print("plot_route_overview — iniciando")
        print("─" * 55)

    lat_col_s = next((c for c in ["latitud", "latitude"] if c in df_stop_route.columns), None)
    lon_col_s = next((c for c in ["longitud", "longitude"] if c in df_stop_route.columns), None)
    lat_col_r = next((c for c in ["latitude", "latitud"] if c in df_rec_route_clean.columns), None)
    lon_col_r = next((c for c in ["longitude", "longitud"] if c in df_rec_route_clean.columns), None)

    if lat_col_s and lon_col_s and len(df_stop_route) > 0:
        lats = df_stop_route[lat_col_s].drop_nulls().to_list()
        lons = df_stop_route[lon_col_s].drop_nulls().to_list()
    elif lat_col_r and lon_col_r and len(df_rec_route_clean) > 0:
        sample = df_rec_route_clean.sample(min(500, len(df_rec_route_clean)))
        lats   = sample[lat_col_r].drop_nulls().to_list()
        lons   = sample[lon_col_r].drop_nulls().to_list()
    else:
        lats, lons = [29.09], [-110.97]  # Hermosillo fallback

    center = _centroid(lats, lons)
    if verbose:
        print(f"  Centro del mapa: {center[0]:.5f}, {center[1]:.5f}")

    #──────────────────────────────────────────────────────────────────────────#
    # ── 5.2 Crear mapa base ── #
    #─────────────────────────#

    m = folium.Map(
        location=center,
        zoom_start=zoom_start,
        tiles="CartoDB positron",
    )

    #──────────────────────────────────────────────────────────────────────────#
    # ── 5.3 Agregar capas ── #
    #────────────────────────#

    # Segmentos y su dirección: siempre visibles (sin checkbox de dir A/B)
    if verbose:
        print("\n[1/3] Segmentos:")
    _add_segment_layer(m, df_segments_route, route_prefix=route_prefix, verbose=verbose)

    # GPS: checkboxes por mes
    if verbose:
        print("\n[2/3] Registros GPS por mes:")
    _add_gps_layers_by_month(
        m, df_rec_route_clean,
        n_boundary_days=n_boundary_days,
        boundary_hours=boundary_hours,
        verbose=verbose,
    )

    # Paradas

    if show_stops:
        if verbose:
            print("\n[3/3] Paradas:")
        _add_stop_layer(m, df_stop_route, verbose=verbose)

    #──────────────────────────────────────────────────────────────────────────#
    # ── 5.4 Control de capas y resumen ── #
    #─────────────────────────────────────#

    LayerControl(collapsed=False).add_to(m)

    if verbose:
        print("\n" + "─" * 55)
        print("✔  Mapa generado correctamente")
        print("─" * 55)

    return m

