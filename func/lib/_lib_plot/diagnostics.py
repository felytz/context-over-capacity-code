"""Diagnósticos de geometría: orden de segmentos + snap de puntos a la polilínea.

Submódulo de `_lib_plot`. Dos preguntas que estos mapas responden:

  1. ¿Los **segmentos** de la ruta están en el **orden correcto**?
     `plot_segment_order` colorea cada segmento con una **escala secuencial**
     (inicio→fin). Si el orden es correcto, el degradado fluye de forma continua
     a lo largo de la ruta; si un color "salta" a otra parte del mapa, ese
     segmento está fuera de orden. Además detecta y marca las discontinuidades
     (gap entre el fin de un segmento y el inicio del siguiente).

  2. ¿Los puntos GPS se están **snapeando bien** a la polilínea?
     `plot_snap_diagnostic` proyecta una muestra de registros sobre el segmento
     más cercano (la MISMA lógica de proyección que usa `enrich_trip_segments`,
     "snap points to polyline") y dibuja, por punto, la línea original→snap
     coloreada por la **distancia perpendicular**. Distancias grandes = el punto
     no es consistente con la polilínea (mala geometría, ruta equivocada, o ruido
     GPS que el filtro de consistencia debería descartar).

`run_snap_segment_diagnostics` es el wrapper de notebook: carga (o recibe) los
segmentos y una muestra de registros limpios de la ruta configurada, genera los
dos mapas, los guarda en `PATHS.maps_dir` y devuelve un dict con todo.

Importar desde el paquete: `from _lib_plot import (plot_segment_order,
plot_snap_diagnostic, run_snap_segment_diagnostics)`.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl
import folium

from _lib_geo import _haversine_m
from .overview import _extract_polylines_from_df, _midpoint, _parse_stop_pair

__all__ = [
    "plot_segment_order",
    "plot_snap_diagnostic",
    "run_snap_segment_diagnostics",
]


# ──────────────────────────────────────────────────────────────────────────#
# ── Helpers de color (escala secuencial) ── #
# ──────────────────────────────────────────────────────────────────────────#

def _get_cmap(name: str):
    """Devuelve un colormap de matplotlib de forma robusta entre versiones."""
    try:
        import matplotlib
        return matplotlib.colormaps[name]
    except Exception:
        import matplotlib.cm as cm
        try:
            return cm.get_cmap(name)
        except Exception:
            return cm.get_cmap("viridis")


def _scale_colors(n: int, cmap_name: str = "turbo") -> List[str]:
    """N colores hex equiespaciados sobre un colormap secuencial (orden 0→n-1)."""
    import matplotlib.colors as mcolors
    cmap = _get_cmap(cmap_name)
    if n <= 1:
        return [mcolors.to_hex(cmap(0.0))]
    return [mcolors.to_hex(cmap(i / (n - 1))) for i in range(n)]


def _color_for_value(v: float, vmax: float, cmap_name: str = "RdYlGn_r") -> str:
    """Color hex para un valor escalar en [0, vmax] (clampeado)."""
    import matplotlib.colors as mcolors
    cmap = _get_cmap(cmap_name)
    t = 0.0 if vmax <= 0 else max(0.0, min(1.0, v / vmax))
    return mcolors.to_hex(cmap(t))


def _gradient_legend_html(
    title: str,
    cmap_name: str,
    left_label: str,
    right_label: str,
    n_stops: int = 24,
) -> str:
    """Barra de color HTML (degradado) para la leyenda del mapa."""
    cols = _scale_colors(n_stops, cmap_name)
    stops = ", ".join(cols)
    return f"""
    <div style="position:fixed; bottom:30px; left:30px; z-index:9999;
                background:white; padding:10px 14px; border-radius:8px;
                border:1px solid #ccc; font-size:12px; font-family:monospace;">
      <b>{title}</b><br>
      <div style="height:12px; width:220px; margin:6px 0;
                  background:linear-gradient(to right, {stops});
                  border:1px solid #999;"></div>
      <div style="display:flex; justify-content:space-between; width:220px;">
        <span>{left_label}</span><span>{right_label}</span>
      </div>
    </div>
    """


# ──────────────────────────────────────────────────────────────────────────#
# ── Geometría de segmentos + proyección (snap) ── #
# ──────────────────────────────────────────────────────────────────────────#

def _cum_dists(pts: List[List[float]]) -> np.ndarray:
    """Distancias acumuladas (m) a lo largo de una polilínea [[lat, lon], ...]."""
    cum = [0.0]
    for i in range(1, len(pts)):
        cum.append(cum[-1] + _haversine_m(pts[i-1][0], pts[i-1][1],
                                           pts[i][0], pts[i][1]))
    return np.asarray(cum)


def _prepare_segments(
    df_segments_route: pl.DataFrame,
) -> List[Dict]:
    """Parsea df_segments_route a una lista ordenada de dicts con geometría.

    El **orden de fila** de df_segments_route ES el orden a lo largo de la ruta
    (lo fija `step_01_load_data` desde segments_route.json). Cada dict trae:
        idx, stop, pts ([[lat,lon],...]), cum, length_m.
    """
    poly_col = next(
        (c for c in ["polyline", "geometry", "coords"] if c in df_segments_route.columns),
        None,
    )
    stop_col = next(
        (c for c in ["stop", "codigo", "code", "parada", "tramo", "segment"]
         if c in df_segments_route.columns),
        None,
    )
    if poly_col is None:
        raise ValueError(
            "df_segments_route debe tener columna 'polyline'. "
            f"Columnas: {df_segments_route.columns}"
        )

    all_polys = _extract_polylines_from_df(df_segments_route, poly_col)
    stop_vals = (df_segments_route[stop_col].to_list()
                 if stop_col else [None] * len(all_polys))

    segs: List[Dict] = []
    for i, pts in enumerate(all_polys):
        if pts is None or len(pts) < 2:
            continue
        cum = _cum_dists(pts)
        segs.append({
            "idx":      i,
            "stop":     str(stop_vals[i]) if stop_vals[i] is not None else f"seg_{i}",
            "pts":      pts,
            "cum":      cum,
            "length_m": float(cum[-1]),
        })
    return segs


def _project_to_segment(
    p_lat: float, p_lon: float,
    pts: List[List[float]], cum: np.ndarray,
) -> Tuple[float, float, float]:
    """Proyecta (p_lat, p_lon) sobre una polilínea.

    Misma matemática que `_lib_features.enrich._project_point` (aproximación
    planar con cos_mid), pero devuelve también las coordenadas del snap para
    poder dibujarlo.

    Returns
    -------
    (snap_lat, snap_lon, perp_m)
    """
    best_perp = math.inf
    best_lat, best_lon = pts[0][0], pts[0][1]

    for i in range(len(pts) - 1):
        la1, lo1 = pts[i]
        la2, lo2 = pts[i + 1]
        if cum[i + 1] - cum[i] < 1e-9:
            continue

        cos_mid = math.cos(math.radians((la1 + la2) / 2.0))
        dx_s = (lo2 - lo1) * cos_mid
        dy_s = la2 - la1
        seg_sq = dx_s ** 2 + dy_s ** 2
        if seg_sq < 1e-20:
            continue

        dx_p = (p_lon - lo1) * cos_mid
        dy_p = p_lat - la1
        t = max(0.0, min(1.0, (dx_p * dx_s + dy_p * dy_s) / seg_sq))

        proj_lat = la1 + t * (la2 - la1)
        proj_lon = lo1 + t * (lo2 - lo1)
        perp = math.sqrt(
            ((p_lon - proj_lon) * cos_mid * 111_320) ** 2
            + ((p_lat - proj_lat) * 110_540) ** 2
        )
        if perp < best_perp:
            best_perp = perp
            best_lat, best_lon = proj_lat, proj_lon

    return best_lat, best_lon, best_perp


def _snap_to_nearest_segment(
    p_lat: float, p_lon: float,
    segs: List[Dict],
) -> Tuple[float, float, float, Dict]:
    """Asigna el punto al segmento más cercano (igual que enrich_trip_segments).

    Returns
    -------
    (snap_lat, snap_lon, perp_m, seg)  — `seg` es el dict del segmento ganador.
    """
    best = (p_lat, p_lon, math.inf, segs[0])
    for seg in segs:
        s_lat, s_lon, perp = _project_to_segment(p_lat, p_lon, seg["pts"], seg["cum"])
        if perp < best[2]:
            best = (s_lat, s_lon, perp, seg)
    return best


# ──────────────────────────────────────────────────────────────────────────#
# ── 1. Mapa: orden de segmentos (escala de color) ── #
# ──────────────────────────────────────────────────────────────────────────#

def plot_segment_order(
    df_segments_route: pl.DataFrame,
    route_prefix: Optional[str] = None,
    cmap: str = "turbo",
    show_labels: bool = True,
    flag_gap_m: float = 150.0,
    zoom_start: int = 13,
    return_gaps: bool = False,
    verbose: bool = False,
):
    """Mapa de los segmentos de la ruta coloreados por **orden** (escala secuencial).

    Cada segmento recibe un color de un colormap secuencial según su índice de
    fila (= su orden a lo largo de la ruta). Verificación visual:

      - **Degradado continuo** que recorre la ruta de punta a punta → orden OK.
      - **Color que "salta"** a otra zona del mapa → ese segmento está fuera de
        orden (o la lista de orden está mal armada).

    También calcula, entre cada par de segmentos consecutivos (i, i+1), el "gap"
    entre el último punto de i y el primer punto de i+1. Un gap grande
    (> `flag_gap_m`) sugiere un salto/desorden y se marca en rojo con una línea
    punteada.

    Parameters
    ----------
        df_segments_route : pl.DataFrame  Segmentos en orden de ruta (cols: stop, polyline).
        route_prefix      : str | None    Prefijo para tooltips (p.ej. '1701').
        cmap              : str           Colormap secuencial (default 'turbo').
        show_labels       : bool          Etiqueta numérica del orden en cada segmento.
        flag_gap_m        : float         Umbral (m) para marcar discontinuidades.
        zoom_start        : int           Zoom inicial.
        return_gaps       : bool          Si True devuelve (mapa, df_gaps).
        verbose           : bool          Imprime el reporte de gaps.

    Returns
    -------
        folium.Map  (o (folium.Map, pl.DataFrame) si return_gaps=True)
    """
    segs = _prepare_segments(df_segments_route)
    n = len(segs)
    if n == 0:
        raise ValueError("df_segments_route no tiene segmentos con geometría válida.")

    colors = _scale_colors(n, cmap)

    # Centro del mapa = promedio de todos los vértices
    all_lat = [p[0] for s in segs for p in s["pts"]]
    all_lon = [p[1] for s in segs for p in s["pts"]]
    center = (float(np.mean(all_lat)), float(np.mean(all_lon)))

    m = folium.Map(location=center, zoom_start=zoom_start, tiles="CartoDB positron")

    layer_seg   = folium.FeatureGroup(name="Segmentos (orden)", show=True)
    layer_lbl   = folium.FeatureGroup(name="Etiquetas de orden", show=show_labels)
    layer_gaps  = folium.FeatureGroup(name="Discontinuidades", show=True)

    gap_rows: List[Dict] = []

    for k, seg in enumerate(segs):
        pts   = seg["pts"]
        color = colors[k]
        cod_ini, cod_fin = _parse_stop_pair(seg["stop"])
        pref = f"{route_prefix}-" if route_prefix else ""
        tip = (f"Orden {k} / {n - 1}<br>"
               f"stop: {seg['stop']}<br>"
               f"{pref}{cod_ini} → {pref}{cod_fin}<br>"
               f"largo: {seg['length_m']:.0f} m")

        folium.PolyLine(
            locations=pts,
            color=color,
            weight=5,
            opacity=0.9,
            tooltip=tip,
        ).add_to(layer_seg)

        # Marcador de inicio del segmento (vértice 0)
        folium.CircleMarker(
            location=pts[0], radius=3, color=color, fill=True,
            fill_color=color, fill_opacity=0.9, weight=1, tooltip=tip,
        ).add_to(layer_seg)

        # Etiqueta numérica de orden en el punto medio
        if show_labels:
            mid = _midpoint(pts)
            folium.Marker(
                location=mid,
                icon=folium.DivIcon(
                    icon_size=(22, 22), icon_anchor=(11, 11),
                    html=(f'<div style="font-size:10px;font-weight:bold;'
                          f'color:#111;background:rgba(255,255,255,0.75);'
                          f'border:1px solid {color};border-radius:9px;'
                          f'width:20px;height:14px;line-height:14px;'
                          f'text-align:center;">{k}</div>'),
                ),
                tooltip=tip,
            ).add_to(layer_lbl)

        # Gap con el siguiente segmento
        if k < n - 1:
            p_end   = seg["pts"][-1]
            p_start = segs[k + 1]["pts"][0]
            gap_m   = _haversine_m(p_end[0], p_end[1], p_start[0], p_start[1])
            gap_rows.append({
                "from_idx": k, "to_idx": k + 1,
                "from_stop": seg["stop"], "to_stop": segs[k + 1]["stop"],
                "gap_m": round(gap_m, 1),
            })
            if gap_m > flag_gap_m:
                folium.PolyLine(
                    [p_end, p_start], color="#000000", weight=2,
                    opacity=0.9, dash_array="3 6",
                    tooltip=f"GAP {gap_m:.0f} m  (seg {k} → {k + 1})",
                ).add_to(layer_gaps)
                folium.CircleMarker(
                    location=p_end, radius=6, color="#000000", fill=True,
                    fill_color="#ff0000", fill_opacity=0.85, weight=1,
                    tooltip=f"Fin seg {k} · salta {gap_m:.0f} m a seg {k + 1}",
                ).add_to(layer_gaps)

    # Inicio (verde) y fin (rojo) de TODA la ruta para orientar
    folium.Marker(
        location=segs[0]["pts"][0],
        icon=folium.Icon(color="green", icon="play", prefix="fa"),
        tooltip="Inicio de la ruta (orden 0)",
    ).add_to(m)
    folium.Marker(
        location=segs[-1]["pts"][-1],
        icon=folium.Icon(color="red", icon="stop", prefix="fa"),
        tooltip=f"Fin de la ruta (orden {n - 1})",
    ).add_to(m)

    layer_seg.add_to(m)
    layer_lbl.add_to(m)
    layer_gaps.add_to(m)

    m.get_root().html.add_child(folium.Element(
        _gradient_legend_html(
            "Orden del segmento", cmap,
            left_label="inicio (0)", right_label=f"fin ({n - 1})",
        )
    ))
    folium.LayerControl(collapsed=False).add_to(m)

    df_gaps = pl.DataFrame(gap_rows) if gap_rows else pl.DataFrame(
        schema={"from_idx": pl.Int64, "to_idx": pl.Int64,
                "from_stop": pl.Utf8, "to_stop": pl.Utf8, "gap_m": pl.Float64})

    if verbose:
        big = df_gaps.filter(pl.col("gap_m") > flag_gap_m) if df_gaps.height else df_gaps
        print("=" * 60)
        print("  DIAGNÓSTICO · ORDEN DE SEGMENTOS")
        print("=" * 60)
        print(f"  Segmentos                 : {n}")
        if df_gaps.height:
            gaps = df_gaps["gap_m"].to_numpy()
            print(f"  Gap entre consecutivos (m): mediana {np.median(gaps):.1f}  "
                  f"P95 {np.percentile(gaps, 95):.1f}  max {np.max(gaps):.1f}")
        print(f"  Discontinuidades > {flag_gap_m:.0f} m  : {big.height}")
        for r in big.sort("gap_m", descending=True).head(10).iter_rows(named=True):
            print(f"     seg {r['from_idx']:>3} → {r['to_idx']:<3}  "
                  f"{r['gap_m']:>8.0f} m   ({r['from_stop']} → {r['to_stop']})")
        if big.height == 0:
            print("  ✔ Sin saltos grandes: los segmentos encadenan en orden.")
        print("=" * 60)

    if return_gaps:
        return m, df_gaps
    return m


# ──────────────────────────────────────────────────────────────────────────#
# ── 2. Mapa: snap de puntos a la polilínea (calidad del snapping) ── #
# ──────────────────────────────────────────────────────────────────────────#

def plot_snap_diagnostic(
    df_rec_clean: pl.DataFrame,
    df_segments_route: pl.DataFrame,
    route_prefix: Optional[str] = None,
    sample_n: int = 1500,
    bus_code: Optional[str] = None,
    max_snap_m: float = 120.0,
    seed: int = 0,
    zoom_start: int = 13,
    verbose: bool = False,
) -> folium.Map:
    """Mapa diagnóstico del **snap de puntos GPS a la polilínea**.

    Proyecta una muestra de registros sobre el segmento más cercano (la misma
    lógica que `enrich_trip_segments`, "snap points to polyline") y dibuja, por
    punto:

      - el punto GPS original (círculo gris hueco),
      - el punto snapeado sobre la ruta (relleno, coloreado por distancia
        perpendicular: verde = cerca/bien, rojo = lejos/mal),
      - el segmento gris que une ambos (desplazamiento del snap).

    La distancia perpendicular es la señal del **filtro de consistencia en
    polilíneas**: puntos con distancia grande no encajan con la geometría y son
    los que el filtro debería marcar/descartar. Si muchos puntos snapean lejos,
    la polilínea/segmentos probablemente no corresponden a estos datos.

    Parameters
    ----------
        df_rec_clean      : pl.DataFrame  Registros limpios (latitude, longitude,
                                          time, bus_code).
        df_segments_route : pl.DataFrame  Segmentos en orden (stop, polyline).
        route_prefix      : str | None    Prefijo para tooltips.
        sample_n          : int           Nº de puntos a muestrear (default 1500).
        bus_code          : str | None    Si se da, limita a ese bus.
        max_snap_m        : float         Distancia que satura el rojo en la escala.
        seed              : int           Semilla del muestreo.
        zoom_start        : int           Zoom inicial.
        verbose           : bool          Imprime estadísticas de snap.

    Returns
    -------
        folium.Map
    """
    segs = _prepare_segments(df_segments_route)
    if len(segs) == 0:
        raise ValueError("df_segments_route no tiene segmentos con geometría válida.")

    lat_col = next((c for c in ["latitude", "latitud", "lat"] if c in df_rec_clean.columns), None)
    lon_col = next((c for c in ["longitude", "longitud", "lon"] if c in df_rec_clean.columns), None)
    if lat_col is None or lon_col is None:
        raise ValueError("df_rec_clean debe tener columnas de latitud/longitud.")

    df = df_rec_clean
    if bus_code is not None and "bus_code" in df.columns:
        df = df.filter(pl.col("bus_code") == bus_code)
    df = df.drop_nulls([lat_col, lon_col])

    n_avail = df.height
    if n_avail == 0:
        raise ValueError("No hay registros con coordenadas para diagnosticar.")
    if n_avail > sample_n:
        df = df.sample(sample_n, seed=seed)

    lats = df[lat_col].cast(pl.Float64).to_numpy()
    lons = df[lon_col].cast(pl.Float64).to_numpy()
    times = df["time"].to_list() if "time" in df.columns else [None] * len(lats)
    buses = df["bus_code"].to_list() if "bus_code" in df.columns else [None] * len(lats)

    # Proyectar cada punto al segmento más cercano
    snap_lat = np.zeros(len(lats))
    snap_lon = np.zeros(len(lats))
    perp_m   = np.zeros(len(lats))
    seg_idx  = np.zeros(len(lats), dtype=int)
    seg_stop = [""] * len(lats)
    for i in range(len(lats)):
        s_lat, s_lon, perp, seg = _snap_to_nearest_segment(lats[i], lons[i], segs)
        snap_lat[i] = s_lat
        snap_lon[i] = s_lon
        perp_m[i]   = perp
        seg_idx[i]  = seg["idx"]
        seg_stop[i] = seg["stop"]

    center = (float(np.mean(snap_lat)), float(np.mean(snap_lon)))
    m = folium.Map(location=center, zoom_start=zoom_start, tiles="CartoDB positron")

    # Polilínea de fondo (todos los segmentos, gris)
    layer_route = folium.FeatureGroup(name="Polilínea de ruta", show=True)
    for seg in segs:
        folium.PolyLine(seg["pts"], color="#9aa0a6", weight=3, opacity=0.55).add_to(layer_route)
    layer_route.add_to(m)

    layer_snap = folium.FeatureGroup(name="Snap de puntos", show=True)
    for i in range(len(lats)):
        color = _color_for_value(perp_m[i], max_snap_m, "RdYlGn_r")
        pref = f"{route_prefix}-" if route_prefix else ""
        tip = (
            f"bus: {buses[i]}<br>"
            f"snap_dist: {perp_m[i]:.1f} m<br>"
            f"segmento orden: {seg_idx[i]}<br>"
            f"next_stop: {pref}{_parse_stop_pair(seg_stop[i])[1]}"
            + (f"<br>hora: {times[i]}" if times[i] is not None else "")
        )

        folium.PolyLine(
            [(lats[i], lons[i]), (snap_lat[i], snap_lon[i])],
            color="#888888", weight=1, opacity=0.6, dash_array="3 5",
            tooltip=tip,
        ).add_to(layer_snap)
        folium.CircleMarker(
            location=(lats[i], lons[i]), radius=2, color="#777777",
            fill=True, fill_color="#ffffff", fill_opacity=0.5, weight=1,
            tooltip=tip,
        ).add_to(layer_snap)
        folium.CircleMarker(
            location=(snap_lat[i], snap_lon[i]), radius=4, color=color,
            fill=True, fill_color=color, fill_opacity=0.9, weight=1,
            tooltip=tip,
        ).add_to(layer_snap)
    layer_snap.add_to(m)

    m.get_root().html.add_child(folium.Element(
        _gradient_legend_html(
            "Distancia de snap (perp.)", "RdYlGn_r",
            left_label="0 m", right_label=f"≥ {max_snap_m:.0f} m",
        )
    ))
    folium.LayerControl(collapsed=False).add_to(m)

    if verbose:
        med = float(np.median(perp_m))
        p95 = float(np.percentile(perp_m, 95))
        mx  = float(np.max(perp_m))
        n30  = int(np.sum(perp_m > 30))
        n60  = int(np.sum(perp_m > 60))
        nmax = int(np.sum(perp_m > max_snap_m))
        N = len(perp_m)
        print("=" * 60)
        print("  DIAGNÓSTICO · SNAP DE PUNTOS A LA POLILÍNEA")
        print("=" * 60)
        print(f"  Puntos muestreados        : {N:,} (de {n_avail:,})"
              + (f"  · bus {bus_code}" if bus_code else ""))
        print(f"  Distancia perpendicular   : mediana {med:.1f} m  "
              f"P95 {p95:.1f} m  max {mx:.1f} m")
        print(f"  > 30 m  : {n30:,} ({n30/N*100:.1f}%)")
        print(f"  > 60 m  : {n60:,} ({n60/N*100:.1f}%)")
        print(f"  > {max_snap_m:.0f} m : {nmax:,} ({nmax/N*100:.1f}%)  ← candidatos a descartar")
        if med < 25 and p95 < 80:
            print("  ✔ Snap consistente: la mayoría de puntos cae sobre la polilínea.")
        else:
            print("  ⚠ Snap disperso: revisa que la polilínea/segmentos sean de esta ruta.")
        print("=" * 60)

    return m


# ──────────────────────────────────────────────────────────────────────────#
# ── 3. Wrapper de notebook ── #
# ──────────────────────────────────────────────────────────────────────────#

def run_snap_segment_diagnostics(
    df_rec_clean: Optional[pl.DataFrame] = None,
    df_segments_route: Optional[pl.DataFrame] = None,
    route_prefix: Optional[str] = None,
    sample_n: int = 1500,
    bus_code: Optional[str] = None,
    flag_gap_m: float = 150.0,
    max_snap_m: float = 120.0,
    cmap: str = "turbo",
    save: bool = True,
    verbose: bool = True,
) -> Dict:
    """Genera los dos mapas de diagnóstico para la ruta **configurada** y los guarda.

    Si no se pasan los DataFrames, los carga de la ruta activa (`config`):
    segmentos vía `step_01_load_data.load_segments` y registros limpios vía
    `load_records_clean` (usa el cache si existe).

    Returns
    -------
    dict con: 'map_order', 'map_snap', 'gaps' (pl.DataFrame), y si save=True
    'path_order' / 'path_snap' (rutas de los HTML guardados en PATHS.maps_dir).
    """
    import config
    from config import PATHS

    if route_prefix is None:
        route_prefix = str(getattr(config, "INT_ROUTE", "") or "") or None

    if df_segments_route is None:
        from step_01_load_data import load_segments
        df_segments_route = load_segments(verbose=verbose)
    if df_rec_clean is None:
        from step_01_load_data import load_records_clean
        df_rec_clean = load_records_clean(verbose=verbose)

    m_order, df_gaps = plot_segment_order(
        df_segments_route, route_prefix=route_prefix, cmap=cmap,
        flag_gap_m=flag_gap_m, return_gaps=True, verbose=verbose,
    )
    m_snap = plot_snap_diagnostic(
        df_rec_clean, df_segments_route, route_prefix=route_prefix,
        sample_n=sample_n, bus_code=bus_code, max_snap_m=max_snap_m,
        verbose=verbose,
    )

    result: Dict = {"map_order": m_order, "map_snap": m_snap, "gaps": df_gaps}

    if save:
        PATHS.ensure_dirs()
        suffix = getattr(PATHS, "analysis_range", "") or ""
        suffix = f"_{suffix}" if suffix else ""
        p_order = PATHS.maps_dir / f"diag_segment_order{suffix}.html"
        p_snap  = PATHS.maps_dir / f"diag_snap_to_polyline{suffix}.html"
        m_order.save(str(p_order))
        m_snap.save(str(p_snap))
        result["path_order"] = p_order
        result["path_snap"]  = p_snap
        if verbose:
            print(f"\n  ✔ Mapas guardados en {PATHS.maps_dir}:")
            print(f"     file:///{p_order.resolve().as_posix()}")
            print(f"     file:///{p_snap.resolve().as_posix()}")

    return result
