"""Features físicas por segmento (link_travel_time, dwell, etc.; camino2).

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
#   3. compute_alternative_features — features físicas por segmento
#   (originalmente: camino2_alternative_features.py)
##############################################################################

#################
### Libraries ###
#################

import math
import numpy as np
import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from typing import Optional, List, Tuple, Dict


###############
### Helpers ###
###############

# _parse_polyline_struct: ver import al inicio del módulo (consolidado en lib/_lib_geo.py)


# _haversine_m: ver import al inicio del módulo (consolidado en lib/_lib_geo.py)


def _segment_lengths_from_geometry(df_segments_route: pl.DataFrame) -> Dict[str, float]:
    """Devuelve dict {stop_code: length_m} a partir de la polilínea."""
    stop_col = next(
        (c for c in ["stop", "codigo", "code", "parada"]
         if c in df_segments_route.columns),
        None
    )
    poly_col = next(
        (c for c in ["polyline", "geometry", "coords", "puntos"]
         if c in df_segments_route.columns),
        None
    )
    if stop_col is None or poly_col is None:
        return {}

    lengths: Dict[str, float] = {}
    for row in df_segments_route.select([stop_col, poly_col]).iter_rows(named=True):
        pts = _parse_polyline_struct(row[poly_col])
        if len(pts) < 2:
            continue
        L = 0.0
        for i in range(1, len(pts)):
            L += _haversine_m(pts[i-1][0], pts[i-1][1], pts[i][0], pts[i][1])
        lengths[str(row[stop_col])] = L
    return lengths


#──────────────────────────────────────────────────────────────────────────#
# ── 1. Cálculo de features ── #
#──────────────────────────────────────────────────────────────────────────#

def compute_alternative_features(
    df_enriched: pl.DataFrame,
    df_segments_route: Optional[pl.DataFrame] = None,
    route_prefix: Optional[str] = None,
    speed_zero_threshold: float = 2.0,
    verbose: bool = False,
) -> pl.DataFrame:
    """
    Agrega columnas alternativas a `s_next_stop` que son invariantes (o
    sustancialmente menos sensibles) al periodo de muestreo del AVL.

    Columnas generadas:
        link_travel_time      (float, s)   : duración total del segmento en este viaje
        segment_length_m      (float, m)   : longitud geométrica del segmento
        link_space_mean_speed (float, km/h): segment_length / link_travel_time
        dwell_time            (float, s)   : tiempo con speed < threshold cerca del fin del segmento
                                             (cuantizado al período GPS ~60s)
        dwell_time_interp     (float, s)   : ídem pero interpolado al cruce del umbral de
                                             velocidad asumiendo variación lineal entre pings
                                             (sin cuantización; análogo a s_next_stop_interp).
        time_in_segment       (float, s)   : t - t_entrada_al_segmento
        progress_in_segment   (float, 0..1): 1 - m_next_stop / segment_length_m
        time_since_trip_start (float, s)
        time_to_trip_end      (float, s)
        hour_of_day           (int, 0..23)
        day_of_week           (int, 0..6)

    Parameters
    ----------
        df_enriched         : pl.DataFrame   Salida de enrich_trip_segments
        df_segments_route   : pl.DataFrame   Geometría de segmentos (opcional pero
                                             recomendada para segment_length_m).
        route_prefix        : str | None     Prefijo aplicado en next_stop (ej. "101").
                                             Si se pasa, se usa para mapear los códigos.
        speed_zero_threshold: float          km/h por debajo del cual se considera
                                             "detenido" para calcular dwell_time.
        verbose             : bool

    Returns
    -------
        df_out (pl.DataFrame)
    """
    required = {"bus_code", "trip_id", "time", "next_stop", "m_next_stop"}
    missing = required - set(df_enriched.columns)
    if missing:
        raise ValueError(f"Columnas faltantes en df_enriched: {missing}")

    if verbose:
        print("=" * 60)
        print("  CAMINO 2 — Features alternativas")
        print("=" * 60)
        print(f"  Registros entrada : {df_enriched.height:,}")

    df_sorted = df_enriched.sort(["bus_code", "trip_id", "time"])

    # ── 1. Longitudes geométricas de segmentos ──
    seg_lengths = {}
    if df_segments_route is not None:
        raw_lengths = _segment_lengths_from_geometry(df_segments_route)
        for k, v in raw_lengths.items():
            full_key = f"{route_prefix}-{k}" if route_prefix else k
            seg_lengths[full_key] = v
        if verbose:
            print(f"  Segmentos con longitud: {len(seg_lengths)}")

    # ── 2. Calcular link_travel_time por (trip_id, next_stop) ──
    times = df_sorted["time"].to_list()
    buses = df_sorted["bus_code"].to_list()
    trips = df_sorted["trip_id"].to_list()
    nstop = df_sorted["next_stop"].to_list()
    n = len(times)

    # tiempo de entrada al segmento (= tiempo del primer registro con ese next_stop
    # dentro del viaje) y de salida (= tiempo del primer registro con un next_stop diferente)
    t_enter = [None] * n
    t_exit  = [None] * n

    i = 0
    while i < n:
        # bloque del mismo bus+viaje
        j = i
        while j < n and buses[j] == buses[i] and trips[j] == trips[i]:
            j += 1
        # Dentro del bloque, encontrar sub-bloques de next_stop constante
        k = i
        while k < j:
            l = k
            while l < j and nstop[l] == nstop[k]:
                l += 1
            # Sub-bloque [k, l) tiene next_stop = nstop[k]
            t_in = times[k]
            t_out = times[l] if l < j else times[l - 1]
            for r in range(k, l):
                t_enter[r] = t_in
                t_exit[r]  = t_out
            k = l
        i = j

    # ── 3. Construir columnas derivadas ──
    link_tt          = np.zeros(n, dtype=np.float64)
    time_in_seg      = np.zeros(n, dtype=np.float64)
    seg_len          = np.full(n, np.nan, dtype=np.float64)
    progress         = np.full(n, np.nan, dtype=np.float64)
    space_mean_speed = np.full(n, np.nan, dtype=np.float64)

    mnext = df_sorted["m_next_stop"].cast(pl.Float64).to_numpy()

    for r in range(n):
        link_tt[r]     = (t_exit[r] - t_enter[r]).total_seconds()
        time_in_seg[r] = (times[r]  - t_enter[r]).total_seconds()
        L = seg_lengths.get(nstop[r])
        if L is not None and L > 0:
            seg_len[r]  = L
            progress[r] = max(0.0, min(1.0, 1.0 - mnext[r] / L))
            if link_tt[r] > 1.0:
                space_mean_speed[r] = (L / link_tt[r]) * 3.6   # m/s → km/h

    # ── 4. dwell_time + dwell_time_interp por (trip_id, next_stop) ──
    #
    # dwell_time        : acumulación clásica — suma de Δt entre pings
    #                     consecutivos mientras speed[rr] < umbral.
    #                     Cuantizado al período de muestreo GPS (~60 s).
    #
    # dwell_time_interp : versión interpolada — en los intervalos de
    #                     transición (bus entra o sale del estado parado)
    #                     se estima la fracción exacta del intervalo que
    #                     el bus pasó por debajo del umbral, asumiendo que
    #                     la velocidad varía linealmente entre pings.
    #                     Análogo a s_next_stop_interp para link_TT.
    #
    # Casos por intervalo (rr → rr+1):
    #   both_stopped  : s0 < thr  AND s1 < thr  → todo el Δt cuenta
    #   enter_stopped : s0 >= thr AND s1 < thr  → fracción (1-alpha) de Δt
    #   exit_stopped  : s0 < thr  AND s1 >= thr → fracción alpha de Δt
    #   both_moving   : s0 >= thr AND s1 >= thr → nada
    #
    # alpha = fracción del intervalo hasta el cruce del umbral (0..1)
    # Interpolación lineal:
    #   alpha = (thr - s0) / (s1 - s0)  [exit:   s0<thr, s1>=thr → la primera
    #                                            fracción alpha pertenece al
    #                                            estado parado]
    #   alpha = (s0 - thr) / (s0 - s1)  [enter:  s0>=thr, s1<thr → la primera
    #                                            fracción alpha es de movimiento;
    #                                            el dwell aporta dt*(1-alpha)]
    if "speed" in df_sorted.columns:
        speeds = df_sorted["speed"].cast(pl.Float64).to_numpy()
    else:
        speeds = np.full(n, np.nan, dtype=np.float64)

    dwell        = np.zeros(n, dtype=np.float64)
    dwell_interp = np.zeros(n, dtype=np.float64)
    thr = speed_zero_threshold

    # Por sub-bloque de (trip_id, next_stop), sumar dwell y dwell_interp.
    i = 0
    while i < n:
        j = i
        while j < n and buses[j] == buses[i] and trips[j] == trips[i]:
            j += 1
        k = i
        while k < j:
            l = k
            while l < j and nstop[l] == nstop[k]:
                l += 1
            # sumar dwell dentro del sub-bloque
            d_total        = 0.0   # cuantizado (clásico)
            d_total_interp = 0.0   # interpolado

            for rr in range(k, l - 1):
                s0 = speeds[rr]
                s1 = speeds[rr + 1]
                dt = (times[rr + 1] - times[rr]).total_seconds()

                s0_ok = not math.isnan(s0)
                s1_ok = not math.isnan(s1)

                # ── dwell_time clásico ───────────────────────────────
                if s0_ok and s0 < thr:
                    d_total += dt

                # ── dwell_time_interp ────────────────────────────────
                if not s0_ok and not s1_ok:
                    pass   # sin info de velocidad, no suma

                elif not s0_ok or not s1_ok:
                    # Solo un lado conocido: misma lógica que clásico
                    spd = s0 if s0_ok else s1
                    if spd < thr:
                        d_total_interp += dt

                else:
                    stopped0 = s0 < thr
                    stopped1 = s1 < thr

                    if stopped0 and stopped1:
                        # Ambos parados → todo el intervalo
                        d_total_interp += dt

                    elif stopped0 and not stopped1:
                        # Bus acelera y sale del estado parado
                        denom = s1 - s0
                        if denom > 1e-6:
                            alpha = max(0.0, min(1.0,
                                        (thr - s0) / denom))
                        else:
                            alpha = 1.0   # velocidades casi iguales
                        d_total_interp += dt * alpha

                    elif not stopped0 and stopped1:
                        # Bus desacelera y entra al estado parado
                        denom = s0 - s1
                        if denom > 1e-6:
                            alpha = max(0.0, min(1.0,
                                        (s0 - thr) / denom))
                        else:
                            alpha = 0.0   # velocidades casi iguales
                        # Está parado durante (1 - alpha) * dt
                        d_total_interp += dt * (1.0 - alpha)

                    # else: ambos en movimiento → no suma

            for rr in range(k, l):
                dwell[rr]        = d_total
                dwell_interp[rr] = d_total_interp
            k = l
        i = j

    # ── 5. Time-since-start y time-to-end ──
    t_since_start = np.zeros(n, dtype=np.float64)
    t_to_end      = np.zeros(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j < n and buses[j] == buses[i] and trips[j] == trips[i]:
            j += 1
        t0 = times[i]
        t_end = times[j-1]
        for r in range(i, j):
            t_since_start[r] = (times[r] - t0).total_seconds()
            t_to_end[r]      = (t_end   - times[r]).total_seconds()
        i = j

    # ── 6. Calendario ──
    df_out = df_sorted.with_columns([
        pl.Series("link_travel_time",      np.round(link_tt, 1)),
        pl.Series("segment_length_m",      np.round(seg_len, 1)),
        pl.Series("link_space_mean_speed", np.round(space_mean_speed, 2)),
        pl.Series("dwell_time",            np.round(dwell, 1)),
        pl.Series("dwell_time_interp",     np.round(dwell_interp, 2)),
        pl.Series("time_in_segment",       np.round(time_in_seg, 1)),
        pl.Series("progress_in_segment",   np.round(progress, 4)),
        pl.Series("time_since_trip_start", np.round(t_since_start, 1)),
        pl.Series("time_to_trip_end",      np.round(t_to_end, 1)),
        pl.col("time").dt.hour().alias("hour_of_day"),
        pl.col("time").dt.weekday().alias("day_of_week"),
    ])

    if verbose:
        print(f"\n  Columnas agregadas: link_travel_time, segment_length_m,")
        print( "                      link_space_mean_speed, dwell_time, dwell_time_interp,")
        print( "                      time_in_segment, progress_in_segment,")
        print( "                      time_since_trip_start, time_to_trip_end,")
        print( "                      hour_of_day, day_of_week")
        valid = ~np.isnan(space_mean_speed)
        if valid.any():
            print(f"\n  link_travel_time      — mediana: {np.median(link_tt):.1f}s")
            print(f"  link_space_mean_speed — mediana: "
                  f"{np.nanmedian(space_mean_speed[valid]):.1f} km/h")
        if (dwell > 0).any():
            dw_raw = dwell[dwell > 0]
            print(f"  dwell_time            — mediana: {np.median(dw_raw):.1f}s")
            dw_int = dwell_interp[dwell_interp > 0]
            if len(dw_int):
                delta = np.median(dw_int) - np.median(dw_raw)
                print(f"  dwell_time_interp     — mediana: {np.median(dw_int):.1f}s "
                      f"(Δ {delta:+.1f}s vs dwell_time)")
        else:
            print(f"  dwell_time            — (sin eventos)")
        print("=" * 60)

    return df_out


#──────────────────────────────────────────────────────────────────────────#
# ── 2. Visualizaciones ── #
#──────────────────────────────────────────────────────────────────────────#

def plot_heatmap_stop_hour(
    df_c2: pl.DataFrame,
    metric: str = "link_travel_time",
    output_path: str = "camino2_heatmap_stop_hour.png",
    direction: Optional[int] = None,
    verbose: bool = False,
) -> str:
    """
    Heatmap parada × hora de la mediana de la métrica indicada.
    """
    df = df_c2
    if direction is not None and "direction" in df.columns:
        df = df.filter(pl.col("direction") == direction)

    df = df.filter(pl.col(metric).is_not_null())

    pivot = (
        df.group_by(["next_stop", "hour_of_day"])
          .agg(pl.col(metric).median().alias("med"))
          .pivot(index="next_stop", on="hour_of_day", values="med")
    )
    # Ordenar paradas por mediana global de m_next_stop (proxy de orden en ruta)
    if "m_next_stop" in df.columns:
        order = (
            df.group_by("next_stop")
              .agg(pl.col("m_next_stop").median().alias("m_med"))
              .sort("m_med", descending=True)
              ["next_stop"].to_list()
        )
        pivot = pivot.with_columns(
            pl.col("next_stop").cast(pl.Categorical)
        )
        pdf = pivot.to_pandas().set_index("next_stop").reindex(order)
    else:
        pdf = pivot.to_pandas().set_index("next_stop")

    # Ordenar columnas (horas) numéricamente
    hour_cols = sorted([c for c in pdf.columns if str(c).isdigit() or
                        isinstance(c, (int, np.integer))],
                       key=lambda x: int(x))
    pdf = pdf[hour_cols]

    fig, ax = plt.subplots(figsize=(13, max(6, 0.25 * len(pdf))))

    data = pdf.values.astype(float)
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd",
                   interpolation="nearest")

    ax.set_xticks(range(len(hour_cols)))
    ax.set_xticklabels([f"{int(h):02d}" for h in hour_cols], fontsize=8)
    ax.set_yticks(range(len(pdf.index)))
    ax.set_yticklabels(pdf.index, fontsize=7)
    ax.set_xlabel("Hora del día")
    ax.set_ylabel("Parada (orden de recorrido)")

    label_map = {
        "link_travel_time":      "Mediana de link_travel_time (s)",
        "link_space_mean_speed": "Mediana de velocidad media espacial (km/h)",
        "dwell_time":            "Mediana de dwell_time (s)",
    }
    ax.set_title(label_map.get(metric, metric), fontsize=12, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label(label_map.get(metric, metric), fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)

    if verbose:
        print(f"  ✔ Heatmap guardado: {output_path}")
    return output_path


def plot_boxplots_by_stop(
    df_c2: pl.DataFrame,
    metric: str = "link_travel_time",
    direction: Optional[int] = None,
    max_stops: int = 25,
    output_path: str = "camino2_boxplots_by_stop.png",
    verbose: bool = False,
) -> str:
    """
    Boxplots de la métrica por parada. Muestra distribución completa, no solo mediana.
    """
    df = df_c2
    if direction is not None and "direction" in df.columns:
        df = df.filter(pl.col("direction") == direction)
    df = df.filter(pl.col(metric).is_not_null())

    # Orden de paradas por m_next_stop mediano
    order = (
        df.group_by("next_stop")
          .agg(pl.col("m_next_stop").median().alias("m_med"))
          .sort("m_med", descending=True)
          ["next_stop"].to_list()
    )
    if len(order) > max_stops:
        # Tomar primeras y últimas del recorrido para no sobrecargar
        keep_n = max_stops // 2
        order = order[:keep_n] + order[-keep_n:]

    data_per_stop = []
    for s in order:
        v = df.filter(pl.col("next_stop") == s)[metric].to_numpy()
        v = v[~np.isnan(v)]
        data_per_stop.append(v)

    fig, ax = plt.subplots(figsize=(max(11, 0.45 * len(order)), 6))
    bp = ax.boxplot(data_per_stop, showfliers=False, patch_artist=True,
                    medianprops=dict(color="#222", linewidth=1.5),
                    boxprops=dict(facecolor="#5bc0de", alpha=0.7,
                                  edgecolor="#31708f"))
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels(order, rotation=70, fontsize=7, ha="right")
    ax.set_xlabel("Parada (orden de recorrido)")

    label_map = {
        "link_travel_time":      "link_travel_time (s)",
        "link_space_mean_speed": "Velocidad media espacial (km/h)",
        "dwell_time":            "dwell_time (s)",
    }
    ax.set_ylabel(label_map.get(metric, metric))
    ax.set_title(f"Distribución de {label_map.get(metric, metric)} por parada",
                 fontsize=12, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)

    if verbose:
        print(f"  ✔ Boxplots guardados: {output_path}")
    return output_path


def plot_correlation_matrix(
    df_c2: pl.DataFrame,
    features: Optional[List[str]] = None,
    output_path: str = "camino2_correlation_matrix.png",
    verbose: bool = False,
) -> str:
    """
    Matriz de correlación de Spearman entre las features nuevas + s_next_stop.
    Spearman captura relaciones monótonas, no solo lineales.
    """
    if features is None:
        features = [
            "s_next_stop", "s_next_stop_interp",
            "link_travel_time", "link_space_mean_speed",
            "dwell_time", "time_in_segment", "progress_in_segment",
            "time_since_trip_start", "time_to_trip_end",
            "m_next_stop", "speed", "hour_of_day",
        ]
    features = [f for f in features if f in df_c2.columns]

    pdf = df_c2.select(features).to_pandas().apply(pd.to_numeric, errors="coerce")
    corr = pdf.corr(method="spearman")

    fig, ax = plt.subplots(figsize=(0.75 * len(features) + 2,
                                     0.75 * len(features) + 1.5))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1,
                   interpolation="nearest")
    ax.set_xticks(range(len(features)))
    ax.set_xticklabels(features, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features, fontsize=8)
    ax.set_title("Matriz de correlación de Spearman",
                 fontsize=12, fontweight="bold")

    # Anotar valores
    for i in range(len(features)):
        for j in range(len(features)):
            v = corr.values[i, j]
            if not np.isnan(v):
                color = "white" if abs(v) > 0.5 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=7, color=color)

    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)

    if verbose:
        print(f"  ✔ Matriz de correlación guardada: {output_path}")
    return output_path


def summary_table(
    df_c2: pl.DataFrame,
    features: Optional[List[str]] = None,
    output_path: str = "camino2_summary_table.csv",
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Tabla de estadísticos descriptivos de cada feature.
    Guarda CSV y también devuelve DataFrame.
    """
    if features is None:
        features = [
            "s_next_stop", "s_next_stop_interp",
            "link_travel_time", "link_space_mean_speed",
            "dwell_time", "time_in_segment", "progress_in_segment",
        ]
    features = [f for f in features if f in df_c2.columns]

    rows = []
    for f in features:
        v = df_c2[f].drop_nulls().to_numpy()
        v = v[~np.isnan(v)]
        if len(v) == 0:
            rows.append({
                "feature": f, "n": 0, "mean": None, "median": None,
                "std": None, "p05": None, "p95": None, "iqr": None,
                "min": None, "max": None,
            })
            continue
        rows.append({
            "feature": f,
            "n":      int(len(v)),
            "mean":   round(float(np.mean(v)), 2),
            "median": round(float(np.median(v)), 2),
            "std":    round(float(np.std(v)), 2),
            "p05":    round(float(np.percentile(v, 5)), 2),
            "p95":    round(float(np.percentile(v, 95)), 2),
            "iqr":    round(float(np.percentile(v, 75) - np.percentile(v, 25)), 2),
            "min":    round(float(np.min(v)), 2),
            "max":    round(float(np.max(v)), 2),
        })

    out = pd.DataFrame(rows)
    out.to_csv(output_path, index=False)

    if verbose:
        print(f"\n  ✔ Tabla resumen guardada: {output_path}")
        print(out.to_string(index=False))

    return out


#──────────────────────────────────────────────────────────────────────────#
# ── 3. Suite completa ── #
#──────────────────────────────────────────────────────────────────────────#

def plot_feature_suite(
    df_c2: pl.DataFrame,
    output_dir: str = "./camino2_plots/",
    direction: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, str]:
    """
    Genera las 3 visualizaciones + tabla resumen.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}

    if verbose:
        print("\n  Generando suite de visualización Camino 2...\n")

    # 1. Heatmaps (uno por métrica clave)
    paths["heatmap_link_tt"] = plot_heatmap_stop_hour(
        df_c2, metric="link_travel_time", direction=direction,
        output_path=str(Path(output_dir) / "01_heatmap_link_travel_time.png"),
        verbose=verbose,
    )
    if "link_space_mean_speed" in df_c2.columns:
        paths["heatmap_speed"] = plot_heatmap_stop_hour(
            df_c2, metric="link_space_mean_speed", direction=direction,
            output_path=str(Path(output_dir) / "02_heatmap_space_mean_speed.png"),
            verbose=verbose,
        )

    # 2. Boxplots
    paths["boxplots"] = plot_boxplots_by_stop(
        df_c2, metric="link_travel_time", direction=direction,
        output_path=str(Path(output_dir) / "03_boxplots_link_tt.png"),
        verbose=verbose,
    )

    # 3. Correlación
    paths["correlation"] = plot_correlation_matrix(
        df_c2,
        output_path=str(Path(output_dir) / "04_correlation_matrix.png"),
        verbose=verbose,
    )

    # 4. Tabla
    summary_table(
        df_c2,
        output_path=str(Path(output_dir) / "05_summary_table.csv"),
        verbose=verbose,
    )
    paths["summary"] = str(Path(output_dir) / "05_summary_table.csv")

    if verbose:
        print(f"\n  ✅ Suite completa en: {output_dir}\n")
    return paths


if __name__ == "__main__":
    print("Uso desde notebook:")
    print("    from _lib_features import (")
    print("        compute_alternative_features, plot_feature_suite")
    print("    )")
    print("    df_c2 = compute_alternative_features(")
    print("        df_enriched, df_segments_route,")
    print("        route_prefix='101', verbose=True)")
    print("    plot_feature_suite(df_c2, output_dir='./camino2_plots/')")


