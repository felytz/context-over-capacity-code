"""Interpolación de cruces de parada (target sin cuantización, camino1).

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
#   2. compute_interpolated_crossings — target sin cuantización
#   (originalmente: camino1_interpolation.py)
##############################################################################

#################
### Libraries ###
#################

import math
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Optional, List, Tuple, Dict


###############
### Helpers ###
###############

from _lib_geo import _haversine_m  # consolidado en lib/_lib_geo.py


#──────────────────────────────────────────────────────────────────────────#
# ── 1. Cálculo de distancia acumulada sobre la trayectoria del viaje ── #
#──────────────────────────────────────────────────────────────────────────#

def _compute_cum_distance(
    df_sorted: pl.DataFrame,
    verbose: bool = False,
) -> np.ndarray:
    """
    Calcula la distancia acumulada (metros) recorrida por cada bus dentro
    de cada viaje. Reinicia en cada cambio de (bus_code, trip_id).

    Parameters
    ----------
        df_sorted : pl.DataFrame   Ya ordenado por bus_code, trip_id, time.
                                   Requiere latitude, longitude.
        verbose   : bool

    Returns
    -------
        cum_m (np.ndarray): Distancia acumulada en metros por registro.
    """
    lats  = df_sorted["latitude"].cast(pl.Float64).to_numpy()
    lons  = df_sorted["longitude"].cast(pl.Float64).to_numpy()
    buses = df_sorted["bus_code"].to_list()
    trips = df_sorted["trip_id"].to_list()
    n = len(lats)

    cum_m = np.zeros(n, dtype=np.float64)

    for i in range(1, n):
        if buses[i] == buses[i-1] and trips[i] == trips[i-1]:
            d = _haversine_m(lats[i-1], lons[i-1], lats[i], lons[i])
            cum_m[i] = cum_m[i-1] + d
        else:
            cum_m[i] = 0.0

    if verbose:
        # Distancia acumulada al final de cada viaje (índice anterior a un cambio de viaje)
        trip_ends = [cum_m[i-1] for i in range(1, n)
                     if (buses[i] != buses[i-1] or trips[i] != trips[i-1])]
        if trip_ends:
            med_end = float(np.median(trip_ends))
        else:
            med_end = float(cum_m[-1]) if n > 0 else 0.0
        print(f"  Distancia acumulada — max: {cum_m.max():.1f}m  "
              f"mediana fin-viaje: {med_end:.1f}m")

    return cum_m


#──────────────────────────────────────────────────────────────────────────#
# ── 2. Función pública: compute_interpolated_crossings ── #
#──────────────────────────────────────────────────────────────────────────#

def compute_interpolated_crossings(
    df_enriched: pl.DataFrame,
    df_segments_route: Optional[pl.DataFrame] = None,
    verbose: bool = False,
) -> pl.DataFrame:
    """
    Reemplaza s_next_stop por una versión interpolada que NO está cuantizada
    por el periodo de muestreo del AVL.

    Estrategia: para cada registro r del bus+viaje, encontrar el siguiente
    registro f donde next_stop cambia. El instante real del cruce se interpola
    linealmente entre r y f usando la distancia acumulada y m_next_stop[r] como
    proxy de "qué tan cerca está r de la frontera".

    Aproximación práctica:
        Sea m_r = m_next_stop[r] (metros restantes hasta frontera desde r)
        Sea m_f = m_next_stop[f-1] (último registro con la parada vieja)
        Sea d_step = distancia acumulada(f) - distancia acumulada(f-1)

        Fracción del paso (f-1 → f) recorrida hasta la frontera:
            alpha = m_{f-1} / d_step    (0 < alpha ≤ 1)

        t_cruce = times[f-1] + alpha * (times[f] - times[f-1])

    Parameters
    ----------
        df_enriched       : pl.DataFrame   Salida de enrich_trip_segments,
                                           debe contener: bus_code, trip_id,
                                           time, latitude, longitude, next_stop,
                                           m_next_stop, s_next_stop.
        df_segments_route : pl.DataFrame | None
                                           No requerido aquí (la geometría ya
                                           se reflejó en m_next_stop). Se acepta
                                           por compatibilidad de API.
        verbose           : bool

    Returns
    -------
        df_out : pl.DataFrame con columnas adicionales:
            cum_dist_m            (float): distancia acumulada en el viaje
            t_cross_interp        (datetime): timestamp interpolado del cruce
            s_next_stop_interp    (float): segundos hasta el cruce (no cuantizado)
            quantization_error_s  (float): s_next_stop - s_next_stop_interp
    """
    required = {"bus_code", "trip_id", "time", "latitude", "longitude",
                "next_stop", "m_next_stop", "s_next_stop"}
    missing = required - set(df_enriched.columns)
    if missing:
        raise ValueError(f"Columnas faltantes en df_enriched: {missing}")

    if verbose:
        print("=" * 60)
        print("  CAMINO 1 — Interpolación lineal del cruce")
        print("=" * 60)
        print(f"  Registros entrada : {df_enriched.height:,}")

    # ── Ordenar y calcular distancia acumulada ──
    df_sorted = df_enriched.sort(["bus_code", "trip_id", "time"])
    cum_m = _compute_cum_distance(df_sorted, verbose=verbose)

    # ── Extraer arrays ──
    times   = df_sorted["time"].to_list()
    buses   = df_sorted["bus_code"].to_list()
    trips   = df_sorted["trip_id"].to_list()
    nstops  = df_sorted["next_stop"].to_list()
    mnext   = df_sorted["m_next_stop"].cast(pl.Float64).to_numpy()
    snext   = df_sorted["s_next_stop"].cast(pl.Float64).to_numpy()
    n = len(times)

    t_cross    = [None] * n
    s_interp   = [float("nan")] * n

    # ── Loop por bloques bus+viaje ──
    i = 0
    while i < n:
        j = i
        while j < n and buses[j] == buses[i] and trips[j] == trips[i]:
            j += 1
        # Bloque [i, j)
        for r in range(i, j):
            current = nstops[r]
            # Buscar primer índice f > r con next_stop distinto
            f = -1
            for k in range(r + 1, j):
                if nstops[k] != current:
                    f = k
                    break
            if f == -1:
                continue   # último segmento del viaje, queda NaN

            # Paso entre f-1 y f (donde ocurrió el cruce)
            d_step = cum_m[f] - cum_m[f - 1]
            if d_step < 1e-3:
                # Bus parado entre los dos registros: usamos el siguiente
                t_cross[r]  = times[f]
                s_interp[r] = (times[f] - times[r]).total_seconds()
                continue

            m_prev = mnext[f - 1]   # metros restantes en r=f-1 antes de cruzar
            alpha  = max(0.0, min(1.0, m_prev / d_step))

            dt_step    = (times[f] - times[f - 1]).total_seconds()
            t_cross_dt = times[f - 1].timestamp() + alpha * dt_step
            # Reconstruir datetime
            from datetime import datetime, timezone
            t_cross[r] = datetime.fromtimestamp(t_cross_dt, tz=times[f].tzinfo)

            s_interp[r] = (t_cross_dt - times[r].timestamp())
        i = j

    # ── Calcular error de cuantización ──
    snext_arr   = snext
    sinterp_arr = np.array(s_interp, dtype=np.float64)
    quant_err   = snext_arr - sinterp_arr   # original - interpolado (sesgo positivo)

    df_out = df_sorted.with_columns([
        pl.Series("cum_dist_m",           cum_m.round(2)),
        pl.Series("t_cross_interp",       t_cross),
        pl.Series("s_next_stop_interp",   np.round(sinterp_arr, 2)),
        pl.Series("quantization_error_s", np.round(quant_err, 2)),
    ])

    if verbose:
        valid = ~np.isnan(sinterp_arr)
        if valid.any():
            print(f"\n  s_next_stop original — mediana: "
                  f"{np.nanmedian(snext_arr[valid]):.1f}s")
            print(f"  s_next_stop interpolado — mediana: "
                  f"{np.nanmedian(sinterp_arr[valid]):.1f}s")
            print(f"  Sesgo medio (original − interp): "
                  f"{np.nanmean(quant_err[valid]):.1f}s")
            print(f"  ➜ La cuantización introducía ~"
                  f"{np.nanmean(quant_err[valid]):.0f}s de sobreestimación promedio")
        print("=" * 60)

    return df_out


#──────────────────────────────────────────────────────────────────────────#
# ── 3. Visualizaciones ── #
#──────────────────────────────────────────────────────────────────────────#

def plot_histogram_comparison(
    df_interp: pl.DataFrame,
    output_path: str = "camino1_hist_comparison.png",
    max_seconds: float = 600.0,
    bin_width: float = 5.0,
    verbose: bool = False,
) -> str:
    """
    Histograma 1: s_next_stop original (escalonado) vs s_next_stop_interp (continuo).

    Muestra visualmente la cuantización del original (picos en múltiplos de ~60 s)
    y la distribución suave del interpolado.
    """
    s_orig   = df_interp["s_next_stop"].drop_nulls().to_numpy()
    s_interp = df_interp["s_next_stop_interp"].drop_nulls().to_numpy()
    s_orig   = s_orig[(s_orig   > 0) & (s_orig   < max_seconds)]
    s_interp = s_interp[(s_interp > 0) & (s_interp < max_seconds)]

    bins = np.arange(0, max_seconds + bin_width, bin_width)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    # ── Subplot 1: Original ──
    axes[0].hist(s_orig, bins=bins, color="#d9534f", alpha=0.85,
                 edgecolor="white", linewidth=0.3)
    axes[0].set_title("s_next_stop ORIGINAL — cuantizado por muestreo AVL (~60s)",
                      fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Frecuencia")
    axes[0].grid(True, alpha=0.3)
    # Marcar líneas de cuantización
    for k in range(1, int(max_seconds // 60) + 1):
        axes[0].axvline(k * 60, color="black", linestyle=":", alpha=0.4, linewidth=0.7)

    # ── Subplot 2: Interpolado ──
    axes[1].hist(s_interp, bins=bins, color="#5cb85c", alpha=0.85,
                 edgecolor="white", linewidth=0.3)
    axes[1].set_title("s_next_stop_interp INTERPOLADO — distribución continua",
                      fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Segundos hasta llegar a la siguiente parada")
    axes[1].set_ylabel("Frecuencia")
    axes[1].grid(True, alpha=0.3)
    for k in range(1, int(max_seconds // 60) + 1):
        axes[1].axvline(k * 60, color="black", linestyle=":", alpha=0.4, linewidth=0.7)

    # ── Estadísticos ──
    stats_text = (
        f"ORIGINAL:    n={len(s_orig):,}  μ={s_orig.mean():.1f}s  "
        f"σ={s_orig.std():.1f}s  med={np.median(s_orig):.1f}s\n"
        f"INTERPOLADO: n={len(s_interp):,}  μ={s_interp.mean():.1f}s  "
        f"σ={s_interp.std():.1f}s  med={np.median(s_interp):.1f}s\n"
        f"SESGO MEDIO (orig - interp): {s_orig.mean() - s_interp.mean():+.1f}s"
    )
    fig.suptitle(stats_text, fontsize=10, y=0.02, ha="center",
                 fontfamily="monospace")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)

    if verbose:
        print(f"  ✔ Histograma guardado: {output_path}")
    return output_path


def plot_geographic_crossings(
    df_interp: pl.DataFrame,
    bus_code: Optional[str] = None,
    trip_id: Optional[int] = None,
    n_crossings: int = 30,
    output_path: str = "camino1_geo_crossings.png",
    verbose: bool = False,
) -> str:
    """
    Scatter geográfico: pares consecutivos (r, r+1) que abarcan un cambio de
    next_stop, con el punto de cruce interpolado mostrado entre ellos.
    """
    df = df_interp.filter(pl.col("s_next_stop_interp").is_not_null())

    # Si no se da bus/trip, tomar el viaje con más cruces
    if bus_code is None or trip_id is None:
        candidates = (
            df.group_by(["bus_code", "trip_id"])
              .agg(pl.col("next_stop").n_unique().alias("n_stops"))
              .sort("n_stops", descending=True)
              .head(1)
        )
        if candidates.height == 0:
            raise ValueError("No hay viajes con cruces interpolados.")
        bus_code = candidates["bus_code"][0]
        trip_id  = candidates["trip_id"][0]

    df_t = df.filter(
        (pl.col("bus_code") == bus_code) & (pl.col("trip_id") == trip_id)
    ).sort("time")

    if df_t.height < 2:
        raise ValueError(f"Viaje {bus_code}/{trip_id} con muy pocos puntos.")

    lats   = df_t["latitude"].to_list()
    lons   = df_t["longitude"].to_list()
    nstops = df_t["next_stop"].to_list()
    mnext  = df_t["m_next_stop"].to_list()
    times  = df_t["time"].to_list()
    cum_m  = df_t["cum_dist_m"].to_list()

    # Identificar transiciones (cruces)
    crossings = []   # lista de (idx_prev, idx_post)
    for k in range(1, len(nstops)):
        if nstops[k] != nstops[k-1]:
            crossings.append((k-1, k))
    crossings = crossings[:n_crossings]

    fig, ax = plt.subplots(figsize=(11, 9))

    # Trayectoria completa
    ax.plot(lons, lats, color="#cccccc", linewidth=1.0,
            zorder=1, label="Trayectoria")
    ax.scatter(lons, lats, color="#888888", s=10, zorder=2,
               label=f"Registros GPS (n={len(lats)})")

    # Cada cruce: dos puntos + cruce interpolado entre ellos
    for k_prev, k_post in crossings:
        # Fracción de cruce
        d_step = cum_m[k_post] - cum_m[k_prev]
        if d_step < 1e-3:
            continue
        alpha = max(0.0, min(1.0, mnext[k_prev] / d_step))

        # Posición geográfica interpolada del cruce
        lat_cross = lats[k_prev] + alpha * (lats[k_post] - lats[k_prev])
        lon_cross = lons[k_prev] + alpha * (lons[k_post] - lons[k_prev])

        ax.plot([lons[k_prev], lons[k_post]],
                [lats[k_prev], lats[k_post]],
                color="#0275d8", linewidth=2.0, alpha=0.6, zorder=3)
        ax.scatter(lons[k_prev], lats[k_prev], color="#d9534f",
                   s=60, zorder=4, edgecolor="white", linewidth=1)
        ax.scatter(lons[k_post], lats[k_post], color="#f0ad4e",
                   s=60, zorder=4, edgecolor="white", linewidth=1)
        ax.scatter(lon_cross, lat_cross, color="#5cb85c",
                   s=120, marker="*", zorder=5,
                   edgecolor="white", linewidth=1.2)

    # Leyenda
    handles = [
        mpatches.Patch(color="#cccccc", label="Trayectoria GPS"),
        mpatches.Patch(color="#d9534f", label="Último registro de la parada"),
        mpatches.Patch(color="#f0ad4e", label="Primer registro de la nueva parada"),
        mpatches.Patch(color="#5cb85c", label="Cruce interpolado (★)"),
    ]
    ax.legend(handles=handles, loc="best", fontsize=9)
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.set_title(f"Cruces de parada interpolados\n"
                 f"bus={bus_code}, trip={trip_id}, "
                 f"cruces mostrados={len(crossings)}",
                 fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")

    fig.tight_layout()
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)

    if verbose:
        print(f"  ✔ Mapa de cruces guardado: {output_path}")
    return output_path


def plot_time_distance_diagram(
    df_interp: pl.DataFrame,
    n_trips: int = 8,
    direction: Optional[int] = None,
    output_path: str = "camino1_time_distance.png",
    verbose: bool = False,
) -> str:
    """
    Diagrama tiempo-distancia (clásico): eje X = distancia acumulada (km),
    eje Y = minutos desde el inicio del viaje. Cada viaje es una línea.
    Marcadores ★ en cada cruce interpolado.
    """
    df = df_interp.filter(pl.col("cum_dist_m").is_not_null())
    if direction is not None and "direction" in df.columns:
        df = df.filter(pl.col("direction") == direction)

    # Tomar n_trips viajes con más registros
    top_trips = (
        df.group_by(["bus_code", "trip_id"])
          .agg(pl.len().alias("n"))
          .sort("n", descending=True)
          .head(n_trips)
    )
    if top_trips.height == 0:
        raise ValueError("No hay viajes para graficar.")

    fig, ax = plt.subplots(figsize=(13, 7))
    cmap = plt.get_cmap("tab10")

    for idx, row in enumerate(top_trips.iter_rows(named=True)):
        bc, tid = row["bus_code"], row["trip_id"]
        df_t = df.filter(
            (pl.col("bus_code") == bc) & (pl.col("trip_id") == tid)
        ).sort("time")

        cum_km = df_t["cum_dist_m"].to_numpy() / 1000.0
        t0     = df_t["time"][0]
        mins   = np.array([(t - t0).total_seconds() / 60.0
                          for t in df_t["time"].to_list()])
        nstops = df_t["next_stop"].to_list()

        color = cmap(idx % 10)
        ax.plot(cum_km, mins, color=color, linewidth=1.4, alpha=0.85,
                label=f"{bc}/{tid}")
        ax.scatter(cum_km, mins, color=color, s=12, alpha=0.6, zorder=3)

        # Cruces
        for k in range(1, len(nstops)):
            if nstops[k] != nstops[k-1]:
                # punto interpolado entre k-1 y k
                d_step = cum_km[k] - cum_km[k-1]
                if d_step <= 0:
                    continue
                m_prev = df_t["m_next_stop"][k-1] / 1000.0
                a = max(0.0, min(1.0, m_prev / d_step))
                x_c = cum_km[k-1] + a * (cum_km[k] - cum_km[k-1])
                y_c = mins[k-1]   + a * (mins[k] - mins[k-1])
                ax.scatter(x_c, y_c, color=color, marker="*", s=120,
                           edgecolor="black", linewidth=0.6, zorder=5)

    ax.set_xlabel("Distancia acumulada (km)")
    ax.set_ylabel("Tiempo desde inicio del viaje (min)")
    ax.set_title("Diagrama tiempo–distancia con cruces interpolados (★)\n"
                 "La pendiente local = velocidad; cambio de pendiente = "
                 "congestión o parada",
                 fontsize=11, fontweight="bold")
    ax.legend(loc="best", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)

    if verbose:
        print(f"  ✔ Diagrama tiempo-distancia guardado: {output_path}")
    return output_path


#──────────────────────────────────────────────────────────────────────────#
# ── 4. Suite completa de validación ── #
#──────────────────────────────────────────────────────────────────────────#

def plot_validation_suite(
    df_interp: pl.DataFrame,
    output_dir: str = "./camino1_plots/",
    verbose: bool = True,
) -> Dict[str, str]:
    """
    Genera las 3 visualizaciones de validación del Camino 1.

    Parameters
    ----------
        df_interp  : pl.DataFrame   Salida de compute_interpolated_crossings.
        output_dir : str            Carpeta destino.
        verbose    : bool

    Returns
    -------
        paths (dict): rutas de los archivos generados.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    paths = {}

    if verbose:
        print("\n  Generando suite de validación Camino 1...\n")

    paths["histogram"] = plot_histogram_comparison(
        df_interp,
        output_path=str(Path(output_dir) / "01_histogram_comparison.png"),
        verbose=verbose,
    )
    paths["geographic"] = plot_geographic_crossings(
        df_interp,
        output_path=str(Path(output_dir) / "02_geographic_crossings.png"),
        verbose=verbose,
    )
    paths["time_distance"] = plot_time_distance_diagram(
        df_interp,
        output_path=str(Path(output_dir) / "03_time_distance_diagram.png"),
        verbose=verbose,
    )

    if verbose:
        print(f"\n  ✅ Suite completa en: {output_dir}\n")
    return paths


#──────────────────────────────────────────────────────────────────────────#
# ── 5. Demo en notebook ── #
#──────────────────────────────────────────────────────────────────────────#

if __name__ == "__main__":
    print("Uso desde notebook:")
    print("    from _lib_features import (")
    print("        compute_interpolated_crossings, plot_validation_suite")
    print("    )")
    print("    df_c1 = compute_interpolated_crossings(df_enriched, verbose=True)")
    print("    plot_validation_suite(df_c1, output_dir='./camino1_plots/')")


