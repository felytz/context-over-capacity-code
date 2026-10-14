"""
arrival_prediction.plots
==========================
Figuras del paper Dublin Bus 46A — versión corregida para coincidir
con el PDF de referencia (PDF correcto).

  - Portada  — tabla de resumen textual
  - Figura 2 — trayectorias con banda P5/mediana/P95, eje Y en segundos
  - Figura 4 — RMSE acumulado por posición, eje Y en segundos
  - Figura 5 — boxplots reales (P25/P50/P75) de |error| por horizonte
  - Figura 6 — barras agrupadas RMSE promedio × algoritmo × grid

Genera PDF estático (matplotlib) + dashboard HTML interactivo (Chart.js).
"""
from __future__ import annotations

import json
import numpy as np
import polars as pl
from pathlib import Path
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages

from .cross_validation import (
    summarize_rmse_by_location,
    summarize_error_by_horizon,
)

# Estilo compartido (paleta colorblind-safe + labels + markers + rcParams).
from _lib_style import (
    ALGO_COLORS, ALGO_LABELS, ALGO_MARKERS, ALGO_LINESTYLES, apply_style,
)

apply_style()


# ── Figura 2 ──────────────────────────────────────────────────────────────

def plot_trajectories(
    T:            np.ndarray,
    l_grid:       np.ndarray,
    n_background: int = 300,
    ax:           Optional[plt.Axes] = None,
    seed:         int = 0,
    metadata:     Optional[dict] = None,
) -> plt.Axes:
    """Trayectorias con banda P5/mediana/P95. Eje Y en segundos."""
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4))

    M = T.shape[0]
    rng = np.random.default_rng(seed)
    bg  = rng.choice(M, size=min(n_background, M), replace=False)

    for i in bg:
        ax.plot(l_grid, T[i], color="#aaaaaa", lw=0.35, alpha=0.35)

    p5  = np.percentile(T,  5, axis=0)
    p50 = np.percentile(T, 50, axis=0)
    p95 = np.percentile(T, 95, axis=0)

    ax.plot(l_grid, p5,  color="#0275d8", lw=2.2, label="Rápido (P5 del total)",  zorder=5)
    ax.plot(l_grid, p50, color="black",   lw=2.5, label="Mediano",                zorder=5)
    ax.plot(l_grid, p95, color="#d9534f", lw=2.2, label="Lento (P95 del total)",  zorder=5)

    n_route = int(metadata["n_route"])    if metadata and "n_route"        in metadata else int(l_grid[-1])
    n_trips = int(metadata["n_trips_output"]) if metadata and "n_trips_output" in metadata else M

    ax.set_xlabel("Posición en la ruta (m)")
    ax.set_ylabel("Tiempo desde inicio del viaje (s)")
    ax.set_title(f"Trayectorias interpoladas — {n_trips:,} viajes (n_route = {n_route:,} m)")
    ax.legend(loc="upper left")
    ax.set_xlim(0, l_grid[-1])
    ax.set_ylim(bottom=0)
    return ax


# ── Figura 4 ──────────────────────────────────────────────────────────────

def plot_rmse_by_location(
    df_errors: pl.DataFrame,
    ax:        Optional[plt.Axes] = None,
) -> plt.Axes:
    """RMSE acumulado vs posición. Eje Y en segundos, banda ±1 std."""
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 5))

    rmse = summarize_rmse_by_location(df_errors)
    order = ["linreg", "delay", "kernel", "knn"]
    algos = [a for a in order if a in rmse["algorithm"].unique().to_list()] + \
            [a for a in rmse["algorithm"].unique().to_list() if a not in order]

    for algo in algos:
        sub   = rmse.filter(pl.col("algorithm") == algo).sort("l_meters")
        l_m   = sub["l_meters"].to_numpy()
        mu    = sub["rmse_mean"].to_numpy()
        sigma = sub["rmse_std"].to_numpy()
        c     = ALGO_COLORS.get(algo, "#888")
        ax.plot(l_m, mu, color=c, lw=2.0,
                marker=ALGO_MARKERS.get(algo, "o"), markersize=4,
                linestyle=ALGO_LINESTYLES.get(algo, "-"),
                label=ALGO_LABELS.get(algo, algo), zorder=4)
        ax.fill_between(l_m, mu - sigma, mu + sigma, color=c, alpha=0.13, zorder=3)

    ax.set_xlabel("Posición del bus en la ruta (m)")
    ax.set_ylabel("RMSE acumulado (s)")
    ax.set_title("RMSE acumulado vs posición — promedio entre folds, banda = ±1 std")
    ax.legend(loc="upper right")
    ax.set_xlim(0)
    ax.set_ylim(bottom=0)
    return ax


# ── Figura 5 ──────────────────────────────────────────────────────────────

def plot_error_boxplots_by_horizon(
    df_errors:        pl.DataFrame,
    horizon_bins_min: Optional[list] = None,
    ax:               Optional[plt.Axes] = None,
) -> plt.Axes:
    """Boxplots reales (P25/P50/P75/bigotes) de |error| por horizonte. Eje Y en minutos."""
    if ax is None:
        _, ax = plt.subplots(figsize=(14, 5))

    if horizon_bins_min is None:
        horizon_bins_min = list(range(5, 55, 5))

    df_b = summarize_error_by_horizon(df_errors, horizon_bins_min)
    bins  = sorted(df_b["horizon_bin_min"].unique().to_list())
    order = ["delay", "linreg", "knn", "kernel"]
    algos = [a for a in order if a in df_b["algorithm"].unique().to_list()]

    n_algos = len(algos)
    width   = 0.8 / n_algos
    gap     = 0.2

    for ai, algo in enumerate(algos):
        c   = ALGO_COLORS.get(algo, "#888")
        sub = df_b.filter(pl.col("algorithm") == algo)

        box_data  = []
        positions = []
        for bi, b in enumerate(bins):
            vals = (sub.filter(pl.col("horizon_bin_min") == b)["error_abs"]
                       .to_numpy() / 60.0)
            if len(vals) == 0:
                continue
            box_data.append(vals)
            positions.append(bi * (n_algos * width + gap) + ai * width + width / 2)

        if not box_data:
            continue

        ax.boxplot(
            box_data,
            positions    = positions,
            widths       = width * 0.85,
            patch_artist = True,
            showfliers   = True,
            flierprops   = dict(marker=".", markersize=1.5, alpha=0.25, color=c),
            medianprops  = dict(color="black", lw=1.5),
            boxprops     = dict(facecolor=c, alpha=0.7, linewidth=0.8),
            whiskerprops = dict(color=c, lw=0.9),
            capprops     = dict(color=c, lw=0.9),
        )

    tick_pos = [bi * (n_algos * width + gap) + (n_algos * width) / 2
                for bi in range(len(bins))]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([f"{int(b)}" for b in bins])
    ax.set_xlabel("Horizonte de predicción (min)")
    ax.set_ylabel("|error| (min)")
    ax.set_title("Error absoluto por horizonte de predicción (boxplots = P25/P50/P75)")
    ax.set_ylim(bottom=0)

    patches = [mpatches.Patch(color=ALGO_COLORS.get(a, "#888"),
                               label=ALGO_LABELS.get(a, a), alpha=0.7)
               for a in algos]
    ax.legend(handles=patches, loc="upper left")
    return ax


# ── Figura 6 ──────────────────────────────────────────────────────────────

def plot_grid_sensitivity(
    df_errors_100: pl.DataFrame,
    df_errors_500: pl.DataFrame,
    ax:            Optional[plt.Axes] = None,
) -> plt.Axes:
    """Barras agrupadas: RMSE promedio por algoritmo × tamaño de grid."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    def _avg(df):
        return {r["algorithm"]: r["avg"]
                for r in (summarize_rmse_by_location(df)
                           .group_by("algorithm")
                           .agg(pl.col("rmse_mean").mean().alias("avg"))
                           .iter_rows(named=True))}

    v100 = _avg(df_errors_100)
    v500 = _avg(df_errors_500)
    order = ["delay", "linreg", "knn", "kernel"]
    algos = [a for a in order if a in v100]

    x = np.arange(len(algos))
    w = 0.35

    b100 = ax.bar(x - w/2, [v100.get(a, 0) for a in algos],
                  width=w, color="#4c8fbd", label="grid 100 m", zorder=3)
    b500 = ax.bar(x + w/2, [v500.get(a, 0) for a in algos],
                  width=w, color="#f0a500", label="grid 500 m", zorder=3, alpha=0.85)

    for bar in list(b100) + list(b500):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 1,
                f"{int(round(h))}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([ALGO_LABELS.get(a, a) for a in algos])
    ax.set_ylabel("RMSE promedio sobre l (s)")
    ax.set_title("Sensibilidad al tamaño de la grid de interpolación")
    ax.legend()
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.3)
    ax.grid(axis="x", visible=False)
    return ax


# ── Portada ───────────────────────────────────────────────────────────────

def _make_cover_page(df_errors, metadata, direction, grid_step_m):
    fig = plt.figure(figsize=(8.5, 11))
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    rmse_summary = summarize_rmse_by_location(df_errors)
    by_algo = (
        rmse_summary.group_by("algorithm")
        .agg(
            pl.col("rmse_mean").mean().alias("avg"),
            pl.col("rmse_mean").min().alias("min"),
            pl.col("rmse_mean").max().alias("max"),
        )
        .sort("avg")
    )

    n_route    = metadata.get("n_route",            "?") if metadata else "?"
    n_trips    = metadata.get("n_trips_output",     "?") if metadata else "?"
    n_dropped  = metadata.get("dropped_short",        0) if metadata else 0
    med_time_s = metadata.get("median_total_time_s",  0) if metadata else 0

    try:
        v_kmh = (int(n_route) / med_time_s) * 3.6 if med_time_s > 0 else 0
        v_str = f"{v_kmh:.1f} km/h"
    except Exception:
        v_str = "—"

    fig.text(0.5, 0.84, "Replicación del paper de predicción",
             ha="center", fontsize=17, fontweight="bold", fontfamily="monospace")
    fig.text(0.5, 0.79, "de tiempos de llegada",
             ha="center", fontsize=17, fontweight="bold", fontfamily="monospace")
    fig.text(0.5, 0.74, "Adaptado a la Ruta 101",
             ha="center", fontsize=13, color="#555", fontfamily="monospace")

    info = (
        f"  Dirección procesada          : {direction}\n"
        f"  Tamaño de grid               : {grid_step_m} m\n"
        f"  n_route                      : {int(n_route):,} m\n"
        f"  Viajes incluidos             : {int(n_trips):,}\n"
        f"  Viajes descartados (corto)   : {int(n_dropped):,}\n"
        f"  Tiempo total (mediana)       : {int(med_time_s)} s ({med_time_s/60:.1f} min)\n"
        f"  Velocidad promedio           : {v_str}\n"
    )
    fig.text(0.1, 0.67, info, ha="left", va="top",
             fontsize=11, fontfamily="monospace", linespacing=1.7)

    # Tabla
    col_w = [28, 10, 10, 10]
    header = (f"  {'Algoritmo':<{col_w[0]}}"
              f"{'RMSE prom':>{col_w[1]}}"
              f"{'RMSE min':>{col_w[2]}}"
              f"{'RMSE max':>{col_w[3]}}")
    sep    = "  " + "─" * (sum(col_w) + 2)
    rows_txt = [header, sep]
    for row in by_algo.iter_rows(named=True):
        label = ALGO_LABELS.get(row["algorithm"], row["algorithm"])
        rows_txt.append(
            f"  {label:<{col_w[0]}}"
            f"{row['avg']:>{col_w[1]}.1f}"
            f"{row['min']:>{col_w[2]}.1f}"
            f"{row['max']:>{col_w[3]}.1f}"
        )
    rows_txt.append("")
    rows_txt.append("  Métrica: RMSE acumulado en segundos, promediado entre folds.")

    fig.text(0.1, 0.46, "\n".join(rows_txt), ha="left", va="top",
             fontsize=11, fontfamily="monospace", linespacing=1.7)
    return fig


# ── build_static_report ───────────────────────────────────────────────────

def build_static_report(
    df_errors:        pl.DataFrame,
    T:                np.ndarray,
    l_grid:           np.ndarray,
    output_dir:       str,
    df_errors_500:    Optional[pl.DataFrame] = None,
    horizon_bins_min: Optional[list]         = None,
    metadata:         Optional[dict]         = None,
    direction:        int                    = 0,
    grid_step_m:      int                    = 100,
    verbose:          bool                   = False,
) -> Dict[str, str]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    pdf_path = out_dir / "arrival_prediction_report.pdf"
    with PdfPages(str(pdf_path)) as pdf:

        fig = _make_cover_page(df_errors, metadata, direction, grid_step_m)
        pdf.savefig(fig, dpi=120); plt.close(fig)

        fig, ax = plt.subplots(figsize=(12, 4))
        plot_trajectories(T, l_grid, ax=ax, metadata=metadata)
        fig.tight_layout(); pdf.savefig(fig, dpi=120); plt.close(fig)

        fig, ax = plt.subplots(figsize=(12, 5))
        plot_rmse_by_location(df_errors, ax=ax)
        fig.tight_layout(); pdf.savefig(fig, dpi=120); plt.close(fig)

        fig, ax = plt.subplots(figsize=(14, 5))
        plot_error_boxplots_by_horizon(df_errors, horizon_bins_min, ax=ax)
        fig.tight_layout(); pdf.savefig(fig, dpi=120); plt.close(fig)

        if df_errors_500 is not None and not df_errors_500.is_empty():
            fig, ax = plt.subplots(figsize=(10, 5))
            plot_grid_sensitivity(df_errors, df_errors_500, ax=ax)
            fig.tight_layout(); pdf.savefig(fig, dpi=120); plt.close(fig)

    paths["pdf"] = str(pdf_path)
    if verbose: print(f"  ✔ PDF: {pdf_path.name}")

    # PNGs individuales
    specs = [
        ("fig2_trajectories.png",
         lambda ax: plot_trajectories(T, l_grid, ax=ax, metadata=metadata),
         (12, 4)),
        ("fig4_rmse_by_location.png",
         lambda ax: plot_rmse_by_location(df_errors, ax=ax),
         (12, 5)),
        ("fig5_error_by_horizon.png",
         lambda ax: plot_error_boxplots_by_horizon(df_errors, horizon_bins_min, ax=ax),
         (14, 5)),
    ]
    if df_errors_500 is not None and not df_errors_500.is_empty():
        specs.append((
            "fig6_grid_sensitivity.png",
            lambda ax: plot_grid_sensitivity(df_errors, df_errors_500, ax=ax),
            (10, 5),
        ))

    for fname, fn, (fw, fh) in specs:
        fig, ax = plt.subplots(figsize=(fw, fh))
        fn(ax)
        fig.tight_layout()
        p = out_dir / fname
        fig.savefig(str(p), dpi=140, bbox_inches="tight")
        plt.close(fig)
        paths[fname.replace(".png", "")] = str(p)
        if verbose: print(f"  ✔ PNG: {fname}")

    return paths


# ── Dashboard HTML ────────────────────────────────────────────────────────

_HTML_TEMPLATE = """<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>Arrival Prediction Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  body{font-family:-apple-system,Segoe UI,sans-serif;background:#1b1b1b;color:#ddd;
       margin:0;padding:20px}
  h1{color:#fff;margin-bottom:4px}
  h2{color:#aaa;font-size:.95rem;font-weight:normal;margin-top:0}
  .grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:16px}
  .card{background:#262626;border-radius:6px;padding:16px;border:1px solid #333}
  .card h3{margin-top:0;color:#fff;font-size:.9rem}
  .stats{display:flex;gap:16px;flex-wrap:wrap;margin-bottom:16px}
  .stat{background:#1f1f1f;padding:10px 14px;border-radius:4px;
        border-left:3px solid #0275d8;min-width:130px}
  .stat .v{font-size:20px;font-weight:bold;color:#fff}
  .stat .l{font-size:11px;color:#888;text-transform:uppercase}
  table{width:100%;border-collapse:collapse;font-size:13px}
  th,td{padding:6px 10px;text-align:right;border-bottom:1px solid #333}
  th{color:#aaa;font-weight:normal} td:first-child,th:first-child{text-align:left}
  canvas{max-height:280px}
</style>
</head>
<body>
<h1>Predicción de tiempos de llegada</h1>
<h2>Replicación Dublin Bus 46A · Ruta 101</h2>
<div class="stats">__STATS__</div>
<div class="grid">
  <div class="card"><h3>RMSE por posición (s) — Figura 4</h3><canvas id="rmse_chart"></canvas></div>
  <div class="card"><h3>Error mediano por horizonte (min) — Figura 5</h3><canvas id="horizon_chart"></canvas></div>
</div>
<div class="card" style="margin-top:16px"><h3>Resumen RMSE (segundos)</h3>__TABLE__</div>
<script>
const D=__DATA__;
const C={delay:"#d9534f",linreg:"#f0ad4e",knn:"#5cb85c",kernel:"#0275d8"};
const L={delay:"Delay-based",linreg:"Linear Regression",knn:"K-NN (K=10)",kernel:"Kernel Regression (b=1)"};
const sc={ticks:{color:"#aaa"},grid:{color:"#333"}};
new Chart(document.getElementById("rmse_chart"),{type:"line",
  data:{labels:D.rmse.l_meters.map(v=>Math.round(v)+"m"),
        datasets:D.rmse.algorithms.map(a=>({label:L[a]||a,data:D.rmse.series[a],
          borderColor:C[a]||"#888",backgroundColor:"transparent",borderWidth:2,pointRadius:0}))},
  options:{responsive:true,plugins:{legend:{labels:{color:"#ccc"}}},
    scales:{x:{...sc,title:{display:true,text:"Posición (m)",color:"#aaa"},ticks:{...sc.ticks,maxTicksLimit:10}},
            y:{...sc,title:{display:true,text:"RMSE (s)",color:"#aaa"}}}}});
new Chart(document.getElementById("horizon_chart"),{type:"bar",
  data:{labels:D.horizon.bins.map(b=>b+" min"),
        datasets:D.horizon.algorithms.map(a=>({label:L[a]||a,data:D.horizon.median[a],
          backgroundColor:(C[a]||"#888")+"cc",borderWidth:0}))},
  options:{responsive:true,plugins:{legend:{labels:{color:"#ccc"}}},
    scales:{x:{...sc,title:{display:true,text:"Horizonte (min)",color:"#aaa"}},
            y:{...sc,title:{display:true,text:"Error mediano (min)",color:"#aaa"}}}}});
</script>
</body>
</html>"""


def _jdumps(obj):
    import math
    def _c(v):
        if isinstance(v, float) and math.isnan(v): return None
        if isinstance(v, dict):  return {k: _c(vv) for k,vv in v.items()}
        if isinstance(v, list):  return [_c(x) for x in v]
        return v
    return json.dumps(_c(obj))


def _build_dashboard_data(df_errors, horizon_bins_min):
    rmse = summarize_rmse_by_location(df_errors)
    algos = sorted(rmse["algorithm"].unique().to_list())
    l_all = sorted(rmse["l_meters"].unique().to_list())
    series = {}
    for a in algos:
        sub = rmse.filter(pl.col("algorithm")==a).sort("l_meters")
        m   = dict(zip(sub["l_meters"].to_list(), sub["rmse_mean"].to_list()))
        series[a] = [round(float(m.get(l, float("nan"))), 1) for l in l_all]
    df_b = summarize_error_by_horizon(df_errors, horizon_bins_min)
    bins = sorted(df_b["horizon_bin_min"].unique().to_list())
    hmed = {}
    for a in algos:
        sub = df_b.filter(pl.col("algorithm")==a)
        m = {}
        for b in bins:
            v = sub.filter(pl.col("horizon_bin_min")==b)["error_abs"].to_numpy()
            m[b] = float(np.median(v))/60.0 if len(v) else float("nan")
        hmed[a] = [round(m.get(b, float("nan")), 2) for b in bins]
    return {"rmse":{"l_meters":l_all,"algorithms":algos,"series":series},
            "horizon":{"bins":[int(b) for b in bins],"algorithms":algos,"median":hmed}}


def _build_stats_cards(df_errors, metadata):
    rmse = summarize_rmse_by_location(df_errors)
    by_algo = (rmse.group_by("algorithm")
               .agg(pl.col("rmse_mean").mean().alias("avg")).sort("avg"))
    items = []
    if by_algo.height > 0:
        best = by_algo[0,"algorithm"]; worst = by_algo[-1,"algorithm"]
        items += [("Mejor algoritmo", ALGO_LABELS.get(best,best)),
                  ("Línea base",      ALGO_LABELS.get(worst,worst)),
                  ("Mejora vs base",  f"{float(by_algo[-1,'avg']-by_algo[0,'avg']):+.0f} s")]
    if metadata:
        items += [("Viajes en CV",  f"{metadata.get('n_trips_output','?'):,}"),
                  ("n_route",       f"{metadata.get('n_route','?'):,} m"),
                  ("Tiempo mediano",f"{metadata.get('median_total_time_s',0)/60:.1f} min")]
    return "".join(f'<div class="stat"><div class="v">{v}</div><div class="l">{l}</div></div>'
                   for l,v in items)


def _build_summary_table(df_errors):
    rmse = summarize_rmse_by_location(df_errors)
    if rmse.height == 0: return "<p>(sin datos)</p>"
    s = (rmse.group_by("algorithm")
         .agg(pl.col("rmse_mean").mean().alias("avg"),
              pl.col("rmse_mean").min().alias("min"),
              pl.col("rmse_mean").max().alias("max"),
              pl.col("rmse_std").mean().alias("std")).sort("avg"))
    head = ("<table><thead><tr><th>Algoritmo</th><th>RMSE prom (s)</th>"
            "<th>RMSE min (s)</th><th>RMSE max (s)</th><th>std (s)</th></tr></thead><tbody>")
    rows = "".join(
        f"<tr><td>{ALGO_LABELS.get(r['algorithm'],r['algorithm'])}</td>"
        f"<td>{r['avg']:.1f}</td><td>{r['min']:.1f}</td>"
        f"<td>{r['max']:.1f}</td><td>{r['std']:.1f}</td></tr>"
        for r in s.iter_rows(named=True))
    return head + rows + "</tbody></table>"


def build_html_dashboard(df_errors, output_path, metadata=None,
                          horizon_bins_min=None, verbose=False):
    data  = _build_dashboard_data(df_errors, horizon_bins_min)
    cards = _build_stats_cards(df_errors, metadata)
    table = _build_summary_table(df_errors)
    html  = (_HTML_TEMPLATE
             .replace("__DATA__",  _jdumps(data))
             .replace("__STATS__", cards)
             .replace("__TABLE__", table))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(html, encoding="utf-8")
    if verbose: print(f"  ✔ Dashboard HTML: {output_path}")
    return output_path
