"""
make_results_figures_frame.py
=============================
Fig. 2 y Figs. S1–S2 del artículo (Figuras 3–5 en un borrador anterior), sobre el frame de evaluación único.

Sustituye a `make_results_figures.py`, que las construía desde los artefactos del
protocolo matched (`matched_tmin_v7_*`). Paleta Okabe-Ito (segura para daltonismo),
300 dpi, PNG + PDF.

  Fig 3  `fig_results_1_dumbbell`            LSTM vs mejor alternativa, por leg
  Fig 4  `fig_results_2_by_position`         MAE relativo por tramo del viaje
  Fig 5  `fig_results_5_ontime_decomposition`  descomposición del on-time

⚠ La Figura 5 es NUEVA. La anterior con ese número contrastaba el MAE del mismo
LSTM en full-grid contra el subconjunto matched: era el diagnóstico de un
artefacto del protocolo viejo, y bajo el frame único no existe tal contraste
(todas las familias se evalúan en las mismas filas por construcción). El hueco lo
ocupa la descomposición del on-time, que es lo que la sección 4.2 necesita
enseñar y no tenía figura.

Los nombres de archivo llevan el desfase heredado (`fig_results_1_*` es la Figura
3 del manuscrito); se conservan para no romper las rutas del manuscrito, salvo la
5, que es nueva y sí va numerada como el manuscrito.

Con la vuelta (docs/PREREGISTRO_VUELTA.md §4) las tres dibujan el panel entero —todas
las rutas incluidas, una vuelta cada una— y no el estrato «validated» de agosto. Los
títulos ya no afirman el resultado («the LSTM's deficit is a level, not a slope»,
«24 validated legs»): describen qué se dibuja y cuentan las rutas de los datos.

Uso
---
    python docs/paper/analysis/make_results_figures_frame.py --root test3/extracted
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import numpy as np                                              # noqa: E402
import polars as pl                                             # noqa: E402
import matplotlib as mpl                                        # noqa: E402
mpl.use("Agg")
import matplotlib.pyplot as plt                                 # noqa: E402

from audit_run import route_label                               # noqa: E402
from honest_ontime import RANGE                                 # noqa: E402
from leg_inclusion import MIN_M                                 # noqa: E402
from make_table_05_permodel import LABEL                        # noqa: E402
from make_table_07_by_position import BINS, HEADLINE, per_leg_table  # noqa: E402
from make_tables_06_09_paired_stats import ALTS, REF, _paired, load  # noqa: E402

FIG = Path("docs/paper/figures")
TAB = Path("docs/paper/tables")

# Okabe-Ito
C_LSTM, C_BEST, C_DH, C_DELAY, GREY = "#0072B2", "#D55E00", "#009E73", "#999999", "#666666"

mpl.rcParams.update({
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#E6E6E6", "grid.linewidth": 0.6,
    "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
})


def save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}")
    plt.close(fig)
    print(f"  ✓ {FIG / (name + '.png')}")


def panel_letters(axes) -> None:
    """Las letras **a**, **b**… que citan los pies (Figs. 2–4), arriba a la izquierda de
    cada panel, en negrita como en la Fig. 1. Van fuera del área de datos (encima del eje
    y), así que no tapan ningún punto."""
    for k, a in enumerate(axes):
        a.text(-0.02, 1.02, "abcdefgh"[k], transform=a.transAxes, fontsize=12,
               fontweight="bold", ha="right", va="bottom")


def _boot_ci(x: np.ndarray, n: int = 10_000, seed: int = 42):
    rng = np.random.default_rng(seed)
    d = np.median(x[rng.integers(0, len(x), size=(n, len(x)))], axis=1)
    return np.percentile(d, 2.5), np.percentile(d, 97.5)


# ────────────────────────────────────────────────────────── Fig 3
def fig_dumbbell(d: pl.DataFrame) -> None:
    """Por ruta: el LSTM y la mejor alternativa, unidos. Un enlace corto = empate."""
    M, keys = _paired(d, "MAE_min")
    O, _ = _paired(d, "pct_ontime_sym180_cal")
    best_i = M[:, 1:].argmin(axis=1)
    best_mae = M[np.arange(len(M)), best_i + 1]
    best_ot = O[:, 1:].max(axis=1)

    order = np.argsort(best_mae / M[:, 0])          # peor ratio para el LSTM abajo
    y = np.arange(len(order))
    labels = [route_label(keys[i].split("/")[0], number=True) for i in order]

    fig, ax = plt.subplots(1, 2, figsize=(9.6, 6.4), sharey=True)

    for k, (vals_l, vals_b, xlab) in enumerate((
            (M[order, 0], best_mae[order], "MAE (min) — lower is better"),
            (O[order, 0], best_ot[order],
             "On-time %, symmetric band, centred — higher is better"))):
        a = ax[k]
        a.hlines(y, np.minimum(vals_l, vals_b), np.maximum(vals_l, vals_b),
                 color=GREY, lw=1.2, zorder=1)
        a.scatter(vals_b, y, s=34, color=C_BEST, zorder=3,
                  label="best alternative")
        a.scatter(vals_l, y, s=34, color=C_LSTM, zorder=3, label="Sequential LSTM")
        a.set_xlabel(xlab)
        a.set_yticks(y)
    ax[0].set_yticklabels(labels, fontsize=7.5)
    panel_letters(ax)
    # La leyenda, DEBAJO de los ejes: dentro de ax[0] («lower right») tapaba el
    # punto del LSTM de la 302, la fila de abajo, que es donde el hueco es mayor.
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=8)

    won = int((best_mae < M[:, 0]).sum())
    fig.suptitle(f"Sequential LSTM vs. best of nine alternatives, "
                 f"{len(order)} routes\n"
                 f"the alternative is more accurate on {won} of {len(order)} routes; "
                 f"routes ordered by MAE ratio", fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "fig_results_1_dumbbell")


# ────────────────────────────────────────────────────────── Fig 4
def fig_by_position(per_leg: pl.DataFrame) -> None:
    """MAE relativo al LSTM en cada tramo del viaje, con IC95% bootstrap."""
    v = per_leg
    n_routes = v.select(["route", "leg"]).unique().height
    names = [b[0] for b in BINS]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    colors = {"xgb_mae_b": C_BEST, "delay_hour": C_DH, "delay": C_DELAY}
    off = {"xgb_mae_b": -0.12, "delay_hour": 0.0, "delay": 0.12}

    for a in HEADLINE:
        xs, ys, los, his = [], [], [], []
        for i, pos in enumerate(names):
            b = v.filter(pl.col("pos") == pos)
            # sort: per_leg comes from a group_by without maintain_order, and the
            # seeded bootstrap draws different replicates from a permuted array
            w = (b.pivot(values="MAE_min", index=["route", "leg"], on="algorithm")
                  .sort(["route", "leg"]))
            rel = ((w[a] - w[REF]) / w[REF] * 100).to_numpy()
            lo, hi = _boot_ci(rel)
            xs.append(i + off[a]); ys.append(np.median(rel))
            los.append(np.median(rel) - lo); his.append(hi - np.median(rel))
        ax.errorbar(xs, ys, yerr=[los, his], fmt="o-", color=colors[a],
                    capsize=3, lw=1.4, ms=5, label=LABEL[a])

    ax.axhline(0, color=C_LSTM, lw=1.6, ls="--")
    ax.text(2.32, 0.4, "Sequential LSTM", color=C_LSTM, fontsize=8, va="bottom",
            ha="right")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(["early\n(first third observed)", "mid", "late\n(last third)"])
    ax.set_ylabel("MAE relative to the LSTM (%)\nnegative = alternative more accurate")
    ax.set_title("MAE relative to the LSTM along the round trip\n"
                 f"median over {n_routes} routes, bootstrap 95% CI", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    save(fig, "fig_results_2_by_position")


# ────────────────────────────────────────────────────────── Fig 5
def fig_ontime_decomposition(d: pl.DataFrame) -> None:
    """APTA → banda simétrica → centrada, por modelo. Aísla asimetría y sesgo."""
    v = d
    n_routes = v.select(["route", "leg"]).unique().height
    cols = ["pct_ontime_apta", "pct_ontime_sym180", "pct_ontime_sym180_cal"]
    g = (v.group_by("algorithm")
         .agg(*[pl.col(c).median().alias(c) for c in cols])
         .sort("pct_ontime_sym180_cal", descending=True))

    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    x = np.arange(3)
    for r in g.iter_rows(named=True):
        a = r["algorithm"]
        ys = [r[c] for c in cols]
        is_key = a in ("lstm_a", "xgb_mae_b", "delay_hour")
        color = {"lstm_a": C_LSTM, "xgb_mae_b": C_BEST,
                 "delay_hour": C_DH}.get(a, "#CCCCCC")
        ax.plot(x, ys, "o-", color=color, lw=2.0 if is_key else 1.0,
                ms=5 if is_key else 3, zorder=3 if is_key else 1,
                label=LABEL.get(a, a) if is_key else None)
        if is_key:
            ax.annotate(LABEL.get(a, a), (x[-1] + 0.04, ys[-1]), fontsize=8,
                        color=color, va="center")

    ax.set_xticks(x)
    ax.set_xticklabels(["operational band\n[−60, +300] s",
                        "symmetric band\n±180 s (same width)",
                        "symmetric band,\nresidual centred"])
    ax.set_xlim(-0.25, 2.9)
    ax.set_ylabel(f"On-time rate (%), median over {n_routes} routes")
    # ⚠ El salto izquierda→centro es POSITIVO para todos: la banda simétrica es
    # más generosa porque el grueso de los residuos está cerca de 0 y la
    # operacional solo concede 60 s por el lado temprano. Lo informativo es
    # CUÁNTO gana cada uno: el que menos gana es el que ya estaba mejor situado
    # respecto del centro de la banda asimétrica. Titular esto como "el premio
    # por caer del lado ancho" describiría algo que la figura no enseña.
    ax.set_title("What the operational on-time rate is measuring\n"
                 "the same band width centred on zero, then with each model's "
                 "residual bias removed out of fold", fontsize=10)
    fig.tight_layout()
    save(fig, "fig_results_5_ontime_decomposition")


def main() -> int:
    repo = Path(__file__).resolve().parents[3]
    ap = argparse.ArgumentParser(description="Fig. 2 y Figs. S1–S2 sobre el frame")
    ap.add_argument("--root", default=None)
    ap.add_argument("--q", type=float, default=0.05)
    ap.add_argument("--cov", type=float, default=1.0)
    ap.add_argument("--range", dest="rng", default=RANGE)
    ap.add_argument("--min-M", type=int, default=MIN_M)
    a = ap.parse_args()
    root = Path(a.root).resolve() if a.root else repo

    src = TAB / "table_10_ontime_honest.csv"
    if not src.is_file():
        raise SystemExit(f"✗ falta {src} — corre antes honest_ontime.py")
    d = load(src)
    inc = pl.read_csv(repo / "docs/paper/analysis/leg_inclusion.csv") \
        .filter(pl.col("included")).select(pl.col("route").cast(pl.Utf8), "leg")
    d = d.with_columns(pl.col("route").cast(pl.Utf8)).join(
        inc, on=["route", "leg"], how="inner")

    fig_dumbbell(d)
    fig_by_position(per_leg_table(root, a.q, a.cov, a.rng, a.min_M))
    fig_ontime_decomposition(d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
