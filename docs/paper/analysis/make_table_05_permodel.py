"""
make_table_05_permodel.py
=========================
Tabla 2 del artículo (archivo `table_05`) — rendimiento por modelo en el panel, mediana [IQR].

Sustituye a `make_table_03_permodel_panel.py` (régimen q=0.95, protocolo matched,
5 modelos). Cambia en tres cosas:

1. **La unidad es la LEG, no la ruta.** Es la unidad de análisis del método
   (terminal→terminal) y la del criterio de inclusión pre-especificado. Agregar por
   ruta mezclaba legs de dificultad muy distinta bajo una sola mediana.
2. **Diez algoritmos**, no cinco: entran las cuatro variantes de XGBoost y
   `delay_hour`, que por decisión de alcance son modelos de primera clase.
3. **El on-time va acompañado de su versión honesta.** La columna APTA es la
   banda operacional asimétrica `[−60, +300]`, que premia a los modelos cuyo
   residuo mediano cae del lado ancho; la columna calibrada es la banda simétrica
   de la MISMA anchura con el residuo centrado leave-one-fold-out, o sea
   precisión pura. La descomposición completa está en la Tabla 10.

Se alimenta de `table_10_ontime_honest.csv` para que las Tablas 5 y 10 no puedan
discrepar: es el mismo objeto, agregado de dos maneras.

Con la vuelta (docs/PREREGISTRO_VUELTA.md §4) cada ruta es una leg —su vuelta— y hay
**un solo panel**: todas las rutas que cumplen el criterio, sin estratos (los de agosto
venían del descubrimiento de terminales, que ya no corta). Las variantes, la 902 y las
rutas `revisar` van dentro. `PANELS` se queda como lista de un elemento porque las
Tablas 6, 7 y 9 iteran sobre ella.

Uso
---
    python docs/paper/analysis/honest_ontime.py      --root artefacts/run_bundle_<…>
    python docs/paper/analysis/make_table_05_permodel.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import polars as pl                                             # noqa: E402

# Los diez modelos de la tabla. ⚠ En agosto esta lista era además el ORDEN de las
# filas, elegido para que «el LSTM quedara entre los que lo baten y los que no»: un
# orden que presupone el resultado. Con la vuelta las filas van por MAE mediano.
ORDER = ["xgb_mae_b", "xgb_mae", "delay_hour", "xgb_mse_b", "kernel", "knn",
         "lstm_a", "xgb_mse", "delay", "linreg"]
LABEL = {
    "lstm_a": "Sequential LSTM", "delay": "Delay", "delay_hour": "Delay (hourly)",
    "kernel": "Kernel", "knn": "kNN", "linreg": "Linear",
    # ⚠ Los nombres siguen la taxonomia de §3.5, no la del codigo. "historical"
    # era un nombre equivocado: ese tier YA lleva condicionamiento historico
    # dentro (el bloque prior_*), y "contextual" sobrevendia un bloque derivado
    # del mismo flujo AVL que todo lo demas.
    "xgb_mae": "XGBoost (MAE, historical)", "xgb_mae_b": "XGBoost (MAE, + context)",
    "xgb_mse": "XGBoost (MSE, historical)", "xgb_mse_b": "XGBoost (MSE, + context)",
    "xgb_mae_np": "XGBoost (MAE, current-trip + delay)",
    "xgb_mae_t0": "XGBoost (MAE, current-trip only)",
}
#: (nombre, filtro). Con la vuelta, uno: todas las rutas que ya dejó pasar el criterio
#: (`table_10_ontime_honest.csv` sólo trae ésas). `None` = sin filtro.
PANELS = [("all", None)]


def in_panel(d: pl.DataFrame, keep) -> pl.DataFrame:
    """Las filas del panel. `keep` es el filtro de `PANELS` (``None`` = todas)."""
    return d if keep is None else d.filter(keep)


def build(src: Path) -> pl.DataFrame:
    d = pl.read_csv(src)
    d = d.with_columns((pl.col("MAE_s") / 60.0).alias("MAE_min"))

    rows: List[dict] = []
    for panel, keep in PANELS:
        p = in_panel(d, keep)
        n_legs = p.select(["route", "leg"]).unique().height
        g = (p.group_by("algorithm").agg(
                pl.col("MAE_min").median().alias("mae_med"),
                pl.col("MAE_min").quantile(0.25).alias("mae_q1"),
                pl.col("MAE_min").quantile(0.75).alias("mae_q3"),
                pl.col("pct_ontime_apta").median().alias("ot_med"),
                pl.col("pct_ontime_apta").quantile(0.25).alias("ot_q1"),
                pl.col("pct_ontime_apta").quantile(0.75).alias("ot_q3"),
                pl.col("pct_ontime_sym180_cal").median().alias("cal_med"),
                pl.col("pct_ontime_sym180_cal").quantile(0.25).alias("cal_q1"),
                pl.col("pct_ontime_sym180_cal").quantile(0.75).alias("cal_q3"),
                pl.len().alias("legs")))
        for r in (g.filter(pl.col("algorithm").is_in(ORDER))
                  .sort(["mae_med", "algorithm"]).iter_rows(named=True)):
            rows.append({"panel": panel, "n_legs_panel": n_legs,
                         "model": LABEL.get(r["algorithm"], r["algorithm"]), **r})
    return pl.DataFrame(rows)


def to_markdown(t: pl.DataFrame) -> str:
    L: List[str] = []
    L.append("**Table 2. Per-model performance across the panel "
             "(median [IQR] over routes).**")
    L.append("")
    n_all = int(t["n_legs_panel"][0])
    L.append(
        f"The unit is the route: each contributes its round trip, from one "
        f"crossing of the route's cut point to the next, with one evaluation "
        f"frame per route. Routes enter under the pre-specified criterion (trip "
        f"cut not rated poor, at least 100 round trips retained by the grid); "
        f"{n_all} qualify, and all of them form a single panel. All models are "
        f"scored on identical round trips, positions, horizons and folds. "
        f"*On-time (APTA)* uses the operational "
        f"band `t_true − t_pred ∈ [−60, +300] s`, which is asymmetric and "
        f"therefore rewards models whose residuals sit on its wide side; "
        f"*on-time (centred)* uses a symmetric band of the same width with the "
        f"residual centred out-of-fold, and so reflects dispersion alone. "
        f"Table 4 decomposes the difference.")
    L.append("")
    L.append("| Model | MAE (min) med [IQR] | On-time % (APTA) med [IQR] "
             "| On-time % (centred) med [IQR] |")
    L.append("|---|---|---|---|")
    for r in t.iter_rows(named=True):
        L.append(
            f"| {r['model']} | "
            f"{r['mae_med']:.2f} [{r['mae_q1']:.2f}, {r['mae_q3']:.2f}] | "
            f"{r['ot_med']:.1f} [{r['ot_q1']:.1f}, {r['ot_q3']:.1f}] | "
            f"{r['cal_med']:.1f} [{r['cal_q1']:.1f}, {r['cal_q3']:.1f}] |")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Tabla 2 — panel por modelo")
    repo = Path(__file__).resolve().parents[3]
    ap.add_argument("--src", default=str(repo / "docs" / "paper" / "tables"
                                         / "table_10_ontime_honest.csv"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    src = Path(a.src)
    if not src.is_file():
        raise SystemExit(f"✗ falta {src} — corre antes honest_ontime.py --root <paquete>")
    t = build(src)
    prefix = Path(a.out) if a.out else (repo / "docs" / "paper" / "tables"
                                        / "table_05_permodel_panel")
    t.write_csv(prefix.with_suffix(".csv"))
    md = to_markdown(t)
    prefix.with_suffix(".md").write_text(md, encoding="utf-8")
    print(md)
    print(f"  ✓ {prefix.with_suffix('.csv')}")
    print(f"  ✓ {prefix.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
