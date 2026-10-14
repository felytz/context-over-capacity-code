"""
make_table_11_ablation.py
=========================
Tabla S14 del artículo (archivo `table_11`) — la ablación por GRUPOS del bloque de contexto, en el frame de la vuelta
(enmienda E6 del pre-registro; plan del manuscrito, decisión V1 y 0.2d).

Hasta el 9-oct-2026 la S9 era un artefacto congelado de agosto (43 legs de 11 rutas,
tope de 1,200 viajes, otra geometría y `segment_count_ahead` con el error de E1).
Ésta sale de la corrida `run_routes.py --exo-ablate --no-baselines --no-frames` sobre
las 35 rutas, con el régimen publicado (q05c100, 2,500 viajes, 600 árboles):

    xgb_mae            Tier A (la referencia de cada grupo)
    xgb_mae_<grupo>    Tier A + UN grupo (cal, headway, dwell, speed, route)
    xgb_mae_b          Tier A + los cinco (Tier B)

Todas en el mismo loop de validación cruzada, sobre las mismas filas y folds, así que
la comparación es pareada por construcción, igual que la Tabla 5.

Comprobación incorporada
------------------------
La corrida también ajusta `xgb_mse`, `xgb_mae`, `xgb_mse_b` y `xgb_mae_b`, que son los
de la corrida publicada (`*_exo_overall`): mismo frame, misma semilla. Tienen que dar
|ΔMAE| = 0 en las 35 rutas; si no, la tabla no se escribe.

Además escribe los descriptivos del bloque de contexto en los cortes
(`*_ablate_exodesc.parquet`, lo que ve el modelo): `analysis/context_descriptives.csv`.

Uso
---
    python docs/paper/analysis/make_table_11_ablation.py --root artefacts/run_bundle_<…>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import numpy as np                                              # noqa: E402
import polars as pl                                             # noqa: E402
from scipy.stats import wilcoxon                                # noqa: E402

from audit_run import ROUTES                                    # noqa: E402
from honest_ontime import RANGE                                 # noqa: E402
from make_tables_06_09_paired_stats import fmt_p, holm          # noqa: E402

TBL = _HERE.parent / "tables"
TAG = "q05c100"

GROUPS = [   # (algoritmo, nombre, features) — los de EXO_GROUPS, nombres de §2.4
    ("xgb_mae_cal", "Calendar", "hour of day (cyclical), day of week, weekend flag"),
    ("xgb_mae_headway", "Headway", "headway ahead and behind (capped at 3,600 s)"),
    ("xgb_mae_dwell", "Dwell", "cumulative and recent dwell"),
    ("xgb_mae_speed", "Recent speed", "recent GPS speed and its dispersion"),
    ("xgb_mae_route", "Route position", "segment count ahead, observed ping count"),
]
SHARED = ["xgb_mse", "xgb_mae", "xgb_mse_b", "xgb_mae_b"]


def _read(root: Path, rng: str, suffix: str, what: str, routes: List[str]) -> pl.DataFrame:
    parts = []
    for r in routes:
        p = (root / "data" / "02_interim" / "routes" / r / "xgb" / rng
             / f"legT0_T0_{TAG}{suffix}_{what}.parquet")
        if p.is_file():
            parts.append(pl.read_parquet(p).with_columns(pl.col("route").cast(pl.Utf8)))
    return pl.concat(parts, how="diagonal_relaxed") if parts else pl.DataFrame()


def check_shared(abl: pl.DataFrame, pub: pl.DataFrame) -> float:
    """Máximo |ΔMAE| de los cuatro modelos compartidos contra la corrida publicada."""
    a = abl.filter(pl.col("algorithm").is_in(SHARED)).select("route", "algorithm", "MAE_s")
    b = pub.filter(pl.col("algorithm").is_in(SHARED)).select(
        "route", "algorithm", pl.col("MAE_s").alias("MAE_pub"))
    j = a.join(b, on=["route", "algorithm"], how="inner")
    if j.height != a.height:
        raise SystemExit(f"✗ {a.height - j.height} filas de la ablación sin par publicado")
    return float((j["MAE_s"] - j["MAE_pub"]).abs().max())


def build(abl: pl.DataFrame) -> pl.DataFrame:
    w = (abl.select("route", "algorithm", "MAE_s")
            .pivot(values="MAE_s", index="route", on="algorithm").sort("route"))
    base = w["xgb_mae"].to_numpy()
    full = (w["xgb_mae_b"].to_numpy() - base) / base * 100.0
    full_med = float(np.median(full))
    rows: List[Dict] = []
    pv = []
    for algo, name, feats in GROUPS:
        x = w[algo].to_numpy()
        rel = (x - base) / base * 100.0
        pv.append(wilcoxon(x - base, alternative="two-sided").pvalue)
        rows.append({"group": name, "features": feats, "algorithm": algo,
                     "mae_median_s": float(np.median(x)),
                     "delta_pct_vs_tierA": float(np.median(rel)),
                     "routes_improved": int((x < base).sum()), "n_routes": len(x),
                     "p_wilcoxon": float(pv[-1]),
                     "share_of_tierB_gain_pct": 100.0 * float(np.median(rel)) / full_med})
    adj = holm(np.array(pv))
    for r, p in zip(rows, adj):
        r["p_holm"] = float(p)
    rows.sort(key=lambda r: r["delta_pct_vs_tierA"])
    rows.append({"group": "Full tier B", "features": "all twelve", "algorithm": "xgb_mae_b",
                 "mae_median_s": float(np.median(w["xgb_mae_b"].to_numpy())),
                 "delta_pct_vs_tierA": full_med,
                 "routes_improved": int((w["xgb_mae_b"].to_numpy() < base).sum()),
                 "n_routes": len(base),
                 "p_wilcoxon": float(wilcoxon(w["xgb_mae_b"].to_numpy() - base).pvalue),
                 "p_holm": float("nan"), "share_of_tierB_gain_pct": 100.0})
    t = pl.DataFrame(rows)
    return t.with_columns(pl.lit(float(np.median(base))).alias("tierA_median_s"))


def to_md(t: pl.DataFrame, max_diff: float) -> str:
    n = int(t["n_routes"][0])
    full = t.filter(pl.col("group") == "Full tier B").row(0, named=True)
    L = [f"**Table S14. Group-wise ablation of the contextual block ({n} routes; "
         f"tier A plus one group at a time).**", ""]
    L.append(
        f"Every variant was fitted in the same cross-validation loop of the common "
        f"evaluation frame, on the same round trips, cut positions and folds, so the "
        f"differences are paired by construction. Deltas are medians over routes of "
        f"the per-route relative change in MAE against tier A (median "
        f"{t['tierA_median_s'][0]:.1f} s). *Routes improved* counts routes where the "
        f"group lowers MAE. Holm adjustment is across the five groups. The share "
        f"column expresses each group median as a fraction of the full tier-B "
        f"improvement ({full['delta_pct_vs_tierA']:+.2f}%). The four models this run "
        + ("shares with the main one reproduce it exactly (|ΔMAE| = 0 in every route)."
           if max_diff == 0 else
           f"shares with the main one reproduce it to |ΔMAE| ≤ {max_diff:.2g} s."))
    L += ["", "| Group | Features | MAE median (s) | Δ vs tier A | Routes improved "
          "| p | p (Holm) | Share of tier-B gain |",
          "|---|---|---|---|---|---|---|---|"]
    for r in t.iter_rows(named=True):
        bold = r["group"] == "Full tier B"
        g = f"**{r['group']}**" if bold else r["group"]
        p = "—" if bold else fmt_p(r["p_wilcoxon"])
        ph = "—" if bold else fmt_p(r["p_holm"])
        L.append(f"| {g} | {r['features']} | {r['mae_median_s']:.2f} | "
                 f"{r['delta_pct_vs_tierA']:+.2f}% | {r['routes_improved']}/{r['n_routes']} "
                 f"| {p} | {ph} | {r['share_of_tierB_gain_pct']:.0f}% |")
    L.append("")
    L.append("*Notes.* The shares need not add to 100%: the groups overlap in what "
             "they tell the model, and per-route increments are not additive.")
    return "\n".join(L) + "\n"


def descriptives(root: Path, rng: str, routes: List[str]) -> pl.DataFrame:
    d = _read(root, rng, "_ablate", "exodesc", routes)
    if d.is_empty():
        return d
    return (d.group_by("feature")
             .agg(pl.len().alias("n_routes"),
                  pl.col("n_cells").sum().alias("n_cells"),
                  *[pl.col(c).median().alias(f"{c}_route_median")
                    for c in ("p10", "p50", "p90", "mean", "share_zero", "share_at_cap")],
                  pl.col("p50").min().alias("p50_min"),
                  pl.col("p50").max().alias("p50_max"))
             .sort("feature"))


def main() -> int:
    ap = argparse.ArgumentParser(description="Tabla S14 — ablación por grupos")
    ap.add_argument("--root", required=True)
    ap.add_argument("--range", default=RANGE)
    ap.add_argument("--partial", action="store_true",
                    help="acepta menos de 35 rutas (una prueba)")
    ap.add_argument("--out", default=str(TBL / "table_11_ablation"))
    a = ap.parse_args()
    root = Path(a.root)

    abl = _read(root, a.range, "_ablate", "overall", ROUTES)
    if abl.is_empty():
        raise SystemExit(f"✗ no hay `*_{TAG}_ablate_overall.parquet` bajo {root}")
    routes = sorted(abl["route"].unique().to_list())
    if len(routes) != len(ROUTES) and not a.partial:
        raise SystemExit(f"✗ ablación de {len(routes)} rutas, se esperan {len(ROUTES)} "
                         f"(--partial para una prueba)")
    pub = _read(root, a.range, "_exo", "overall", routes)
    max_diff = check_shared(abl, pub)
    if max_diff > 0:
        raise SystemExit(f"✗ los modelos compartidos no reproducen la corrida publicada "
                         f"(|ΔMAE| máx = {max_diff:.4g} s): otro frame u otro código")

    t = build(abl)
    Path(a.out + ".csv").write_text(t.write_csv(), encoding="utf-8")
    md = to_md(t, max_diff)
    Path(a.out + ".md").write_text(md, encoding="utf-8")
    print(md)
    desc = descriptives(root, a.range, routes)
    if not desc.is_empty():
        (_HERE / "context_descriptives.csv").write_text(desc.write_csv(), encoding="utf-8")
        print(desc)
    print(f"→ {a.out}.{{csv,md}}, {_HERE / 'context_descriptives.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
