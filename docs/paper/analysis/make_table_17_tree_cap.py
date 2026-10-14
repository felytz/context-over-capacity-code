"""
make_table_17_tree_cap.py
=========================
Tabla S8 del artículo (archivo `table_17`) — ¿dependen las conclusiones del tope de 600 árboles? (enmienda E6,
segunda parte; sensibilidad, no la corrida principal).

Por qué
-------
La S5 medida (`make_appendix_f_model_fits.py`) muestra que el early stopping NO paró
a muchos ajustes con objetivo de error absoluto antes del tope de 600 árboles: 59 de
175 en XGBoost (MAE, + contexto), 55 en el nivel 1 y 71 en el nivel 0 de la Tabla 5,
contra 10 en el nivel 2. Si un nivel queda más recortado que otro, el reparto de la
Tabla 5 (36 % historia, 64 % contexto) y la paridad del nivel 0 con el LSTM
podrían ser en parte del tope y no de la información.

Esta tabla lee la corrida con el tope en N árboles (`--n-estimators N`, todo lo demás
igual: mismo frame, semilla, folds y early stopping de 30 rondas) y rehace, en las
dos versiones, lo que el tope podría mover:

  1. cuántos ajustes llegan al tope y cuánto cambia el MAE de cada variante;
  2. la escalera de información (Tabla 5): MAE por nivel y el reparto;
  3. el nivel 0 contra el LSTM, y XGBoost (MAE, + contexto) contra el LSTM y contra
     la media por hora;
  4. la ablación por grupos (Tabla S14): el Δ de cada grupo.

Corrida: `run_routes.py --exo-ablate --no-prior --trip-only --no-baselines
--no-frames --n-estimators N` → `*_q05c100_ablate_np_t0_n<N>_*`.

Uso
---
    python docs/paper/analysis/make_table_17_tree_cap.py --root artefacts/run_bundle_<…>
"""
from __future__ import annotations

import argparse
import json
import re
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
from make_table_05_permodel import LABEL                        # noqa: E402
from make_tables_06_09_paired_stats import fmt_p                # noqa: E402

TBL = _HERE.parent / "tables"
TAG = "q05c100"
ES_ROUNDS = 30
LEVELS = ["xgb_mae_t0", "xgb_mae_np", "xgb_mae", "xgb_mae_b"]
GROUPS = ["xgb_mae_cal", "xgb_mae_headway", "xgb_mae_dwell", "xgb_mae_speed",
          "xgb_mae_route"]
MODELS = ["xgb_mse", "xgb_mae", "xgb_mse_b", "xgb_mae_b", "xgb_mae_np", "xgb_mae_t0",
          *GROUPS]
GLABEL = {"xgb_mae_cal": "Tier A + calendar", "xgb_mae_headway": "Tier A + headway",
          "xgb_mae_dwell": "Tier A + dwell", "xgb_mae_speed": "Tier A + recent speed",
          "xgb_mae_route": "Tier A + route position"}


def _dir(root: Path, r: str, rng: str, fam: str = "xgb") -> Path:
    return root / "data" / "02_interim" / "routes" / r / fam / rng


def _overall(root: Path, rng: str, suffix: str, routes: List[str],
             fam: str = "xgb") -> pl.DataFrame:
    parts = []
    for r in routes:
        p = _dir(root, r, rng, fam) / f"legT0_T0_{TAG}{suffix}_overall.parquet"
        if p.is_file():
            parts.append(pl.read_parquet(p).with_columns(pl.col("route").cast(pl.Utf8))
                         .select("route", "algorithm", "MAE_s"))
    return pl.concat(parts) if parts else pl.DataFrame()


def _iters(root: Path, rng: str, suffix: str, routes: List[str]) -> pl.DataFrame:
    rows = []
    for r in routes:
        p = _dir(root, r, rng) / f"legT0_T0_{TAG}{suffix}_meta.json"
        if not p.is_file():
            continue
        m = json.loads(p.read_text(encoding="utf-8"))
        cap = int(m["xgb_params"]["n_estimators"])
        for a, its in m["best_iterations"].items():
            for it in its:
                rows.append({"route": r, "algorithm": a, "trees": int(it) + 1,
                             "at_cap": int(it) >= cap - ES_ROUNDS, "cap": cap})
    return pl.DataFrame(rows)


def _wide(d: pl.DataFrame) -> pl.DataFrame:
    return d.pivot(values="MAE_s", index="route", on="algorithm").sort("route")


def ladder(w: pl.DataFrame, lstm: np.ndarray, look: np.ndarray) -> Dict[str, float]:
    med = {k: float(np.median(w[k].to_numpy())) for k in LEVELS}
    tot = med["xgb_mae_t0"] - med["xgb_mae_b"]
    l0, best = w["xgb_mae_t0"].to_numpy(), w["xgb_mae_b"].to_numpy()
    out = {f"mae_{k}_s": v for k, v in med.items()}
    out.update({
        "share_history_pct": 100.0 * (med["xgb_mae_t0"] - med["xgb_mae"]) / tot,
        "share_context_pct": 100.0 * (med["xgb_mae"] - med["xgb_mae_b"]) / tot,
        "total_gain_pct": 100.0 * float(np.median((best - l0) / l0)),
        "level0_beats_lstm": int((l0 < lstm).sum()),
        "level0_vs_lstm_pct": 100.0 * float(np.median((l0 - lstm) / lstm)),
        "level0_vs_lstm_p": float(wilcoxon(l0 - lstm).pvalue),
        "best_beats_lstm": int((best < lstm).sum()),
        "best_vs_lstm_pct": 100.0 * float(np.median((best - lstm) / lstm)),
        "best_beats_lookup": int((best < look).sum()),
        "best_vs_lookup_pct": 100.0 * float(np.median((best - look) / look)),
    })
    if all(g in w.columns for g in GROUPS):
        base = w["xgb_mae"].to_numpy()
        for g in GROUPS:
            out[f"delta_{g}_pct"] = 100.0 * float(np.median(
                (w[g].to_numpy() - base) / base))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Tabla S8 — sensibilidad al tope de árboles")
    ap.add_argument("--root", required=True)
    ap.add_argument("--range", default=RANGE)
    ap.add_argument("--partial", action="store_true")
    ap.add_argument("--out", default=str(TBL / "table_17_tree_cap"))
    a = ap.parse_args()
    root = Path(a.root)

    # La corrida con el tope alto: `_ablate_np_t0_n<N>`, un solo N.
    found = sorted({m.group(1) for r in ROUTES
                    for p in _dir(root, r, a.range).glob(
                        f"legT0_T0_{TAG}_ablate_np_t0_n*_overall.parquet")
                    if (m := re.search(r"_n(\d+)_overall", p.name))})
    if len(found) != 1:
        raise SystemExit(f"✗ se espera UNA corrida `_ablate_np_t0_n<N>`; hay {found}")
    n_hi = int(found[0])
    hi_suf = f"_ablate_np_t0_n{n_hi}"
    hi = _overall(root, a.range, hi_suf, ROUTES)
    routes = sorted(hi["route"].unique().to_list())
    if len(routes) != len(ROUTES) and not a.partial:
        raise SystemExit(f"✗ {len(routes)} rutas con tope {n_hi}, se esperan {len(ROUTES)}")

    # El tope publicado: la corrida principal, la de la Tabla 5 y la de la S14.
    lo = pl.concat([
        _overall(root, a.range, "_exo", routes),
        _overall(root, a.range, "_exo_np_t0", routes).filter(
            pl.col("algorithm").is_in(["xgb_mae_np", "xgb_mae_t0"])),
        _overall(root, a.range, "_ablate", routes).filter(
            pl.col("algorithm").is_in(GROUPS)),
    ])
    lstm = (_overall(root, a.range, "_es15", routes, fam="lstm_frame")
            .filter(pl.col("algorithm") == "lstm_a").sort("route"))
    w_lo = _wide(lo).filter(pl.col("route").is_in(routes))
    w_hi = _wide(hi)
    assert w_lo["route"].to_list() == w_hi["route"].to_list() == lstm["route"].to_list()
    look = w_lo["delay_hour"].to_numpy()
    ls = lstm["MAE_s"].to_numpy()

    # Los árboles con tope 600, de la corrida de donde sale cada modelo publicado
    # (xgb_mae, por ejemplo, está en las tres, idéntico: se cuenta una vez).
    src600 = {"_exo": ["xgb_mse", "xgb_mae", "xgb_mse_b", "xgb_mae_b"],
              "_exo_np_t0": ["xgb_mae_np", "xgb_mae_t0"], "_ablate": GROUPS}
    it = pl.concat([_iters(root, a.range, s, routes).filter(pl.col("algorithm").is_in(ms))
                    for s, ms in src600.items()]
                   + [_iters(root, a.range, hi_suf, routes)])
    rows: List[Dict] = []
    for m in MODELS:
        if m not in w_hi.columns or m not in w_lo.columns:
            continue
        x_lo, x_hi = w_lo[m].to_numpy(), w_hi[m].to_numpy()
        i_lo = it.filter((pl.col("algorithm") == m) & (pl.col("cap") == 600))
        i_hi = it.filter((pl.col("algorithm") == m) & (pl.col("cap") == n_hi))
        rows.append({
            "model": m, "n_routes": len(x_lo),
            "at_cap_600": int(i_lo["at_cap"].sum()),
            "fits_600": i_lo.height,
            "trees_median_hi": float(i_hi["trees"].median()),
            "at_cap_hi": int(i_hi["at_cap"].sum()), "fits_hi": i_hi.height,
            "mae_median_600_s": float(np.median(x_lo)),
            "mae_median_hi_s": float(np.median(x_hi)),
            "delta_pct": 100.0 * float(np.median((x_hi - x_lo) / x_lo)),
            "routes_improved": int((x_hi < x_lo).sum()),
        })
    per_model = pl.DataFrame(rows)

    claims = {"600": ladder(w_lo, ls, look), str(n_hi): ladder(w_hi, ls, look)}
    keys = list(claims["600"])
    cl = pl.DataFrame({"quantity": keys,
                       "cap_600": [claims["600"][k] for k in keys],
                       f"cap_{n_hi}": [claims[str(n_hi)].get(k, float("nan"))
                                       for k in keys]})
    Path(a.out + ".csv").write_text(per_model.write_csv(), encoding="utf-8")
    Path(a.out + "_claims.csv").write_text(cl.write_csv(), encoding="utf-8")

    n = len(routes)
    L = [f"**Table S8. Sensitivity to the cap on boosted trees ({n} routes; "
         f"600, as published, against {n_hi}).**", ""]
    L.append(
        f"Everything but the cap is unchanged: the same frame, round trips, folds, "
        f"seed and early stopping (30 rounds without improvement on 15% of the "
        f"training trips). A fit *reached the cap* when early stopping never "
        f"triggered. Δ is the median over routes of the per-route relative change in "
        f"MAE from 600 to {n_hi} trees.")
    L += ["", f"| Model | Reached 600 | Trees at cap {n_hi} (median) | Reached {n_hi} | "
          f"MAE 600 (s) | MAE {n_hi} (s) | Δ | Routes improved |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in per_model.iter_rows(named=True):
        name = LABEL.get(r["model"], GLABEL.get(r["model"], r["model"]))
        L.append(f"| {name} | {r['at_cap_600']}/{r['fits_600']} | "
                 f"{r['trees_median_hi']:.0f} | {r['at_cap_hi']}/{r['fits_hi']} | "
                 f"{r['mae_median_600_s']:.1f} | {r['mae_median_hi_s']:.1f} | "
                 f"{r['delta_pct']:+.2f}% | {r['routes_improved']}/{r['n_routes']} |")
    c0, c1 = claims["600"], claims[str(n_hi)]
    L += ["", "| What it could move | 600 trees | " + f"{n_hi} trees |", "|---|---|---|"]

    def row(label: str, f) -> None:
        L.append(f"| {label} | {f(c0)} | {f(c1)} |")
    row("Table 5: MAE of levels 0 / 1 / 2 / 3 (s)",
        lambda c: " / ".join(f"{c[f'mae_{k}_s']:.1f}" for k in LEVELS))
    row("Table 5: share of history / context",
        lambda c: f"{c['share_history_pct']:.0f}% / {c['share_context_pct']:.0f}%")
    row("Table 5: level 0 → level 3", lambda c: f"{c['total_gain_pct']:+.1f}%")
    row("Level 0 against the LSTM",
        lambda c: (f"better on {c['level0_beats_lstm']}/{n}, "
                   f"{c['level0_vs_lstm_pct']:+.1f}% (p {fmt_p(c['level0_vs_lstm_p'])})"))
    row("XGBoost (MAE, + context) against the LSTM",
        lambda c: f"better on {c['best_beats_lstm']}/{n}, {c['best_vs_lstm_pct']:+.1f}%")
    row("XGBoost (MAE, + context) against Delay (hourly)",
        lambda c: f"better on {c['best_beats_lookup']}/{n}, {c['best_vs_lookup_pct']:+.1f}%")
    if "delta_xgb_mae_cal_pct" in c0:
        row("Table S14: Δ of each group vs tier A (cal / headway / dwell / speed / route)",
            lambda c: " / ".join(f"{c[f'delta_{g}_pct']:+.2f}%" for g in GROUPS))
    L.append("")
    L.append("*Notes.* The sequence model and Delay (hourly) are not refitted: they "
             "have no trees. The published tables keep the pre-specified cap of 600.")
    md = "\n".join(L) + "\n"
    Path(a.out + ".md").write_text(md, encoding="utf-8")
    print(md)
    print(f"→ {a.out}.{{csv,md}}, {a.out}_claims.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
