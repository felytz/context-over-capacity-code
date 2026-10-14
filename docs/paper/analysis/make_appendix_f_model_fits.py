"""
make_appendix_f_model_fits.py
=============================
Lo que la Tabla S9 del artículo dice de los ajustes, MEDIDO en la corrida y no escrito a mano
(plan del manuscrito, fase 0: 0.2c, 0.2g y 0.2h).

  1. **Árboles que eligió el early stopping** de cada variante de XGBoost, sobre
     los 175 ajustes (35 rutas × 5 folds), del `*_meta.json` de cada ruta
     (`best_iterations`). El 9k decía «73 bajo error cuadrático, 181 bajo error
     absoluto: muy por debajo del tope», sobre 435 ajustes de agosto.
  2. **Épocas del LSTM** (`es15`): en cuál paró y cuál fue la mejor, por fold,
     leídas del log de cada tarea (`logs/lstm_fr_<ruta>_<job>_<idx>.out`, dentro
     del bundle). El meta no las guarda.
  3. **La secuencia del LSTM**: `max_sequence_length` y `attention_chunks` del meta
     (la vuelta entera, sin la ventana de 200 celdas del 9k; E3).
  4. **La ganancia por splits, por bloque** (`*_exo_importance.parquet`): para la
     frase de §4 «the fitted model leans heavily on its historical block, which
     takes most of the split gain». D3 la había retirado por no tener artefacto.

⚠ Un ajuste «llegó al tope» si su mejor iteración (base 0) es ≥ n_estimators −
early_stopping_rounds (600 − 30 = 570): con eso el entrenamiento alcanzó los 600
árboles sin 30 rondas seguidas sin mejora, o sea, el early stopping no lo paró.

Uso
---
    python docs/paper/analysis/make_appendix_f_model_fits.py --root artefacts/run_bundle_<…>
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

from audit_run import ROUTES                                    # noqa: E402
from honest_ontime import RANGE                                 # noqa: E402
from make_table_05_permodel import LABEL                        # noqa: E402

TBL = _HERE.parent / "tables"
ANA = _HERE

N_ESTIMATORS = 600          # func/xgb_prediction/model.py, XGB_PARAMS
ES_ROUNDS = 30              # func/xgb_prediction/model.py, early_stopping_rounds
AT_CAP = N_ESTIMATORS - ES_ROUNDS
XGB = ["xgb_mse", "xgb_mae", "xgb_mse_b", "xgb_mae_b"]
ABL = ["xgb_mae_np", "xgb_mae_t0"]

#: Bloques para la ganancia, los niveles de la Tabla 5 y los grupos de §2.4.
HIST = ["prior_mean", "prior_p50", "prior_p90", "prior_std"]
HOOK = ["delay_vs_mu"]
CONTEXT = {
    "calendar": ["hour_sin", "hour_cos", "day_of_week", "is_weekend"],
    "headway": ["headway_ahead", "headway_behind"],
    "dwell": ["dwell_cum_s", "dwell_recent_s"],
    "recent speed": ["speed_recent_5min", "speed_std_obs"],
    "route position": ["segment_count_ahead", "n_pings_obs"],
}


def _route_dir(root: Path, r: str) -> Path:
    return root / "data" / "02_interim" / "routes" / r


def trees(root: Path) -> pl.DataFrame:
    rows: List[Dict] = []
    for r in ROUTES:
        for tag, algos in (("_exo", XGB), ("_exo_np_t0", ABL)):
            p = _route_dir(root, r) / "xgb" / RANGE / f"legT0_T0_q05c100{tag}_meta.json"
            if not p.is_file():
                raise SystemExit(f"✗ falta {p}")
            m = json.loads(p.read_text(encoding="utf-8"))
            if int(m["xgb_params"]["n_estimators"]) != N_ESTIMATORS:
                raise SystemExit(f"✗ {p.name}: n_estimators ≠ {N_ESTIMATORS}")
            for a in algos:
                for fold, it in enumerate(m["best_iterations"][a]):
                    rows.append({"route": r, "algorithm": a, "fold": fold,
                                 "best_iteration": int(it)})
    return pl.DataFrame(rows)


_STOP = re.compile(r"Early stopping en epoch (\d+) \(sin mejora desde epoch (\d+)\)")
_FULL = re.compile(r"Entrenamiento completo en .* \((\d+) epochs\)")
_BEST = re.compile(r"Mejor val_loss = [\d.]+ en epoch (\d+)")
_ROUTE = re.compile(r"^Ruta\s*:\s*(\S+)", re.M)
_TAG = re.compile(r"artefactos → \*_(q05c100[^*]*)_\*\.parquet")


def epochs(root: Path) -> pl.DataFrame:
    """Una fila por (ruta, fold) del LSTM `es15`, desde los logs del bundle."""
    rows: List[Dict] = []
    for log in sorted((root / "logs").glob("lstm_fr_*.out")):
        txt = log.read_text(encoding="utf-8", errors="replace")
        tag, route = _TAG.search(txt), _ROUTE.search(txt)
        if not tag or not route or tag.group(1) != "q05c100_es15":
            continue
        trained = [int(x) for x in _FULL.findall(txt)]
        best = [int(x) for x in _BEST.findall(txt)]
        stopped = len(_STOP.findall(txt))
        if len(trained) != 5 or len(best) != 5:
            raise SystemExit(f"✗ {log.name}: {len(trained)} folds entrenados, se esperan 5")
        n_ep = int(re.search(r"epochs=(\d+)", txt).group(1))
        for f, (e, b) in enumerate(zip(trained, best)):
            rows.append({"route": route.group(1), "fold": f, "max_epochs": n_ep,
                         "epochs_trained": e, "best_epoch": b,
                         "early_stopped": f < stopped})
    d = pl.DataFrame(rows)
    if d.is_empty() or d["route"].n_unique() != len(ROUTES):
        raise SystemExit(f"✗ logs del LSTM es15 de {d['route'].n_unique() if not d.is_empty() else 0}"
                         f" rutas, se esperan {len(ROUTES)}")
    return d


def lstm_meta(root: Path) -> pl.DataFrame:
    rows = []
    for r in ROUTES:
        p = (_route_dir(root, r) / "lstm_frame" / RANGE
             / "legT0_T0_q05c100_es15_meta.json")
        m = json.loads(p.read_text(encoding="utf-8"))
        rows.append({"route": r, "n_cells": int(m["frame"]["n_cells"]),
                     "max_sequence_length": int(m["max_sequence_length"]),
                     "attention_chunks": int(m.get("attention_chunks", 1)),
                     "batch_size": int(m["batch_size"]), "es_frac": m["es_frac"]})
    return pl.DataFrame(rows)


def gain(root: Path) -> pl.DataFrame:
    """Fracción de la ganancia por bloque, por ruta y variante."""
    parts = []
    for r in ROUTES:
        p = (_route_dir(root, r) / "xgb" / RANGE
             / "legT0_T0_q05c100_exo_importance.parquet")
        parts.append(pl.read_parquet(p).with_columns(pl.col("route").cast(pl.Utf8)))
    d = pl.concat(parts)
    block = {f: "historical prior" for f in HIST}
    block.update({f: "delay vs. mean profile" for f in HOOK})
    for g, fs in CONTEXT.items():
        block.update({f: f"context: {g}" for f in fs})
    d = d.with_columns(pl.col("feature").replace_strict(
        block, default="current trip").alias("block"))
    return (d.group_by("route", "algorithm", "block")
             .agg(pl.col("gain_frac").sum())
             .sort("route", "algorithm", "block"))


def _q(x: np.ndarray) -> Dict[str, float]:
    return {"median": float(np.median(x)), "p25": float(np.percentile(x, 25)),
            "p75": float(np.percentile(x, 75)), "min": float(x.min()),
            "max": float(x.max())}


def summarize(tr: pl.DataFrame, ep: pl.DataFrame, lm: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for a in XGB + ABL:
        x = tr.filter(pl.col("algorithm") == a)["best_iteration"].to_numpy() + 1
        rows.append({"model": a, "quantity": "trees selected", "n_fits": len(x),
                     **_q(x), "n_at_cap": int((x - 1 >= AT_CAP).sum()),
                     "cap": N_ESTIMATORS})
    for col, what in (("epochs_trained", "epochs trained"),
                      ("best_epoch", "best epoch")):
        x = ep[col].to_numpy()
        rows.append({"model": "lstm_a", "quantity": what, "n_fits": len(x), **_q(x),
                     "n_at_cap": int((~ep["early_stopped"]).sum()),
                     "cap": int(ep["max_epochs"].max())})
    x = lm["max_sequence_length"].to_numpy()
    rows.append({"model": "lstm_a", "quantity": "sequence length (cells)",
                 "n_fits": len(x), **_q(x),
                 "n_at_cap": int((lm["attention_chunks"] > 1).sum()), "cap": -1})
    return pl.DataFrame(rows)


def gain_summary(g: pl.DataFrame) -> pl.DataFrame:
    return (g.group_by("algorithm", "block")
             .agg(pl.col("gain_frac").median().alias("median"),
                  pl.col("gain_frac").quantile(0.25, "linear").alias("p25"),
                  pl.col("gain_frac").quantile(0.75, "linear").alias("p75"),
                  pl.len().alias("n_routes"))
             .sort(["algorithm", "median", "block"],
                   descending=[False, True, False]))


def to_md(s: pl.DataFrame, gs: pl.DataFrame, lm: pl.DataFrame) -> str:
    n_routes = lm.height
    L = ["**Table S9 (measured). What the fits actually did "
         f"({n_routes} routes × 5 folds).**", ""]
    L.append(
        f"Trees: the number the early stopping selected, on 15% of the training "
        f"trips held out, out of a cap of {N_ESTIMATORS}; a fit *reached the cap* "
        f"when the 30-round patience never triggered before tree {N_ESTIMATORS}. "
        f"Epochs: the sequence model (honest early stopping), read from each "
        f"task's log; *reached the cap* counts fits that were not stopped early. "
        f"Sequence length: the cells of each route's round trip fed to the network "
        f"(no window); the last column counts the routes whose attention is "
        f"computed in horizon chunks (amendment E3).")
    L += ["", "| Model | Quantity | Fits | Median [IQR] | Range | Reached the cap |",
          "|---|---|---:|---|---|---:|"]
    for r in s.iter_rows(named=True):
        cap = (f"{r['n_at_cap']}/{r['n_fits']} (cap {r['cap']})" if r["cap"] > 0
               else f"{r['n_at_cap']} routes chunked")
        L.append(f"| {LABEL.get(r['model'], r['model'])} | {r['quantity']} | "
                 f"{r['n_fits']} | {r['median']:.0f} [{r['p25']:.0f}, {r['p75']:.0f}] | "
                 f"{r['min']:.0f}–{r['max']:.0f} | {cap} |")
    L += ["", "**Split-gain share by block (median [IQR] over routes; the "
          "harness's `gain_frac`, summed within each block).**", "",
          "| Model | Block | Median | IQR |", "|---|---|---:|---|"]
    for r in gs.iter_rows(named=True):
        L.append(f"| {LABEL.get(r['algorithm'], r['algorithm'])} | {r['block']} | "
                 f"{100 * r['median']:.1f}% | [{100 * r['p25']:.1f}, "
                 f"{100 * r['p75']:.1f}]% |")
    L.append("")
    L.append(
        "*Notes.* Split gain measures how a fitted model allocates its splits among "
        "the features it was given, not what each block is worth: that is the "
        "ablation of Table 5.")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Tabla S9 medida")
    ap.add_argument("--root", required=True, help="bundle descomprimido de la corrida")
    ap.add_argument("--out", default=str(TBL / "appendix_f_model_fits"))
    a = ap.parse_args()
    root = Path(a.root)

    tr, ep, lm, g = trees(root), epochs(root), lstm_meta(root), gain(root)
    s, gs = summarize(tr, ep, lm), gain_summary(g)
    Path(a.out + ".csv").write_text(s.write_csv(), encoding="utf-8")
    (ANA / "xgb_gain_by_block.csv").write_text(gs.write_csv(), encoding="utf-8")
    (ANA / "model_fits_per_fold.csv").write_text(
        tr.sort("route", "algorithm", "fold").write_csv(), encoding="utf-8")
    (ANA / "lstm_epochs_per_fold.csv").write_text(
        ep.sort("route", "fold").write_csv(), encoding="utf-8")
    md = to_md(s, gs, lm)
    Path(a.out + ".md").write_text(md, encoding="utf-8")
    print(md)
    print(f"→ {a.out}.{{csv,md}}, {ANA / 'xgb_gain_by_block.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
