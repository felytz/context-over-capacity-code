"""
make_appendix_h_replication.py
==============================
Tabla S10 del artículo — la réplica: el mismo pipeline, ejecutado otra vez (V3 del manuscrito).

Por qué
-------
El 9k reportaba una réplica de la corrida de agosto (375 de 378 pares idénticos). Esa
corrida no se publica (V2), así que la réplica se rehace con lo que sí va al depósito:

  1. **E6, corrida A** (dentro del bundle publicado): un job aparte que vuelve a ajustar,
     sobre los frames publicados, los cuatro XGBoost de la comparación principal en las
     35 rutas. Es lo que `make_table_11_ablation.py` ya exige (|ΔMAE| = 0); aquí se
     cuentan todas las métricas, no sólo el MAE.
  2. **Los dos pilotos del servidor** (bundles aparte en `artefacts/`, en el MANIFEST):
     ejecuciones desde el crudo AVL —fase de datos, frames y todos los modelos— de cuatro
     rutas (601, 902, 1601, 180101) y, otra vez, de la 601. Los deterministas se comparan
     fila por fila; el LSTM, que no es bit-reproducible en GPU, se mide: cuánto se mueve
     su MAE y su on-time, y si cambia su puesto entre los diez modelos.

Los pilotos corrieron antes de las enmiendas E3 y E4 del pre-registro (§6). E3 (la
atención por trozos) no cambia nada en las rutas que caben en un trozo, y el piloto de
la 601 ya la llevaba; E4 sólo rebina el desglose por horizonte, que esta tabla no usa.

Uso
---
    python docs/paper/analysis/make_appendix_h_replication.py --root artefacts/run_bundle_<…> \\
        --pilots artefacts/run_bundle_<piloto1> artefacts/run_bundle_<piloto2>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import polars as pl

TBL = Path(__file__).resolve().parent.parent / "tables"
RANGE = "2024_01-08_tmin"
TAG = "q05c100"
METRICS = ["n_pairs", "MAE_s", "RMSE_s", "MedAE_s", "P90AE_s", "bias_s", "pct_ontime_dev"]
SHARED = ["xgb_mse", "xgb_mae", "xgb_mse_b", "xgb_mae_b"]
# Las salidas deterministas que un piloto puede traer: sufijo → qué es.
DET = {"exo": "main comparison (nine models)",
       "exo_np_t0": "information ablation (Table 5)",
       "tw_exo": "rolling origin, tabular models"}
LSTM = {"es15": "main comparison", "tw": "rolling origin"}
FRAMES = ["", "_tw"]


def _routes_dir(root: Path) -> Path:
    return root / "data" / "02_interim" / "routes"


def _overall(root: Path, fam: str, suffix: str) -> pl.DataFrame:
    parts = []
    for p in sorted(_routes_dir(root).glob(f"*/{fam}/{RANGE}/legT0_T0_{TAG}_{suffix}_overall.parquet")):
        parts.append(pl.read_parquet(p).with_columns(pl.col("route").cast(pl.Utf8))
                     .select("route", "algorithm", *METRICS))
    return pl.concat(parts) if parts else pl.DataFrame()


def _frame_hash(root: Path, route: str, sfx: str) -> str | None:
    p = _routes_dir(root) / route / "frames" / RANGE / f"legT0_T0_{TAG}{sfx}_meta.json"
    return json.loads(p.read_text(encoding="utf-8"))["frame_hash"] if p.is_file() else None


def _match(a: pl.DataFrame, b: pl.DataFrame) -> Dict[str, float]:
    """Pares (ruta, algoritmo) presentes en las dos; idénticos = todas las métricas iguales."""
    j = a.join(b, on=["route", "algorithm"], suffix="_b")
    same = pl.all_horizontal([pl.col(m) == pl.col(f"{m}_b") for m in METRICS])
    return {"routes": j["route"].n_unique(), "pairs": j.height,
            "identical": int(j.select(same.sum()).item()),
            "max_dmae_s": float(j.select((pl.col("MAE_s") - pl.col("MAE_s_b")).abs().max()).item())}


def _rank(det: pl.DataFrame, route: str, mae: float) -> int:
    """Puesto del LSTM (1 = mejor) entre los nueve deterministas de su ruta y él."""
    return 1 + int(det.filter((pl.col("route") == route) & (pl.col("MAE_s") < mae)).height)


def build(root: Path, pilots: List[Path]) -> tuple[str, pl.DataFrame, pl.DataFrame]:
    det_rows: List[dict] = []

    # 1 · E6, corrida A: los cuatro XGBoost compartidos, sobre los frames publicados.
    abl = _overall(root, "xgb", "ablate").filter(pl.col("algorithm").is_in(SHARED))
    pub = _overall(root, "xgb", "exo").filter(pl.col("algorithm").is_in(SHARED))
    if abl.height:
        det_rows.append({"execution": "E6, run A (published bundle)",
                         "from": "the published frames", "outputs": "four boosted models",
                         "frames_identical": "same files", **_match(pub, abl)})

    # 2 · los pilotos, desde el crudo.
    lstm_rows: List[dict] = []
    pub_main = _overall(root, "xgb", "exo")
    for i, pil in enumerate(pilots, 1):
        name = f"pilot {i} ({pil.name.removeprefix('run_bundle_' + TAG + '_')})"
        routes = sorted(p.name for p in _routes_dir(pil).iterdir() if p.is_dir())
        fr = [(r, s) for r in routes for s in FRAMES if _frame_hash(pil, r, s)]
        same_fr = sum(_frame_hash(pil, r, s) == _frame_hash(root, r, s) for r, s in fr)
        for sfx, what in DET.items():
            a, b = _overall(root, "xgb", sfx), _overall(pil, "xgb", sfx)
            if b.height:
                det_rows.append({"execution": name, "from": "the raw AVL records",
                                 "outputs": what, "frames_identical": f"{same_fr} of {len(fr)}",
                                 **_match(a, b)})
        pil_main = _overall(pil, "xgb", "exo")
        for sfx, what in LSTM.items():
            a, b = _overall(root, "lstm_frame", sfx), _overall(pil, "lstm_frame", sfx)
            j = a.join(b, on=["route", "algorithm"], suffix="_b")
            for r in j.iter_rows(named=True):
                row = {"execution": name, "protocol": what, "route": r["route"],
                       "mae_published_s": r["MAE_s"], "mae_pilot_s": r["MAE_s_b"],
                       "d_mae_s": r["MAE_s"] - r["MAE_s_b"],
                       "d_mae_pct": 100 * (r["MAE_s"] - r["MAE_s_b"]) / r["MAE_s_b"],
                       "d_ontime_pp": r["pct_ontime_dev"] - r["pct_ontime_dev_b"],
                       "same_pairs": r["n_pairs"] == r["n_pairs_b"]}
                if sfx == "es15":
                    row["rank_published"] = _rank(pub_main, r["route"], r["MAE_s"])
                    row["rank_pilot"] = _rank(pil_main, r["route"], r["MAE_s_b"])
                lstm_rows.append(row)

    det = pl.DataFrame(det_rows)
    if not det.height or not lstm_rows:
        sys.exit("✗ nada que comparar: ¿el bundle trae E6 y los pilotos sus rutas?")
    order = {w: i for i, w in enumerate(LSTM.values())}
    lstm = (pl.DataFrame(lstm_rows)
            .with_columns(pl.col("protocol").replace_strict(order).alias("_o"),
                          pl.col("route").cast(pl.Int64).alias("_r"))
            .sort(["_o", "_r"]).drop("_o", "_r"))
    if not lstm["same_pairs"].all():
        sys.exit("✗ el LSTM de un piloto no evalúa los mismos pares que el publicado")

    L: List[str] = ["**Appendix H. Replication: the same pipeline executed again.**", ""]
    L.append(
        "Two kinds of re-execution are compared with the published run. **E6, run A**, a "
        "separate cluster job inside the published bundle, re-fitted the four boosted models "
        "of the main comparison on the published frames in every route. **The two pilots** "
        "executed the whole chain from the raw AVL records — cleaning, round-trip cut, frames "
        "and every model — on four routes, and again on route 601; their bundles are "
        "deposited with the published one. A pair is a (route, model) row of the results; it "
        "is *identical* when the number of prediction pairs and all six error metrics agree "
        "exactly. The deterministic models are reproducible given the same frame, seeds and "
        "number of threads; the sequence model is not bit-reproducible on GPU, so it is "
        "measured instead.")
    L.append("")
    L.append("| Execution | From | Outputs | Routes | Frames identical (hash) | Pairs | "
             "Identical | max \\|ΔMAE\\| (s) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in det.iter_rows(named=True):
        L.append(f"| {r['execution']} | {r['from']} | {r['outputs']} | {r['routes']} | "
                 f"{r['frames_identical']} | {r['pairs']} | {r['identical']} | "
                 f"{r['max_dmae_s']:.2f} |")
    L.append("")
    L.append("| Sequence model | Route | MAE published (s) | MAE pilot (s) | ΔMAE (s) | "
             "ΔMAE (%) | Δ on-time (pp) | Rank of ten, published → pilot |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in lstm.iter_rows(named=True):
        rank = (f"{r['rank_published']} → {r['rank_pilot']}"
                if r.get("rank_published") is not None else "—")
        L.append(f"| {r['protocol']} | {r['route']} | {r['mae_published_s']:.2f} | "
                 f"{r['mae_pilot_s']:.2f} | {r['d_mae_s']:+.2f} | {r['d_mae_pct']:+.2f} | "
                 f"{r['d_ontime_pp']:+.2f} | {rank} |")
    L.append("")

    n_pairs, n_same = int(det["pairs"].sum()), int(det["identical"].sum())
    mx = lstm.select(pl.col("d_mae_s").abs().max()).item()
    mxp = lstm.select(pl.col("d_mae_pct").abs().max()).item()
    mxo = lstm.select(pl.col("d_ontime_pp").abs().max()).item()
    es = lstm.filter(pl.col("protocol") == LSTM["es15"])
    flips = es.filter(pl.col("rank_published") != pl.col("rank_pilot"))
    note = (f"*Note.* Deterministic models: {n_same} of {n_pairs} pairs identical, every "
            f"frame hash equal. Sequence model: on the same frames and the same prediction "
            f"pairs its MAE moves by at most {mx:.1f} s ({mxp:.1f}%) and its on-time rate by "
            f"at most {mxo:.1f} percentage points")
    if flips.height:
        f = flips.row(0, named=True)
        note += (f"; in route {f['route']} that is enough to change its rank among the ten "
                 f"models of the main comparison (rank {f['rank_published']} in the published "
                 f"run, {f['rank_pilot']} in the pilot).")
    else:
        note += "."
    note += (" The pilots predate amendments E3 and E4 of the protocol: E3 leaves routes "
             "whose attention fits in one chunk unchanged, and the pilot of route 601 "
             "already carried it; E4 only re-bins the breakdown by horizon, not used here.")
    L.append(note)
    return "\n".join(L) + "\n", det, lstm


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--root", required=True, type=Path, help="el bundle publicado")
    ap.add_argument("--pilots", nargs="+", required=True, type=Path)
    a = ap.parse_args()
    md, det, lstm = build(a.root, a.pilots)
    (TBL / "appendix_h_replication.md").write_text(md, encoding="utf-8", newline="\n")
    det.write_csv(TBL / "appendix_h_replication_deterministic.csv")
    lstm.write_csv(TBL / "appendix_h_replication_lstm.csv")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
