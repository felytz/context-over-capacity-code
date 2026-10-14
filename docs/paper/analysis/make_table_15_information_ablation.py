"""
make_table_15_information_ablation.py
=====================================
Tabla 5 del artículo (archivo `table_15`) — la escalera de INFORMACIÓN, cuatro niveles sobre el panel principal.

Por qué esta tabla y no la 11
-----------------------------
La Tabla 11 ablaciona el bloque contextual por grupos, pero corre sobre 43 legs
de 11 rutas, con tope de 1,200 viajes y **fuera del frame común**. Para una
afirmación secundaria eso basta; para la tesis del paper —que lo que repaga es
la información y no la profundidad— no, y el dictamen de agosto-2026 lo señala.

Esta corre sobre el **panel pre-registrado**, en el mismo loop de validación
cruzada que el resto, así que la comparación entre niveles es pareada por
construcción igual que todas las demás del paper. Con la vuelta el panel son las
rutas que deja pasar `leg_inclusion.py` (una vuelta cada una) y que tienen el LSTM
`es15`; en agosto se tomaba «las legs que tienen LSTM», que entonces coincidía con el
panel porque el `es15` sólo se corrió ahí.

Los cuatro niveles
------------------
    0  xgb_mae_t0   trayectoria del viaje en curso, y nada histórico
    1  xgb_mae_np   + `delay_vs_mu` = t_l − μ_l
    2  xgb_mae      + el bloque `prior_*` completo (posición × horizonte)
    3  xgb_mae_b    + el bloque de contexto operacional

⚠ El nivel 0 existe porque el 1 no era lo que su nombre decía. `delay_vs_mu`
se lee como una cantidad del viaje en curso —"el retraso actual"— pero se
calcula contra la media por posición del fold de ENTRENAMIENTO. Sin separarlo,
el "nivel trayectoria" llevaba condicionamiento histórico dentro y la
atribución salía sesgada a la baja.

⚠ Y el reparto se lee por MEDIANAS DE PANEL, no sumando efectos por leg: los
incrementos por leg no son aditivos y su suma no reconstruye el total.

Uso
---
    python docs/paper/analysis/make_table_15_information_ablation.py \
        --root final_pipeline/run_bundle_q05c100_20260825_173936
"""
from __future__ import annotations

import argparse
import glob
import re
import sys
from pathlib import Path
from typing import Dict, List

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import numpy as np                                              # noqa: E402
import polars as pl                                             # noqa: E402
from scipy.stats import wilcoxon                                # noqa: E402

from honest_ontime import RANGE                                 # noqa: E402
from leg_inclusion import MIN_M, apply_criteria, collect_legs   # noqa: E402

TBL = _HERE.parent / "tables"

#: (columna, etiqueta, qué añade respecto del nivel anterior)
LEVELS = [
    ("xgb_mae_t0", "0 · current-trip only",
     "the trip's own distance–time trace on the grid, and nothing estimated "
     "from other trips"),
    ("xgb_mae_np", "1 · + delay vs. mean profile",
     "one historical hook: the trip's current deviation from the training "
     "fold's mean profile"),
    ("xgb_mae", "2 · + historical conditioning",
     "the full distribution of remaining time by position and horizon"),
    ("xgb_mae_b", "3 · + operational context",
     # Los cinco grupos de EXO_GROUPS (func/xgb_prediction/exo_features.py), con
     # los nombres de §2.4 del manuscrito. El quinto, `route`, son
     # `segment_count_ahead` y `n_pings_obs`: decía sólo «data density».
     "calendar, headway ahead and behind, dwell, recent speed and route "
     "position"),
]


def load(root: str, fam: str, tag: str) -> pl.DataFrame:
    """Los `*_overall.parquet` del régimen, con el sufijo ANCLADO.

    Sin anclar, un tag vacío casa también con `_es15`, `_tw` y `_exo_np_t0`, y
    la tabla saldría mezclando corridas sin avisar.
    """
    out = []
    for p in glob.glob(f"{root}/**/{fam}/**/*_overall.parquet", recursive=True):
        stem = p.replace("\\", "/").rsplit("/", 1)[1]
        if re.search(rf"_q05c100{re.escape(tag)}_overall\.parquet$", stem):
            out.append(pl.read_parquet(p))
    if not out:
        raise SystemExit(f"sin artefactos bajo {root} con tag={tag!r}")
    return pl.concat(out)


def build(root: str, lstm_tag: str) -> tuple:
    abl = load(root, "xgb", "_exo_np_t0")
    lstm = load(root, "lstm_frame", lstm_tag).filter(
        pl.col("algorithm") == "lstm_a")

    # El panel: las rutas del criterio pre-registrado que tienen el LSTM.
    inc = (apply_criteria(collect_legs(Path(root), 0.05, 1.0, RANGE), MIN_M)
           .filter(pl.col("included")).select(["route", "leg"]))
    lstm = (lstm.select(pl.col("route").cast(pl.Utf8), pl.col("leg").cast(pl.Utf8),
                        (pl.col("MAE_s") / 60.0).alias("lstm_min"))
                .join(inc, on=["route", "leg"], how="inner"))

    cols = [c for c, _, _ in LEVELS]
    m = (abl.with_columns(pl.col("route").cast(pl.Utf8), pl.col("leg").cast(pl.Utf8))
            .select("route", "leg", "algorithm", "MAE_s")
            .pivot(values="MAE_s", index=["route", "leg"], on="algorithm"))
    missing = [c for c in cols if c not in m.columns]
    if missing:
        raise SystemExit(
            f"faltan niveles en los artefactos: {missing}. ¿Corriste con "
            f"--exo --no-prior --trip-only?")
    # ⚠ Alineado por JOIN, no por orden: antes se ordenaban las dos tablas por
    # (route, leg) y se suponía que tenían las mismas filas.
    m = m.join(lstm, on=["route", "leg"], how="inner").sort("route", "leg")
    if m.is_empty():
        raise SystemExit(f"ninguna ruta del panel tiene a la vez la ablación y el "
                         f"LSTM `{lstm_tag}` bajo {root}")
    lstm_mae = m["lstm_min"].to_numpy()

    v = {c: m[c].to_numpy() / 60.0 for c in cols}
    total = np.median(v[cols[0]]) - np.median(v[cols[-1]])

    def _wil(d: np.ndarray):
        """Wilcoxon; None si no se puede (todas las diferencias cero o n < 2)."""
        try:
            return float(wilcoxon(d).pvalue)
        except ValueError:
            return None

    rows: List[Dict] = []
    prev = None
    for c, label, adds in LEVELS:
        x = v[c]
        r = {"level": label, "algorithm": c, "adds": adds,
             "n_legs": m.height, "mae_min": float(np.median(x)),
             "rel_pct": None, "improved": None, "p": None, "share_pct": None}
        if prev is not None:
            d = x - prev
            r["rel_pct"] = float(np.median((x - prev) / prev) * 100.0)
            r["improved"] = int((x < prev).sum())
            r["p"] = _wil(d)
            r["share_pct"] = float(
                (np.median(prev) - np.median(x)) / total * 100.0)
        rows.append(r)
        prev = x

    t = pl.DataFrame(rows)
    extra = {
        "n_routes": m["route"].n_unique(),
        "total_rel": float(np.median((v[cols[-1]] - v[cols[0]]) / v[cols[0]])
                           * 100.0),
        "hist_share": float((np.median(v[cols[0]]) - np.median(v["xgb_mae"]))
                            / total * 100.0),
        "lstm_mae": float(np.median(lstm_mae)),
        "lvl0_beats_lstm": int((v[cols[0]] < lstm_mae).sum()),
        "lvl0_vs_lstm_pct": float(
            np.median((v[cols[0]] - lstm_mae) / lstm_mae) * 100.0),
        "lvl0_p": _wil(lstm_mae - v[cols[0]]),
    }
    return t, extra


def fmt_p(x) -> str:
    return "—" if x is None else ("< 0.0001" if x < 0.00005 else f"{x:.4f}")


def to_md(t: pl.DataFrame, e: Dict) -> str:
    n = int(t["n_legs"][0])
    L = [f"**Table 5. What each source of information buys "
         f"({n} routes; paired within one cross-validation).**", ""]
    L.append(
        "Each level adds one source to the level above it, and all four are "
        "fitted in the same cross-validation loop on the same round trips, cut "
        "positions and folds, so the increments are paired by construction. "
        "*Gain* is the median per-route change in MAE against the level above; "
        "*share* is that level's contribution to the total improvement from "
        "level 0 to level 3, read from panel medians.")
    L += ["", "| Level | What it adds | MAE (min) | Gain | Routes improved "
          "| *p* | Share of total |",
          "|---|---|---:|---:|---:|---:|---:|"]
    for r in t.iter_rows(named=True):
        gain = "—" if r["rel_pct"] is None else f"{r['rel_pct']:+.1f}%"
        imp = "—" if r["improved"] is None else f"{r['improved']}/{r['n_legs']}"
        sh = "—" if r["share_pct"] is None else f"{r['share_pct']:.0f}%"
        L.append(f"| {r['level']} | {r['adds']} | {r['mae_min']:.3f} | "
                 f"{gain} | {imp} | {fmt_p(r['p'])} | {sh} |")

    # ⚠ La nota de agosto afirmaba «the step from level 0 to level 1 is not
    # significant» como frase fija, y «level 0 is more accurate [than the LSTM]»: con
    # otra corrida podían quedar falsas sin que nada fallara. Cada una sale ahora sólo
    # si los números la sostienen.
    total = e["total_rel"]
    verb = "improves" if total < 0 else "worsens"
    note = (f"*Notes.* Going from level 0 to level 3 {verb} MAE by "
            f"{abs(total):.1f}%. Of that change, historical conditioning — levels 1 "
            f"and 2 together — accounts for {e['hist_share']:.0f}%, and operational "
            f"context for the rest.")
    p01 = t["p"][1]
    if p01 is not None and p01 >= 0.05:
        note += (" The step from level 0 to level 1 is not significant: "
                 "`delay_vs_mu` is the trip's elapsed time minus a constant that "
                 "depends only on the cut position, and both of those are already "
                 "features at level 0, so it carries almost no information a tree "
                 "can use.")
    note += (" The comparison is of information, not of model class: all four "
             "levels are the same gradient-boosted learner under the same "
             "absolute-error objective.")
    L += ["", note]

    k, pct = e["lvl0_beats_lstm"], e["lvl0_vs_lstm_pct"]
    side = ("more accurate" if pct < 0 else "less accurate")
    L += ["", f"For reference on the same panel, the sequence model reaches "
          f"{e['lstm_mae']:.3f} min. Level 0 — which is given no historical "
          f"summary and no context at all — is more accurate than it on {k} of "
          f"{n} routes, and {side} by a median {abs(pct):.1f}% "
          f"(*p* {fmt_p(e['lvl0_p'])}). The "
          f"sequence model receives the full trajectory as a sequence and the "
          f"tabular model seven summaries of it, so this contrast is not a "
          f"difference in what was observed but in how it was represented."]
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="paquete con los artefactos `_exo_np_t0`")
    ap.add_argument("--lstm-tag", default="_es15")
    ap.add_argument("--out", default=str(TBL / "table_15_information_ablation"))
    a = ap.parse_args()

    t, e = build(a.root, a.lstm_tag)
    Path(a.out + ".csv").write_text(t.write_csv(), encoding="utf-8")
    Path(a.out + ".md").write_text(to_md(t, e), encoding="utf-8")

    print(f"panel: {e['n_routes']} rutas\n")
    print(f"  {'nivel':30s} {'MAE':>7s} {'ganancia':>10s} {'mejora':>8s} "
          f"{'p':>10s} {'cuota':>7s}")
    for r in t.iter_rows(named=True):
        g = "—" if r["rel_pct"] is None else f"{r['rel_pct']:+.1f}%"
        i = "—" if r["improved"] is None else f"{r['improved']}/{r['n_legs']}"
        s = "—" if r["share_pct"] is None else f"{r['share_pct']:.0f}%"
        print(f"  {r['level']:30s} {r['mae_min']:7.3f} {g:>10s} {i:>8s} "
              f"{fmt_p(r['p']):>10s} {s:>7s}")
    print(f"\n  total 0->3            : {e['total_rel']:+.1f}%")
    print(f"  cuota del historico   : {e['hist_share']:.0f}%")
    print(f"  cuota del contexto    : {100 - e['hist_share']:.0f}%")
    print(f"\n  LSTM                  : {e['lstm_mae']:.3f} min")
    print(f"  nivel 0 le gana en    : {e['lvl0_beats_lstm']}/"
          f"{int(t['n_legs'][0])} rutas "
          f"({e['lvl0_vs_lstm_pct']:+.1f}%, p={fmt_p(e['lvl0_p'])})")
    print(f"\n→ {a.out}.{{csv,md}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
