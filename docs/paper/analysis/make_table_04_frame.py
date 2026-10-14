"""
make_table_04_frame.py
======================
Tabla S11 del artículo (archivo `table_04`) — comparación pareada por ruta sobre el FRAME de evaluación.

Sustituye a `make_table_02_matched.py`, que implementaba el protocolo *matched*
a posteriori (`idx_off` / `dt_rem` / columna `cov`). Ese protocolo era el parche
para cruzar dos corridas con poblaciones y particiones distintas; con el frame
único (`func/eval_frame.py`) las tres familias comparten muestra, geometría,
cortes y folds **por construcción**, así que:

  · `cov` desaparece — es 1.0 para todas las legs, por definición;
  · `n_pairs` es idéntico entre familias (verificado en 83/83 legs);
  · no hay certificado que emitir: lo sustituye un `assert` en el harness.

La tabla vieja se conserva como apéndice del régimen anterior (q=0.95).

Unidad y agregación
-------------------
Con la vuelta la unidad es la **ruta** (una leg ``T0->T0`` cada una), así que esta
tabla es la fila de cada unidad: deja ver que el resultado no lo arrastran dos o tres
rutas. Solo entran las que cumplen el criterio pre-especificado (`leg_inclusion.py`:
el corte no sale mal, `M >= 100`); las `revisar` llevan su marca. El LSTM es el `es15`
(`honest_ontime._read_overall` lo toma por defecto). El agrupado ponderado por
`n_pairs`, que en agosto juntaba las dos legs de una ruta, se queda: con una leg por
ruta no cambia nada.

Uso
---
    python docs/paper/analysis/make_table_04_frame.py --root artefacts/run_bundle_<…>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import polars as pl                                             # noqa: E402

from audit_run import qtag, route_label, route_order           # noqa: E402
from honest_ontime import RANGE, _read_overall                  # noqa: E402
from leg_inclusion import MIN_M, apply_criteria, collect_legs    # noqa: E402

# Los cuatro que cuentan la historia del 2×2 features × arquitectura.
# `delay` = piso naïve · `delay_hour` = el mismo piso con prior por hora, sin
# entrenamiento · `xgb_mae_b` = el mejor tabular · `lstm_a` = la arquitectura.
ALGOS = ["delay", "delay_hour", "xgb_mae_b", "lstm_a"]
REF = "lstm_a"


def build(root: Path, q: float, cov: float, rng: str, min_M: int,
          lstm_root: Path = None, lstm_tag: str = None) -> pl.DataFrame:
    legs = apply_criteria(collect_legs(root, q, cov, rng), min_M)
    keep = legs.filter(pl.col("included")).select(["route", "leg", "M", "flag"])

    ov = _read_overall(root, qtag(q, cov), rng,
                       lstm_root=lstm_root, lstm_tag=lstm_tag)
    if ov.is_empty():
        raise SystemExit(f"✗ no hay agregados bajo {root}")
    d = ov.join(keep, on=["route", "leg"], how="inner")

    # Solo legs con las cuatro familias: comparar medianas o pools sobre
    # conjuntos distintos de legs no dice nada (HANDOFF_FRAME.md §5.5).
    ok = (d.filter(pl.col("algorithm").is_in(ALGOS))
          .group_by(["route", "leg"]).agg(pl.col("algorithm").n_unique().alias("k"))
          .filter(pl.col("k") == len(ALGOS)).select(["route", "leg"]))
    d = d.join(ok, on=["route", "leg"], how="inner")

    per_route = (
        d.filter(pl.col("algorithm").is_in(ALGOS))
        .group_by(["route", "algorithm"])
        .agg((pl.col("MAE_s") * pl.col("n_pairs")).sum().alias("_w"),
             pl.col("n_pairs").sum().alias("n_pairs"),
             pl.col("leg").n_unique().alias("legs"))
        .with_columns((pl.col("_w") / pl.col("n_pairs") / 60.0).alias("MAE_min"))
        .drop("_w"))

    # Columnas en el orden de ALGOS: el pivot las deja en el orden en que el group_by
    # de arriba encuentra los algoritmos, que cambia de una corrida a otra
    wide = (per_route.pivot(values="MAE_min", index="route", on="algorithm")
            .select("route", *ALGOS)
            .join(per_route.group_by("route")
                  .agg(pl.col("n_pairs").max(), pl.col("legs").max()),
                  on="route"))
    trips = (keep.join(ok, on=["route", "leg"], how="inner")
             .group_by("route").agg(pl.col("M").sum().alias("trips"),
                                    pl.col("flag").first().alias("flag")))
    wide = wide.join(trips, on="route", how="left")

    for a in ALGOS:
        if a == REF:
            continue
        wide = wide.with_columns(
            ((pl.col(a) - pl.col(REF)) / pl.col(REF) * 100).alias(f"d_{a}"))
    best = pl.min_horizontal([pl.col(a) for a in ALGOS])
    wide = wide.with_columns(
        best.alias("best_mae"),
        pl.concat_list([pl.col(a) for a in ALGOS])
        .list.arg_min().alias("_bi"))
    wide = wide.with_columns(
        pl.col("_bi").map_elements(lambda i: ALGOS[int(i)],
                                   return_dtype=pl.Utf8).alias("best")).drop("_bi")
    # Orden: el de las líneas (audit_run.ROUTES, cada variante junto a su principal),
    # no el de `route` como texto, que deja "1001, 101, 102, 110101…".
    return wide.with_columns(
        pl.col("route").map_elements(route_order, return_dtype=pl.Int32).alias("_o"),
    ).sort("_o").drop("_o")


def to_markdown(w: pl.DataFrame, min_M: int) -> str:
    n_routes = w.height
    L: List[str] = []
    L.append(f"**Table S11. Paired per-route comparison on the common evaluation "
             f"frame (January–August 2024).**")
    L.append("")
    L.append(
        f"All algorithms are trained and scored on identical round trips, an identical "
        f"distance grid, identical cut positions and identical folds, defined once "
        f"per route by the evaluation frame; the number of evaluated (trip, position, "
        f"horizon) pairs is therefore identical across algorithms by construction "
        f"rather than by post-hoc matching. Routes enter under a pre-specified "
        f"criterion (trip cut not rated poor, at least {min_M} round trips "
        f"retained by the grid); {n_routes} routes qualify and have all four "
        f"algorithms, each with one unit, its round trip. Δ columns are relative "
        f"to the LSTM: negative means the competing model is more accurate. In "
        f"the summary row, counts are totals and MAE and Δ are medians across "
        f"routes. † marks routes whose trip cut is rated *review* (Table S4).")
    L.append("")
    L.append("| Route | Round trips | n pairs | LSTM | delay | "
             "delay_hour | xgb_mae_b | Δ delay_hour | Δ xgb_mae_b | Best |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in w.iter_rows(named=True):
        L.append(
            f"| {route_label(r['route'], number=True)}{' †' if r.get('flag') else ''} | "
            f"{int(r['trips']):,} | {int(r['n_pairs']):,} | "
            f"{r['lstm_a']:.2f} | {r['delay']:.2f} | {r['delay_hour']:.2f} | "
            f"{r['xgb_mae_b']:.2f} | {r['d_delay_hour']:+.1f}% | "
            f"{r['d_xgb_mae_b']:+.1f}% | `{r['best']}` |")

    med = {a: float(w[a].median()) for a in ALGOS}
    L.append(
        f"| **All (totals / medians)** | **{int(w['trips'].sum()):,}** | "
        f"**{int(w['n_pairs'].sum()):,}** | **{med['lstm_a']:.2f}** | "
        f"**{med['delay']:.2f}** | **{med['delay_hour']:.2f}** | "
        f"**{med['xgb_mae_b']:.2f}** | "
        f"**{float(w['d_delay_hour'].median()):+.1f}%** | "
        f"**{float(w['d_xgb_mae_b'].median()):+.1f}%** |  |")

    n_win_dh = int((w["d_delay_hour"] < 0).sum())
    n_win_xg = int((w["d_xgb_mae_b"] < 0).sum())
    n_win_dl = int((w["delay"] < w["lstm_a"]).sum())
    n_lstm_best = int((w["best"] == "lstm_a").sum())
    L.append("")
    L.append(
        f"*Notes.* MAE is in minutes. The LSTM is the best of the four in "
        f"{n_lstm_best} of {n_routes} routes. An hour-of-day historical mean "
        f"(`delay_hour`), which requires no training and no GPU, is more accurate "
        f"in {n_win_dh} of {n_routes} routes; the gradient-boosted tabular model "
        f"with contextual features (`xgb_mae_b`) in {n_win_xg} of {n_routes}; and "
        f"the plain historical mean (`delay`) in {n_win_dl}. On-time percentages "
        f"are deliberately absent here: under the "
        f"asymmetric operational band they are confounded with residual centring, "
        f"and are reported with that decomposition in Table 4.")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Tabla S11 — comparación sobre el frame")
    ap.add_argument("--root", default=None)
    ap.add_argument("--lstm-root", default=None, dest="lstm_root",
                    help="raiz alterna SOLO para la familia LSTM")
    ap.add_argument("--lstm-tag", default=None, dest="lstm_tag",
                    help="sufijo de regimen del LSTM, p. ej. q05c100_es15")
    ap.add_argument("--q", type=float, default=0.05)
    ap.add_argument("--cov", type=float, default=1.0)
    ap.add_argument("--range", dest="rng", default=RANGE)
    ap.add_argument("--min-M", type=int, default=MIN_M)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    repo = Path(__file__).resolve().parents[3]
    root = Path(a.root).resolve() if a.root else repo
    lroot = Path(a.lstm_root).resolve() if a.lstm_root else None
    w = build(root, a.q, a.cov, a.rng, a.min_M, lroot, a.lstm_tag)

    prefix = Path(a.out) if a.out else (repo / "docs" / "paper" / "tables"
                                        / "table_04_frame_comparison")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    w.write_csv(prefix.with_suffix(".csv"))
    md = to_markdown(w, a.min_M)
    prefix.with_suffix(".md").write_text(md, encoding="utf-8")
    print(md)
    print(f"  ✓ {prefix.with_suffix('.csv')}")
    print(f"  ✓ {prefix.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
