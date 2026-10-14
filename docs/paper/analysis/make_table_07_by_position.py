"""
make_table_07_by_position.py
============================
Tabla S16 del artículo (archivo `table_07`) — rendimiento por POSICIÓN DE CORTE a lo largo del viaje.

Sustituye a la versión del régimen anterior (`table_05_by_pfrac`), que contrastaba
el LSTM contra el mejor baseline sobre el subconjunto matched.

Fuente: los agregados `*_by_position.parquet` de las dos familias, que el harness
emite por (algoritmo, `l_idx`) con el MISMO esquema. La posición se normaliza a
`pfrac = l_idx / n_cells` con el `n_cells` del frame de esa ruta, porque las rutas
tienen longitudes de grid muy distintas y `l_idx` crudo no es comparable entre
ellas. Se agrupa en tres tramos: temprano (`pfrac ≤ 1/3`), medio y tardío
(`> 2/3`). Dentro de un tramo las posiciones se agrupan ponderando por `n_pairs`,
que es exacto para el MAE.

Con la vuelta (docs/PREREGISTRO_VUELTA.md §5, P5 y P8)
------------------------------------------------------
* **El LSTM es el `es15`** (`audit_run.lstm_qtag`). En agosto esta tabla leía el `es0`
  del paquete de julio, que ya no se corre: el titular y los desgloses salen ahora del
  mismo modelo.
* **La malla es la vuelta entera**, el doble de larga que una leg de agosto, así que las
  30 posiciones de corte (`eval_frame.make_l_indices`) quedan al doble de distancia
  entre sí. Los tercios siguen siendo tercios.
* **La terminal intermedia (P8).** En las siete rutas de dos terminales la vuelta pasa
  por la otra terminal, y la espera ahí queda dentro del tiempo que se predice. El
  segundo bloque separa, en esas rutas, las predicciones emitidas ANTES de la terminal
  intermedia (su horizonte cruza la espera) de las emitidas DESPUÉS (ya no). El km de la
  otra terminal sale de `cortes/cortes_vuelta.csv` («cortando en km X», la alternativa
  que perdió por cobertura) y el largo del lazo, del funnel de la ruta.

⚠ **El on-time de esta tabla es el CRUDO** (banda operacional asimétrica). Los
histogramas de `dev` se persisten por (algoritmo, fold), no por posición —
guardarlos por posición multiplicaría por 30 el coste de memoria del harness—, así
que la versión centrada de la Tabla 10 no se puede derivar aquí. Por eso el MAE
es la métrica primaria de esta tabla: no depende de dónde esté centrada la banda.
La lectura del on-time por posición hay que hacerla con la Tabla 10 al lado.

Uso
---
    python docs/paper/analysis/make_table_07_by_position.py --root artefacts/run_bundle_<…>
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import numpy as np                                              # noqa: E402
import polars as pl                                             # noqa: E402

from audit_run import ROUTES, lstm_qtag, qtag, route_label      # noqa: E402
from honest_ontime import RANGE                                 # noqa: E402
from leg_inclusion import MIN_M, apply_criteria, collect_legs    # noqa: E402
from make_table_05_permodel import LABEL, PANELS, in_panel      # noqa: E402

REF = "lstm_a"
HEADLINE = ["delay", "delay_hour", "xgb_mae_b"]
BINS = [("early", 0.0, 1 / 3), ("mid", 1 / 3, 2 / 3), ("late", 2 / 3, 1.01)]

CORTES_CSV = _HERE.parents[2] / "cortes" / "cortes_vuelta.csv"
#: «… cortando en km 50.07)»: la otra terminal de una ruta de dos (cortes_vuelta.py).
_OTRA = re.compile(r"cortando en km ([0-9]+(?:\.[0-9]+)?)")


def _read_by_position(root: Path, tag: str, ltag: str, rng: str) -> pl.DataFrame:
    parts: List[pl.DataFrame] = []
    for r in ROUTES:
        base = root / "data" / "02_interim" / "routes" / r
        for d, pat in ((base / "xgb" / rng, f"*_{tag}_exo_by_position.parquet"),
                       (base / "lstm_frame" / rng, f"*_{ltag}_by_position.parquet")):
            if not d.is_dir():
                continue
            for p in sorted(d.glob(pat)):
                try:
                    x = pl.read_parquet(p)
                except Exception:
                    continue
                parts.append(x.select(["leg", "algorithm", "l_idx", "n_pairs",
                                       "MAE_s", "pct_ontime_dev"])
                             .with_columns(pl.lit(r).alias("route"),
                                           pl.col("leg").cast(pl.Utf8),
                                           pl.col("n_pairs").cast(pl.Int64)))
    if not parts:
        return pl.DataFrame()
    return pl.concat(parts, how="diagonal_relaxed")


def _n_cells(root: Path, tag: str, rng: str) -> pl.DataFrame:
    rows = []
    base = root / "data" / "02_interim" / "routes"
    for p in sorted(base.glob(f"*/frames/{rng}/*_{tag}_meta.json")):
        m = json.loads(p.read_text(encoding="utf-8"))
        rows.append({"route": str(m["route"]), "leg": str(m["leg"]),
                     "n_cells": int(m["n_cells"]),
                     "grid_step_m": float(m.get("grid_step_m") or 100.0)})
    return pl.DataFrame(rows)


def per_leg_table(root: Path, q: float, cov: float, rng: str,
                  min_M: int) -> pl.DataFrame:
    """Detalle por (ruta, tramo, algoritmo) — lo consume `build` y las figuras.

    Se expone porque la figura necesita la dispersión ENTRE rutas para los
    intervalos; agregada a medianas esa información ya no está.
    """
    return _per_leg(_positions(root, q, cov, rng, min_M))


def _positions(root: Path, q: float, cov: float, rng: str,
               min_M: int) -> pl.DataFrame:
    """Las posiciones de corte de las rutas incluidas con todas las familias."""
    tag = qtag(q, cov)
    d = _read_by_position(root, tag, lstm_qtag(q, cov), rng)
    if d.is_empty():
        raise SystemExit(f"✗ no hay agregados by_position bajo {root}")
    d = d.join(_n_cells(root, tag, rng), on=["route", "leg"], how="inner")

    legs = apply_criteria(collect_legs(root, q, cov, rng), min_M)
    keep = legs.filter(pl.col("included")).select(["route", "leg"])
    d = d.join(keep, on=["route", "leg"], how="inner")

    # Solo rutas con TODAS las familias, o los tramos compararían conjuntos
    # distintos de rutas entre sí (HANDOFF_FRAME.md §5.5).
    need = set(HEADLINE) | {REF}
    ok = (d.group_by(["route", "leg"])
          .agg(pl.col("algorithm").unique().alias("a"))
          .filter(pl.col("a").list.set_intersection(list(need)).list.len() == len(need))
          .select(["route", "leg"]))
    if ok.is_empty():
        raise SystemExit(f"✗ ninguna ruta incluida tiene by_position de "
                         f"{sorted(need)} (¿falta el LSTM `{lstm_qtag(q, cov)}`?)")
    return d.join(ok, on=["route", "leg"], how="inner")


def _pool(d: pl.DataFrame, keys: List[str]) -> pl.DataFrame:
    """Agrupa posiciones ponderando por n_pairs (exacto para MAE y on-time)."""
    return (d.group_by(keys)
            .agg((pl.col("MAE_s") * pl.col("n_pairs")).sum().alias("_w"),
                 (pl.col("pct_ontime_dev") * pl.col("n_pairs")).sum().alias("_o"),
                 pl.col("n_pairs").sum().alias("n_pairs"))
            .with_columns(
                (pl.col("_w") / pl.col("n_pairs") / 60.0).alias("MAE_min"),
                (pl.col("_o") / pl.col("n_pairs")).alias("ontime"))
            .drop(["_w", "_o"]))


def _per_leg(d: pl.DataFrame) -> pl.DataFrame:
    d = d.with_columns((pl.col("l_idx") / pl.col("n_cells")).alias("pfrac"))
    d = d.with_columns(
        pl.when(pl.col("pfrac") <= BINS[0][2]).then(pl.lit("early"))
        .when(pl.col("pfrac") <= BINS[1][2]).then(pl.lit("mid"))
        .otherwise(pl.lit("late")).alias("pos"))
    return _pool(d, ["route", "leg", "pos", "algorithm"])


def build(root: Path, q: float, cov: float, rng: str, min_M: int) -> pl.DataFrame:
    per_leg = per_leg_table(root, q, cov, rng, min_M)

    rows: List[Dict] = []
    for panel, keep_s in PANELS:
        p = in_panel(per_leg, keep_s)
        for pos, _lo, _hi in BINS:
            b = p.filter(pl.col("pos") == pos)
            w_mae = b.pivot(values="MAE_min", index=["route", "leg"], on="algorithm")
            w_ot = b.pivot(values="ontime", index=["route", "leg"], on="algorithm")
            if REF not in w_mae.columns:
                continue
            alts = [c for c in w_mae.columns if c not in ("route", "leg")
                    and c != REF]
            rec = {"panel": panel, "pos": pos, "legs": w_mae.height,
                   "lstm_mae_min": float(w_mae[REF].median()),
                   "lstm_ontime": float(w_ot[REF].median())}
            for a in HEADLINE:
                rec[f"d_{a}_pct"] = float(
                    ((w_mae[a] - w_mae[REF]) / w_mae[REF] * 100).median())
            best = pl.min_horizontal([pl.col(c) for c in alts])
            rec["ratio_best"] = float(
                w_mae.select((best / pl.col(REF)).alias("r"))["r"].median())
            best_ot = pl.max_horizontal([pl.col(c) for c in alts])
            rec["d_ontime_pp"] = float(
                w_ot.select((pl.col(REF) - best_ot).alias("d"))["d"].median())
            rows.append(rec)
    return pl.DataFrame(rows)


# ── la terminal intermedia (P8) ────────────────────────────────────────────

def intermediate_terminals(root: Path, rng: str,
                           cortes_csv: Path = CORTES_CSV) -> Dict[str, Dict]:
    """Ruta → dónde cae su otra terminal sobre la vuelta, en metros desde el corte.

    Sólo las de dos terminales (`tipo = dos` en cortes_vuelta.csv). La posición es
    (km de la otra − km del corte) mod el largo del lazo (`cruce.lazo_km` del funnel),
    sobre el mismo eje que la malla (`GRID_AXIS=traza`: el avance sobre la traza desde
    el corte). Una ruta sin funnel en `root` no sale."""
    out: Dict[str, Dict] = {}
    if not cortes_csv.is_file():
        return out
    with open(cortes_csv, encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            if row.get("tipo") != "dos":
                continue
            m = _OTRA.search(row.get("fuente") or "")
            fp = (root / "data" / "02_interim" / "routes" / row["ruta"] / "perf"
                  / f"funnel_{rng}.json")
            if not m or not fp.is_file():
                continue
            lazo = (json.loads(fp.read_text(encoding="utf-8")).get("cruce") or {}
                    ).get("lazo_km")
            if not lazo:
                continue
            km_cut, km_otra = float(row["km"]), float(m.group(1))
            out[row["ruta"]] = {"km_cut": km_cut, "km_other": km_otra,
                                "lazo_km": float(lazo),
                                "at_m": ((km_otra - km_cut) % float(lazo)) * 1000.0}
    return out


def build_intermediate(root: Path, q: float, cov: float, rng: str,
                       min_M: int) -> pl.DataFrame:
    """Por ruta de dos terminales y algoritmo: el ESCALÓN del MAE al cruzar la terminal
    intermedia, contra el escalón típico entre dos cortes consecutivos de la misma ruta.

    ⚠ No se comparan las dos mitades de la vuelta: el MAE cae a lo largo de la vuelta
    en TODAS las rutas porque el horizonte que queda se acorta, así que «después /
    antes» sale ~0.5 con espera o sin ella. Lo que delata la espera es el salto entre
    el último corte antes de la terminal (su horizonte la cruza) y el primero después
    (ya no): en el piloto, 1601 cae 44 % ahí contra 2–3 % entre cortes consecutivos.
    Vacío si no hay rutas de dos terminales en `root`."""
    term = intermediate_terminals(root, rng)
    if not term:
        return pl.DataFrame()
    d = _positions(root, q, cov, rng, min_M).filter(pl.col("route").is_in(list(term)))
    rows: List[Dict] = []
    for (route, leg, algo), g in d.group_by(["route", "leg", "algorithm"]):
        g = g.sort("l_idx")
        pos = (g["l_idx"] * g["grid_step_m"]).to_numpy()
        mae = (g["MAE_s"] / 60.0).to_numpy()
        at = term[str(route)]["at_m"]
        i = int(np.searchsorted(pos, at))           # el primer corte a partir de ahí
        if i < 1 or i >= len(mae):
            continue
        steps = mae[1:] / mae[:-1]
        rows.append({"route": str(route), "leg": str(leg), "algorithm": str(algo),
                     "at_m": at, "cut_before_m": float(pos[i - 1]),
                     "cut_after_m": float(pos[i]),
                     "mae_before_min": float(mae[i - 1]),
                     "mae_after_min": float(mae[i]),
                     "step_at_terminal": float(steps[i - 1]),
                     "step_typical": float(np.median(np.delete(steps, i - 1))),
                     "n_cuts": len(mae)})
    return pl.DataFrame(rows).sort(["route", "algorithm"]) if rows else pl.DataFrame()


def to_markdown(t: pl.DataFrame, inter: pl.DataFrame) -> str:
    L = ["**Table S16. Performance by cut position along the round trip "
         "(median over routes).**", ""]
    L.append(
        "Cut positions are normalized by the length of the route's grid, which "
        "spans the whole round trip, and grouped into early (first third), mid "
        "and late. Within a band, positions are pooled weighting by pair count, "
        "which is exact for MAE. Δ columns "
        "are the alternative's MAE relative to the LSTM's, so negative means the "
        "alternative is more accurate; the ratio column takes the best "
        "alternative per route. On-time here is the **uncalibrated operational "
        "band**: residual histograms are stored per fold, not per position, so "
        "the centred version of Table 4 cannot be derived at this resolution. "
        "MAE is therefore the metric to read across positions, since it does not "
        "depend on where the band is centred.")
    L.append("")
    L.append("| Position | Routes | LSTM MAE (min) | Δ delay | "
             "Δ delay_hour | Δ xgb_mae_b | Ratio best/LSTM | LSTM on-time % | "
             "Δ on-time vs best (pp) |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for r in t.iter_rows(named=True):
        L.append(
            f"| {r['pos']} | {int(r['legs'])} | "
            f"{r['lstm_mae_min']:.2f} | {r['d_delay_pct']:+.1f}% | "
            f"{r['d_delay_hour_pct']:+.1f}% | {r['d_xgb_mae_b_pct']:+.1f}% | "
            f"{r['ratio_best']:.2f} | {r['lstm_ontime']:.1f} | "
            f"{r['d_ontime_pp']:+.1f} |")
    if inter.is_empty():
        return "\n".join(L) + "\n"

    show = [REF] + HEADLINE
    L += ["", "*Routes with two terminals.* On these routes the round trip passes "
          "the second terminal, and any wait there is part of the time being "
          "predicted. A prediction issued at the last cut position before that "
          "terminal has a horizon spanning the wait; one issued at the first cut "
          "position after it does not. Since error falls along the round trip on "
          "every route as the remaining horizon shortens, the step between those two "
          "cut positions (MAE after / MAE before) is read against the typical step "
          "between consecutive cut positions on the same route (the median of the "
          "others).", "",
          "| Route | Intermediate terminal (km into the round trip) | Model "
          "| MAE, last cut before (min) | MAE, first cut after (min) "
          "| Step at the terminal | Typical step |",
          "|---|---|---|---|---|---|---|"]
    for route in [r for r in ROUTES if r in set(inter["route"].to_list())]:
        g = inter.filter(pl.col("route") == route)
        first = True
        for a in show:
            x = g.filter(pl.col("algorithm") == a)
            if x.is_empty():
                continue
            x = x.row(0, named=True)
            # Sin f-strings anidadas: el python del servidor es 3.10 (PEP 701 es 3.12).
            name = route_label(route, number=True) if first else ""
            km = "{:.1f}".format(x["at_m"] / 1000) if first else ""
            L.append(f"| {name} | {km} | {LABEL.get(a, a)} | "
                     f"{x['mae_before_min']:.2f} | {x['mae_after_min']:.2f} | "
                     f"{x['step_at_terminal']:.2f} | {x['step_typical']:.2f} |")
            first = False
    return "\n".join(L) + "\n"


def main() -> int:
    repo = Path(__file__).resolve().parents[3]
    ap = argparse.ArgumentParser(description="Tabla S16 — por posición de corte")
    ap.add_argument("--root", default=None)
    ap.add_argument("--q", type=float, default=0.05)
    ap.add_argument("--cov", type=float, default=1.0)
    ap.add_argument("--range", dest="rng", default=RANGE)
    ap.add_argument("--min-M", type=int, default=MIN_M)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    root = Path(a.root).resolve() if a.root else repo
    t = build(root, a.q, a.cov, a.rng, a.min_M)
    inter = build_intermediate(root, a.q, a.cov, a.rng, a.min_M)
    prefix = Path(a.out) if a.out else (repo / "docs" / "paper" / "tables"
                                        / "table_07_by_position")
    t.write_csv(prefix.with_suffix(".csv"))
    if not inter.is_empty():
        inter.write_csv(prefix.parent / f"{prefix.name}_intermediate.csv")
    md = to_markdown(t, inter)
    prefix.with_suffix(".md").write_text(md, encoding="utf-8")
    print(md)
    print(f"  ✓ {prefix.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
