"""
honest_ontime.py
================
Tabla de on-time HONESTO (fase 5): banda simétrica ±180 s y on-time calibrado
por sesgo, derivados de los histogramas de ``dev`` que escriben las dos
familias (`*_devhist.parquet`).

Qué produce
-----------
Una fila por (leg, algoritmo) con la descomposición en cadena:

    pct_ontime_apta        on-time tal cual (banda [−60, +300], asimétrica)
      ↓ premio por caer del lado ancho
    pct_ontime_sym180      misma ANCHURA (360 s), centrada en 0
      ↓ coste del sesgo residual
    pct_ontime_sym180_cal  ya centrado (leave-one-fold-out)  =  precisión pura

y el agregado por algoritmo sobre las legs **incluidas por el criterio
pre-especificado** (`leg_inclusion.py`) que además tienen las dos familias — sin
esa restricción se estarían comparando medianas sobre conjuntos distintos de
legs, que es la trampa documentada en `HANDOFF_FRAME.md` §5.5.

Con la vuelta cada ruta es una leg (``T0->T0``), así que «por leg» es «por ruta», y
el LSTM es el `es15` (`audit_run.lstm_qtag`; el `tw` con ``--split expanding``).

Verificación incorporada
------------------------
El on-time APTA recalculado desde el histograma se compara con el
``pct_ontime_dev`` que el harness escribió en los agregados. Son dos caminos
independientes sobre los mismos residuos (contador al vuelo vs histograma), así
que una discrepancia > 0.01 pp delata un desajuste de binning o de población.

Uso
---
    python docs/paper/analysis/honest_ontime.py --root artefacts/run_bundle_<…>
    python docs/paper/analysis/honest_ontime.py --root artefacts/run_bundle_<…> --all-legs
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[2] / "func"))

import polars as pl                                             # noqa: E402

import honest_metrics as hm                                     # noqa: E402
from audit_run import ROUTES, qtag                              # noqa: E402
from leg_inclusion import MIN_M, apply_criteria, collect_legs   # noqa: E402

RANGE = "2024_01-08_tmin"
TOL_PP = 0.01     # tolerancia de la verificación hist ↔ agregados, en pp


def _lstm_tag_of(tag: str) -> str:
    """El régimen del LSTM que acompaña al de los tabulares (pre-registro §5):
    ``q05c100`` → ``q05c100_es15`` y ``q05c100_tw`` → ``q05c100_tw``. Es
    `audit_run.lstm_qtag` a partir del tag ya armado."""
    return tag if tag.endswith("_tw") else tag + "_es15"


def _read_devhists(root: Path, tag: str, rng: str,
                   routes: Optional[List[str]] = None,
                   lstm_root: Optional[Path] = None,
                   lstm_tag: Optional[str] = None) -> pl.DataFrame:
    """
    ⚠ `lstm_root`/`lstm_tag` sustituyen SOLO la familia del modelo profundo.
    En agosto servían para tomar el LSTM `_es15` de otro paquete sin arrastrar el
    resto de esa corrida. Con la vuelta todo sale de un solo paquete y el `_es15`
    es el de por defecto (`lstm_tag=None` → `_lstm_tag_of`): quedan para
    comparar contra el LSTM de otra corrida.
    """
    parts: List[pl.DataFrame] = []
    for r in (routes or ROUTES):
        base = root / "data" / "02_interim" / "routes" / r
        lbase = ((lstm_root / "data" / "02_interim" / "routes" / r)
                 if lstm_root is not None else base)
        ltag = lstm_tag or _lstm_tag_of(tag)
        for d, pat in ((base / "xgb" / rng, f"*_{tag}_exo_devhist.parquet"),
                       (base / "xgb" / rng, f"*_{tag}_devhist.parquet"),
                       (lbase / "lstm_frame" / rng, f"*_{ltag}_devhist.parquet")):
            if not d.is_dir():
                continue
            for p in sorted(d.glob(pat)):
                # `*_{tag}_devhist` en xgb/ también casa con `*_{tag}_exo_devhist`;
                # el `_exo` es el que manda (es el que corrió con Tier B).
                if pat.endswith(f"_{tag}_devhist.parquet") and "_exo_" in p.name:
                    continue
                try:
                    parts.append(pl.read_parquet(p).with_columns(
                        pl.lit(r).alias("route")))
                except Exception as e:
                    print(f"  ⚠ {p.name}: {e}")
    if not parts:
        return pl.DataFrame()
    return pl.concat(parts, how="diagonal_relaxed")


def _read_overall(root: Path, tag: str, rng: str,
                  routes: Optional[List[str]] = None,
                  lstm_root: Optional[Path] = None,
                  lstm_tag: Optional[str] = None) -> pl.DataFrame:
    """`pct_ontime_dev`, `bias_s` y `MAE_s` de los agregados, para la
    verificación cruzada y para acompañar el on-time con su sesgo."""
    keep = ["route", "leg", "algorithm", "n_pairs", "MAE_s", "bias_s",
            "pct_ontime_dev"]
    parts: List[pl.DataFrame] = []
    for r in (routes or ROUTES):
        base = root / "data" / "02_interim" / "routes" / r
        lbase = ((lstm_root / "data" / "02_interim" / "routes" / r)
                 if lstm_root is not None else base)
        ltag = lstm_tag or _lstm_tag_of(tag)
        for d, pat in ((base / "xgb" / rng, f"*_{tag}_exo_overall.parquet"),
                       (lbase / "lstm_frame" / rng, f"*_{ltag}_overall.parquet")):
            if not d.is_dir():
                continue
            for p in sorted(d.glob(pat)):
                try:
                    x = pl.read_parquet(p)
                except Exception:
                    continue
                cols = [c for c in keep if c in x.columns]
                parts.append(x.select(cols).with_columns(
                    pl.lit(r).alias("route"),
                    pl.col("leg").cast(pl.Utf8)))
    if not parts:
        return pl.DataFrame()
    return pl.concat(parts, how="diagonal_relaxed")


def verify(honest: pl.DataFrame, overall: pl.DataFrame) -> bool:
    """El APTA del histograma debe reproducir el de los agregados."""
    if overall.is_empty():
        print("  ⚠ verificación: no hay agregados que comparar")
        return True
    j = honest.join(overall.select(["route", "leg", "algorithm", "n_pairs",
                                    "pct_ontime_dev"]),
                    on=["route", "leg", "algorithm"], how="inner",
                    suffix="_agg")
    if j.is_empty():
        print("  ⚠ verificación: ningún devhist casa con un agregado")
        return True
    j = j.with_columns(
        (pl.col("pct_ontime_apta") - pl.col("pct_ontime_dev")).abs().alias("d_pp"),
        (pl.col("n_pairs") - pl.col("n_pairs_agg")).abs().alias("d_n"))
    bad_ot = j.filter(pl.col("d_pp") > TOL_PP)
    bad_n  = j.filter(pl.col("d_n") > 0)
    ok = bad_ot.is_empty() and bad_n.is_empty()
    print(f"  {'✔' if ok else '✗'} verificación hist ↔ agregados: "
          f"{j.height - max(bad_ot.height, bad_n.height)}/{j.height} "
          f"· máx |Δ on-time| = {j['d_pp'].max():.4f} pp "
          f"· n_pairs idéntico en {j.height - bad_n.height}/{j.height}")
    for row in bad_ot.head(5).iter_rows(named=True):
        print(f"        {row['route']}/{row['leg']}/{row['algorithm']}: "
              f"hist={row['pct_ontime_apta']:.4f} agg={row['pct_ontime_dev']:.4f}")
    for row in bad_n.head(5).iter_rows(named=True):
        print(f"        {row['route']}/{row['leg']}/{row['algorithm']}: "
              f"n_pairs hist={row['n_pairs']:,} agg={row['n_pairs_agg']:,}")
    return ok


def summarize(df: pl.DataFrame, ref: str = "lstm_a") -> pl.DataFrame:
    """Medianas por algoritmo sobre el MISMO conjunto de legs."""
    cols = ["pct_ontime_apta", "pct_ontime_sym180", "pct_ontime_sym180_cal",
            "pct_ontime_apta_cal", "bias_s", "MAE_s"]
    agg = [pl.col(c).median().alias(c) for c in cols if c in df.columns]
    out = (df.group_by("algorithm").agg(*agg, pl.len().alias("legs"))
           .with_columns((pl.col("pct_ontime_sym180")
                          - pl.col("pct_ontime_apta")).alias("d_asimetria"),
                         (pl.col("pct_ontime_sym180_cal")
                          - pl.col("pct_ontime_sym180")).alias("d_sesgo"))
           .sort("pct_ontime_sym180_cal", descending=True))
    return out


def to_markdown(summ: pl.DataFrame, per_leg: pl.DataFrame, n_legs: int,
                min_M: int) -> str:
    out: List[str] = []
    out.append("**Table 4. On-time rate under three bands: operational, "
               "width-matched symmetric, and centred.**\n")
    out.append(f"Medians over the **{n_legs} routes** meeting the pre-specified "
               f"criterion (trip cut not rated poor, at least {min_M} round trips "
               f"retained by the grid) and having both model families computed. "
               f"Each route is one unit: its round trip, from one crossing of the "
               f"route's cut point to the next.\n")
    out.append("\nThe symmetric ±180 s band has the **same width** (360 s) as "
               "the operational band `[−60, +300]`; only the centre differs, so "
               "the step between those two columns measures the reward for "
               "landing on the band's wide side rather than any gain in "
               "precision. The next step, to the centred column, removes what "
               "remained of each model's residual bias, estimated out of fold. "
               "The final column is therefore dispersion alone.\n")
    out.append("\n| Model | MAE (s) | Bias (s) | Operational % | ±180 s % | "
               "Δ asym. | ±180 s centred % | Δ bias |")
    out.append("|---|---|---|---|---|---|---|---|")
    def _f(v, fmt):
        return "—" if v is None else format(v, fmt)

    for r in summ.iter_rows(named=True):
        out.append(
            f"| `{r['algorithm']}` | {_f(r.get('MAE_s'), '.2f')} | "
            f"{_f(r.get('bias_s'), '+.1f')} | "
            f"{r['pct_ontime_apta']:.1f} | {r['pct_ontime_sym180']:.1f} | "
            f"{r['d_asimetria']:+.1f} | {r['pct_ontime_sym180_cal']:.1f} | "
            f"{r['d_sesgo']:+.1f} |")
    # La «segunda instancia» de V2 del manuscrito (10-oct): el sesgo es la MEDIA del
    # residuo, y con una cola derecha no es el error típico. Se lee de los mismos datos
    # que la tabla (la mediana del residuo por ruta es `dev_median_s` del histograma).
    note_bias = ""
    if "MAE_s" in summ.columns and "dev_median_s" in per_leg.columns:
        best = summ.sort("MAE_s").row(0, named=True)
        med = per_leg.filter(pl.col("algorithm") == best["algorithm"])["dev_median_s"]
        note_bias = (
            f" *Bias* is the mean residual `t_true − t_pred`; with a right-skewed "
            f"residual it is not the typical error. For `{best['algorithm']}`, the "
            f"most accurate model, the mean residual is {best['bias_s']:+.1f} s but "
            f"the median residual is {med.median():+.0f} s (per route "
            f"{med.min():+.0f} to {med.max():+.0f} s), and centring moves its "
            f"on-time rate by "
            + ("less than 0.1 points." if abs(best['d_sesgo']) < 0.05
               else f"{best['d_sesgo']:+.1f} points."))
    out.append(f"\n\n*Notes.*{note_bias} Per-route detail for all {per_leg.height} "
               f"(route, model) rows is available in the supplementary "
               f"material.\n")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description="On-time honesto (fase 5)")
    ap.add_argument("--root", default=None)
    ap.add_argument("--lstm-root", default=None, dest="lstm_root",
                    help="raiz alterna SOLO para la familia LSTM (p. ej. el "
                         "bundle con early stopping honesto)")
    ap.add_argument("--lstm-tag", default=None, dest="lstm_tag",
                    help="régimen del LSTM (default: el `_es15` de la corrida, o "
                         "el `_tw` con --split expanding)")
    ap.add_argument("--q", type=float, default=0.05)
    ap.add_argument("--cov", type=float, default=1.0)
    ap.add_argument("--range", dest="rng", default=RANGE)
    ap.add_argument("--min-M", type=int, default=MIN_M)
    ap.add_argument("--all-legs", action="store_true",
                    help="ignora el criterio pre-especificado (diagnóstico)")
    ap.add_argument("--any-family", dest="require_all", action="store_false",
                    default=True,
                    help="no exige que la ruta tenga las dos familias. Por "
                         "defecto SÍ se exige: comparar medianas sobre "
                         "conjuntos distintos de legs no dice nada "
                         "(HANDOFF_FRAME.md §5.5).")
    ap.add_argument("--split", default="random",
                    choices=["random", "expanding"],
                    help="'expanding' lee los artefactos `_tw` Y calibra en "
                         "causal: el desplazamiento de cada escalón se estima "
                         "solo con los anteriores, nunca con los posteriores")
    ap.add_argument("--out", default=None,
                    help="prefijo de salida; default docs/paper/tables/"
                         "table_10_ontime_honest")
    a = ap.parse_args()

    repo = Path(__file__).resolve().parents[3]
    root = Path(a.root).resolve() if a.root else repo
    tag = qtag(a.q, a.cov, a.split)
    causal = (a.split == "expanding")

    print("=" * 78)
    print(f"  ON-TIME HONESTO · {tag} · {a.rng}"
          + ("  · calibración CAUSAL (solo escalones anteriores)"
             if causal else ""))
    print(f"  raíz: {root}")
    print("=" * 78)

    lroot = Path(a.lstm_root).resolve() if a.lstm_root else None
    if lroot is not None:
        print(f"  LSTM desde: {lroot}  (tag {a.lstm_tag or _lstm_tag_of(tag)})")
    hist = _read_devhists(root, tag, a.rng,
                          lstm_root=lroot, lstm_tag=a.lstm_tag)
    if hist.is_empty():
        print("\n  ✗ No hay ningún `*_devhist.parquet` bajo esa raíz.")
        print("    Los histogramas de `dev` los escriben las corridas a partir")
        print("    de la fase 5; las anteriores no persistían residuos (el")
        print("    harness es streaming: ~300M pares por leg y algoritmo).")
        print("    → re-corre la fase CPU y la fase GPU; los runners ya NO dan")
        print("      por hecha una leg sin devhist (ver run_routes.py y")
        print("      lstm_prediction_tf/run_frame.py).")
        return 1

    print(f"  histogramas: {hist.height:,} bins · "
          f"{hist.select(['route', 'leg']).unique().height} legs · "
          f"{hist['algorithm'].n_unique()} algoritmos")

    honest = hm.honest_table(hist, causal=causal)
    overall = _read_overall(root, tag, a.rng,
                            lstm_root=lroot, lstm_tag=a.lstm_tag)
    ok = verify(honest, overall)

    if not overall.is_empty():
        honest = honest.join(
            overall.select(["route", "leg", "algorithm", "MAE_s", "bias_s"]),
            on=["route", "leg", "algorithm"], how="left")

    # ── criterio de inclusión pre-especificado ──
    legs = apply_criteria(collect_legs(root, a.q, a.cov, a.rng, split=a.split),
                          a.min_M)
    if not a.all_legs:
        keep = legs.filter(pl.col("included")).select(["route", "leg"])
        honest = honest.join(keep, on=["route", "leg"], how="inner")

    if a.require_all:
        fam = (honest.group_by(["route", "leg"])
               .agg(pl.col("algorithm").unique().alias("a"))
               .filter(pl.col("a").list.contains("lstm_a")
                       & pl.col("a").list.contains("xgb_mae_b")
                       & pl.col("a").list.contains("delay_hour"))
               .select(["route", "leg"]))
        honest = honest.join(fam, on=["route", "leg"], how="inner")

    n_legs = honest.select(["route", "leg"]).unique().height
    print(f"\n  rutas analizadas: {n_legs}"
          + ("  (todas)" if a.all_legs else
             f"  (criterio pre-especificado: el corte no sale mal, M >= {a.min_M})")
          + ("  · con las dos familias" if a.require_all else ""))
    if n_legs == 0:
        print("  (nada que resumir todavía)")
        return 1

    sat = honest["sat_frac"].max()
    print(f"  saturación máxima del histograma: {sat:.2e} "
          f"(bins de ±{hm.DEV_MAX_ABS_S:.0f} s)")

    summ = summarize(honest)
    print()
    print(f"  {'algoritmo':14} {'MAE':>8} {'bias':>7} {'APTA%':>7} {'±180%':>7} "
          f"{'Δasim':>7} {'±180cal':>8} {'Δsesgo':>7} {'legs':>5}")
    print("  " + "-" * 78)
    for r in summ.iter_rows(named=True):
        print("  {:14} {:8.2f} {:7.1f} {:7.2f} {:7.2f} {:+7.2f} {:8.2f} "
              "{:+7.2f} {:5d}".format(
                  r["algorithm"], r.get("MAE_s") or float("nan"),
                  r.get("bias_s") or float("nan"),
                  r["pct_ontime_apta"], r["pct_ontime_sym180"],
                  r["d_asimetria"], r["pct_ontime_sym180_cal"],
                  r["d_sesgo"], int(r["legs"])))
    print()
    print("  Δasim  = ±180 − APTA : premio por caer del lado ancho de la banda")
    print("  Δsesgo = calib − ±180: lo que costaba el sesgo residual")
    print("  La última columna (±180 calibrado) es PRECISIÓN pura.")

    prefix = Path(a.out) if a.out else (repo / "docs" / "paper" / "tables"
                                        / "table_10_ontime_honest")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    honest.sort(["route", "leg", "algorithm"]).write_csv(prefix.with_suffix(".csv"))
    prefix.with_suffix(".md").write_text(
        to_markdown(summ, honest, n_legs, a.min_M), encoding="utf-8")
    print(f"\n  ✓ {prefix.with_suffix('.csv')}")
    print(f"  ✓ {prefix.with_suffix('.md')}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
