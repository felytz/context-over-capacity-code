"""
make_table_13_temporal.py
=========================
Tabla 6, Fig. 3 y Tabla S15 del artículo (archivo `table_13`) — **la escalera temporal**: entrenar con los meses
anteriores y predecir el siguiente.

Qué contesta
------------
El manuscrito valida con folds sorteados por viaje y declara la ausencia de
partición temporal como limitación (§6.1), con la evaluación temporal como
trabajo futuro. Esta tabla la convierte en resultado. Y no es solo realismo de
despliegue: con folds aleatorios el prior histórico de un viaje de enero se
calcula en parte con viajes de agosto, y ese bloque es el 77% de la ganancia del
modelo que gana. La escalera lo cierra.

Cómo se lee — y esto importa más que la tabla
---------------------------------------------
**La comparación DENTRO de un escalón es pareada; ENTRE escalones no lo es.**
Cada escalón evalúa un mes distinto, y los meses no son igual de difíciles, así
que un MAE que sube del escalón 3 al 4 no dice que el modelo se degrade: dice
que abril no es marzo. Lo que sí es comparable a lo largo de la escalera es la
**ventaja relativa** entre modelos, porque dentro de cada escalón todos ven los
mismos viajes, los mismos cortes y los mismos horizontes.

Por eso la columna que manda es `Δ vs ref` y la figura dibuja eso. El MAE
absoluto va al lado como referencia, no como serie temporal.

⚠ Además el tamaño de entrenamiento CRECE escalón a escalón por construcción,
así que cualquier tendencia mezcla dos cosas: más datos y otro mes. La escalera
no separa esos dos efectos y esta tabla no pretende que lo haga.

Panel
-----
Por defecto, las rutas que tienen **todos** los escalones de la escalera y llegan en
cada uno al umbral pre-registrado (``--min-train``, 100 vueltas;
docs/PREREGISTRO_VUELTA.md §4): es el único subconjunto **constante a lo largo de los
siete escalones**, y comparar escalones sobre paneles distintos no dice nada.
`--all-legs` usa todas las incluidas y reporta cuántas caen por escalón (el panel de
sensibilidad del pre-registro). Con la vuelta cada ruta es una leg y el LSTM de la
escalera es el `tw`.

⚠ **El escalón es el mes evaluado, no el número de fold** (enmienda E5).
`make_folds_expanding` numera los periodos **de cada ruta**: la 901 (AVL desde abril)
tiene cuatro escalones, y su escalón 1 evalúa mayo. Hasta E5 esta tabla agrupaba por
el número de fold, así que la 901 entraba con mayo–agosto en las filas de
febrero–mayo; y como su escalón más flaco pasaba el umbral, entraba al panel
«constante» con cuatro de siete escalones (34 rutas en 1–4, 33 en 5–7). Ahora cada
fold se traduce a su `test_period` con el meta del frame de su ruta, la escalera es la
unión de los meses evaluados, y a una ruta que no tiene un escalón le cuenta como
cero vueltas de entrenamiento en ese escalón.

⚠ Los dos paneles se escriben con **nombres distintos** y no se pisan: el
pre-registrado en ``table_13_temporal_ladder.*`` y el de todas las rutas en
``table_13_temporal_ladder_all.*`` (en agosto, ``…_all42``, por sus 42 legs; el nombre
llevaba una cifra que ya no vale). Antes compartían nombre, así que un
``--all-legs`` desde el repo sustituía la tabla PUBLICADA por la otra sin avisar —
mismas columnas, otro panel. ``--stem`` fuerza el nombre si hace falta.

Uso
---
    python docs/paper/analysis/make_table_13_temporal.py --root runs/temporal
    python docs/paper/analysis/make_table_13_temporal.py --root runs/temporal \\
           --ref lstm_a          # cuando exista la corrida GPU
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[2] / "func"))

import numpy as np                                              # noqa: E402
import polars as pl                                             # noqa: E402
import matplotlib as mpl                                        # noqa: E402
mpl.use("Agg")
import matplotlib.pyplot as plt                                 # noqa: E402

from audit_run import qtag                                      # noqa: E402
from audit_run import pg                                        # noqa: E402
from honest_ontime import RANGE                                 # noqa: E402
from leg_inclusion import MIN_M, apply_criteria, collect_legs   # noqa: E402
from make_results_figures_frame import (                        # noqa: E402
    C_BEST, C_DELAY, C_DH, C_LSTM, GREY, TAB, panel_letters, save,
)
from make_table_05_permodel import LABEL                        # noqa: E402

SPLIT = "expanding"
#: Color por familia y trazo por variante: las dos variantes de XGBoost comparten
#: color a propósito —son la misma familia— y se separan por el trazo, que es
#: como el resto de figuras del paper distingue historical-only de +exógenos.
STYLE = {
    "xgb_mae_b":  (C_BEST,  "-"),
    "xgb_mae":    (C_BEST,  "--"),
    "delay_hour": (C_DH,    "-"),
    "lstm_a":     (C_LSTM,  "-"),
    "delay":      (C_DELAY, "-"),
}
SHOW = list(STYLE)


# ── recolección ──────────────────────────────────────────────────────

def route_rungs(root: Path, legs: pl.DataFrame, tag: str,
                rng: str) -> Dict[str, Dict[int, dict]]:
    """Ruta → {fold: escalón del meta}, leído del meta del frame de CADA ruta (no
    re-derivado ni copiado de otra). Una ruta incluida sin meta legible aborta: sin él
    no se sabe qué mes evalúa cada fold, y adivinarlo es justo el defecto de E5."""
    out: Dict[str, Dict[int, dict]] = {}
    for r in legs.iter_rows(named=True):
        p = (root / "data" / "02_interim" / "routes" / r["route"] / "frames"
             / rng / f"{r['label']}_{tag}_meta.json")
        try:
            m = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            raise SystemExit(f"  ✗ {r['route']}: no se puede leer {p} ({e})")
        rungs = m.get("rungs") or []
        if not rungs:
            raise SystemExit(f"  ✗ {r['route']}: {p.name} no trae `rungs`")
        out[r["route"]] = {int(x["rung"]): x for x in rungs}
    return out


def ladder_months(rungs: Dict[str, Dict[int, dict]]) -> List[str]:
    """Los meses de la escalera: la unión de los evaluados por las rutas incluidas."""
    return sorted({str(x["test_period"]) for m in rungs.values() for x in m.values()})


def panel(root: Path, q: float, cov: float, rng: str, tag: str,
          min_train: int, all_legs: bool) -> tuple:
    """Devuelve (panel, incluidas, escalones por ruta, meses de la escalera) — panel e
    incluidas por separado, para poder distinguir "no hay corrida" de "la corrida existe
    pero ninguna leg llega al umbral".

    El panel constante (pre-registro §4): ≥ `min_train` vueltas de entrenamiento en
    TODOS los escalones. Un escalón que la ruta no tiene cuenta como cero, así que una
    ruta con menos meses que la escalera no entra aunque su escalón más flaco pase."""
    df = apply_criteria(collect_legs(root, q, cov, rng, split=SPLIT), MIN_M)
    inc = df.filter(pl.col("included"))
    if inc.is_empty():
        return inc, inc, {}, []
    rungs = route_rungs(root, inc, tag, rng)
    months = ladder_months(rungs)
    full = [r for r, m in rungs.items()
            if {str(x["test_period"]) for x in m.values()} == set(months)]
    inc = inc.with_columns(
        pl.col("route").is_in(full).alias("all_rungs"),
        pl.col("route").map_elements(
            lambda r: ",".join(sorted(str(x["test_period"]) for x in rungs[r].values())),
            return_dtype=pl.Utf8).alias("test_periods"))
    if all_legs:
        return inc, inc, rungs, months
    keep = inc.filter(pl.col("all_rungs") & (pl.col("min_n_train") >= min_train))
    return keep, inc, rungs, months


def read_by_fold(root: Path, legs: pl.DataFrame, tag: str, rng: str,
                 rungs: Dict[str, Dict[int, dict]],
                 months: List[str]) -> pl.DataFrame:
    """Métricas por (leg, algoritmo, escalón). Cada `fold` se traduce al mes que evalúa
    según el meta de SU ruta, y `fold` pasa a ser la posición de ese mes en la escalera
    (1 = el primer mes evaluado de la red), igual para todas las rutas."""
    pos = {mth: i + 1 for i, mth in enumerate(months)}
    parts: List[pl.DataFrame] = []
    for r in legs.iter_rows(named=True):
        base = root / "data" / "02_interim" / "routes" / r["route"]
        fmap = rungs[r["route"]]
        for p in (base / "xgb" / rng / f"{r['label']}_{tag}_exo_by_fold.parquet",
                  base / "lstm_frame" / rng / f"{r['label']}_{tag}_by_fold.parquet"):
            if not pg.parquet_ok(p):
                continue
            d = pl.read_parquet(p)
            unknown = sorted(set(d["fold"].to_list()) - set(fmap))
            if unknown:
                raise SystemExit(f"  ✗ {r['route']}: {p.name} trae folds {unknown} "
                                 f"que el meta del frame no tiene")
            tp = {f: str(fmap[f]["test_period"]) for f in fmap}
            d = d.with_columns(
                pl.col("fold").replace_strict(tp, return_dtype=pl.Utf8)
                .alias("test_period"))
            d = d.with_columns(
                pl.col("test_period").replace_strict(pos, return_dtype=pl.Int64)
                .alias("fold"))
            parts.append(d)
    if not parts:
        return pl.DataFrame()
    return pl.concat(parts, how="diagonal_relaxed")


# ── agregación ───────────────────────────────────────────────────────

def build(bf: pl.DataFrame, ref: str) -> pl.DataFrame:
    """Una fila por (escalón, algoritmo), pareada contra `ref` dentro del escalón."""
    r = (bf.filter(pl.col("algorithm") == ref)
           .select(["route", "leg", "fold", pl.col("MAE_s").alias("ref_s")]))
    d = bf.join(r, on=["route", "leg", "fold"], how="inner")
    d = d.with_columns(
        (100.0 * (pl.col("MAE_s") - pl.col("ref_s")) / pl.col("ref_s"))
        .alias("delta_pct"),
        (pl.col("MAE_s") < pl.col("ref_s")).alias("beats_ref"),
    )
    return (d.group_by(["fold", "test_period", "algorithm"]).agg(
        pl.col("MAE_s").median().alias("MAE_s"),
        pl.col("delta_pct").median().alias("delta_pct"),
        pl.col("beats_ref").sum().alias("wins"),
        pl.len().alias("legs"),
        pl.col("n_pairs").sum().alias("n_pairs"),
    ).sort(["fold", "MAE_s"]))


def ordering_holds(t: pl.DataFrame, ref: str) -> Dict[int, bool]:
    """¿El modelo tabular gana en cada escalón? Es la pregunta del revisor."""
    out = {}
    for fold in sorted(t["fold"].unique().to_list()):
        g = t.filter(pl.col("fold") == fold).sort("MAE_s")
        winners = [a for a in g["algorithm"].to_list()]
        out[int(fold)] = bool(winners and winners[0].startswith("xgb"))
    return out


# ── salidas ──────────────────────────────────────────────────────────

def left_out_note(included: pl.DataFrame, legs: pl.DataFrame) -> str:
    """Una frase con las rutas incluidas que no entran al panel constante, y por qué —
    leída de los metas, no escrita a mano."""
    out = included.filter(~pl.col("route").is_in(legs["route"].to_list()))
    if out.is_empty():
        return ""
    why = []
    for r in out.sort("route").iter_rows(named=True):
        have = [m for m in r["test_periods"].split(",") if m]
        if not r["all_rungs"]:
            why.append(f"{r['name']} (its records start later: rungs "
                       f"{have[0]} to {have[-1]} only)")
        else:
            why.append(f"{r['name']} ({r['min_n_train']} training round trips "
                       f"at its first rung)")
    n = out.height
    return (f" {n} included {'route falls' if n == 1 else 'routes fall'} short: "
            + "; ".join(why) + ".")


def to_markdown(t: pl.DataFrame, ref: str, n_legs: int, min_train: int,
                all_legs: bool = False, left_out: str = "") -> str:
    o: List[str] = []
    o.append(f"**Table {'S15' if all_legs else '6'}. Rolling-origin evaluation: train on the "
             f"months so far, predict the next one ({n_legs} routes).**\n")
    o.append(
        "Each rung fits every model on all round trips up to a month and scores it "
        "on the following month; the first month is training only, so eight months "
        "give seven rungs. Within a rung every model sees the same round trips, cut "
        "positions and horizons, so the comparison is paired by construction. "
        "**Across rungs it is not**: each rung scores a different month, and "
        "months differ in difficulty, so a change in absolute MAE from one rung "
        "to the next is not degradation. The comparable quantity along the "
        "ladder is the relative difference, reported as *Δ vs "
        f"{LABEL.get(ref, ref)}*. Training size also grows by construction, so "
        "any trend mixes more data with a different month; this design does not "
        "separate the two. Rungs are calendar months: a route whose records "
        "start later enters the ladder at the first month it can be scored on. " + (
            "This is the sensitivity panel: it takes **every** included route, "
            f"whether or not it reaches {min_train} training round trips at each "
            "rung, so the set does not stay constant along the ladder."
            if all_legs else
            f"The panel is restricted to routes reaching {min_train} training "
            "round trips at **every** rung, which is the only set that stays "
            "constant along the ladder." + left_out) + "\n")
    o.append("\n| Rung | Test month | Model | MAE (min) | Δ vs ref | "
             "Beats ref | Routes | Pairs |")
    o.append("|---|---|---|---:|---:|---:|---:|---:|")
    for fold in sorted(t["fold"].unique().to_list()):
        g = t.filter(pl.col("fold") == fold).sort("MAE_s")
        first = True
        for r in g.iter_rows(named=True):
            if r["algorithm"] not in SHOW:
                continue
            delta = ("—" if r["algorithm"] == ref
                     else f"{r['delta_pct']:+.1f}%")
            wins = "—" if r["algorithm"] == ref else f"{r['wins']}/{r['legs']}"
            o.append(
                f"| {fold if first else ''} | "
                f"{r['test_period'] if first else ''} | "
                f"{LABEL.get(r['algorithm'], r['algorithm'])} | "
                f"{r['MAE_s'] / 60:.2f} | {delta} | {wins} | "
                f"{r['legs']} | {r['n_pairs']:,} |")
            first = False
    return "\n".join(o) + "\n"


def figure(t: pl.DataFrame, ref: str, n_legs: int,
           name: str = "fig_results_7_temporal_ladder") -> None:
    rungs = (t.select("fold", "test_period").unique().sort("fold"))
    folds = rungs["fold"].to_list()
    xpos = {f: i for i, f in enumerate(folds)}
    xs = list(range(len(folds)))
    ticks = rungs["test_period"].to_list()

    fig, ax = plt.subplots(1, 2, figsize=(10.2, 4.1))
    for algo in SHOW:
        g = t.filter(pl.col("algorithm") == algo).sort("fold")
        if g.is_empty():
            continue
        col, ls = STYLE.get(algo, (GREY, "-"))
        # x por el escalón de cada fila, no por su orden: un modelo al que le falta
        # un escalón no corre los demás un mes a la izquierda.
        gx = [xpos[f] for f in g["fold"].to_list()]
        ax[0].plot(gx, g["delta_pct"], marker="o", ms=4.5, lw=1.4,
                   color=col, ls=ls, label=LABEL.get(algo, algo))
        ax[1].plot(gx, [v / 60 for v in g["MAE_s"]], marker="o",
                   ms=4.5, lw=1.4, color=col, ls=ls)
    ax[0].axhline(0, color="#999999", lw=0.9, ls="--")
    ax[0].set_ylabel(f"median Δ MAE vs {LABEL.get(ref, ref)} (%)")
    ax[0].set_title("Relative advantage — comparable across rungs", fontsize=9.5)
    ax[1].set_ylabel("median MAE (min)")
    ax[1].set_title("Absolute MAE — each rung is a different month", fontsize=9.5)
    for k in (0, 1):
        ax[k].set_xticks(xs)
        ax[k].set_xticklabels(ticks, rotation=45, ha="right", fontsize=7.5)
        ax[k].set_xlabel("month predicted (trained on everything before it)")
    ax[1].text(0.02, 0.04, "months differ in difficulty:\nread the left panel",
               transform=ax[1].transAxes, fontsize=6.5, color="#777777")
    panel_letters(ax)
    # La leyenda va debajo, fuera de los ejes: con loc="best" tapaba el punto de junio
    # de XGBoost (−7.8 %), como la de la Fig. 2 tapa el de la 302.
    fig.legend(*ax[0].get_legend_handles_labels(), fontsize=7.5, frameon=False,
               loc="lower center", ncol=len(SHOW), bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"Rolling-origin ladder · {n_legs} "
                 f"{'route' if n_legs == 1 else 'routes'}", fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, name)


# ── main ─────────────────────────────────────────────────────────────

def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    ap = argparse.ArgumentParser(description="Tabla 6 / S15 · escalera temporal")
    ap.add_argument("--root", default=None)
    ap.add_argument("--q", type=float, default=0.05)
    ap.add_argument("--cov", type=float, default=1.0)
    ap.add_argument("--range", dest="rng", default=RANGE)
    ap.add_argument("--ref", default="delay",
                    help="modelo de referencia de la comparación pareada "
                         "(default `delay`, que siempre está; usa `lstm_a` "
                         "cuando exista la corrida GPU)")
    ap.add_argument("--min-train", type=int, default=100,
                    help="viajes de entrenamiento exigidos en TODOS los "
                         "escalones (default: el umbral pre-registrado)")
    ap.add_argument("--all-legs", action="store_true")
    ap.add_argument("--no-figure", action="store_true")
    ap.add_argument("--stem", default=None,
                    help="nombre base de la salida (default "
                         "`table_13_temporal_ladder`; con --all-legs, "
                         "`table_13_temporal_ladder_all`).")
    a = ap.parse_args()
    stem = a.stem or ("table_13_temporal_ladder_all" if a.all_legs
                      else "table_13_temporal_ladder")

    repo = Path(__file__).resolve().parents[3]
    root = Path(a.root).resolve() if a.root else repo
    tag = qtag(a.q, a.cov, SPLIT)

    legs, included, rungs, months = panel(root, a.q, a.cov, a.rng, tag,
                                          a.min_train, a.all_legs)
    if legs.is_empty() and not included.is_empty():
        # La corrida existe; lo que falta es tamaño. Es un resultado, no un
        # fallo, y decirlo con las cifras delante evita el diagnóstico
        # equivocado de "no se ha corrido nada".
        worst = included.sort("min_n_train", descending=True)
        print("=" * 74)
        print(f"  ✗ hay {included.height} ruta(s) con escalera, pero NINGUNA "
              f"llega a {a.min_train} viajes de entrenamiento")
        print(f"    en todos los escalones. El escalón 1 es el que manda: "
              f"entrena solo con el primer mes.")
        print()
        print(f"  {'ruta':>6} {'leg':<10} {'M':>6} {'escalones':>10} "
              f"{'min train':>10}")
        for r in worst.head(10).iter_rows(named=True):
            print(f"  {r['route']:>6} {r['leg']:<10} {r['M']:6d} "
                  f"{r['n_rungs']:10d} {r['min_n_train']:10d}")
        print()
        print("  Opciones: --min-train 0 para verlas todas (los escalones "
              "dejan de ser")
        print("  comparables entre sí), o correr más rutas.")
        print("=" * 74)
        return 2
    if legs.is_empty():
        print("=" * 74)
        print(f"  ✗ no hay frames `{tag}` bajo {root}")
        print()
        print("  La corrida con ventana expansiva aún no existe. Se lanza con:")
        print()
        print("    python func/xgb_prediction/run_routes.py --routes all \\")
        print("      --n-route-quantile 0.05 --min-coverage 1.0 \\")
        print("      --max-trips 2500 --exo --split expanding")
        print()
        print("  No necesita `--force`: el sufijo `_tw` no colisiona con la")
        print("  corrida aleatoria, así que no se salta como `cached`.")
        print("=" * 74)
        return 2

    bf = read_by_fold(root, legs, tag, a.rng, rungs, months)
    if bf.is_empty():
        print(f"  ✗ hay frames pero ningún `*_{tag}_*_by_fold.parquet`: "
              f"la corrida se quedó a medias.")
        return 2
    if a.ref not in bf["algorithm"].unique().to_list():
        print(f"  ✗ el modelo de referencia {a.ref!r} no está en los "
              f"artefactos. Hay: {sorted(bf['algorithm'].unique().to_list())}")
        return 1

    t = build(bf, a.ref)
    n_legs = int(bf.select(pl.struct("route", "leg").n_unique()).item())
    left_out = ("" if a.all_legs
                else left_out_note(included, legs))

    print("=" * 74)
    print(f"  TABLA 13 · escalera temporal · {tag} · {a.rng}")
    print(f"  rutas del panel: {legs.height} · con métricas: {n_legs} · "
          f"referencia: {a.ref}")
    print(f"  escalera: {len(months)} meses ({months[0]} a {months[-1]})"
          if months else "  escalera: —")
    if left_out:
        print(f"  fuera del panel constante:{left_out}")
    print("=" * 74)
    holds = ordering_holds(t, a.ref)
    for fold in sorted(t["fold"].unique().to_list()):
        g = t.filter(pl.col("fold") == fold).sort("MAE_s")
        best = g.row(0, named=True)
        print(f"  escalón {fold} → {best['test_period']:>8}  "
              f"gana {LABEL.get(best['algorithm'], best['algorithm']):<28} "
              f"MAE={best['MAE_s'] / 60:5.2f} min  "
              f"Δ={best['delta_pct']:+6.1f}%  "
              f"{'◆' if holds[int(fold)] else ''}")
    n_ok = sum(holds.values())
    print(f"\n  el modelo tabular gana en {n_ok} de {len(holds)} escalones")

    TAB.mkdir(parents=True, exist_ok=True)
    t.write_csv(TAB / f"{stem}.csv")
    (TAB / f"{stem}.md").write_text(
        to_markdown(t, a.ref, n_legs, a.min_train, a.all_legs, left_out),
        encoding="utf-8")
    print(f"\n  ✓ {TAB / f'{stem}.csv'}")
    print(f"  ✓ {TAB / f'{stem}.md'}")
    if not a.no_figure:
        figure(t, a.ref, n_legs,
               name=("fig_results_7_temporal_ladder" if stem
                     == "table_13_temporal_ladder" else f"fig_{stem}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
