"""
make_table_14_error_bands.py
============================
Tabla S12 del artículo (archivo `table_14`) — cuántas predicciones caen dentro de ±1, ±2, ±3 y ±5 minutos.

Por qué
-------
La banda `[−1, +5]` que reporta el manuscrito sale del *Transit Capacity and
Quality of Service Manual* y fue concebida para **adherencia operacional**: mide
si un autobús llega cuando dijo, con una tolerancia asimétrica porque llegar
pronto molesta más que llegar tarde. No fue concebida para **error de
pronóstico**. El dictamen de agosto-2026 pide, con razón, una lectura que un
operador o un usuario puedan interpretar sin conocer esa procedencia.

Estas bandas son simétricas y directas: *«en el X% de los casos la predicción
cae a menos de k minutos del tiempo real»*. No sustituyen a la banda APTA —que
sigue siendo la métrica operacional del paper— sino que la traducen.

No hace falta volver a correr nada
----------------------------------
El `devhist` guarda el histograma CON SIGNO de ``dev = t_true − t_pred`` en bins
de 1 s sobre ±7200 s, y existe precisamente para derivar bandas arbitrarias a
posteriori. Las cuatro columnas son una suma sobre ese histograma.

⚠ El agregado es por LEG y luego se toma la mediana sobre legs, igual que las
Tablas 5 y 10. Agrupar todos los pares de la red daría más peso a las legs con
más viajes, y el panel no está diseñado para eso. Con la vuelta cada ruta es una leg
(su vuelta), así que «por leg» es «por ruta», y el LSTM es el `es15` (P5).

Uso
---
    python docs/paper/analysis/make_table_14_error_bands.py --root test5/extracted
"""
from __future__ import annotations

import argparse
import glob
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent.parent.parent / "func"))

import numpy as np                                              # noqa: E402
import polars as pl                                             # noqa: E402

from honest_metrics import pct_in_band                          # noqa: E402
from make_table_05_permodel import LABEL                        # noqa: E402

TBL = _HERE.parent / "tables"
BANDS = [60.0, 120.0, 180.0, 300.0]          # ±1, ±2, ±3, ±5 min
HEADLINE = ["xgb_mae_b", "delay_hour", "lstm_a"]


#: Sufijo del artefacto POR FAMILIA, y no es cosmético: la corrida de XGBoost
#: escribe `_q05c100_exo_devhist` (lleva Tier B) y la del LSTM `_q05c100_es15_devhist`.
#: Con un solo sufijo para las dos, el patrón caza una familia y la otra
#: desaparece — y la tabla sale con un modelo en vez de diez, que es un fallo
#: visible; peor sería al revés, con nueve de diez y nadie contando.
#: (En agosto el del LSTM era `""`, el `es0`, que con la vuelta ya no se corre.)
FAM_TAG = {"xgb": "_exo", "lstm_frame": "_es15"}


def load_devhist(root: str, tag: Optional[str] = None,
                 families: Optional[Dict[str, str]] = None) -> tuple:
    """Todos los `*_devhist.parquet` del régimen, de las dos familias.

    Devuelve ``(tabla, ilegibles)``. Los ilegibles NO se tragan en silencio: el
    llamador tiene que cruzarlos contra el panel y abortar si alguno era suyo.
    En `test5/extracted` hay cuatro `devhist` del LSTM truncados (§7.9 del
    handoff), los cuatro en legs self-loop que el criterio C1 deja fuera del
    panel — pero eso hay que COMPROBARLO cada vez, no suponerlo: un truncado
    nuevo en una leg del panel se vería exactamente igual desde aquí.
    """
    fams = families if families is not None else FAM_TAG
    out, bad = [], []
    for fam, fam_tag in fams.items():
        t = fam_tag if tag is None else tag
        for p in glob.glob(f"{root}/**/{fam}/**/*_devhist.parquet",
                           recursive=True):
            stem = p.replace("\\", "/").rsplit("/", 1)[1]
            if not re.search(rf"_q05c100{re.escape(t)}_devhist\.parquet$",
                             stem):
                continue
            try:
                out.append(pl.read_parquet(p))
            except Exception as exc:
                bad.append((p, exc.__class__.__name__))
    if not out:
        raise SystemExit(f"sin devhist legibles bajo {root} con tag={tag!r}")
    return pl.concat(out), bad


def _check_unreadable(bad: List[tuple], panel: set) -> None:
    """Aborta si algún artefacto ilegible pertenece a una leg DEL PANEL."""
    if not bad:
        return
    print(f"⚠ {len(bad)} devhist ilegibles (truncados):")
    offend = []
    for p, why in bad:
        parts = p.replace("\\", "/").split("/")
        route = parts[parts.index("routes") + 1]
        leg = parts[-1].split("_q05c100")[0].replace("leg", "").replace("_", "->")
        inside = (route, leg) in panel
        print(f"    {'PANEL' if inside else 'fuera'}  {route}/{leg}  ({why})")
        if inside:
            offend.append((route, leg))
    if offend:
        raise SystemExit(
            f"ABORTA: {len(offend)} leg(s) del panel con devhist ilegible "
            f"{offend}. La tabla saldría con menos legs que las demás y nadie "
            f"lo notaría al leerla.")
    print("    ninguno pertenece al panel -> la tabla no se ve afectada\n")


def per_leg(d: pl.DataFrame, panel: set) -> pl.DataFrame:
    """% dentro de cada banda, por (route, leg, algorithm).

    Los folds se SUMAN antes de contar: la banda es una proporción sobre todos
    los pares de la leg, no un promedio de proporciones por fold (que pesaría
    igual a folds de distinto tamaño).
    """
    rows: List[Dict] = []
    for (route, leg, algo), g in d.group_by(["route", "leg", "algorithm"]):
        if (str(route), str(leg)) not in panel:
            continue
        agg = (g.group_by("dev_s").agg(pl.col("count").sum())
                .sort("dev_s"))
        dev = agg["dev_s"].to_numpy().astype(float)
        cnt = agg["count"].to_numpy().astype(float)
        r = {"route": str(route), "leg": str(leg), "algorithm": str(algo),
             "n_pairs": int(cnt.sum())}
        for b in BANDS:
            r[f"pct_{int(b // 60)}min"] = pct_in_band(dev, cnt, -b, b)
        rows.append(r)
    return pl.DataFrame(rows)


def to_md(t: pl.DataFrame, n_legs: int, n_routes: int) -> str:
    L = [f"**Table S12. Share of predictions within a symmetric error band "
         f"(median over {n_legs} routes).**", ""]
    L.append(
        "Each cell is the percentage of (trip, cut, horizon) predictions whose "
        "absolute error is at most the stated tolerance, computed per route and "
        "then taken as the median across routes. Unlike the "
        "[-1, +5] min band of Table 2, which follows the operational "
        "on-time definition and is deliberately asymmetric, these bands are "
        "symmetric and are meant to be read directly: a value of 40 at "
        "&plusmn;2 min means two predictions in five land within two minutes "
        "of the actual arrival.")
    L += ["", "| Model | &plusmn;1 min | &plusmn;2 min | &plusmn;3 min "
          "| &plusmn;5 min |", "|---|---:|---:|---:|---:|"]
    for r in t.iter_rows(named=True):
        L.append(f"| {LABEL.get(r['algorithm'], r['algorithm'])} | "
                 f"{r['pct_1min']:.1f} | {r['pct_2min']:.1f} | "
                 f"{r['pct_3min']:.1f} | {r['pct_5min']:.1f} |")
    L += ["", f"*Notes.* Panel of {n_routes} routes, one round trip each. "
          "Values are medians over routes, so rows need not be consistent with "
          "any single route."]
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="test5/extracted")
    ap.add_argument("--lstm-root", default=None,
                    help="raiz alterna para el LSTM (p.ej. el bundle honesto)")
    ap.add_argument("--lstm-tag", default="_es15",
                    help="sufijo del LSTM con --lstm-root (default _es15)")
    ap.add_argument("--panel", default=str(TBL / "table_10_ontime_honest.csv"))
    ap.add_argument("--out", default=str(TBL / "table_14_error_bands"))
    ap.add_argument("--all-models", action="store_true",
                    help="incluye los diez, no solo los tres headline")
    a = ap.parse_args()

    pan = pl.read_csv(a.panel).select("route", "leg").unique()
    panel = {(str(r), str(l)) for r, l in zip(pan["route"], pan["leg"])}

    d, bad = load_devhist(a.root)
    if a.lstm_root:
        # El LSTM honesto vive en otro paquete y con otro sufijo; se sustituye
        # solo esa familia para no arrastrar el resto de la corrida nueva.
        d = d.filter(pl.col("algorithm") != "lstm_a")
        d2, bad2 = load_devhist(a.lstm_root, a.lstm_tag,
                                families={"lstm_frame": a.lstm_tag})
        d = pl.concat([d, d2.filter(pl.col("algorithm") == "lstm_a")])
        bad += bad2
    _check_unreadable(bad, panel)

    leg = per_leg(d, panel)
    cols = [f"pct_{int(b // 60)}min" for b in BANDS]
    med = (leg.group_by("algorithm")
              .agg(*[pl.col(c).median() for c in cols],
                   pl.len().alias("n_legs"))
              .sort("pct_5min", descending=True))

    keep = med if a.all_models else med.filter(
        pl.col("algorithm").is_in(HEADLINE))
    n_legs = int(med["n_legs"].max())
    n_routes = len({r for r, _ in panel})

    # Ordenada: el group_by de per_leg recorre los grupos en otro orden cada corrida
    leg.sort("route", "leg", "algorithm").write_csv(a.out + "_perleg.csv")
    Path(a.out + ".csv").write_text(keep.write_csv(), encoding="utf-8")
    Path(a.out + ".md").write_text(to_md(keep, n_legs, n_routes),
                                   encoding="utf-8")

    print(f"panel: {n_routes} rutas\n")
    print(f"{'modelo':22s} {'+-1min':>8s} {'+-2min':>8s} {'+-3min':>8s} "
          f"{'+-5min':>8s}")
    for r in med.iter_rows(named=True):
        mark = "  *" if r["algorithm"] in HEADLINE else ""
        print(f"{LABEL.get(r['algorithm'], r['algorithm'])[:22]:22s} "
              f"{r['pct_1min']:8.1f} {r['pct_2min']:8.1f} "
              f"{r['pct_3min']:8.1f} {r['pct_5min']:8.1f}{mark}")
    print(f"\n→ {a.out}.{{csv,md}} + _perleg.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
