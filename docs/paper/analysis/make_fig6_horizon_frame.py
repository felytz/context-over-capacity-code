"""
make_fig6_horizon_frame.py
==========================
Fig. S3 del artículo («Figura 6» en un borrador anterior) — rendimiento por HORIZONTE de predicción, sobre el frame.

Consume los `*_by_horizon.parquet` que las dos familias emiten desde 2026-07-25.
Si no existen, aborta con el comando exacto que los produce en vez de dibujar una
figura vacía.

El horizonte es el TIEMPO RESTANTE REAL hasta el objetivo, bineado por `dt_true`
(nunca por la predicción: el estrato dependería del modelo y los algoritmos
dejarían de compararse sobre las mismas filas). Los cortes: 0-2, 2-5, 5-10, 10-20,
20-40, 40-60, 60-90 y > 90 min (`honest_metrics.HORIZON_*`, fijados en el harness:
cambiarlos exige re-correr los modelos). Los cinco primeros son los de agosto; 60 y
90, la enmienda E4 del pre-registro.

Con la vuelta (docs/PREREGISTRO_VUELTA.md §4–§5)
-------------------------------------------------
* El LSTM es el `es15` (`audit_run.lstm_qtag`), no el `es0` del paquete de julio.
* Un solo panel: todas las rutas incluidas. Los dos de agosto (validated y el
  agrupado con group B) venían de los estratos, que ya no existen.
* ⚠ La vuelta dura 60–180 min, así que «> 40 min» dejó de ser una cola: en el piloto
  del servidor juntaba el 6 % de los pares de la 902, el 49 % de la 1801-ALT y el 58 %
  de la 601 (~50 % en las 35 rutas). Por eso la E4 lo parte en 40-60, 60-90 y > 90.
  La figura dice en el eje la fracción de pares de cada cubo.
* Las marcas del eje de MAE salen de los datos (en agosto iban a mano hasta 7 min).

⚠ El on-time de esta figura es el CRUDO (banda operacional asimétrica): igual
que en la Tabla 7, los histogramas de `dev` se guardan por fold y no por
horizonte, así que la corrección de centrado de la Tabla 10 no se puede aplicar a
esta resolución. El MAE es la métrica primaria; el panel de on-time se lee con la
Tabla 10 al lado.

Uso
---
    python docs/paper/analysis/make_fig6_horizon_frame.py --root artefacts/run_bundle_<…>
"""
from __future__ import annotations

import argparse
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

import honest_metrics as hm                                     # noqa: E402
from audit_run import ROUTES, lstm_qtag, qtag                   # noqa: E402
from honest_ontime import RANGE                                 # noqa: E402
from leg_inclusion import MIN_M, apply_criteria, collect_legs    # noqa: E402
from make_results_figures_frame import C_BEST, C_DELAY, C_DH, C_LSTM, save  # noqa: E402
from make_table_05_permodel import LABEL                        # noqa: E402

REF = "lstm_a"
SHOW = ["xgb_mae_b", "delay_hour", "delay", REF]
COLORS = {"xgb_mae_b": C_BEST, "delay_hour": C_DH, "delay": C_DELAY,
          REF: C_LSTM}
TAB = Path("docs/paper/tables")
#: Marcas candidatas del eje de MAE (log). Se dibujan las que caen en el rango de los
#: datos: con las de agosto, a mano hasta 7 min, la vuelta se salía del eje.
LOG_TICKS = [0.2, 0.4, 0.6, 1, 2, 4, 7, 10, 15, 20, 30]


def read_by_horizon(root: Path, tag: str, ltag: str, rng: str) -> pl.DataFrame:
    parts: List[pl.DataFrame] = []
    for r in ROUTES:
        base = root / "data" / "02_interim" / "routes" / r
        for d, pat in ((base / "xgb" / rng, f"*_{tag}_exo_by_horizon.parquet"),
                       (base / "lstm_frame" / rng, f"*_{ltag}_by_horizon.parquet")):
            if not d.is_dir():
                continue
            for p in sorted(d.glob(pat)):
                try:
                    x = pl.read_parquet(p)
                except Exception:
                    continue
                parts.append(x.select(["leg", "algorithm", "horizon", "n_pairs",
                                       "MAE_s", "pct_ontime_dev"])
                             .with_columns(pl.lit(r).alias("route"),
                                           pl.col("leg").cast(pl.Utf8),
                                           pl.col("n_pairs").cast(pl.Int64)))
    if not parts:
        return pl.DataFrame()
    return pl.concat(parts, how="diagonal_relaxed")


def main() -> int:
    ap = argparse.ArgumentParser(description="Fig. S3 — por horizonte")
    ap.add_argument("--root", default=None)
    ap.add_argument("--q", type=float, default=0.05)
    ap.add_argument("--cov", type=float, default=1.0)
    ap.add_argument("--range", dest="rng", default=RANGE)
    ap.add_argument("--min-M", type=int, default=MIN_M)
    a = ap.parse_args()
    repo = Path(__file__).resolve().parents[3]
    root = Path(a.root).resolve() if a.root else repo
    tag = qtag(a.q, a.cov)

    d = read_by_horizon(root, tag, lstm_qtag(a.q, a.cov), a.rng)
    if d.is_empty():
        raise SystemExit(
            "\n✗ No hay ningún `*_by_horizon.parquet` bajo esa raíz.\n"
            "  El desglose por horizonte lo emiten las corridas desde el\n"
            "  2026-07-25; las anteriores agregaban sobre horizontes en cada\n"
            "  corte y no lo pueden reconstruir (los residuos no se persisten).\n"
            "  → bash run_pipeline.sh --skip-data\n"
            "    y después re-descargar el paquete.")
    # Un cubo que no está en HORIZON_LABELS se saltaría en silencio más abajo: con un
    # bundle de antes de la E4 (el piloto del 7-oct) se perdería «> 40 min», la mitad
    # de los pares. Mejor parar.
    ajenos = sorted(set(d["horizon"].to_list()) - set(hm.HORIZON_LABELS))
    if ajenos:
        raise SystemExit(
            f"\n✗ El bundle trae cubos de horizonte que no son los de honest_metrics: "
            f"{ajenos}.\n"
            f"  Los de ahora son {list(hm.HORIZON_LABELS)} (enmienda E4, 8-oct-2026).\n"
            f"  Es una corrida de antes de la E4: sus by_horizon no se pueden re-binar.")

    legs = apply_criteria(collect_legs(root, a.q, a.cov, a.rng), a.min_M)
    keep = legs.filter(pl.col("included")).select(["route", "leg"])
    d = d.join(keep, on=["route", "leg"], how="inner")
    # Solo rutas con las cuatro curvas, o cada cubo promediaría otro conjunto.
    ok = (d.filter(pl.col("algorithm").is_in(SHOW))
          .group_by(["route", "leg"]).agg(pl.col("algorithm").n_unique().alias("k"))
          .filter(pl.col("k") == len(SHOW)).select(["route", "leg"]))
    d = d.join(ok, on=["route", "leg"], how="inner")
    n_routes = ok.height
    if not n_routes:
        raise SystemExit(f"✗ ninguna ruta incluida tiene by_horizon de {SHOW}")

    labels = list(hm.HORIZON_LABELS)
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 4.4))
    rows: List[Dict] = []
    for a_ in SHOW:
        mae, ont, xs = [], [], []
        for i, h in enumerate(labels):
            b = d.filter((pl.col("algorithm") == a_) & (pl.col("horizon") == h))
            if b.is_empty() or b["n_pairs"].sum() == 0:
                continue
            xs.append(i)
            mae.append(float((b["MAE_s"] * b["n_pairs"]).sum()
                             / b["n_pairs"].sum() / 60.0))
            ont.append(float((b["pct_ontime_dev"] * b["n_pairs"]).sum()
                             / b["n_pairs"].sum()))
            rows.append({"panel": "all", "algorithm": a_, "horizon": h,
                         "MAE_min": mae[-1], "ontime": ont[-1],
                         "n_pairs": int(b["n_pairs"].sum())})
        lw = 2.2 if a_ == REF else 1.5
        for k, ys in enumerate((mae, ont)):
            ax[k].plot(xs, ys, marker="o", ls="-", color=COLORS[a_], lw=lw, ms=4,
                       label=LABEL.get(a_, a_) if k == 0 else None)

    t = pl.DataFrame(rows)
    # La fracción de pares de cada cubo (la de `delay`: es la misma en las cuatro,
    # porque el frame es común): con la vuelta los cubos largos pesan tanto como los cortos.
    share = (t.filter(pl.col("algorithm") == "delay")
              .with_columns((100 * pl.col("n_pairs") / pl.col("n_pairs").sum())
                            .alias("pct")))
    pct = dict(zip(share["horizon"].to_list(), share["pct"].to_list()))
    ticklab = [f"{h}\n{pct[h]:.0f}% of pairs" if h in pct else h for h in labels]
    for k, lab in enumerate(("MAE (min)", "On-time %, operational band")):
        ax[k].set_xticks(range(len(labels)))
        ax[k].set_xticklabels(ticklab, rotation=30, ha="right", fontsize=7.5)
        ax[k].set_ylabel(lab)
        ax[k].set_xlabel("time remaining to the target")
    # Log en MAE porque el rango es de un orden de magnitud o más. Con las marcas por
    # defecto el eje se queda con una sola etiqueta (10⁰) y no se puede leer ningún
    # valor: se eligen de LOG_TICKS las que caen en el rango y se apagan las menores.
    ax[0].set_yscale("log")
    lo, hi = float(t["MAE_min"].min()), float(t["MAE_min"].max())
    ticks = [v for v in LOG_TICKS if lo / 1.3 <= v <= hi * 1.3] or [lo, hi]
    ax[0].set_yticks(ticks)
    ax[0].set_yticklabels([f"{v:g}" for v in ticks])
    ax[0].yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle(f"Performance vs. prediction horizon, {n_routes} "
                 f"{'route' if n_routes == 1 else 'routes'}", fontsize=10)
    fig.tight_layout()
    save(fig, "fig_results_4_by_horizon")

    TAB.mkdir(parents=True, exist_ok=True)
    t.write_csv(TAB / "fig06_horizon_values.csv")
    print(f"  ✓ {TAB / 'fig06_horizon_values.csv'}  (cifras para el texto de §4.5)")
    print()
    print(f"todas ({n_routes} rutas)")
    print(t.pivot(values="MAE_min", index="horizon", on="algorithm"))
    print({h: round(v, 1) for h, v in pct.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
