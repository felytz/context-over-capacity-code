"""
make_table_12_cost.py
=====================
Tabla 7 y Fig. 4 del artículo (archivo `table_12`) — **precisión contra coste computacional**.

Por qué existe
--------------
§3.9 del manuscrito promete que el coste de cada familia se reporta junto a su
precisión ("Wall-clock time and peak memory are recorded per pipeline stage and
per route, which allows the computational cost of each model family to be
reported alongside its accuracy") y el manuscrito no lo entrega. Lo único
medido era el LSTM. Esta tabla cierra la promesa y sostiene la afirmación
central del paper en el eje que le faltaba: la media horaria recupera el 41% de
la brecha **sin ajustar un parámetro**, y el modelo tabular alcanza el mejor
error a una fracción del coste del modelo profundo.

De dónde salen los números
--------------------------
* CPU — `*_timing.parquet`, una fila por (fold, algoritmo), que emite
  `xgb_prediction.cross_validation` desde 2026-08-17. Relojes `perf_counter`
  pegados a la llamada.
* GPU — `elapsed_s` del `_overall.parquet` del LSTM, que ya existía.

⚠ **Las tres asimetrías que hay que declarar en el pie**, porque ninguna se
arregla con más cómputo:

1. **Pared ≠ tiempo de CPU.** XGBoost corre con `n_jobs=-1` bajo
   `OMP_NUM_THREADS=16`; los baselines son numpy y en su mayoría de un hilo. La
   columna `core_s` multiplica el reloj de XGBoost por los hilos que el
   artefacto grabó (`omp_threads`) y es una **cota superior**: el paralelismo
   real de XGBoost nunca es perfecto. Se reportan las dos.
2. **El LSTM no se puede partir.** Su `elapsed_s` es de leg entero —
   entrenamiento y evaluación juntos —, así que entra en el panel de coste
   total y NO en el de inferencia. Partirlo exigiría re-correr en GPU.
   ⚠ Eso es **partir el reloj ya medido**, y no es lo mismo que **medir la
   latencia de nuevo**: un forward pass no depende de los pesos, así que se
   cronometraría en local y en CPU sin reentrenar nada (y CPU es el blanco
   correcto, porque ahí están cronometrados sus rivales). No se hace por
   decisión — D4 en `paper_9k_public_transport/DECISIONES.md`, opción (b) —,
   no por falta de servidor.
3. **Ningún modelo lleva presupuesto de búsqueda de hiperparámetros**, porque
   el protocolo no ajusta ninguno (Tabla S9). Incluirlo alteraría la comparación
   de coste. ⚠ La nota ya NO afirma a quién penalizaría (decisión del autor,
   28-sep: es una conjetura sin medir), ni compara el precio del hardware —
   el artículo no habla de precios, solo de reloj observado.

La featurización de XGBoost se atribuye ENTERA a cada variante (lo que costaría
desplegar esa sola). `--amortize` la divide entre las variantes que comparten
tier, que es la otra lectura defendible.

Con la vuelta (docs/PREREGISTRO_VUELTA.md §5) la unidad es la ruta, y el LSTM, el
`es15` del mismo paquete (por defecto; en agosto venía de otro con `--lstm-root`). La
nota ya no afirma quién gana ni que el LSTM esté dominado: lo dice sólo si los datos
lo dicen (`_notes`).

Uso
---
    python docs/paper/analysis/make_table_12_cost.py --root runs/nueva
    python docs/paper/analysis/make_table_12_cost.py --root runs/nueva --amortize
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[2] / "func"))

import numpy as np                                              # noqa: E402
import polars as pl                                             # noqa: E402
import matplotlib as mpl                                        # noqa: E402
mpl.use("Agg")
import matplotlib.pyplot as plt                                 # noqa: E402

from audit_run import ROUTES, qtag                              # noqa: E402
from audit_run import pg                                        # noqa: E402
from honest_ontime import RANGE                                 # noqa: E402
from leg_inclusion import MIN_M, apply_criteria, collect_legs   # noqa: E402
from make_results_figures_frame import (                        # noqa: E402
    C_BEST, C_DELAY, C_DH, C_LSTM, GREY, FIG, TAB, panel_letters, save,
)
from make_table_05_permodel import LABEL                        # noqa: E402

LSTM = "lstm_a"
# El orden de familia fija color y marcador; el eje de coste ya separa a todos.
FAMILY = {
    "delay": "historical", "delay_hour": "historical",
    "knn": "trajectory", "kernel": "trajectory", "linreg": "trajectory",
    "xgb_mae": "tabular", "xgb_mae_b": "tabular",
    "xgb_mse": "tabular", "xgb_mse_b": "tabular",
    LSTM: "deep",
}
STYLE = {
    "historical":  (C_DH,    "s"),
    "trajectory":  (GREY,    "^"),
    "tabular":     (C_BEST,  "o"),
    "deep":        (C_LSTM,  "D"),
}


# ── recolección ──────────────────────────────────────────────────────

def panel_legs(root: Path, q: float, cov: float, rng: str) -> pl.DataFrame:
    """Las rutas pre-registradas que tienen las dos familias — el panel.

    Se reutiliza `leg_inclusion` en vez de re-derivar el criterio: es el mismo
    que fija las Tablas 4-10, y una tabla de coste sobre otro conjunto de legs
    no sería comparable con ninguna de ellas (§7.7, primera trampa).
    """
    df = apply_criteria(collect_legs(root, q, cov, rng), MIN_M)
    return df.filter(pl.col("included") & pl.col("has_xgb") & pl.col("has_lstm"))


def read_timing(root: Path, legs: pl.DataFrame, tag: str,
                rng: str) -> pl.DataFrame:
    """Suma los relojes sobre folds → una fila por (leg, algoritmo)."""
    parts: List[pl.DataFrame] = []
    missing: List[str] = []
    for r in legs.iter_rows(named=True):
        p = (root / "data" / "02_interim" / "routes" / r["route"] / "xgb" / rng
             / f"{r['label']}_{tag}_exo_timing.parquet")
        if not pg.parquet_ok(p):
            missing.append(f"{r['route']}/{r['leg']}")
            continue
        parts.append(pl.read_parquet(p))
    if not parts:
        return pl.DataFrame()
    df = pl.concat(parts, how="vertical_relaxed")
    # ⚠ `.sort().sum()` y no `.sum()`: el group_by de polars suma en paralelo y el orden
    # de una suma en coma flotante cambiaba entre corridas; la Fig. 4 salía con unos
    # píxeles distintos cada vez (fase 1b, al correr reproduce_tables dos veces).
    df = (df.group_by(["route", "leg", "algorithm", "family", "tier"])
            .agg(pl.col("fit_s").sort().sum(),
                 pl.col("predict_s").sort().sum(),
                 pl.col("prior_s").sort().sum(),
                 pl.col("featurize_train_s").sort().sum(),
                 pl.col("featurize_eval_s").sort().sum(),
                 pl.col("n_share_tier").max(),
                 pl.col("omp_threads").max(),
                 pl.len().alias("n_folds")))
    return df.with_columns(pl.Series("_missing", [len(missing)] * df.height))


def read_lstm_cost(root: Path, legs: pl.DataFrame, tag: str,
                   rng: str, lstm_root: Path = None,
                   lstm_tag: str = None) -> pl.DataFrame:
    root = lstm_root or root
    tag = lstm_tag or (tag + "_es15")         # el LSTM del titular (P5)
    rows = []
    for r in legs.iter_rows(named=True):
        p = (root / "data" / "02_interim" / "routes" / r["route"] / "lstm_frame"
             / rng / f"{r['label']}_{tag}_overall.parquet")
        if not pg.parquet_ok(p):
            continue
        d = pl.read_parquet(p)
        if "elapsed_s" not in d.columns:
            continue
        for row in d.iter_rows(named=True):
            rows.append({"route": r["route"], "leg": r["leg"],
                         "algorithm": row["algorithm"],
                         "total_s": float(row["elapsed_s"])})
    return pl.DataFrame(rows)


def read_accuracy(root: Path, legs: pl.DataFrame, tag: str,
                  rng: str, lstm_root: Path = None,
                  lstm_tag: str = None) -> pl.DataFrame:
    """MAE por (leg, algoritmo) de las dos familias, sobre las MISMAS legs."""
    rows = []
    lroot = lstm_root or root
    ltag = lstm_tag or (tag + "_es15")        # el LSTM del titular (P5)
    for r in legs.iter_rows(named=True):
        base = root / "data" / "02_interim" / "routes" / r["route"]
        lbase = lroot / "data" / "02_interim" / "routes" / r["route"]
        for fam, p in (
            ("xgb",  base / "xgb" / rng / f"{r['label']}_{tag}_exo_overall.parquet"),
            ("lstm", lbase / "lstm_frame" / rng / f"{r['label']}_{ltag}_overall.parquet"),
        ):
            if not pg.parquet_ok(p):
                continue
            for row in pl.read_parquet(p).iter_rows(named=True):
                rows.append({"route": r["route"], "leg": r["leg"],
                             "algorithm": row["algorithm"],
                             "MAE_s": float(row["MAE_s"])})
    return pl.DataFrame(rows)


# ── agregación ───────────────────────────────────────────────────────

def build(timing: pl.DataFrame, lstm: pl.DataFrame, acc: pl.DataFrame,
          amortize: bool = False) -> pl.DataFrame:
    """Una fila por algoritmo: precisión mediana y coste sobre el panel."""
    share = pl.col("n_share_tier") if amortize else pl.lit(1)
    t = timing.with_columns(
        ((pl.col("prior_s") + pl.col("featurize_train_s")) / share)
        .alias("featurize_train_s"),
        (pl.col("featurize_eval_s") / share).alias("featurize_eval_s"),
    ).with_columns(
        (pl.col("fit_s") + pl.col("featurize_train_s")).alias("train_s"),
        (pl.col("predict_s") + pl.col("featurize_eval_s")).alias("infer_s"),
    ).with_columns(
        (pl.col("train_s") + pl.col("infer_s")).alias("total_s"),
        # Cota superior de tiempo de CPU: solo XGBoost usa los hilos de OMP.
        pl.when(pl.col("family") == "xgb")
          .then(pl.max_horizontal(pl.col("omp_threads"), pl.lit(1)))
          .otherwise(pl.lit(1)).alias("threads"),
    )

    per_algo = (t.group_by("algorithm").agg(
        pl.col("train_s").sort().sum().alias("train_s"),
        pl.col("infer_s").sort().sum().alias("infer_s"),
        pl.col("total_s").sort().sum().alias("total_s"),
        pl.col("total_s").median().alias("total_s_leg"),
        (pl.col("total_s") * pl.col("threads")).sort().sum().alias("core_s"),
        pl.col("threads").max().alias("threads"),
        pl.len().alias("legs"),
    ).with_columns(pl.lit("CPU").alias("device")))

    if not lstm.is_empty():
        l = lstm.group_by("algorithm").agg(
            pl.col("total_s").sort().sum().alias("total_s"),
            pl.col("total_s").median().alias("total_s_leg"),
            pl.len().alias("legs"),
        ).with_columns(
            # El LSTM no se puede partir en train/infer (ver cabecera): sus dos
            # columnas quedan nulas a propósito, no en cero, para que el panel
            # de inferencia lo OMITA en vez de dibujarlo en el origen.
            pl.lit(None, dtype=pl.Float64).alias("train_s"),
            pl.lit(None, dtype=pl.Float64).alias("infer_s"),
            pl.col("total_s").alias("core_s"),
            pl.lit(1).alias("threads"),
            pl.lit("GPU").alias("device"),
        )
        per_algo = pl.concat([per_algo, l.select(per_algo.columns)],
                             how="vertical_relaxed")

    mae = (acc.group_by("algorithm")
              .agg(pl.col("MAE_s").median().alias("MAE_s"),
                   pl.len().alias("legs_mae")))
    out = (per_algo.join(mae, on="algorithm", how="inner")
           .with_columns((pl.col("MAE_s") / 60.0).alias("MAE_min"),
                         pl.col("algorithm").replace_strict(
                             FAMILY, default="?").alias("model_family"))
           .sort("MAE_s"))
    return out


def pareto(df: pl.DataFrame, cost: str) -> List[str]:
    """Algoritmos no dominados: nadie es a la vez más barato y más preciso."""
    d = df.filter(pl.col(cost).is_not_null()).sort(cost)
    best, front = np.inf, []
    for r in d.iter_rows(named=True):
        if r["MAE_s"] < best:
            best = r["MAE_s"]
            front.append(r["algorithm"])
    return front


# ── salidas ──────────────────────────────────────────────────────────

def _accuracy_provenance(df: pl.DataFrame) -> str:
    """Frase de la nota sobre de dónde sale la columna de precisión — MEDIDA.

    ⚠ Por qué esto no es una constante. La frase anterior afirmaba que la
    columna venía de la corrida INSTRUMENTADA y que el modelo profundo caía en
    otra posición que en la Tabla 5, con el «3.73 min there» **escrito a mano**.
    Desde que `read_accuracy` toma los resúmenes `*_overall.parquet` —los MISMOS
    que agrega la Tabla 5— y el LSTM entra por `--lstm-root`, las dos coinciden
    dígito a dígito y la frase quedó falsa sin que nada fallara: no hay corrida
    que se caiga por una nota mentirosa. La lección del paso 51 es que una cifra
    a mano dentro de un fichero generado se pudre en silencio, así que aquí se
    compara contra el CSV de la Tabla 5 y la frase se elige con el resultado.

    Devuelve la frase; nunca lanza. Si la Tabla 5 no está o no cuadran los
    algoritmos, se emite la versión sin la afirmación de acuerdo — decir de
    menos es barato, decir de más es lo que este paso vino a arreglar.
    """
    neutral = ("Accuracy here is the median over the per-route error summaries of "
               "the run reported throughout Section 4; only the timings come "
               "from the instrumented execution.")
    try:
        t5 = (pl.read_csv(TAB / "table_05_permodel_panel.csv")
                .filter(pl.col("panel") == "all")
                .select("algorithm", "mae_med"))
        j = df.select("algorithm", "MAE_min").join(t5, on="algorithm", how="inner")
        if j.height != df.height:
            return neutral
        gap = float((j["MAE_min"] - j["mae_med"]).abs().max())
    except Exception:
        return neutral
    if gap < 0.005:                      # media décima de segundo
        return ("Accuracy here is the median over the same per-route error "
                "summaries that Table 2 aggregates — only the timings come from "
                "the instrumented execution — so the two agree to two decimals "
                "for every model.")
    return (f"Accuracy here comes from the instrumented run that produced the "
            f"timings and differs from Table 2 by up to {gap:.2f} min for some "
            f"model, so the two columns are not interchangeable.")


def to_markdown(df: pl.DataFrame, n_legs: int, amortize: bool) -> str:
    front = set(pareto(df, "total_s"))
    o: List[str] = []
    o.append("**Table 7. Accuracy against computational cost "
             f"({n_legs} pre-specified routes).**\n")
    o.append(
        "Cost is wall-clock time measured inside the cross-validation loop, "
        "summed over folds and routes, on the panel that Tables 2-4, S11, S16 and S17 use. "
        "*Training* is model fitting plus the feature construction it needs; "
        "*inference* is prediction at the evaluated cut positions. "
        "*Core-hours* multiplies by the threads the process was given and is an "
        "upper bound: only the gradient-boosted models are threaded "
        "(`n_jobs=-1` under `OMP_NUM_THREADS`), the baselines are single-threaded "
        "NumPy. The sequence model runs on a GPU and its clock covers training "
        "and evaluation together, so it has no inference column and its hours "
        "are not commensurable with the CPU ones — they are reported side by "
        "side, not summed. No model was tuned (Table S9); a hyperparameter-search "
        "budget would alter the cost comparison and was not included for any "
        "model. **On the frontier** marks models that no other model beats "
        "on both axes at once.\n")
    if amortize:
        o.append("\n*Feature construction is amortised across the variants that "
                 "share a tier.*\n")
    o.append("\n| Model | MAE (min) | Training | Inference | Total | "
             "Resource-hours | s/route | Device | On the frontier |")
    o.append("|---|---:|---:|---:|---:|---:|---:|---|---|")

    def _h(v: Optional[float]) -> str:
        # Bajo el centésimo de hora, "0.00 h" no dice nada: los baselines viven
        # ahí y es justo el extremo que la tabla existe para mostrar.
        if v is None:
            return "—"
        # 0.3 del manuscrito (10-oct): «0 s» para un tiempo medido (0.47 s, `delay`)
        # se leía como coste nulo; lo medido y no nulo bajo medio segundo es «< 1 s».
        if 0 < v < 0.5:
            return "< 1 s"
        return f"{v / 3600:.2f} h" if v >= 36.0 else f"{v:.0f} s"

    for r in df.iter_rows(named=True):
        o.append(
            f"| {LABEL.get(r['algorithm'], r['algorithm'])} | "
            f"{r['MAE_min']:.2f} | {_h(r['train_s'])} | {_h(r['infer_s'])} | "
            f"{_h(r['total_s'])} | {r['core_s'] / 3600:.2f} "
            f"{'GPU-h' if r['device'] == 'GPU' else 'core-h'} | "
            f"{r['total_s_leg']:.1f} | {r['device']} | "
            f"{'**yes**' if r['algorithm'] in front else 'no'} |")

    o.append(_notes(df, front))
    return "\n".join(o) + "\n"


def _notes(df: pl.DataFrame, front: set) -> str:
    """La nota de la tabla, con cada afirmación elegida por los datos.

    ⚠ La de agosto decía «it is already more accurate» (la media horaria contra el
    LSTM), «while being X% more accurate» y «The sequence model is on the frontier
    of neither axis: it is dominated, not traded off» como frases fijas: con otra
    corrida podían quedar falsas sin que nada fallara (paso 51). Aquí cada una sale
    sólo si los números la sostienen; las cifras de coste se dan siempre."""
    provenance = _accuracy_provenance(df)
    tail = (f" {provenance} The sequence model is not bit-reproducible on GPU "
            f"(Section 4.2), so re-executing its training moves its row by a second "
            f"or two — enough to reorder the foot of the table, which is why the "
            f"ranking there is not cited.")
    rows = {r["algorithm"]: r for r in df.iter_rows(named=True)}
    c, b, d = rows.get("delay_hour"), rows.get("xgb_mae_b"), rows.get(LSTM)
    if not (c and b and d):
        return "\n*Notes.*" + tail

    def _cmp(x: dict, y: dict) -> str:
        """«more» / «less accurate than» de x contra y, con el porcentaje."""
        pct = 100 * abs(1 - x["MAE_s"] / y["MAE_s"])
        return (f"{pct:.1f}% more accurate than" if x["MAE_s"] < y["MAE_s"]
                else f"{pct:.1f}% less accurate than")

    # (10-oct) La media por hora cuesta 13 s en todo el panel, y «0.00 h» se leía como
    # coste nulo: mismas unidades que la columna de la tabla (`_h`).
    s = c["total_s"]
    coste_c = (f"{s / 3600:.2f} h" if s >= 36.0
               else "under a second" if s < 0.5 else f"{s:.0f} s")
    txt = (f"\n*Notes.* The hour-of-day historical mean fits no parameters: its "
           f"whole cost is {coste_c} over the panel, "
           f"{d['total_s'] / c['total_s']:.0f}× less than the sequence model, and "
           f"it is {_cmp(c, d)} the sequence model. "
           f"{LABEL.get('xgb_mae_b')} costs {b['total_s'] / 3600:.2f} h against the "
           f"sequence model's {d['total_s'] / 3600:.2f} h — "
           f"{d['total_s'] / b['total_s']:.1f}× less observed wall-clock time on the "
           f"reported hardware — and is {_cmp(b, d)} it. "
           # 0.4 del manuscrito (10-oct): dos estadísticos dan dos cifras para el mismo
           # margen; la nota dice cuál es el suyo y dónde está el otro.
           f"These percentages compare panel medians of the accuracy column; the "
           f"paired per-route differences, with their intervals, are in Table 3.")
    if LSTM not in front:
        # dominado: alguien es a la vez más barato y más preciso
        dom = [a for a, r in rows.items() if a != LSTM
               and r["total_s"] <= d["total_s"] and r["MAE_s"] < d["MAE_s"]]
        if dom:
            txt += (f" The sequence model is not on the frontier: "
                    f"{len(dom)} model{'s are' if len(dom) != 1 else ' is'} both "
                    f"cheaper and more accurate.")
    return txt + tail


def _label_points(ax, pts: List[tuple]) -> None:
    """Etiqueta cada punto empujando en vertical las que se pisarían.

    Con diez modelos hay pares que caen casi encima — `kNN` y `kernel` están a
    una centésima de minuto, y el LSTM aterriza sobre `xgb_mse` — y las
    etiquetas se solapan hasta ser ilegibles. Nadie audita una figura, solo se
    mira (§7.7 del handoff), así que la separación va aquí y no en el ojo del
    que la genere. Cuando una etiqueta se desplaza, un filete la une a su punto.
    """
    if not pts:
        return
    xs = np.log10(np.array([p[0] for p in pts], dtype=float))
    ys = np.array([p[1] for p in pts], dtype=float)
    xr = float(xs.max() - xs.min()) or 1.0
    yr = float(ys.max() - ys.min()) or 1.0
    step = 0.052 * yr

    placed: List[tuple] = []
    for i in np.argsort(-ys):                       # de arriba a abajo
        y = ys[i]
        for _ in range(40):
            if not any(abs(y - py) < step and abs(xs[i] - px) < 0.34 * xr
                       for px, py in placed):
                break
            y -= step
        placed.append((xs[i], y))
        moved = abs(y - ys[i]) > 1e-9
        ax.annotate(
            pts[i][2], xy=(pts[i][0], ys[i]),
            xytext=(pts[i][0] * 1.35, y), textcoords="data",
            fontsize=6.5, color="#333333", va="center", ha="left",
            arrowprops=dict(arrowstyle="-", lw=0.5, color="#BBBBBB",
                            shrinkA=0, shrinkB=3) if moved else None,
        )


def figure(df: pl.DataFrame, n_legs: int) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(10.6, 4.3))
    for k, (cost, title) in enumerate(
            [("total_s", "Total cost (training + inference)"),
             ("infer_s", "Inference cost only")]):
        d = df.filter(pl.col(cost).is_not_null() & (pl.col(cost) > 0))
        front = pareto(d, cost)
        fr = d.filter(pl.col("algorithm").is_in(front)).sort(cost)
        ax[k].plot([r / 3600 for r in fr[cost]], fr["MAE_min"],
                   ls="--", lw=1.0, color="#BBBBBB", zorder=1,
                   drawstyle="steps-post")
        pts = []
        for r in d.iter_rows(named=True):
            col, mk = STYLE[r["model_family"]]
            ax[k].scatter(r[cost] / 3600, r["MAE_min"], s=54, marker=mk,
                          color=col, zorder=3,
                          edgecolors="white", linewidths=0.8)
            pts.append((r[cost] / 3600, r["MAE_min"],
                        LABEL.get(r["algorithm"], r["algorithm"])))
        # Sitio para las etiquetas: en log, headroom es un factor, no una suma.
        lo = min(p[0] for p in pts)
        hi = max(p[0] for p in pts)
        ax[k].set_xscale("log")
        ax[k].set_xlim(lo / 2.5, hi * 22)
        _label_points(ax[k], pts)
        ax[k].margins(y=0.16)
        ax[k].set_xlabel("wall-clock hours over the panel (log scale)")
        ax[k].set_title(title, fontsize=9.5)
    ax[0].set_ylabel("median MAE over routes (min)")
    ax[1].text(0.02, 0.03,
               "the sequence model is absent here:\nits clock cannot be split",
               transform=ax[1].transAxes, fontsize=6.5, color="#777777")
    panel_letters(ax)
    fig.suptitle(f"Accuracy against computational cost · {n_legs} routes",
                 fontsize=10)
    fig.tight_layout()
    save(fig, "fig_results_6_cost_accuracy")


# ── main ─────────────────────────────────────────────────────────────

def main() -> int:
    # Mismo guard que `reversal_params_sensitivity.py`: fuera de una terminal
    # UTF-8 el `✗` del mensaje de aborto reventaba con UnicodeEncodeError, y el
    # mensaje que se pierde es justo el que explica qué falta.
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    ap = argparse.ArgumentParser(description="Tabla 7 · coste vs precisión")
    ap.add_argument("--root", default=None)
    ap.add_argument("--lstm-root", default=None, dest="lstm_root")
    ap.add_argument("--lstm-tag", default=None, dest="lstm_tag")
    ap.add_argument("--q", type=float, default=0.05)
    ap.add_argument("--cov", type=float, default=1.0)
    ap.add_argument("--range", dest="rng", default=RANGE)
    ap.add_argument("--amortize", action="store_true",
                    help="reparte la featurización entre las variantes que "
                         "comparten tier (default: se atribuye entera)")
    ap.add_argument("--no-figure", action="store_true")
    a = ap.parse_args()

    repo = Path(__file__).resolve().parents[3]
    root = Path(a.root).resolve() if a.root else repo
    tag = qtag(a.q, a.cov)

    legs = panel_legs(root, a.q, a.cov, a.rng)
    if legs.is_empty():
        print(f"  ✗ no hay legs del panel bajo {root}")
        return 1

    timing = read_timing(root, legs, tag, a.rng)
    if timing.is_empty():
        # No dibujes una figura vacía: di qué falta y cómo se produce.
        print("=" * 74)
        print("  ✗ no hay un solo `*_timing.parquet` bajo")
        print(f"    {root}")
        print()
        print("  Los emite `xgb_prediction.cross_validation` desde 2026-08-17;")
        print("  los paquetes anteriores NO los llevan. Para producirlos hay que")
        print("  re-correr la fase de CPU con el código instrumentado:")
        print()
        print("    cd final_pipeline && bash run_pipeline.sh --only 1 --dry-run")
        print()
        print("  y lanzar el sbatch que imprime AÑADIENDO `--force` al final:")
        print("  sin él `_already_done` da las rutas por hechas y no escribe")
        print("  ningún timing (la trampa de docs/paper/HANDOFF_FRAME.md §7.8).")
        print("=" * 74)
        return 2

    n_missing = int(timing["_missing"].max() or 0)
    if n_missing:
        print(f"  ⚠ {n_missing} legs del panel sin timing: la tabla sale sobre "
              f"las que lo tienen. Re-corre esas legs antes de reportar.")
    timing = timing.drop("_missing")

    lroot = Path(a.lstm_root).resolve() if a.lstm_root else None
    lstm = read_lstm_cost(root, legs, tag, a.rng, lroot, a.lstm_tag)
    acc  = read_accuracy(root, legs, tag, a.rng, lroot, a.lstm_tag)
    df   = build(timing, lstm, acc, amortize=a.amortize)

    n_legs = int(timing.select(pl.struct("route", "leg").n_unique()).item())
    print("=" * 74)
    print(f"  TABLA 12 · coste vs precisión · {tag} · {a.rng}")
    print(f"  legs del panel: {legs.height} · con timing: {n_legs}")
    print("=" * 74)
    front = set(pareto(df, "total_s"))
    print(f"  {'modelo':<28} {'MAE min':>8} {'total h':>9} {'core h':>8}  frontera")
    for r in df.iter_rows(named=True):
        print(f"  {LABEL.get(r['algorithm'], r['algorithm']):<28} "
              f"{r['MAE_min']:8.2f} {r['total_s'] / 3600:9.2f} "
              f"{r['core_s'] / 3600:8.2f}  "
              f"{'◆' if r['algorithm'] in front else ''}")

    TAB.mkdir(parents=True, exist_ok=True)
    df.write_csv(TAB / "table_12_cost_accuracy.csv")
    (TAB / "table_12_cost_accuracy.md").write_text(
        to_markdown(df, n_legs, a.amortize), encoding="utf-8")
    print(f"\n  ✓ {TAB / 'table_12_cost_accuracy.csv'}")
    print(f"  ✓ {TAB / 'table_12_cost_accuracy.md'}")
    if not a.no_figure:
        figure(df, n_legs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
