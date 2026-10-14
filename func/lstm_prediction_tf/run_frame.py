"""
lstm_prediction_tf.run_frame
=============================
Corre el LSTM sobre el FRAME DE EVALUACIÓN ÚNICO (`func/eval_frame.py`), es
decir sobre exactamente los mismos viajes, la misma grid, los mismos cortes y
los mismos folds que los baselines y XGBoost.

Qué cambia respecto de la corrida certificada
---------------------------------------------
=====================  ===============================  ==========================
                       corrida certificada (v7)         esta corrida (frame)
=====================  ===============================  ==========================
población              todos los viajes con             los del frame
                       ``n_cells >= 5`` (~7,700/leg)    (~M del frame)
folds                  ``permutation(trip_ids)`` propia  los del frame
extensión de la grid   longitud propia de cada viaje    truncada a ``n_route_m``
cortes de validación   ``val_l_strategy="all"``         los 30 del frame
alineación con         a posteriori: protocolo matched  por construcción
baselines              (``idx_off`` / ``dt_rem``)
=====================  ===============================  ==========================

El desfase de índices, explícito
--------------------------------
El builder por rejilla guarda ``times[j] = t_grid[j+1] − t_grid[j]``, así que el
corte ``c`` observa ``times[0..c]`` y eso suma el tiempo hasta la distancia
``(c+1)·step``. En la matriz ``T`` de los baselines, ``l_idx`` es el tiempo a
distancia ``l_idx·step``. Por lo tanto:

    LSTM (l_idx = c, h_idx = k)   ≡   baselines (l_idx = c + 1, h_idx = k)

Esa unidad de desfase es el ``idx_off`` que hizo falta certificar en la saga v6/v7.
Aquí se aplica explícitamente: se piden los cortes ``l − 1`` y al persistir se
devuelve ``l_idx + 1`` a coordenadas de frame. El horizonte NO lleva desfase: el
``cumsum`` de ``times_target`` da tiempo RESTANTE desde el corte, igual que el
``dt_rem`` del harness de XGBoost.

Uso
---
    BUS_ROUTE=101 BUS_ANALYSIS_RANGE=2024_01-08_tmin \
      python func/lstm_prediction_tf/run_frame.py \
             --n-route-quantile 0.05 --max-trips 2500 --n-epochs 40
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np                                            # noqa: E402
import polars as pl                                           # noqa: E402

from config import PATHS, INT_ROUTE, ANALYSIS_RANGE            # noqa: E402
import eval_frame as ef                                       # noqa: E402
import honest_metrics as hm                                   # noqa: E402
import parquet_guard as pg                                    # noqa: E402

APTA_EARLY_S = -60.0
APTA_LATE_S  = 300.0


def _out_dir(q: float) -> Path:
    d = PATHS.interim_route_dir / "lstm_frame" / ANALYSIS_RANGE
    d.mkdir(parents=True, exist_ok=True)
    return d


def _stem(leg: Any, q: float, cov: float, split: str = "random",
          es_frac: float | None = None) -> str:
    """Nombre base de los artefactos de SALIDA de esta corrida.

    ⚠ `es_frac` entra aquí y NO en `ef.frame_path`. El frame de evaluación
    —viajes, cortes, folds— es el mismo objeto se pare como se pare el
    entrenamiento, así que sigue siendo `q05c100` y `--require-frame` lo
    encuentra. Lo que cambia con el early stopping es el RESULTADO, y por eso
    el sufijo solo toca a los `*_overall/_devhist/_by_horizon`.
    """
    return f"{ef.safe_label(leg)}_{ef.q_tag(q, cov, split, es_frac)}"


# ──────────────────────────────────────────────────────────────────
# Métricas — MISMA convención que xgb_prediction y el paso 10
# ──────────────────────────────────────────────────────────────────

def _agg_exprs() -> List[pl.Expr]:
    return [
        pl.len().alias("n_pairs"),
        pl.col("_res").abs().mean().alias("MAE_s"),
        (pl.col("_res") ** 2).mean().sqrt().alias("RMSE_s"),
        pl.col("_res").abs().median().alias("MedAE_s"),
        pl.col("_res").abs().quantile(0.90).alias("P90AE_s"),
        pl.col("_res").mean().alias("bias_s"),
        (pl.col("_dev").is_between(APTA_EARLY_S, APTA_LATE_S).mean() * 100.0)
        .alias("pct_ontime_dev"),
    ]


def _prep(df_err: pl.DataFrame) -> pl.DataFrame:
    """`t_true`/`t_pred` del LSTM ya están en tiempo RESTANTE desde el corte
    (cumsum de los tramos), igual que el `dt_rem` del harness de XGBoost, así que
    MAE/RMSE/bias/on-time son directamente comparables. `res = pred − true`
    (bias<0 ⇒ subestima el restante ⇒ cae del lado TARDE, el ancho de la banda)."""
    return df_err.with_columns(
        (pl.col("t_pred") - pl.col("t_true")).alias("_res"),
        (pl.col("t_true") - pl.col("t_pred")).alias("_dev"),
        # Vuelta a coordenadas de frame: l_idx_frame = l_idx_lstm + 1.
        (pl.col("l_idx") + 1).alias("l_idx_frame"),
    )


def _overall(d: pl.DataFrame, algo: str, leg: str) -> Dict[str, Any]:
    row = d.select(_agg_exprs()).to_dicts()[0]
    row.update({"route": str(INT_ROUTE), "leg": leg, "algorithm": algo})
    return row


def _by_fold(d: pl.DataFrame, algo: str, leg: str) -> pl.DataFrame:
    return (d.group_by("fold").agg(_agg_exprs())
            .with_columns(pl.lit(str(INT_ROUTE)).alias("route"),
                          pl.lit(leg).alias("leg"),
                          pl.lit(algo).alias("algorithm"))
            .sort("fold"))


def _dev_hist(d: pl.DataFrame, algo: str, leg: str) -> pl.DataFrame:
    """Histograma con signo de ``dev`` por fold, bins de 1 s — fase 5.

    MISMO esquema y MISMO binning que el harness de XGBoost
    (`xgb_prediction.cross_validation.ErrorAccumulator.dev_bins`), para que las
    métricas honestas se deriven igual en las dos familias: el bin se etiqueta
    por su BORDE IZQUIERDO (`floor`, no `round`, que además es half-to-even en
    polars y half-away-from-zero en numpy) y satura en ±`DEV_MAX_ABS_S`. Aquí
    los residuos SÍ están en memoria, así que sale de un group_by; en XGBoost
    hay que acumularlo al vuelo porque son ~300M pares.
    """
    lim = hm.DEV_MAX_ABS_S
    return (d.with_columns(
                ((pl.col("_dev") / hm.DEV_BIN_S).floor() * hm.DEV_BIN_S)
                .clip(-lim, lim).cast(pl.Int32).alias("dev_s"))
            .group_by(["fold", "dev_s"]).agg(pl.len().alias("count"))
            .with_columns(pl.lit(str(INT_ROUTE)).alias("route"),
                          pl.lit(leg).alias("leg"),
                          pl.lit(algo).alias("algorithm"),
                          pl.col("fold").cast(pl.Int32),
                          pl.col("count").cast(pl.Int64))
            .select(["route", "leg", "algorithm", "fold", "dev_s", "count"])
            .sort(["fold", "dev_s"]))


def _by_horizon(d: pl.DataFrame, algo: str, leg: str) -> pl.DataFrame:
    """Métricas por bin de TIEMPO RESTANTE REAL — §4.5, mismo esquema que XGB.

    Se binea por `t_true`, nunca por `t_pred`: con la predicción, el estrato
    dependería del modelo y los algoritmos dejarían de compararse sobre las
    mismas filas.
    """
    lab = pl.Series("h", hm.HORIZON_LABELS)
    dd = d.with_columns(
        # `left_closed=True` es OBLIGATORIO: polars cierra por la derecha por
        # defecto y `np.searchsorted(side="right")` del harness de XGBoost cierra
        # por la izquierda, así que un valor justo en el borde (120.0 s) caería
        # en bins distintos en cada familia. Con las etiquetas "0-2 / 2-5" la
        # convención correcta es [0,120), [120,300), … es decir izquierda.
        pl.col("t_true").cut(list(hm.HORIZON_EDGES_S[1:]),
                             labels=list(hm.HORIZON_LABELS),
                             left_closed=True)
        .cast(pl.Utf8).alias("horizon"))
    return (dd.group_by("horizon").agg(_agg_exprs())
            .with_columns(pl.lit(str(INT_ROUTE)).alias("route"),
                          pl.lit(leg).alias("leg"),
                          pl.lit(algo).alias("algorithm"))
            .sort("horizon"))


def _by_position(d: pl.DataFrame, algo: str, leg: str) -> pl.DataFrame:
    return (d.group_by("l_idx_frame").agg(_agg_exprs())
            .rename({"l_idx_frame": "l_idx"})
            .with_columns(pl.lit(str(INT_ROUTE)).alias("route"),
                          pl.lit(leg).alias("leg"),
                          pl.lit(algo).alias("algorithm"))
            .sort("l_idx"))


# ──────────────────────────────────────────────────────────────────

def run_leg(
    df_full: pl.DataFrame, leg: str, *,
    n_route_quantile: float, max_trips: Optional[int],
    n_epochs: int, n_folds: int, seed: int, batch_size: int,
    with_calendar: bool, grid_step_m: int, min_coverage: float,
    require_frame: bool, allow_unpaired_cuts: bool,
    split: str = "random", es_frac: float = 0.0, verbose: bool = True,
) -> Optional[Dict[str, Any]]:
    from lstm_prediction_tf.run_experiment import build_lstm_prediction_pipeline

    t0 = time.time()

    # ── 1) Frame: se lee si existe, se construye y persiste si no ──
    try:
        fr = ef.load_frame(INT_ROUTE, leg, n_route_quantile, ANALYSIS_RANGE,
                           min_coverage=min_coverage, split=split)
    except FileNotFoundError:
        if require_frame:
            # En el server esto es lo que se quiere: el frame lo escribe la
            # corrida CPU (baselines+XGB), que es la barata. Si falta, construirlo
            # aquí funcionaría, pero dependería de que los flags coincidan —y una
            # discrepancia silenciosa arruina horas de GPU.
            print(f"  ✗ {leg}: NO hay frame en disco y --require-frame está "
                  f"activo. Corre primero la fase CPU "
                  f"(jobs/job_xgb_prediction.slrm escribe los frames).",
                  flush=True)
            return None
        if verbose:
            print(f"  frame ausente para {leg}: construyéndolo …", flush=True)
        try:
            built = ef.build_frame(
                df_full, leg, n_route_quantile=n_route_quantile,
                grid_step_m=grid_step_m, n_folds=n_folds, seed=seed,
                max_trips=max_trips, min_coverage=min_coverage,
                route=str(INT_ROUTE), split=split,
                analysis_range=ANALYSIS_RANGE, verbose=verbose,
            )
        except ValueError as e:
            print(f"  ⚠ {leg} SALTADA (frame): {e}", flush=True)
            return None
        ef.save_frame(built, route=str(INT_ROUTE),
                      analysis_range=ANALYSIS_RANGE, verbose=verbose)
        fr = ef.load_frame(INT_ROUTE, leg, n_route_quantile, ANALYSIS_RANGE,
                           min_coverage=min_coverage, split=split)

    meta      = fr["meta"]
    n_cells   = int(meta["n_cells"])
    l_frame   = [int(l) for l in meta["l_indices"]]
    trip_ids  = fr["trip_ids"]
    fold_of   = fr["fold_of"]

    # El número de folds lo fija el FRAME, no el flag. Hasta ahora esto se
    # AVISABA unas líneas más abajo ("MANDA EL FRAME") pero no se hacía: al
    # pipeline le seguía llegando el valor del CLI. Con los dos regímenes en 5
    # daba igual; con la ventana expansiva son 8 —uno por mes— y usar 5 dejaría
    # los tres últimos escalones sin evaluar, en silencio y con el artefacto
    # dándose por completo.
    frame_folds = int(meta.get("n_folds") or n_folds)
    if frame_folds != n_folds:
        print(f"  · {leg}: el frame define {frame_folds} folds y se pidió "
              f"{n_folds} → manda el frame", flush=True)
    n_folds = frame_folds

    if meta["M"] < n_folds:
        print(f"  ⚠ {leg} SALTADA: M={meta['M']} < n_folds={n_folds}", flush=True)
        return None

    # ── Pre-vuelo de paridad de cortes (ANTES de tocar la GPU) ──
    # `l_strategy="uniform"` calcula los cortes POR VIAJE: `max_l` sale de la
    # longitud de esa secuencia. Solo si TODOS los viajes llegan a `n_cells`
    # comparten los mismos 30 cortes que el frame. Con `min_coverage < 1.0` hay
    # viajes más cortos (np.interp los rellena por saturación en la matriz T,
    # pero el builder del LSTM no fabrica nada) ⇒ la unión de cortes se dispara
    # —medido: 53 en vez de 30— y las métricas dejan de ser comparables.
    # Entrenar eso cuesta ~1.4 h de GPU para un resultado inservible.
    frame_cov = meta.get("min_coverage")
    if frame_cov is not None and float(frame_cov) < 1.0 and not allow_unpaired_cuts:
        print(f"  ✗ {leg} SALTADA: el frame tiene min_coverage={frame_cov} < 1.0, "
              f"así que los cortes del LSTM NO van a cuadrar con los del frame "
              f"(l_strategy=uniform los deriva por viaje). Re-corre la fase CPU "
              f"con --min-coverage 1.0, o pasa --allow-unpaired-cuts si de verdad "
              f"quieres el resultado no pareado.", flush=True)
        return None

    # El FRAME es la autoridad, no los flags: si difieren, manda el frame (es lo
    # que corrieron los baselines/XGB) y se avisa. Silenciarlo rompería la
    # comparación pareada sin que se note.
    #
    # ⚠ El nombre del archivo lleva la leg y el cuantil, pero NO `min_coverage`
    # ni `max_trips`: un frame escrito con otra cobertura ocupa la MISMA ruta. El
    # hash detecta corrupción, no régimen — por eso este aviso es la única señal.
    # `n_folds` ya NO entra aquí: se toma del frame unas líneas más arriba, que
    # es lo que este bloque prometía y no hacía.
    for key, cli in (("max_trips", max_trips),
                     ("min_coverage", min_coverage), ("seed", seed),
                     ("grid_step_m", grid_step_m)):
        got = meta.get(key)
        if cli is not None and got is not None and got != cli:
            print(f"  ⚠ {leg}: el frame dice {key}={got} y se pidió {cli}. "
                  f"MANDA EL FRAME (es la muestra que corrieron los baselines). "
                  f"Si no es lo que quieres, re-corre la fase CPU con el régimen "
                  f"correcto antes de gastar GPU.", flush=True)

    if verbose:
        print(f"\n  {'─' * 62}")
        print(f"  leg {leg}  ·  frame {meta['frame_hash']}  ·  M={meta['M']:,}  "
              f"·  cov={meta.get('min_coverage')}  "
              f"·  n_route={meta['n_route_m']:,} m ({n_cells} celdas)")
        print(f"  cortes del frame: {len(l_frame)}  "
              f"[{l_frame[0]} … {l_frame[-1]}]  →  LSTM [{l_frame[0]-1} … "
              f"{l_frame[-1]-1}]")
        print(f"  {'─' * 62}", flush=True)

    # ── 2) Pipeline con la muestra, la grid y los folds del frame ──
    # max_sequence_length >= n_cells es OBLIGATORIO: si el builder trunca a las
    # últimas celdas, `l_idx` deja de ser la celda absoluta y se rompe la
    # equivalencia l_idx_frame = l_idx_lstm + 1 (sería el idx_off variable que
    # obligó al protocolo matched).
    with tempfile.TemporaryDirectory() as td:
        res = build_lstm_prediction_pipeline(
            df_full, group_col="leg", group_value=leg,
            model_type="sequential",
            with_calendar=with_calendar,
            trip_whitelist=trip_ids,
            max_cells=n_cells,
            fold_assignments=fold_of,
            frame_hash=meta["frame_hash"],
            min_l=1,
            # "uniform" con min_l=1 reproduce EXACTAMENTE los cortes del frame:
            # linspace(2, n-2, 30) − 1 == linspace(1, n-3, 30). Se verifica abajo.
            l_strategy="uniform",
            val_l_strategy=None,
            max_sequence_length=max(n_cells, 200),
            grid_step_m=grid_step_m,
            n_epochs=n_epochs, n_folds=n_folds, seed=seed,
            split=split, es_frac=es_frac,
            batch_size=batch_size, output_dir=td, verbose=verbose,
        )
        df_err = res["df_errors"]

    algo = "lstm_cal" if with_calendar else "lstm_a"
    d = _prep(df_err)

    # ── 3) Verificación de paridad de cortes ──
    got = sorted(d["l_idx_frame"].unique().to_list())
    exp = sorted(l_frame)
    cuts_match = got == exp
    if not cuts_match:
        print(f"  ⚠ CORTES NO PAREADOS en {leg}: LSTM tiene {len(got)} cortes "
              f"y el frame {len(exp)}. Solo en LSTM: {sorted(set(got)-set(exp))[:5]} "
              f"· solo en frame: {sorted(set(exp)-set(got))[:5]}", flush=True)

    trips_eval = d["trip_id"].n_unique()

    overall = _overall(d, algo, leg)
    overall.update({
        "M": int(meta["M"]),
        "n_trips_eval": int(trips_eval),
        "frame_hash": meta["frame_hash"],
        "cuts_match_frame": bool(cuts_match),
        "n_cuts": len(got),
        "elapsed_s": round(time.time() - t0, 1),
    })

    if verbose:
        print(f"  {algo}: MAE={overall['MAE_s']:.2f}  "
              f"RMSE={overall['RMSE_s']:.2f}  bias={overall['bias_s']:.2f}  "
              f"on-time={overall['pct_ontime_dev']:.2f}%  "
              f"n_pairs={overall['n_pairs']:,}  cortes={'OK' if cuts_match else 'MAL'}",
              flush=True)

    return {
        "overall":     pl.DataFrame([overall]),
        "by_fold":     _by_fold(d, algo, leg),
        "by_position": _by_position(d, algo, leg),
        "dev_hist":    _dev_hist(d, algo, leg),
        "by_horizon":  _by_horizon(d, algo, leg),
        "meta": {
            "route": str(INT_ROUTE), "leg": leg, "algorithm": algo,
            "analysis_range": ANALYSIS_RANGE,
            "n_route_quantile": n_route_quantile,
            "frame": meta,
            "n_epochs": n_epochs, "n_folds": n_folds, "seed": seed,
            "split": split, "es_frac": es_frac,
            "batch_size": batch_size, "with_calendar": with_calendar,
            "max_trips": max_trips,
            "min_l": 1, "l_strategy": "uniform",
            "max_sequence_length": max(n_cells, 200),
            # enmienda E3: >1 = la atención por trozos (rutas de > 512 celdas)
            "attention_chunks": int(res.get("metadata", {}).get("attention_chunks", 1)),
            "idx_convention": "l_idx_frame = l_idx_lstm + 1",
            "cuts_match_frame": bool(cuts_match),
            "n_trips_eval": int(trips_eval),
            "dataset_metadata": res.get("metadata", {}).get("dataset", {}),
            "elapsed_s": round(time.time() - t0, 1),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="LSTM sobre el frame de evaluación único")
    ap.add_argument("--legs", default=None, help="lista separada por comas")
    ap.add_argument("--n-route-quantile", type=float, default=0.05)
    ap.add_argument("--max-trips", type=int, default=2500)
    ap.add_argument("--grid-step-m", type=int, default=100)
    ap.add_argument("--n-epochs", type=int, default=40)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--split", default="random",
                    choices=["random", "expanding"],
                    help="'expanding' lee el frame `_tw` y corre la escalera: el escalón k entrena con los meses anteriores y evalúa el siguiente. Exige --es-frac > 0.")
    ap.add_argument("--es-frac", type=float, default=None,
                    help="fracción de VIAJES de entrenamiento reservada para el early stopping. Default: 0.0 en `random` (histórico: el early stopping mira el fold evaluado) y 0.15 en `expanding`, que es lo que usa XGBoost.")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--with-calendar", action="store_true",
                    help="variante lstm_cal (fase 4). Por defecto lstm_a, que "
                         "es la arquitectura certificada (context_dim=0)")
    ap.add_argument("--min-M", type=int, default=0,
                    help="salta legs con M < este valor (criterio de inclusión "
                         "pre-registrado; 0 = sin filtro)")
    ap.add_argument("--directional-only", action="store_true",
                    help="salta las legs self-loop (T0->T0, T1->T1)")
    ap.add_argument("--min-coverage", type=float, default=1.0,
                    help="solo se usa si hay que CONSTRUIR el frame; si ya "
                         "existe en disco, manda el frame. 1.0 = sin saturación "
                         "de np.interp y n_pairs exacto contra los baselines.")
    ap.add_argument("--allow-unpaired-cuts", action="store_true",
                    help="entrena aunque el frame tenga min_coverage<1.0 y los "
                         "cortes no vayan a cuadrar. El resultado NO es "
                         "comparable con los baselines; solo para diagnóstico.")
    ap.add_argument("--require-frame", action="store_true",
                    help="aborta la leg si no hay frame en disco en vez de "
                         "construirlo. Recomendado en el server: el frame lo "
                         "escribe la fase CPU y así no puede divergir.")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--accept-missing-devhist", action="store_true",
                    help="da por buena una leg ya calculada aunque no tenga "
                         "histograma de dev (corrida anterior a la fase 5). Esa "
                         "leg queda SIN banda simétrica ni on-time calibrado.")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    verbose = not a.quiet

    # El default de `--es-frac` depende del régimen: 0.0 conserva el
    # comportamiento histórico de la corrida aleatoria (y con él, sus cifras
    # publicadas); en la escalera temporal 0.0 no es admisible, porque el
    # early stopping estaría mirando el mes que se va a predecir.
    #
    # ⚠ El default sale de `ef.DEFAULT_ES_FRAC`, que es TAMBIÉN lo que decide si
    # `es_frac` entra en el nombre. Duplicar el valor aquí es la forma de que un
    # día el runner use un default y el nombre suponga otro — y ese desacuerdo
    # se manifiesta como una corrida que se salta como `cached`, en silencio.
    if a.es_frac is None:
        a.es_frac = ef.DEFAULT_ES_FRAC[a.split]

    out_tag = ef.q_tag(a.n_route_quantile, a.min_coverage, a.split, a.es_frac)
    print("=" * 72)
    print(f"  LSTM · FRAME {ef.q_tag(a.n_route_quantile, a.min_coverage, a.split)} "
          f"· ruta {INT_ROUTE} "
          f"· {ANALYSIS_RANGE}")
    print(f"  epochs={a.n_epochs} folds={a.n_folds} max_trips={a.max_trips} "
          f"calendar={a.with_calendar} split={a.split} es_frac={a.es_frac}")
    nota = ("early stopping propio de esta corrida"
            if ef.es_tag(a.es_frac, a.split)
            else "early stopping = el default del régimen")
    print(f"  artefactos → *_{out_tag}_*.parquet   ({nota})")
    print("=" * 72, flush=True)

    df_full = pl.read_parquet(PATHS.df_full)
    if a.legs:
        legs = [s.strip() for s in a.legs.split(",") if s.strip()]
    else:
        legs = (df_full.filter(pl.col("trip_valid"))
                .group_by("leg").agg(pl.col("trip_id").n_unique().alias("m"))
                .sort("m", descending=True)["leg"].to_list())

    if a.directional_only:
        keep = [l for l in legs if len(set(str(l).split("->"))) > 1]
        if verbose and len(keep) != len(legs):
            print(f"  --directional-only: {len(legs)} → {len(keep)} legs "
                  f"(saltadas: {[l for l in legs if l not in keep]})")
        legs = keep

    out_dir = _out_dir(a.n_route_quantile)
    n_ok = n_skip = 0
    for leg in legs:
        stem = _stem(leg, a.n_route_quantile, a.min_coverage, a.split,
                     a.es_frac)
        done_p = out_dir / f"{stem}_overall.parquet"
        if done_p.is_file() and not a.force:
            # ⚠ Se pregunta si el artefacto SE LEE, no si existe. Un parquet
            # truncado (nodo desalojado, escritura que Lustre no vació) es más
            # pegajoso que uno ausente: `is_file()` lo daba por hecho y el
            # reintento del array lo saltaba para siempre. Ver func/parquet_guard.py.
            why = pg.parquet_problem(done_p)
            if why is not None:
                print(f"  ⚠ {leg}: resultado previo CORRUPTO ({done_p.name}: "
                      f"{why}) → se recalcula", flush=True)
            else:
                # Un artefacto con cortes NO pareados no cuenta como calculado:
                # sus métricas no son comparables con los baselines. Si se dejara
                # pasar, una corrida en mal régimen quedaría "hecha" para siempre
                # y el error viajaría hasta las tablas del paper.
                try:
                    prev = pl.read_parquet(done_p)
                    ok_cuts = (bool(prev["cuts_match_frame"][0])
                               if "cuts_match_frame" in prev.columns else True)
                except Exception as exc:
                    # La envoltura estaba bien, así que esto es daño más profundo
                    # (grupo de filas ilegible). No se da por bueno.
                    print(f"  ⚠ {leg}: {done_p.name} no se deja leer "
                          f"({exc.__class__.__name__}) → se recalcula", flush=True)
                    ok_cuts = False
                else:
                    # Un artefacto sin `devhist` es de ANTES de la fase 5: sus
                    # residuos se perdieron (el CV corre en un tempdir y borra los
                    # checkpoints al terminar), así que de él no se puede derivar
                    # ni la banda simétrica ni el on-time calibrado. Darlo por
                    # hecho dejaría la leg sin métricas honestas para siempre y en
                    # silencio. Mismo criterio de lectura, no de existencia.
                    dh_p = out_dir / f"{stem}_devhist.parquet"
                    bh_p = out_dir / f"{stem}_by_horizon.parquet"
                    dh_why = pg.parquet_problem(dh_p) or pg.parquet_problem(bh_p)
                    has_dh = dh_why is None
                    if ok_cuts and (has_dh or a.accept_missing_devhist):
                        print(f"  {leg}: ya calculada (--force para rehacer)",
                              flush=True)
                        n_ok += 1
                        continue
                    if not ok_cuts:
                        print(f"  ⚠ {leg}: hay resultado previo con CORTES NO "
                              f"PAREADOS ({done_p.name}) → se recalcula",
                              flush=True)
                    else:
                        print(f"  ⚠ {leg}: resultado previo sin devhist/by_horizon "
                              f"utilizable ({dh_why}) → se recalcula. Para "
                              f"conservarlo tal cual y renunciar al on-time "
                              f"calibrado en esta leg, usa "
                              f"--accept-missing-devhist.", flush=True)

        if a.min_M > 0:
            try:
                fr = ef.load_frame(INT_ROUTE, leg, a.n_route_quantile,
                                   ANALYSIS_RANGE,
                                   min_coverage=a.min_coverage,
                                   split=a.split)
                if fr["meta"]["M"] < a.min_M:
                    print(f"  {leg}: SALTADA por --min-M "
                          f"(M={fr['meta']['M']} < {a.min_M})", flush=True)
                    n_skip += 1
                    continue
            except FileNotFoundError:
                pass   # se construye dentro de run_leg y se filtra ahí

        r = run_leg(
            df_full, leg,
            n_route_quantile=a.n_route_quantile, max_trips=a.max_trips,
            n_epochs=a.n_epochs, n_folds=a.n_folds, seed=a.seed,
            batch_size=a.batch_size, with_calendar=a.with_calendar,
            grid_step_m=a.grid_step_m, min_coverage=a.min_coverage,
            require_frame=a.require_frame,
            allow_unpaired_cuts=a.allow_unpaired_cuts,
            split=a.split, es_frac=a.es_frac, verbose=verbose,
        )
        if r is None:
            n_skip += 1
            continue

        for key, suf in (("overall", "overall"), ("by_fold", "by_fold"),
                         ("by_position", "by_position"), ("dev_hist", "devhist"),
                         ("by_horizon", "by_horizon")):
            r[key].write_parquet(out_dir / f"{stem}_{suf}.parquet")
        (out_dir / f"{stem}_meta.json").write_text(
            json.dumps(r["meta"], indent=2, ensure_ascii=False,
                       default=str), encoding="utf-8")

        # Releer lo recién escrito. En Lustre una escritura puede quedar sin
        # vaciar y el fichero acaba en ceros; si eso pasa se quiere saber AQUÍ,
        # con 1.4 h-GPU de trabajo aún en memoria, y no tres auditorías después.
        written = pg.scan(out_dir, f"{stem}_*.parquet")
        if written:
            for p, why in written:
                print(f"  ✗ {leg}: {p.name} salió CORRUPTO ({why})", flush=True)
            # Se quita SOLO el disparador de reanudación (`_overall.parquet`):
            # con él fuera el reintento rehace la leg, y lo corrupto se queda en
            # disco por si hay que mirarlo. Borrarlo todo sería tirar evidencia.
            (out_dir / f"{stem}_overall.parquet").unlink(missing_ok=True)
            print(f"  ✗ {leg}: NO se da por calculada (se retira su "
                  f"_overall.parquet para que el reintento la rehaga)", flush=True)
            n_skip += 1
            continue
        n_ok += 1

    print(f"\n  ✅ {n_ok}/{len(legs)} legs ({n_skip} saltadas) → {out_dir}")
    if n_ok == 0:
        # Ninguna leg calculada (ni hecha de antes) es un job que NO entrenó nada. Antes
        # salía con rc=0, en verde: con la vuelta todas las legs son T0->T0, y un
        # --directional-only olvidado las saltaba todas sin que nada se quejara.
        print(f"  ✗ ninguna leg calculada en la ruta {INT_ROUTE} "
              f"(legs: {legs or 'ninguna'}): sale con rc=4 para que se vea",
              flush=True)
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
