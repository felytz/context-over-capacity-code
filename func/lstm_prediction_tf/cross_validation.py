"""
lstm_prediction_tf.cross_validation
=====================================
Etapa E: K-fold cross-validation por VIAJES (TensorFlow).

Misma lógica que la versión PyTorch — particiones por trip_id para evitar
leakage, schema de df_errors idéntico al paso 10 para concatenación
directa, agregadores reutilizables.

Diferencias clave respecto a la versión PyTorch
-----------------------------------------------
- Instancia modelos de Keras en lugar de PyTorch.
- Construye `tf.data.Dataset` en lugar de `DataLoader`.
- El "modelo fresco por fold" se logra creando un nuevo `keras.Model`
  desde `model_class(**model_kwargs)`.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import numpy as np
import polars as pl
import tensorflow as tf

from .data_loader import SequenceDataset, make_tf_dataset
from .models import BaseLSTMModel
from .trainer import train_model, evaluate_model, TrainingHistory


class InsufficientDataError(ValueError):
    """Menos viajes únicos que n_folds tras el dataset_builder.

    Mismo tipo que `arrival_prediction.cross_validation.InsufficientDataError`
    (paso 10), para que los wrappers (step_10/step_11) puedan capturarla y
    saltar el grupo (leg/direction) en vez de tumbar el run completo.
    """


# ──────────────────────────────────────────────────────────────────────
# Utilidades de partición
# ──────────────────────────────────────────────────────────────────────

def get_fold_assignments(
    trip_ids: List[Any],
    n_folds:  int = 5,
    seed:     int = 42,
) -> Dict[Any, int]:
    """Asigna cada trip_id a un fold determinístico."""
    if n_folds < 2:
        raise ValueError(f"n_folds debe ser ≥ 2, recibido {n_folds}")

    trip_ids_sorted = sorted(set(trip_ids), key=str)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(trip_ids_sorted))
    folds = np.array_split(perm, n_folds)

    assignments: Dict[Any, int] = {}
    for fold_id, indices in enumerate(folds):
        for idx in indices:
            assignments[trip_ids_sorted[idx]] = fold_id
    return assignments


def _split_trips_by_fold(
    assignments: Dict[Any, int],
    val_fold:    int,
    expanding:   bool = False,
) -> tuple:
    """Devuelve (train_trip_ids, val_trip_ids) para un fold dado.

    Con `expanding=True` el fold es el PERIODO del viaje y el entrenamiento son
    solo los anteriores — la misma regla que `xgb_prediction.cross_validation`,
    para que las dos familias sigan viendo exactamente lo mismo.
    """
    train, val = [], []
    for trip_id, fold_id in assignments.items():
        if fold_id == val_fold:
            val.append(trip_id)
        elif (fold_id < val_fold) if expanding else True:
            train.append(trip_id)
    return train, val


def _carve_es_holdout(train_trips: List[Any], es_frac: float,
                      seed: int) -> tuple:
    """Parte el entrenamiento en (ajuste, holdout de early stopping) POR VIAJE.

    Réplica de lo que hace XGBoost (`es_frac=0.15` sobre viajes, no sobre
    filas: las filas de un viaje son fuertemente dependientes). Existe porque
    sin ella el early stopping del LSTM mira el fold de EVALUACIÓN — ver la
    nota de `run_lstm_cross_validation`.
    """
    if es_frac <= 0.0 or len(train_trips) < 4:
        return list(train_trips), []
    order = sorted(train_trips, key=str)
    perm = np.random.default_rng(seed).permutation(len(order))
    n_es = max(1, int(round(es_frac * len(order))))
    es = [order[i] for i in perm[:n_es]]
    fit = [order[i] for i in perm[n_es:]]
    return fit, es


# ──────────────────────────────────────────────────────────────────────
# Orquestador principal
# ──────────────────────────────────────────────────────────────────────

def run_lstm_cross_validation(
    dataset:         SequenceDataset,
    model_class:     Type[BaseLSTMModel],
    model_kwargs:    Dict[str, Any],
    n_folds:         int   = 5,
    n_epochs:        int   = 50,
    batch_size:      int   = 64,
    lr:              float = 1e-3,
    weight_decay:    float = 0.0,
    patience:        int   = 5,
    grad_clip:       float = 1.0,
    seed:            int   = 42,
    model_name:      str   = "sequential_lstm",
    val_l_strategy:  Optional[str] = None,
    val_l_indices:   Optional[Dict[Any, List[int]]] = None,
    checkpoint_dir:  Optional[Path] = None,
    fold_assignments: Optional[Dict[Any, int]] = None,
    frame_hash:      Optional[str] = None,
    split:           str   = "random",
    es_frac:         float = 0.0,
    verbose:         bool  = True,
) -> Dict[str, Any]:
    """Corre K-fold cross-validation por viajes con TensorFlow.

    ⚠ **`es_frac` y de dónde sale el early stopping.** Con `es_frac=0.0` —el
    valor histórico y el que reproduce las cifras publicadas— el early stopping
    y el scheduler de learning rate miran el **fold de evaluación**. Es una
    asimetría real contra XGBoost, que reserva el 15% de sus viajes de
    entrenamiento para eso y nunca toca el fold evaluado: al modelo profundo se
    le está dando una ventaja. Como el resultado del paper es que aun así
    pierde, la asimetría lo hace CONSERVADOR y por eso el valor por defecto no
    cambia — cambiarlo movería cifras ya publicadas.

    Con ventana expansiva la asimetría deja de ser tolerable: parar de entrenar
    mirando el mes que se va a predecir es exactamente el look-ahead que la
    escalera existe para eliminar. Por eso `split="expanding"` exige
    `es_frac > 0` y el holdout se recorta del PASADO, por viaje, igual que en
    XGBoost.

    Si `checkpoint_dir` viene, cada fold terminado deja
    `cv_ckpt_fold<k>_errors.parquet` + `cv_ckpt_fold<k>_meta.json` en ese
    directorio y, al relanzar tras un crash, los folds ya completados se
    REANUDAN desde disco en vez de re-entrenarse (un fold de esta escala
    cuesta 4–9 h de CPU; un crash en el fold 4 perdía los 3 anteriores).
    El meta valida (seed, n_folds, n_trips) para no reanudar de otra
    configuración; al completar el CV los checkpoints se borran."""
    all_trips = dataset.get_trip_ids()
    if len(all_trips) < n_folds:
        raise InsufficientDataError(
            f"Solo hay {len(all_trips)} viajes únicos pero n_folds={n_folds}."
        )

    if fold_assignments is not None:
        # Partición IMPUESTA por el frame de evaluación (`func/eval_frame.py`).
        # Los baselines/XGB particionan filas de T; este builder particiona
        # trip_ids ordenados. Con folds distintos la comparación no es pareada y
        # hay que recertificarla a posteriori (la saga v6/v7). Con el frame la
        # partición es UNA y la comparación queda pareada por construcción.
        missing = [t for t in all_trips if t not in fold_assignments]
        if missing:
            raise InsufficientDataError(
                f"{len(missing)} viaje(s) del dataset no están en el frame "
                f"(p.ej. {missing[:3]}). El dataset debe construirse con el "
                f"whitelist del frame: si sobran viajes, la comparación deja de "
                f"ser pareada."
            )
        assignments = {t: int(fold_assignments[t]) for t in all_trips}
        if verbose:
            print(f"  Folds del FRAME      : {frame_hash or 'sin hash'} "
                  f"({len(assignments):,} viajes)")
    else:
        assignments = get_fold_assignments(all_trips, n_folds=n_folds, seed=seed)

    # El hash del frame entra en la clave del checkpoint: un fold guardado por
    # otra partición (o otro régimen de grid) NO debe reanudarse.
    ckpt_key = {"seed": seed, "n_folds": n_folds, "n_trips": len(all_trips),
                "model_name": model_name, "frame_hash": frame_hash}

    def _ckpt_paths(fold_id: int):
        d = Path(checkpoint_dir)
        return (d / f"cv_ckpt_fold{fold_id}_errors.parquet",
                d / f"cv_ckpt_fold{fold_id}_meta.json")

    if verbose:
        print("=" * 60)
        print(f"  CROSS-VALIDATION ({model_name})")
        print("=" * 60)
        print(f"  Viajes totales       : {len(all_trips):,}")
        print(f"  N folds              : {n_folds}")
        print(f"  Seed                 : {seed}")
        print(f"  Modelo               : {model_class.__name__}")
        print(f"  n_epochs (máx)       : {n_epochs}")
        print(f"  Patience             : {patience}")
        print(f"  Batch size           : {batch_size}")

    expanding = (split == "expanding")
    if expanding and es_frac <= 0.0:
        raise ValueError(
            "split='expanding' con es_frac=0 pararía de entrenar mirando el mes "
            "que se va a predecir: el early stopping tiene que salir del pasado. "
            "Pasa es_frac>0 (0.15 es lo que usa XGBoost)."
        )

    all_errors:        List[pl.DataFrame]    = []
    metrics_per_fold:  List[Dict[str, Any]]  = []
    histories:         List[TrainingHistory] = []

    t_start_global = time.time()

    for fold_id in range(n_folds):
        if expanding and fold_id == 0:
            # El primer periodo es entrenamiento siempre y evaluación nunca.
            continue
        if verbose:
            print(f"\n  ── Fold {fold_id + 1}/{n_folds} ──")

        # ── Resume: fold ya completado en una corrida interrumpida ──
        if checkpoint_dir is not None:
            ck_pq, ck_meta = _ckpt_paths(fold_id)
            if ck_pq.exists() and ck_meta.exists():
                meta = json.loads(ck_meta.read_text(encoding="utf-8"))
                if meta.get("key") == ckpt_key:
                    if verbose:
                        print(f"    ↩ reanudado desde checkpoint "
                              f"({ck_pq.name})")
                    all_errors.append(pl.read_parquet(ck_pq))
                    metrics_per_fold.append(meta["metrics"])
                    histories.append(TrainingHistory(**meta["history"]))
                    continue
                elif verbose:
                    print(f"    ⚠ checkpoint de fold {fold_id} ignorado "
                          f"(configuración distinta)")

        train_trips, val_trips = _split_trips_by_fold(assignments, fold_id,
                                                      expanding)
        # Holdout de early stopping recortado del ENTRENAMIENTO (ver la nota de
        # la función). Con es_frac=0 no se recorta nada y el comportamiento es
        # el histórico, bit a bit.
        fit_trips, es_trips = _carve_es_holdout(train_trips, es_frac,
                                                seed + 1000 * fold_id)
        ds_train_fold = dataset.filter_by_trip_ids(fit_trips)
        ds_val_fold   = dataset.filter_by_trip_ids(val_trips)
        ds_es_fold    = (dataset.filter_by_trip_ids(es_trips) if es_trips
                         else None)

        if val_l_strategy is not None and val_l_strategy != dataset.l_strategy:
            ds_val_fold_strat = SequenceDataset.__new__(SequenceDataset)
            ds_val_fold_strat.vocab              = dataset.vocab
            ds_val_fold_strat.min_l              = dataset.min_l
            ds_val_fold_strat.max_l_offset       = dataset.max_l_offset
            ds_val_fold_strat.max_context_length = dataset.max_context_length
            ds_val_fold_strat.l_strategy         = val_l_strategy
            ds_val_fold_strat._encoded_seqs      = dataset._encoded_seqs

            new_index: List[tuple] = []
            val_trips_set = set(val_trips)
            for trip_idx, eseq in enumerate(dataset._encoded_seqs):
                if eseq["trip_id"] not in val_trips_set:
                    continue
                n = len(eseq["stops"])
                max_l = n - 1 - dataset.max_l_offset
                if max_l < dataset.min_l:
                    continue
                if val_l_strategy == "uniform":
                    n_uniform = min(30, max_l - dataset.min_l + 1)
                    l_values = np.linspace(
                        dataset.min_l, max_l, n_uniform, dtype=int
                    ).tolist()
                    l_values = sorted(set(l_values))
                elif val_l_strategy == "all":
                    l_values = range(dataset.min_l, max_l + 1)
                elif val_l_strategy == "explicit":
                    if val_l_indices is None or eseq["trip_id"] not in val_l_indices:
                        continue
                    l_values = [
                        l for l in val_l_indices[eseq["trip_id"]]
                        if dataset.min_l <= l <= max_l
                    ]
                else:
                    raise ValueError(f"val_l_strategy inválida: {val_l_strategy}")
                for l in l_values:
                    new_index.append((trip_idx, l))
            ds_val_fold_strat._index = new_index
            ds_val_fold = ds_val_fold_strat

        if verbose:
            print(f"    Train: {len(fit_trips):,} viajes, "
                  f"{len(ds_train_fold):,} muestras")
            if ds_es_fold is not None:
                print(f"    ES   : {len(es_trips):,} viajes, "
                      f"{len(ds_es_fold):,} muestras (holdout del pasado)")
            print(f"    Val  : {len(val_trips):,} viajes, "
                  f"{len(ds_val_fold):,} muestras")

        tf_train = make_tf_dataset(ds_train_fold, batch_size=batch_size, shuffle=True)
        tf_val   = make_tf_dataset(ds_val_fold,   batch_size=batch_size, shuffle=False)
        # Lo que vigila el early stopping: el holdout si existe, y si no el fold
        # de evaluación (comportamiento histórico). La EVALUACIÓN es siempre
        # `tf_val`, pase lo que pase.
        tf_es    = (make_tf_dataset(ds_es_fold, batch_size=batch_size,
                                    shuffle=False) if ds_es_fold is not None
                    else tf_val)

        # Modelo fresco por fold con seed distinto
        tf.keras.utils.set_random_seed(seed + fold_id)
        model = model_class(**model_kwargs)

        train_result = train_model(
            model         = model,
            train_dataset = tf_train,
            val_dataset   = tf_es,
            n_epochs      = n_epochs,
            lr            = lr,
            weight_decay  = weight_decay,
            patience      = patience,
            grad_clip     = grad_clip,
            verbose       = verbose,
            log_every     = max(1, n_epochs // 10),
        )

        histories.append(train_result["history"])

        eval_result = evaluate_model(
            model      = train_result["model"],
            dataset    = tf_val,
            model_name = model_name,
            fold       = fold_id,
            verbose    = verbose,
        )
        all_errors.append(eval_result["df_errors"])
        metrics_per_fold.append({
            "fold":    fold_id,
            "n_train": len(fit_trips),
            "n_es":    len(es_trips),
            "n_val":   len(val_trips),
            **eval_result["metrics"],
        })

        # ── Checkpoint del fold recién terminado ────────────────────
        if checkpoint_dir is not None:
            ck_pq, ck_meta = _ckpt_paths(fold_id)
            eval_result["df_errors"].write_parquet(ck_pq)
            ck_meta.write_text(json.dumps({
                "key":     ckpt_key,
                "metrics": metrics_per_fold[-1],
                "history": train_result["history"].to_dict(),
            }), encoding="utf-8")

    df_errors_all = pl.concat(all_errors)

    # CV completo: los checkpoints ya no hacen falta
    if checkpoint_dir is not None:
        for fold_id in range(n_folds):
            for p in _ckpt_paths(fold_id):
                p.unlink(missing_ok=True)
    summary = _summarize_folds(metrics_per_fold)

    t_total = time.time() - t_start_global
    if verbose:
        print("\n" + "=" * 60)
        print(f"  CV completo en {t_total:.1f}s")
        rmse_list = [f"{m['rmse_s']:.2f}" for m in metrics_per_fold]
        print(f"  RMSE por fold: {rmse_list}")
        print(f"  RMSE promedio: {summary['rmse_s_mean']:.2f} ± "
              f"{summary['rmse_s_std']:.2f} s")
        print(f"  MAE promedio : {summary['mae_s_mean']:.2f} ± "
              f"{summary['mae_s_std']:.2f} s")
        print(f"  MAPE promedio: {summary['mape_mean']:.2f} ± "
              f"{summary['mape_std']:.2f} %")
        print("=" * 60)

    return {
        "df_errors":        df_errors_all,
        "metrics_per_fold": metrics_per_fold,
        "histories":        histories,
        "fold_assignments": assignments,
        "summary":          summary,
        "split":            split,
        "es_frac":          es_frac,
        "evaluated_folds":  [m["fold"] for m in metrics_per_fold],
    }


# ──────────────────────────────────────────────────────────────────────
# Agregadores (idénticos a la versión PyTorch)
# ──────────────────────────────────────────────────────────────────────

def _summarize_folds(
    metrics_per_fold: List[Dict[str, Any]],
) -> Dict[str, float]:
    if not metrics_per_fold:
        return {}

    rmse_vals = np.array([m["rmse_s"] for m in metrics_per_fold])
    mae_vals  = np.array([m["mae_s"]  for m in metrics_per_fold])
    mape_vals = np.array([m["mape"]   for m in metrics_per_fold])

    return {
        "rmse_s_mean": float(np.nanmean(rmse_vals)),
        "rmse_s_std":  float(np.nanstd(rmse_vals, ddof=1) if len(rmse_vals) > 1 else 0.0),
        "mae_s_mean":  float(np.nanmean(mae_vals)),
        "mae_s_std":   float(np.nanstd(mae_vals, ddof=1) if len(mae_vals) > 1 else 0.0),
        "mape_mean":   float(np.nanmean(mape_vals)),
        "mape_std":    float(np.nanstd(mape_vals, ddof=1) if len(mape_vals) > 1 else 0.0),
        "n_folds":     len(metrics_per_fold),
    }


def summarize_rmse_by_location(df_errors: pl.DataFrame) -> pl.DataFrame:
    """RMSE promedio por posición (compatible con df_errors mezclado)."""
    per_fold = (
        df_errors
        .group_by(["algorithm", "fold", "l_idx"])
        .agg(pl.col("error_sq").mean().sqrt().alias("rmse"))
    )
    return (
        per_fold
        .group_by(["algorithm", "l_idx"])
        .agg(
            pl.col("rmse").mean().alias("rmse_mean"),
            pl.col("rmse").std(ddof=1).alias("rmse_std"),
            pl.len().alias("n_folds"),
        )
        .sort(["algorithm", "l_idx"])
    )


def summarize_error_by_horizon(
    df_errors:    pl.DataFrame,
    horizon_bins: Optional[List[float]] = None,
) -> pl.DataFrame:
    """Agrupa error_abs por bins de horizonte (en minutos)."""
    if horizon_bins is None:
        horizon_bins = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0]

    df = df_errors.filter(pl.col("horizon_min") > 0)
    bin_edges = [0.0] + list(horizon_bins)
    bin_labels = list(horizon_bins)

    expr = None
    for i in range(len(bin_labels)):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]
        label = bin_labels[i]
        cond = (pl.col("horizon_min") > lower) & (pl.col("horizon_min") <= upper)
        if expr is None:
            expr = pl.when(cond).then(pl.lit(label))
        else:
            expr = expr.when(cond).then(pl.lit(label))
    expr = expr.otherwise(None).alias("horizon_bin_min")

    df = df.with_columns(expr).drop_nulls("horizon_bin_min")

    return (
        df.group_by(["algorithm", "horizon_bin_min"])
          .agg(
              pl.col("error_abs").mean().alias("error_abs_mean"),
              pl.col("error_abs").std(ddof=1).alias("error_abs_std"),
              pl.col("error_sq").mean().sqrt().alias("rmse"),
              pl.len().alias("n_predictions"),
          )
          .sort(["algorithm", "horizon_bin_min"])
    )
