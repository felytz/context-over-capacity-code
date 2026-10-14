"""
lstm_prediction_tf.trainer
============================
Etapa D: loop de entrenamiento y evaluación (TensorFlow).

Responsabilidades
-----------------
1. `train_model`: entrenamiento con Adam + early stopping + gradient clipping.
2. `evaluate_model`: forward sobre val/test con df_errors compatible con
   el output del paso 10.

Decisiones operativas
---------------------
- Loop custom con `tf.GradientTape` en lugar de `model.fit`. Razones:
  * Control fino sobre máscaras y métricas custom.
  * Compatibilidad con el formato de outputs del paso 10.
  * Misma estructura visual que la versión PyTorch.
- Optimizador: Adam con lr=1e-3 + ReduceLROnPlateau manual.
- Early stopping: para si val_loss no mejora en `patience` epochs.
- Gradient clipping: norma global máxima 1.0.

Métricas
--------
- RMSE  (root mean squared error)
- MAE   (mean absolute error)
- MAPE  (mean absolute percentage error)

Compatibilidad con paso 10
--------------------------
`evaluate_model` produce un df_errors con columnas:
    algorithm, fold, trip_id, l_idx, h_idx, t_obs_l, t_true, t_pred,
    error_abs, error_sq, horizon_min
"""
from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl
import tensorflow as tf
from tensorflow import keras

from .data_loader import masked_mse_loss
from .models import BaseLSTMModel


# ──────────────────────────────────────────────────────────────────────
# Configuración y registro
# ──────────────────────────────────────────────────────────────────────

@dataclass
class TrainingHistory:
    """Histórico por epoch del entrenamiento."""
    train_loss:    List[float] = field(default_factory=list)
    val_loss:      List[float] = field(default_factory=list)
    val_rmse:      List[float] = field(default_factory=list)
    val_mae:       List[float] = field(default_factory=list)
    lr:            List[float] = field(default_factory=list)
    epoch_time_s:  List[float] = field(default_factory=list)
    best_epoch:    int         = -1
    best_val_loss: float       = float("inf")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "train_loss":    list(self.train_loss),
            "val_loss":      list(self.val_loss),
            "val_rmse":      list(self.val_rmse),
            "val_mae":       list(self.val_mae),
            "lr":            list(self.lr),
            "epoch_time_s":  list(self.epoch_time_s),
            "best_epoch":    int(self.best_epoch),
            "best_val_loss": float(self.best_val_loss),
        }


# ──────────────────────────────────────────────────────────────────────
# Utilidades
# ──────────────────────────────────────────────────────────────────────

def _format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(seconds, 60)
    return f"{int(m)}m{s:04.1f}s"


# ──────────────────────────────────────────────────────────────────────
# Loop de entrenamiento
# ──────────────────────────────────────────────────────────────────────

def train_model(
    model:           BaseLSTMModel,
    train_dataset:   tf.data.Dataset,
    val_dataset:     tf.data.Dataset,
    n_epochs:        int   = 50,
    lr:              float = 1e-3,
    weight_decay:    float = 0.0,
    patience:        int   = 5,
    patience_lr:     int   = 3,
    lr_factor:       float = 0.5,
    min_lr:          float = 1e-6,
    grad_clip:       float = 1.0,
    verbose:         bool = True,
    log_every:       int  = 1,
) -> Dict[str, Any]:
    """
    Entrena `model` con early stopping y ReduceLROnPlateau manual.

    Parameters
    ----------
    model : BaseLSTMModel
    train_dataset, val_dataset : tf.data.Dataset
        Producen batches de 8 elementos (ver make_tf_dataset).
    n_epochs : int, default=50
    lr : float, default=1e-3
    weight_decay : float, default=0.0
        L2 regularization (vía AdamW).
    patience : int, default=5
        Epochs sin mejora antes de early stopping.
    patience_lr : int, default=3
        Epochs sin mejora antes de reducir lr.
    lr_factor : float, default=0.5
    min_lr : float, default=1e-6
    grad_clip : float, default=1.0
        Norma global máxima del gradiente.
    verbose : bool
    log_every : int

    Returns
    -------
    dict con:
      - "model"          : modelo con los pesos del best epoch (restaurados)
      - "history"        : TrainingHistory
      - "best_weights"   : lista de tensores de pesos óptimos
    """
    # Optimizador con weight_decay opcional
    if weight_decay > 0:
        optimizer = keras.optimizers.AdamW(
            learning_rate=lr, weight_decay=weight_decay,
        )
    else:
        optimizer = keras.optimizers.Adam(learning_rate=lr)

    if verbose:
        # Forzar construcción del modelo para mostrar n_params
        sample_batch = next(iter(train_dataset))
        s_obs, t_obs, s_hor, o_mask, h_mask, _, _, _, ctx = sample_batch
        _ = model((s_obs, t_obs, s_hor, o_mask, h_mask, ctx), training=False)
        print(f"  Parámetros entrenables: {model.num_parameters():,}")

    # ── Train step compilado con tf.function (acelera ~10x en CPU) ──
    # Definimos input_signature para evitar retracing en cada batch (que
    # tienen longitudes variables).
    @tf.function(reduce_retracing=True)
    def train_step(s_obs, t_obs, s_hor, o_mask, h_mask, target, ctx):
        with tf.GradientTape() as tape:
            pred = model((s_obs, t_obs, s_hor, o_mask, h_mask, ctx), training=True)
            loss = masked_mse_loss(target, pred, h_mask, reduction="mean")
        grads = tape.gradient(loss, model.trainable_variables)
        if grad_clip is not None:
            grads, _ = tf.clip_by_global_norm(grads, grad_clip)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        n_valid = tf.reduce_sum(tf.cast(h_mask, tf.float32))
        return loss, n_valid

    history = TrainingHistory()
    best_weights: Optional[List[np.ndarray]] = None
    epochs_no_improve     = 0
    epochs_no_improve_lr  = 0
    current_lr            = lr

    for epoch in range(1, n_epochs + 1):
        t_start = time.time()

        # ── Train ─────────────────────────────────────────────────────
        train_total_sq_err = 0.0
        train_n_valid      = 0.0

        for batch in train_dataset:
            s_obs, t_obs, s_hor, o_mask, h_mask, target, _, _, ctx = batch
            loss, n_valid = train_step(s_obs, t_obs, s_hor, o_mask, h_mask,
                                       target, ctx)
            train_total_sq_err += float(loss.numpy()) * float(n_valid.numpy())
            train_n_valid       += float(n_valid.numpy())

        train_loss = (train_total_sq_err / max(train_n_valid, 1.0)
                      if train_n_valid else float("nan"))

        # ── Validation ────────────────────────────────────────────────
        val_metrics = _evaluate_loss(model, val_dataset)
        val_loss = val_metrics["loss"]
        val_rmse = val_metrics["rmse"]
        val_mae  = val_metrics["mae"]

        # ── Tracking + early stopping + lr scheduling manual ──────────
        epoch_time = time.time() - t_start
        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        history.val_rmse.append(val_rmse)
        history.val_mae.append(val_mae)
        history.lr.append(float(current_lr))
        history.epoch_time_s.append(epoch_time)

        improved = val_loss < history.best_val_loss - 1e-6
        if improved:
            history.best_val_loss = val_loss
            history.best_epoch    = epoch
            best_weights = [v.numpy().copy() for v in model.trainable_variables]
            epochs_no_improve     = 0
            epochs_no_improve_lr  = 0
        else:
            epochs_no_improve     += 1
            epochs_no_improve_lr  += 1

        # ReduceLROnPlateau manual
        if epochs_no_improve_lr >= patience_lr and current_lr > min_lr:
            new_lr = max(current_lr * lr_factor, min_lr)
            if new_lr < current_lr:
                current_lr = new_lr
                optimizer.learning_rate.assign(current_lr)
                epochs_no_improve_lr = 0

        if verbose and (epoch % log_every == 0 or epoch == 1):
            star = " ★" if improved else "  "
            print(
                f"  Epoch {epoch:3d}/{n_epochs}{star} "
                f"train={train_loss:8.2f}  "
                f"val={val_loss:8.2f}  "
                f"rmse={val_rmse:6.1f}s  "
                f"mae={val_mae:6.1f}s  "
                f"lr={current_lr:.1e}  "
                f"t={_format_time(epoch_time)}"
            )

        if epochs_no_improve >= patience:
            if verbose:
                print(f"  ↳ Early stopping en epoch {epoch} "
                      f"(sin mejora desde epoch {history.best_epoch})")
            break

    # Restaurar mejores pesos
    if best_weights is not None:
        for var, w in zip(model.trainable_variables, best_weights):
            var.assign(w)

    if verbose:
        total_time = sum(history.epoch_time_s)
        print(f"  ↳ Entrenamiento completo en {_format_time(total_time)} "
              f"({len(history.train_loss)} epochs)")
        print(f"  ↳ Mejor val_loss = {history.best_val_loss:.2f} "
              f"en epoch {history.best_epoch}")

    return {
        "model":         model,
        "history":       history,
        "best_weights":  best_weights,
    }


# ──────────────────────────────────────────────────────────────────────
# Evaluación de loss (interno)
# ──────────────────────────────────────────────────────────────────────

def _evaluate_loss(
    model:   BaseLSTMModel,
    dataset: tf.data.Dataset,
) -> Dict[str, float]:
    """Evalúa loss, RMSE y MAE agregados sobre todo el dataset."""

    @tf.function(reduce_retracing=True)
    def _eval_step(s_obs, t_obs, s_hor, o_mask, h_mask, target, ctx):
        pred = model((s_obs, t_obs, s_hor, o_mask, h_mask, ctx), training=False)
        mask_f = tf.cast(h_mask, tf.float32)
        diff = (pred - target) * mask_f
        sq_err  = tf.reduce_sum(tf.square(diff))
        abs_err = tf.reduce_sum(tf.abs(diff))
        n_valid = tf.reduce_sum(mask_f)
        return sq_err, abs_err, n_valid

    total_sq_err  = 0.0
    total_abs_err = 0.0
    total_n_valid = 0.0

    for batch in dataset:
        s_obs, t_obs, s_hor, o_mask, h_mask, target, _, _, ctx = batch
        sq, ab, nv = _eval_step(s_obs, t_obs, s_hor, o_mask, h_mask, target, ctx)
        total_sq_err  += float(sq.numpy())
        total_abs_err += float(ab.numpy())
        total_n_valid += float(nv.numpy())

    if total_n_valid == 0:
        return {"loss": float("nan"), "rmse": float("nan"), "mae": float("nan")}

    mse = total_sq_err / total_n_valid
    return {
        "loss": mse,
        "rmse": math.sqrt(mse),
        "mae":  total_abs_err / total_n_valid,
    }


# ──────────────────────────────────────────────────────────────────────
# Evaluación pública con df_errors detallado
# ──────────────────────────────────────────────────────────────────────

def evaluate_model(
    model:       BaseLSTMModel,
    dataset:     tf.data.Dataset,
    model_name:  str = "sequential_lstm",
    fold:        int = -1,
    verbose:     bool = False,
) -> Dict[str, Any]:
    """
    Evalúa el modelo y produce df_errors con schema compatible con paso 10.

    Returns
    -------
    dict con:
      - "metrics"    : {"rmse_s", "mae_s", "mape", "n_samples", "n_predictions"}
      - "df_errors"  : pl.DataFrame con una fila por predicción
    """
    rows: List[Dict[str, Any]] = []

    for batch in dataset:
        s_obs, t_obs, s_hor, o_mask, h_mask, target, l_indices, trip_ids, ctx = batch
        pred = model((s_obs, t_obs, s_hor, o_mask, h_mask, ctx), training=False)

        # A NumPy
        pred_np         = pred.numpy()
        target_np       = target.numpy()
        times_obs_np    = t_obs.numpy()
        obs_mask_np     = o_mask.numpy()
        horizon_mask_np = h_mask.numpy()
        l_indices_np    = l_indices.numpy()
        trip_ids_np     = trip_ids.numpy()

        B, H_max = pred_np.shape

        # Suma prefija (acumulado) tanto del target como de la predicción.
        # El acumulado es el tiempo de LLEGADA a la posición l+1+h medido desde
        # la posición observada l; es la cantidad comparable con los baselines
        # del paso 10 (cuyo t̂_{l,l+h} = t_l + Δ → el t_l se cancela en la resta).
        target_cumsum = np.cumsum(target_np * horizon_mask_np, axis=1)
        pred_cumsum   = np.cumsum(pred_np   * horizon_mask_np, axis=1)

        for i in range(B):
            trip_id = trip_ids_np[i].decode("utf-8") if isinstance(trip_ids_np[i], bytes) else str(trip_ids_np[i])
            l_idx   = int(l_indices_np[i])

            n_obs = int(obs_mask_np[i].sum())
            t_obs_l = float(times_obs_np[i, :n_obs].sum())

            for h in range(H_max):
                if not horizon_mask_np[i, h]:
                    continue
                # ── por-tramo (métrica nativa del LSTM: un solo link) ──
                t_true_seg = float(target_np[i, h])
                t_pred_seg = float(pred_np[i, h])
                err_seg    = t_pred_seg - t_true_seg
                # ── acumulado (tiempo de llegada; comparable con baselines) ──
                t_true_cum = float(target_cumsum[i, h])
                t_pred_cum = float(pred_cumsum[i, h])
                err_cum    = t_pred_cum - t_true_cum

                horizon_min = t_true_cum / 60.0
                rows.append({
                    "algorithm":   model_name,
                    "fold":        fold,
                    "trip_id":     trip_id,
                    "l_idx":       l_idx,
                    "h_idx":       h + 1,
                    "t_obs_l":     t_obs_l,
                    # CANÓNICAS = acumuladas → comparación honesta vs paso 10
                    "t_true":      t_true_cum,
                    "t_pred":      t_pred_cum,
                    "error_abs":   abs(err_cum),
                    "error_sq":    err_cum * err_cum,
                    # POR-TRAMO → alimentan el reporte de segmento
                    "t_true_seg":     t_true_seg,
                    "t_pred_seg":     t_pred_seg,
                    "error_abs_seg":  abs(err_seg),
                    "error_sq_seg":   err_seg * err_seg,
                    "horizon_min": horizon_min,
                })

    if not rows:
        df_errors = pl.DataFrame(schema={
            "algorithm":   pl.Utf8,
            "fold":        pl.Int64,
            "trip_id":     pl.Utf8,
            "l_idx":       pl.Int64,
            "h_idx":       pl.Int64,
            "t_obs_l":     pl.Float64,
            "t_true":      pl.Float64,
            "t_pred":      pl.Float64,
            "error_abs":   pl.Float64,
            "error_sq":    pl.Float64,
            "t_true_seg":     pl.Float64,
            "t_pred_seg":     pl.Float64,
            "error_abs_seg":  pl.Float64,
            "error_sq_seg":   pl.Float64,
            "horizon_min": pl.Float64,
        })
        metrics = {"rmse_s": float("nan"), "mae_s": float("nan"),
                   "mape": float("nan"),
                   "rmse_s_seg": float("nan"), "mae_s_seg": float("nan"),
                   "n_samples": 0, "n_predictions": 0}
        return {"metrics": metrics, "df_errors": df_errors}

    df_errors = pl.DataFrame(rows)

    n_pred = df_errors.height
    # Canónicas = acumuladas (comparable con baselines)
    rmse   = float(np.sqrt(df_errors["error_sq"].mean()))
    mae    = float(df_errors["error_abs"].mean())
    # Por-tramo (métrica nativa del modelo)
    rmse_seg = float(np.sqrt(df_errors["error_sq_seg"].mean()))
    mae_seg  = float(df_errors["error_abs_seg"].mean())

    valid_mape = df_errors.filter(pl.col("t_true") >= 1.0)
    if valid_mape.height > 0:
        mape = float(
            (valid_mape["error_abs"] / valid_mape["t_true"]).mean() * 100
        )
    else:
        mape = float("nan")

    n_samples = df_errors.select(["trip_id", "l_idx"]).unique().height

    metrics = {
        "rmse_s":        rmse,
        "mae_s":         mae,
        "mape":          mape,
        "rmse_s_seg":    rmse_seg,
        "mae_s_seg":     mae_seg,
        "n_samples":     n_samples,
        "n_predictions": n_pred,
    }

    if verbose:
        print(f"  Evaluación [{model_name}, fold={fold}]:")
        print(f"    RMSE (acum)   : {rmse:.2f} s")
        print(f"    RMSE (tramo)  : {rmse_seg:.2f} s")
        print(f"    MAE  (acum)   : {mae:.2f} s")
        print(f"    MAPE          : {mape:.2f} %")
        print(f"    n_samples     : {n_samples:,}")
        print(f"    n_predictions : {n_pred:,}")

    return {"metrics": metrics, "df_errors": df_errors}


# ──────────────────────────────────────────────────────────────────────
# Vistas por métrica (para generar reportes acumulado vs. por-tramo)
# ──────────────────────────────────────────────────────────────────────

_SEG_COLS = ["t_true_seg", "t_pred_seg", "error_abs_seg", "error_sq_seg"]


def segment_error_view(df_errors: pl.DataFrame) -> pl.DataFrame:
    """
    Devuelve una copia de df_errors con las columnas CANÓNICAS
    (t_true/t_pred/error_abs/error_sq) reemplazadas por las versiones
    POR-TRAMO. Sirve para que `plots.py`, que siempre lee las canónicas,
    genere el reporte de error por-tramo sin cambios.
    """
    if df_errors.height == 0 or "error_sq_seg" not in df_errors.columns:
        return df_errors
    return df_errors.with_columns([
        pl.col("t_true_seg").alias("t_true"),
        pl.col("t_pred_seg").alias("t_pred"),
        pl.col("error_abs_seg").alias("error_abs"),
        pl.col("error_sq_seg").alias("error_sq"),
    ])


def strip_segment_cols(df_errors: pl.DataFrame) -> pl.DataFrame:
    """
    Quita las columnas por-tramo, dejando solo el schema acumulado canónico
    (idéntico al del paso 10). Úsese antes de concatenar con los baselines
    para evitar avisos de schema y mantener el combinado limpio.
    """
    drop = [c for c in _SEG_COLS if c in df_errors.columns]
    return df_errors.drop(drop) if drop else df_errors
