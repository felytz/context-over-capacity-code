"""
lstm_prediction_tf
====================
Adaptación del Sequential LSTM de Sun & Kim (2021) a la predicción de
tiempos de llegada en ruta fija de transporte público — implementación
TensorFlow / Keras 3.

Esta es la versión TF del paquete `lstm_prediction` (que usa PyTorch).
Misma arquitectura, mismos resultados esperados, misma compatibilidad
con el paso 10 — solo cambia el backend de cómputo neural.

Diferencias respecto al paper original
--------------------------------------
- Se elimina la cabeza de clasificación (siguiente parada es trivial en
  ruta fija).
- Se predice el horizonte COMPLETO (h = 1..n-l), no solo el siguiente
  paso, para paridad con los baselines del paso 10.
- Se usa attention con query por destino en lugar de cascada autoregresiva.

Diferencias técnicas respecto a la versión PyTorch
--------------------------------------------------
- Backend: TensorFlow 2.x + Keras 3 (en lugar de PyTorch).
- Datasets: `tf.data.Dataset` con `padded_batch` (en lugar de
  `torch.utils.data.DataLoader` con `collate_fn` manual).
- Training loop: `tf.GradientTape` con `@tf.function` para velocidad
  (en lugar de `loss.backward()`).
- PAD masking: `mask_zero=True` en `Embedding` (en lugar de `padding_idx`).
- Modelo guardado: `model.save_weights('m.weights.h5')` (en lugar de
  `torch.save(state_dict)`).

API pública
-----------
La función de más alto nivel es `build_lstm_prediction_pipeline`.

    from func.lstm_prediction_tf import build_lstm_prediction_pipeline

    result = build_lstm_prediction_pipeline(
        df_full     = df_full,
        direction   = 0,
        output_dir  = "data/02_interim/routes/101/lstm_tf/dir0/",
        n_folds     = 5,
        seed        = 42,
    )

Si necesitas control granular, los símbolos abajo están todos exportados.

Referencia
----------
Sun, J. & Kim, J. (2021). "Joint prediction of next location and travel
time from urban vehicle trajectories using long short-term memory neural
networks". Transportation Research Part C, 128.
"""
# ── Etapa A: dataset_builder ─────────────────────────────────────────
from .dataset_builder import build_sequence_dataset

# ── Etapa B: vocab ───────────────────────────────────────────────────
from .vocab import (
    StopVocab,
    PAD_TOKEN, UNK_TOKEN,
    PAD_IDX,   UNK_IDX,
)

# ── Etapa C: models ──────────────────────────────────────────────────
from .models import (
    BaseLSTMModel,
    SequentialLSTM,
    HybridLSTM,
    AdditiveAttention,
    init_weights,
)

# ── Etapa D: data_loader ─────────────────────────────────────────────
from .data_loader import (
    SequenceDataset,
    TrainingSample,
    make_tf_dataset,
    masked_mse_loss,
    masked_rmse,
    masked_mae,
)

# ── Etapa D: trainer ─────────────────────────────────────────────────
from .trainer import (
    train_model,
    evaluate_model,
    TrainingHistory,
)

# ── Etapa E: cross_validation ────────────────────────────────────────
from .cross_validation import (
    run_lstm_cross_validation,
    get_fold_assignments,
    summarize_rmse_by_location,
    summarize_error_by_horizon,
)

# ── Etapa F: plots ───────────────────────────────────────────────────
from .plots import (
    plot_training_curves,
    plot_time_error_distribution,
    plot_attention_heatmap,
    plot_attention_examples,
    plot_lstm_vs_baselines_rmse,
    plot_lstm_vs_baselines_summary_bars,
    build_lstm_static_report,
    build_lstm_html_dashboard,
)

# ── Etapa G: orquestador ─────────────────────────────────────────────
from .run_experiment import (
    build_lstm_prediction_pipeline,
    MODEL_REGISTRY,
)


__all__ = [
    # Etapa A
    "build_sequence_dataset",
    # Etapa B
    "StopVocab", "PAD_TOKEN", "UNK_TOKEN", "PAD_IDX", "UNK_IDX",
    # Etapa C
    "BaseLSTMModel", "SequentialLSTM", "HybridLSTM",
    "AdditiveAttention", "init_weights",
    # Etapa D
    "SequenceDataset", "TrainingSample",
    "make_tf_dataset",
    "masked_mse_loss", "masked_rmse", "masked_mae",
    "train_model", "evaluate_model", "TrainingHistory",
    # Etapa E
    "run_lstm_cross_validation", "get_fold_assignments",
    "summarize_rmse_by_location", "summarize_error_by_horizon",
    # Etapa F
    "plot_training_curves", "plot_time_error_distribution",
    "plot_attention_heatmap", "plot_attention_examples",
    "plot_lstm_vs_baselines_rmse", "plot_lstm_vs_baselines_summary_bars",
    "build_lstm_static_report", "build_lstm_html_dashboard",
    # Etapa G
    "build_lstm_prediction_pipeline", "MODEL_REGISTRY",
]


__version__ = "0.1.0"
