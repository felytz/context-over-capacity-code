"""
lstm_prediction_tf.run_experiment
===================================
Etapa G: orquestador único del pipeline LSTM (TensorFlow).

Función pública: build_lstm_prediction_pipeline()

Reproduce todo el experimento adaptado de Sun & Kim (2021) sobre tu df_full:
    1. Construye secuencias por viaje (Etapa A: dataset_builder)
    2. Construye vocabulario stop_code ↔ int (Etapa B: vocab)
    3. Instancia modelo Sequential/Hybrid (Etapa C: models)
    4. Corre K-fold CV (Etapa E: cross_validation)
    5. Genera reporte estático (PDF) y dashboard HTML (Etapa F: plots)
    6. Persiste df_errors, vocab, training history

Diferencias respecto a la versión PyTorch
-----------------------------------------
- No expone parámetro `device` (TF maneja automáticamente).
- Guarda pesos del modelo con `save_weights('model.weights.h5')` (Keras 3
  format) en lugar de `torch.save(state_dict)`.
- Imports relativos a `lstm_prediction_tf`.

Outputs físicos
---------------
output_dir/
├── df_errors.parquet         ← compatible con paso 10 (mismo schema)
├── vocab.json                ← reconstrucción del Embedding
├── training_history.json     ← curvas de loss/lr por fold
├── fold_assignments.json     ← trip_id → fold (paridad con paso 10)
├── metadata.json             ← config completa del experimento
├── dashboard.html            ← dashboard interactivo
└── static/
    └── report.pdf            ← reporte estático
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Type, Union

import numpy as np
import polars as pl
import tensorflow as tf

from .dataset_builder    import build_sequence_dataset
from .vocab              import StopVocab
from .data_loader        import SequenceDataset
from .models             import (
    BaseLSTMModel, SequentialLSTM, HybridLSTM, attention_chunks,
)
from .cross_validation   import (
    run_lstm_cross_validation,
    summarize_rmse_by_location,
)
from .plots              import (
    build_lstm_static_report,
    build_lstm_html_dashboard,
)
from .trainer            import segment_error_view, strip_segment_cols


# ──────────────────────────────────────────────────────────────────────
# Resolución del tipo de modelo
# ──────────────────────────────────────────────────────────────────────

MODEL_REGISTRY: Dict[str, Type[BaseLSTMModel]] = {
    "sequential":      SequentialLSTM,
    "sequential_lstm": SequentialLSTM,
    "hybrid":          HybridLSTM,
    "hybrid_lstm":     HybridLSTM,
}


def _resolve_model_class(model_type: str) -> Type[BaseLSTMModel]:
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"model_type {model_type!r} desconocido. "
            f"Opciones: {sorted(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[model_type]


# ──────────────────────────────────────────────────────────────────────
# Orquestador principal
# ──────────────────────────────────────────────────────────────────────

def build_lstm_prediction_pipeline(
    df_full:             pl.DataFrame,
    direction:           int   = 0,
    group_col:           str   = "direction",
    group_value:         Any   = None,
    model_type:          str   = "sequential",
    # Hiperparámetros del modelo
    embed_dim:           int   = 64,
    time_proj_dim:       int   = 64,
    hidden_dim:          int   = 128,
    n_lstm_layers:       int   = 1,
    dropout:             float = 0.1,
    attention_hidden_dim: int  = 128,
    # Hiperparámetros del entrenamiento
    n_epochs:            int   = 50,
    batch_size:          int   = 64,
    lr:                  float = 1e-3,
    weight_decay:        float = 0.0,
    patience:            int   = 5,
    grad_clip:           float = 1.0,
    n_folds:             int   = 5,
    # Hiperparámetros del dataset
    min_sequence_length: int   = 5,
    max_sequence_length: int   = 200,
    max_context_length:  Optional[int] = None,
    aggregation:         str   = "first",
    tokenize_by:         str   = "grid",
    with_calendar:       bool  = False,
    grid_step_m:         int   = 100,
    # Frame de evaluación único (`func/eval_frame.py`): impone la muestra, la
    # geometría de la grid y los folds de los baselines/XGB. Sin esto cada
    # familia define su propia población y la comparación exige el protocolo
    # matched a posteriori.
    trip_whitelist:      Optional[Iterable[Any]] = None,
    max_cells:           Optional[int] = None,
    fold_assignments:    Optional[Dict[Any, int]] = None,
    frame_hash:          Optional[str] = None,
    # Régimen de partición. `expanding` = el fold es el periodo del viaje y el
    # escalón k entrena solo con los anteriores; exige `es_frac > 0` para que el
    # early stopping no mire el mes que se va a predecir.
    split:               str   = "random",
    es_frac:             float = 0.0,
    min_l:               int   = 2,
    l_strategy:          str   = "all",
    val_l_strategy:      Optional[str] = None,
    val_l_indices:       Optional[Dict[Any, List[int]]] = None,
    # Otros
    output_dir:          Union[str, Path] = "./lstm_prediction_tf_output/",
    seed:                int   = 42,
    df_errors_baseline:  Optional[pl.DataFrame] = None,
    verbose:             bool  = True,
) -> Dict[str, Any]:
    """
    Orquestador completo. Adapta el Sequential LSTM de Sun & Kim 2021
    a tu df_full y produce todos los outputs para la tesis.

    Parameters
    ----------
    df_full : pl.DataFrame
        Salida del paso 7 enriquecido. Columnas requeridas:
        trip_id, direction, trip_valid, time, next_stop, link_travel_time.
    direction : int, default=0
    model_type : str, default="sequential"
        "sequential" (Sun & Kim §3.3) o "hybrid" (§3.2).

    embed_dim, time_proj_dim, hidden_dim, n_lstm_layers, dropout,
    attention_hidden_dim : hiperparámetros del modelo.

    n_epochs, batch_size, lr, weight_decay, patience, grad_clip :
        hiperparámetros del entrenamiento.

    n_folds : int, default=5
        Para paridad con paso 10, mantén 5.

    min_sequence_length, max_sequence_length, max_context_length,
    aggregation, l_strategy : ver `dataset_builder.build_sequence_dataset`
    y `data_loader.SequenceDataset`.

    val_l_strategy, val_l_indices : sobre-escriben la estrategia de val
    (paridad con paso 10).

    output_dir : str | Path
    seed : int, default=42
        Para reproducibilidad. Mismo seed = mismas particiones de CV.
    df_errors_baseline : pl.DataFrame | None
        Si se pasa (típicamente del paso 10), se concatena para
        generar las páginas de comparativo del reporte.
    verbose : bool

    Returns
    -------
    dict con:
      - "df_errors", "df_errors_combined", "vocab", "histories",
        "fold_assignments", "summary", "metadata", "paths"
    """
    # ── 0) Setup ─────────────────────────────────────────────────────
    t_start = time.time()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "static").mkdir(exist_ok=True)

    # Unidad de análisis: leg (terminal→terminal) o, por compat, direction 0/1.
    gval = group_value if group_value is not None else direction

    if verbose:
        print("\n" + "█" * 60)
        print(f"  LSTM PREDICTION PIPELINE (TF) — {group_col}={gval}, model={model_type}")
        print("█" * 60)

    # Reproducibilidad global
    tf.keras.utils.set_random_seed(seed)
    np.random.seed(seed)

    # ── A) Construir secuencias ──────────────────────────────────────
    if verbose:
        print(f"\n  [A] Construyendo secuencias por viaje …")

    seq_data = build_sequence_dataset(
        df_full             = df_full,
        direction           = direction,
        group_col           = group_col,
        group_value         = gval,
        min_sequence_length = min_sequence_length,
        max_sequence_length = max_sequence_length,
        max_context_length  = max_context_length,
        aggregation         = aggregation,
        context_features    = None,
        use_interp_time     = True,
        tokenize_by         = tokenize_by,
        with_calendar       = with_calendar,
        grid_step_m         = grid_step_m,
        trip_whitelist      = trip_whitelist,
        max_cells           = max_cells,
        verbose             = verbose,
    )

    if not seq_data["sequences"]:
        raise ValueError(
            f"No se generaron secuencias para direction={direction}."
        )

    # ── B) Vocabulario ───────────────────────────────────────────────
    if verbose:
        print(f"\n  [B] Construyendo vocabulario de paradas …")
    vocab = StopVocab.from_stop_counts(
        seq_data["stop_counts"], add_unk=True,
    )
    vocab_path = out_dir / "vocab.json"
    vocab.save(vocab_path)
    if verbose:
        print(f"      n_real_stops = {vocab.n_real_stops}, "
              f"n_total (con PAD/UNK) = {vocab.n_stops}")
        print(f"      ✔ Guardado: {vocab_path.name}")

    # ── C) Dataset ───────────────────────────────────────────────────
    if verbose:
        print(f"\n  [C] Construyendo SequenceDataset (l_strategy={l_strategy}) …")
    dataset = SequenceDataset(
        sequences          = seq_data["sequences"],
        vocab              = vocab,
        # min_l = 1 alinea los cortes con los de los baselines: el corte `c` de
        # este dataset observa hasta la distancia (c+1)·step, así que equivale a
        # `l_idx = c+1` de la matriz T. Con min_l=2 (default histórico) el primer
        # corte de los baselines (l=2) no existiría aquí y n_pairs no cuadraría.
        min_l              = min_l,
        max_context_length = max_context_length,
        l_strategy         = l_strategy,
    )
    if verbose:
        print(f"      n_trips={dataset.n_trips()}, "
              f"n_muestras={len(dataset):,}")

    # ── D) Cross-validation ──────────────────────────────────────────
    if verbose:
        print(f"\n  [D] Cross-validation ({n_folds}-fold) …")
    model_class = _resolve_model_class(model_type)
    model_kwargs = dict(
        n_stops              = vocab.n_stops,
        embed_dim            = embed_dim,
        time_proj_dim        = time_proj_dim,
        hidden_dim           = hidden_dim,
        n_lstm_layers        = n_lstm_layers,
        dropout              = dropout,
        attention_hidden_dim = attention_hidden_dim,
        pad_idx              = vocab.pad_idx,
        context_dim          = seq_data.get("context_dim", 0),
    )
    # Enmienda E3: la atención por trozos sólo donde (B, H, L, d_attn) no cabe en
    # int32 (con la vuelta, más de 512 celdas). En las demás rutas model_kwargs
    # queda como antes y el modelo es el de siempre.
    n_attn_chunks = attention_chunks(
        batch_size, max_cells or max_sequence_length, attention_hidden_dim,
    )
    if n_attn_chunks > 1:
        model_kwargs["attention_chunks"] = n_attn_chunks
        if verbose:
            print(f"      atención por trozos: {n_attn_chunks} "
                  f"(B·H·L·d_attn > 2^31 con {max_cells or max_sequence_length} "
                  f"celdas; enmienda E3)")

    cv_result = run_lstm_cross_validation(
        dataset        = dataset,
        model_class    = model_class,
        model_kwargs   = model_kwargs,
        n_folds        = n_folds,
        n_epochs       = n_epochs,
        batch_size     = batch_size,
        lr             = lr,
        weight_decay   = weight_decay,
        patience       = patience,
        grad_clip      = grad_clip,
        seed           = seed,
        model_name     = model_class.name,
        val_l_strategy = val_l_strategy,
        val_l_indices  = val_l_indices,
        # Folds terminados sobreviven a un crash (se reanudan del parquet);
        # se limpian solos al completar el CV.
        checkpoint_dir = out_dir,
        fold_assignments = fold_assignments,
        split          = split,
        es_frac        = es_frac,
        frame_hash     = frame_hash,
        verbose        = verbose,
    )

    df_errors        = cv_result["df_errors"]
    histories        = cv_result["histories"]
    fold_assignments = cv_result["fold_assignments"]
    summary          = cv_result["summary"]

    # ── E) Persistencia ──────────────────────────────────────────────
    parquet_path = out_dir / "df_errors.parquet"
    from _lib_data import save_df
    save_df(df_errors, parquet_path)
    if verbose:
        print(f"\n  [E] Persistencia …")
        print(f"      ✔ df_errors → {parquet_path.name}")

    history_path = out_dir / "training_history.json"
    history_payload = [h.to_dict() for h in histories]
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history_payload, f, ensure_ascii=False, indent=2)
    if verbose:
        print(f"      ✔ training_history → {history_path.name}")

    folds_path = out_dir / "fold_assignments.json"
    with folds_path.open("w", encoding="utf-8") as f:
        json.dump(
            {str(tid): int(f) for tid, f in fold_assignments.items()},
            f, ensure_ascii=False, indent=2,
        )
    if verbose:
        print(f"      ✔ fold_assignments → {folds_path.name}")

    # ── F) df_errors_combined (si se pasó baseline) ──────────────────
    # El combinado usa solo el schema ACUMULADO canónico (idéntico al
    # paso 10). Las columnas por-tramo se quitan aquí; viven en df_errors
    # y alimentan el reporte de segmento más abajo.
    df_errors_canonical = strip_segment_cols(df_errors)
    df_errors_combined: Optional[pl.DataFrame] = None
    if df_errors_baseline is not None:
        if set(df_errors_baseline.columns) != set(df_errors_canonical.columns):
            if verbose:
                missing = set(df_errors_canonical.columns) - set(df_errors_baseline.columns)
                extra   = set(df_errors_baseline.columns) - set(df_errors_canonical.columns)
                print(f"\n  ⚠ Schema de df_errors_baseline difiere:")
                print(f"     falta en baseline : {missing}")
                print(f"     extra en baseline : {extra}")
        df_errors_combined = pl.concat(
            [df_errors_canonical, df_errors_baseline], how="diagonal_relaxed",
        )
        if verbose:
            n_algos = df_errors_combined["algorithm"].n_unique()
            print(f"      ✔ df_errors_combined: {n_algos} algoritmos, "
                  f"{df_errors_combined.height:,} filas")

    # ── G) Metadata completo del experimento ─────────────────────────
    metadata: Dict[str, Any] = {
        "framework":           "tensorflow",
        "tf_version":          tf.__version__,
        "direction":           direction,
        "group_col":           group_col,
        "group_value":         gval,
        "model_type":          model_type,
        "model_class":         model_class.__name__,
        "n_stops":             int(vocab.n_stops),
        "n_real_stops":        int(vocab.n_real_stops),
        "n_trips":             int(dataset.n_trips()),
        "n_samples":           int(len(dataset)),
        "n_folds":             int(n_folds),
        "embed_dim":           int(embed_dim),
        "hidden_dim":          int(hidden_dim),
        "n_lstm_layers":       int(n_lstm_layers),
        "dropout":             float(dropout),
        "batch_size":          int(batch_size),
        "attention_chunks":    int(n_attn_chunks),
        "lr":                  float(lr),
        "n_epochs":            int(n_epochs),
        "patience":            int(patience),
        "grad_clip":           float(grad_clip) if grad_clip is not None else None,
        "max_context_length":  max_context_length,
        "l_strategy":          l_strategy,
        "aggregation":         aggregation,
        "tokenize_by":         tokenize_by,
        "with_calendar":       with_calendar,
        "context_dim":         seq_data.get("context_dim", 0),
        "grid_step_m":         int(grid_step_m),
        "seed":                int(seed),
        "summary":             summary,
        "elapsed_seconds":     None,
    }

    # ── H) Reportes (PDF + HTML) ─────────────────────────────────────
    # Se generan DOS juegos desde el MISMO entrenamiento:
    #   • acumulado/  → métrica de tiempo de llegada, comparable con baselines
    #   • segmento/   → error por-tramo (link), métrica nativa del LSTM, sin baselines
    if verbose:
        print(f"\n  [H] Generando reportes (acumulado + por-tramo) …")

    cum_dir = out_dir / "acumulado"
    seg_dir = out_dir / "segmento"
    (cum_dir / "static").mkdir(parents=True, exist_ok=True)
    (seg_dir / "static").mkdir(parents=True, exist_ok=True)

    # --- H.1) Reporte ACUMULADO (vs baselines) ---
    pdf_path = cum_dir / "static" / "report.pdf"
    build_lstm_static_report(
        df_errors          = df_errors_canonical,
        histories          = histories,
        output_path        = pdf_path,
        metadata           = {**metadata, "metric": "acumulado (tiempo de llegada)"},
        df_errors_combined = df_errors_combined,
        verbose            = verbose,
    )
    html_path = cum_dir / "dashboard.html"
    build_lstm_html_dashboard(
        df_errors          = df_errors_canonical,
        histories          = histories,
        output_path        = html_path,
        metadata           = {
            **metadata,
            "title":    f"LSTM (TF) — {group_col} {gval} · ACUMULADO",
            "subtitle": f"{model_class.name} · {n_folds}-fold CV · vs baselines",
        },
        df_errors_combined = df_errors_combined,
        verbose            = verbose,
    )

    # --- H.2) Reporte POR-TRAMO (LSTM solo; baselines no tienen salida por-link) ---
    df_errors_seg = segment_error_view(df_errors)
    pdf_path_seg = seg_dir / "static" / "report.pdf"
    build_lstm_static_report(
        df_errors          = df_errors_seg,
        histories          = histories,
        output_path        = pdf_path_seg,
        metadata           = {**metadata, "metric": "por-tramo (link)"},
        df_errors_combined = None,
        verbose            = verbose,
    )
    html_path_seg = seg_dir / "dashboard.html"
    build_lstm_html_dashboard(
        df_errors          = df_errors_seg,
        histories          = histories,
        output_path        = html_path_seg,
        metadata           = {
            **metadata,
            "title":    f"LSTM (TF) — {group_col} {gval} · POR-TRAMO",
            "subtitle": f"{model_class.name} · {n_folds}-fold CV · error por link",
        },
        df_errors_combined = None,
        verbose            = verbose,
    )

    # ── I) Persistir metadata final ──────────────────────────────────
    elapsed = time.time() - t_start
    metadata["elapsed_seconds"] = float(elapsed)
    metadata_path = out_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2, default=str)

    # ── J) Resumen final por consola ─────────────────────────────────
    paths: Dict[str, Path] = {
        "parquet":          parquet_path,
        "vocab":            vocab_path,
        "training_history": history_path,
        "fold_assignments": folds_path,
        "metadata":         metadata_path,
        # Claves "legacy" → apuntan al reporte ACUMULADO (comparación principal)
        "pdf_report":       pdf_path,
        "dashboard":        html_path,
        # Reporte acumulado (explícito)
        "pdf_report_acumulado": pdf_path,
        "dashboard_acumulado":  html_path,
        # Reporte por-tramo
        "pdf_report_segmento":  pdf_path_seg,
        "dashboard_segmento":   html_path_seg,
    }

    if verbose:
        print("\n" + "─" * 60)
        print(f"  Resumen final ({model_class.name}, {group_col}={gval})")
        print("─" * 60)
        print(f"    RMSE  : {summary['rmse_s_mean']:.2f} ± "
              f"{summary['rmse_s_std']:.2f} s")
        print(f"    MAE   : {summary['mae_s_mean']:.2f} ± "
              f"{summary['mae_s_std']:.2f} s")
        print(f"    MAPE  : {summary['mape_mean']:.2f} ± "
              f"{summary['mape_std']:.2f} %")
        print(f"    Tiempo total : {elapsed:.1f}s")
        if df_errors_combined is not None:
            print(f"\n  Comparativo con baselines:")
            cmp = (
                df_errors_combined
                .group_by("algorithm")
                .agg(pl.col("error_sq").mean().sqrt().alias("rmse"))
                .sort("rmse")
            )
            for r in cmp.iter_rows(named=True):
                marker = "★" if r["algorithm"].endswith("_lstm") else " "
                print(f"    {marker} {r['algorithm']:22s} → RMSE = {r['rmse']:6.2f} s")
        print(f"\n  ✅ Todo en: {out_dir}")
        print("█" * 60 + "\n")

    return {
        "df_errors":          df_errors,
        "df_errors_combined": df_errors_combined,
        "vocab":              vocab,
        "histories":          histories,
        "fold_assignments":   fold_assignments,
        "summary":            summary,
        "metadata":           metadata,
        "paths":              paths,
    }
