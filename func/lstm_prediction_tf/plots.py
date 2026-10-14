"""
lstm_prediction.plots
=======================
Etapa F: visualización específica del LSTM.

Las figuras de RMSE por posición y boxplots por horizonte ya existen en
`arrival_prediction.plots` y funcionan sobre el df_errors común (gracias
al schema compartido). Aquí solo agregamos lo específico del LSTM:

  - Curvas de entrenamiento (train/val loss + lr)
  - Distribución de error de tiempo (histograma + KDE)
  - Heatmaps de atención (interpretabilidad)
  - Comparativo LSTM vs. baselines (lado a lado)
  - Reporte estático (PDF) y dashboard HTML interactivo

Convención de colores
---------------------
- LSTM en azul oscuro distintivo (#1f4e79)
- Baselines reusan los colores de arrival_prediction.plots para
  que las figuras combinadas sean visualmente coherentes.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import polars as pl

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from .cross_validation import (
    summarize_rmse_by_location,
    summarize_error_by_horizon,
)
from .trainer import TrainingHistory


# Estilo compartido con arrival_prediction (paleta colorblind-safe + labels +
# markers + rcParams). LSTM_COLOR sigue disponible (lo usan las curvas de train).
from _lib_style import (
    ALGO_COLORS, ALGO_LABELS, ALGO_MARKERS, ALGO_LINESTYLES, LSTM_COLOR, apply_style,
)

apply_style()


# ──────────────────────────────────────────────────────────────────────
# 1) Curvas de entrenamiento
# ──────────────────────────────────────────────────────────────────────

def plot_training_curves(
    history:     Union[TrainingHistory, Dict[str, Any], List[TrainingHistory]],
    output_path: Optional[Union[str, Path]] = None,
    title:       str = "Curvas de entrenamiento",
) -> plt.Figure:
    """
    Grafica train/val loss + lr a lo largo del entrenamiento.

    Si `history` es una lista (varios folds), grafica todas con la mediana
    en negro grueso y las trazas individuales semitransparentes.

    Returns
    -------
    matplotlib.Figure
    """
    histories: List[TrainingHistory] = (
        history if isinstance(history, list) else [history]
    )

    # Convertir dicts a algo accesible uniformemente
    def _get(h, attr):
        if isinstance(h, dict):
            return h.get(attr, [])
        return getattr(h, attr, [])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # ── Panel izquierdo: train/val loss ─────────────────────────────
    train_lists = [_get(h, "train_loss") for h in histories]
    val_lists   = [_get(h, "val_loss")   for h in histories]

    n_folds = len(histories)
    for fold_i, (tr, va) in enumerate(zip(train_lists, val_lists)):
        epochs = np.arange(1, len(tr) + 1)
        alpha = 0.4 if n_folds > 1 else 1.0
        ax1.plot(epochs, tr, "--", color="#888", alpha=alpha,
                 label="train" if fold_i == 0 else None)
        ax1.plot(epochs, va, "-",  color=LSTM_COLOR, alpha=alpha,
                 label="val"   if fold_i == 0 else None)

    # Marca el mejor epoch (si solo hay un fold)
    if n_folds == 1:
        best_epoch = _get(histories[0], "best_epoch")
        if isinstance(best_epoch, int) and best_epoch > 0:
            ax1.axvline(best_epoch, color="red", linestyle=":", alpha=0.6,
                        label=f"best epoch ({best_epoch})")

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (MSE de tiempo, s²)")
    ax1.set_title("Loss por epoch")
    ax1.legend(loc="upper right")
    ax1.set_yscale("log")

    # ── Panel derecho: learning rate ────────────────────────────────
    for fold_i, h in enumerate(histories):
        lrs = _get(h, "lr")
        if not lrs:
            continue
        epochs = np.arange(1, len(lrs) + 1)
        alpha = 0.4 if n_folds > 1 else 1.0
        ax2.plot(epochs, lrs, "-", color="#444", alpha=alpha,
                 label=f"fold {fold_i}" if n_folds > 1 else "lr")

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Learning rate")
    ax2.set_yscale("log")
    ax2.set_title("Learning rate schedule (ReduceLROnPlateau)")
    if n_folds <= 5:
        ax2.legend(loc="best", fontsize=8)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path)
    return fig


# ──────────────────────────────────────────────────────────────────────
# 2) Distribución del error de tiempo
# ──────────────────────────────────────────────────────────────────────

def plot_time_error_distribution(
    df_errors:   pl.DataFrame,
    output_path: Optional[Union[str, Path]] = None,
    bins:        int = 60,
    clip_seconds: float = 300.0,
    title:       str = "Distribución del error de predicción de tiempo",
) -> plt.Figure:
    """
    Histograma del error firmado (t_pred - t_true) con líneas verticales
    en mediana y P95.

    Si df_errors contiene múltiples algoritmos, muestra uno por color.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Guard: DF vacío → figura "sin datos" en lugar de warnings de matplotlib
    if df_errors.height == 0:
        for ax in (ax1, ax2):
            ax.text(0.5, 0.5, "(sin datos)", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12, color="#999")
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
        fig.tight_layout()
        if output_path is not None:
            fig.savefig(output_path)
        return fig

    # Error firmado (sesgo) y absoluto
    df = df_errors.with_columns(
        (pl.col("t_pred") - pl.col("t_true")).alias("error_signed"),
    )

    algos = df["algorithm"].unique().sort().to_list()

    # ── Panel izquierdo: histograma del error firmado ───────────────
    bin_edges = np.linspace(-clip_seconds, clip_seconds, bins + 1)
    for algo in algos:
        d = df.filter(pl.col("algorithm") == algo)
        errs = d["error_signed"].to_numpy()
        errs = np.clip(errs, -clip_seconds, clip_seconds)
        color = ALGO_COLORS.get(algo, "#666")
        label = ALGO_LABELS.get(algo, algo)
        ax1.hist(errs, bins=bin_edges, color=color, alpha=0.5,
                 label=label, density=True)

    ax1.axvline(0, color="black", linewidth=1.0)
    ax1.set_xlabel("Error firmado, t̂ − t (s)")
    ax1.set_ylabel("Densidad")
    ax1.set_title(f"Sesgo (clip ±{clip_seconds:.0f}s)")
    ax1.legend(loc="upper right", fontsize=8)

    # ── Panel derecho: CDF del error absoluto ───────────────────────
    for algo in algos:
        d = df.filter(pl.col("algorithm") == algo)
        abs_err = np.sort(d["error_abs"].to_numpy())
        cdf = np.arange(1, len(abs_err) + 1) / len(abs_err)
        color = ALGO_COLORS.get(algo, "#666")
        label = ALGO_LABELS.get(algo, algo)

        # Marcas de mediana y P95
        med = float(np.median(abs_err))
        p95 = float(np.quantile(abs_err, 0.95))

        ax2.plot(abs_err, cdf, "-", color=color, label=label, linewidth=1.5)
        ax2.axvline(p95, color=color, linestyle=":", alpha=0.4)

    ax2.set_xlabel("Error absoluto |t̂ − t| (s)")
    ax2.set_ylabel("CDF")
    ax2.set_title("CDF del error absoluto (líneas punteadas = P95)")
    ax2.set_xlim(0, clip_seconds)
    ax2.legend(loc="lower right", fontsize=8)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path)
    return fig


# ──────────────────────────────────────────────────────────────────────
# 3) Heatmap de atención (interpretabilidad)
# ──────────────────────────────────────────────────────────────────────

def plot_attention_heatmap(
    attention_weights: np.ndarray,        # (H, L) ya normalizado por fila
    stops_obs:         List[str],         # (L,)
    stops_horizon:     List[str],         # (H,)
    output_path:       Optional[Union[str, Path]] = None,
    title:             str = "Atención del modelo",
    max_obs_to_show:   int = 30,
    max_h_to_show:     int = 20,
) -> plt.Figure:
    """
    Heatmap de los pesos de atención. Filas = pasos del horizonte
    (paradas futuras), columnas = paradas observadas.

    Sirve para mostrar QUÉ paradas pasadas atendió el modelo al predecir
    cada parada futura. Es la figura "interpretable" central de la tesis.

    Parameters
    ----------
    attention_weights : np.ndarray (H, L)
        Pesos de atención post-softmax (suma 1 por fila).
    stops_obs, stops_horizon : list[str]
        Códigos de parada para los ejes.
    max_obs_to_show, max_h_to_show : int
        Si las secuencias son muy largas se trunca para que el heatmap
        sea legible (mostramos el inicio para obs, el final para horizonte).
    """
    A = np.asarray(attention_weights)
    H, L = A.shape
    assert len(stops_obs)     == L, f"len(stops_obs)={len(stops_obs)} vs L={L}"
    assert len(stops_horizon) == H, f"len(stops_horizon)={len(stops_horizon)} vs H={H}"

    # Trunca para legibilidad
    L_show = min(L, max_obs_to_show)
    H_show = min(H, max_h_to_show)
    A_show          = A[:H_show, -L_show:]                 # últimos L observados
    obs_show        = list(stops_obs[-L_show:])
    horizon_show    = list(stops_horizon[:H_show])

    fig, ax = plt.subplots(figsize=(max(8, L_show * 0.4),
                                    max(4, H_show * 0.35)))
    im = ax.imshow(A_show, aspect="auto", cmap="viridis",
                   interpolation="nearest")

    ax.set_xticks(np.arange(L_show))
    ax.set_xticklabels(obs_show, rotation=90, fontsize=7)
    ax.set_yticks(np.arange(H_show))
    ax.set_yticklabels(horizon_show, fontsize=7)
    ax.set_xlabel("Parada observada (contexto)")
    ax.set_ylabel("Parada futura (target)")

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Peso de atención (softmax)")
    ax.set_title(title)

    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path)
    return fig


def plot_attention_examples(
    model,
    dataset,                              # SequenceDataset (TF)
    n_examples:    int = 3,
    sample_indices: Optional[List[int]] = None,
    output_path:   Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Genera múltiples heatmaps de atención sobre muestras seleccionadas.

    Si sample_indices es None, toma muestras espaciadas uniformemente
    del dataset.

    Útil para incluir en el reporte estático: 3-5 ejemplos representativos
    de cómo el modelo presta atención a la historia observada.

    Versión TF: no requiere device handling — TF maneja CPU/GPU automáticamente.
    Llamamos al modelo con training=False (equivalente a model.eval()) y
    sin GradientTape (equivalente a torch.no_grad()).
    """
    import tensorflow as tf

    if sample_indices is None:
        if len(dataset) == 0:
            raise ValueError("Dataset vacío, no se pueden generar ejemplos")
        sample_indices = np.linspace(
            0, len(dataset) - 1, min(n_examples, len(dataset)), dtype=int
        ).tolist()

    samples = [dataset[i] for i in sample_indices]
    n = len(samples)

    fig, axes = plt.subplots(n, 1, figsize=(12, 4 * n), squeeze=False)
    vocab = dataset.vocab

    for ax_i, s in enumerate(samples):
        # Construir un batch de tamaño 1 manualmente
        L = len(s.stops_obs)
        H = len(s.stops_horizon)
        stops_obs     = tf.constant(s.stops_obs.reshape(1, -1), dtype=tf.int64)
        times_obs     = tf.constant(s.times_obs.reshape(1, -1), dtype=tf.float32)
        stops_horizon = tf.constant(s.stops_horizon.reshape(1, -1), dtype=tf.int64)
        obs_mask      = tf.ones((1, L), dtype=tf.bool)
        horizon_mask  = tf.ones((1, H), dtype=tf.bool)

        # Forward con return_attention=True
        pred, attn = model(
            (stops_obs, times_obs, stops_horizon, obs_mask, horizon_mask),
            training=False,
            return_attention=True,
        )
        # attn: (1, H, L)
        A = attn.numpy()[0]
        H_act, L_act = A.shape

        obs_codes = vocab.decode_sequence(s.stops_obs.tolist())
        hor_codes = vocab.decode_sequence(s.stops_horizon.tolist())

        # Trunca para legibilidad
        L_show = min(L_act, 30)
        H_show = min(H_act, 15)
        A_show = A[:H_show, -L_show:]

        ax = axes[ax_i, 0]
        im = ax.imshow(A_show, aspect="auto", cmap="viridis")
        ax.set_xticks(np.arange(L_show))
        ax.set_xticklabels(obs_codes[-L_show:], rotation=90, fontsize=6)
        ax.set_yticks(np.arange(H_show))
        ax.set_yticklabels(hor_codes[:H_show], fontsize=6)
        ax.set_title(
            f"trip={s.trip_id}  l_idx={s.l_idx}  "
            f"L={L_act}, H={H_act}",
            fontsize=9,
        )
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)

    fig.suptitle("Pesos de atención (muestras de validación)",
                 fontsize=13, fontweight="bold", y=1.0)
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path)
    return fig
    return fig


# ──────────────────────────────────────────────────────────────────────
# 4) Comparativo LSTM vs baselines
# ──────────────────────────────────────────────────────────────────────

def plot_lstm_vs_baselines_rmse(
    df_errors_combined: pl.DataFrame,
    output_path:        Optional[Union[str, Path]] = None,
    aggregation:        str = "by_location",
    title:              str = "RMSE: LSTM vs. baselines",
) -> plt.Figure:
    """
    Compara RMSE por posición (o por horizonte) entre LSTM y baselines.

    df_errors_combined debe contener filas de ambos (concatenadas con
    pl.concat), distinguibles por la columna `algorithm`.

    Parameters
    ----------
    df_errors_combined : pl.DataFrame
        Schema común del paso 10 + paso 11.
    aggregation : "by_location" | "by_horizon"
        Eje X del comparativo.
    """
    if aggregation == "by_location":
        summary = summarize_rmse_by_location(df_errors_combined)
        x_col, y_col = "l_idx", "rmse_mean"
        std_col      = "rmse_std"
        xlabel       = "Posición de corte l_idx (parada)"
    elif aggregation == "by_horizon":
        summary = summarize_error_by_horizon(df_errors_combined)
        x_col, y_col = "horizon_bin_min", "rmse"
        std_col      = None
        xlabel       = "Horizonte (minutos)"
    else:
        raise ValueError(f"aggregation desconocida: {aggregation!r}")

    algos = summary["algorithm"].unique().sort().to_list()

    fig, ax = plt.subplots(figsize=(11, 5))
    for algo in algos:
        d = summary.filter(pl.col("algorithm") == algo).sort(x_col)
        x = d[x_col].to_numpy()
        y = d[y_col].to_numpy()
        color = ALGO_COLORS.get(algo, "#666")
        label = ALGO_LABELS.get(algo, algo)
        is_lstm = algo.endswith("_lstm")
        lw = 2.5 if is_lstm else 1.5
        ax.plot(x, y, "-o", color=color, label=label,
                linewidth=lw, markersize=4)
        if std_col and std_col in d.columns:
            std = d[std_col].fill_null(0).to_numpy()
            ax.fill_between(x, y - std, y + std, color=color, alpha=0.12)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("RMSE (s)")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)

    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path)
    return fig


def plot_lstm_vs_baselines_summary_bars(
    df_errors_combined: pl.DataFrame,
    output_path:        Optional[Union[str, Path]] = None,
    title:              str = "Métricas agregadas por algoritmo",
) -> plt.Figure:
    """
    Barras agrupadas con RMSE, MAE y MAPE promedio por algoritmo.
    Resumen ejecutivo de la comparación.
    """
    metrics = (
        df_errors_combined
        .group_by(["algorithm", "fold"])
        .agg(
            pl.col("error_sq").mean().sqrt().alias("rmse"),
            pl.col("error_abs").mean().alias("mae"),
            (pl.col("error_abs") / pl.col("t_true").clip(1.0, None))
                .mean().alias("mape_frac"),
        )
        .group_by("algorithm")
        .agg(
            pl.col("rmse").mean().alias("rmse_mean"),
            pl.col("rmse").std(ddof=1).alias("rmse_std"),
            pl.col("mae").mean().alias("mae_mean"),
            pl.col("mae").std(ddof=1).alias("mae_std"),
            (pl.col("mape_frac").mean() * 100).alias("mape_mean"),
            (pl.col("mape_frac").std(ddof=1) * 100).alias("mape_std"),
        )
        .sort("rmse_mean")
    )

    algos = metrics["algorithm"].to_list()
    colors = [ALGO_COLORS.get(a, "#666") for a in algos]
    labels = [ALGO_LABELS.get(a, a) for a in algos]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    metric_specs = [
        ("rmse_mean", "rmse_std", "RMSE (s)",  axes[0]),
        ("mae_mean",  "mae_std",  "MAE (s)",   axes[1]),
        ("mape_mean", "mape_std", "MAPE (%)",  axes[2]),
    ]
    for value_col, err_col, ylabel, ax in metric_specs:
        vals = metrics[value_col].to_numpy()
        errs = metrics[err_col].fill_null(0).to_numpy()
        positions = np.arange(len(algos))
        bars = ax.bar(positions, vals, yerr=errs, capsize=5,
                      color=colors, alpha=0.85, edgecolor="white")
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        # Anotar valores encima de cada barra
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path)
    return fig


# ──────────────────────────────────────────────────────────────────────
# 5) Reporte estático (PDF agregador)
# ──────────────────────────────────────────────────────────────────────

def build_lstm_static_report(
    df_errors:        pl.DataFrame,
    histories:        List[TrainingHistory],
    output_path:      Union[str, Path],
    metadata:         Optional[Dict[str, Any]] = None,
    df_errors_combined: Optional[pl.DataFrame] = None,
    verbose:          bool = True,
) -> Path:
    """
    Genera un PDF con todas las figuras relevantes del experimento LSTM.

    Si se pasa `df_errors_combined` (LSTM + baselines), agrega las páginas
    de comparativo.

    Parameters
    ----------
    df_errors : pl.DataFrame
        Resultados del CV solo del LSTM (para curvas, error dist).
    histories : list[TrainingHistory]
        Historiales de los folds.
    output_path : str | Path
        Ruta del PDF a generar.
    metadata : dict | None
        Información sobre el experimento (n_trips, hyperparams, etc.)
        para la portada.
    df_errors_combined : pl.DataFrame | None
        Si se pasa, agrega comparativo LSTM vs baselines al final.

    Returns
    -------
    Path del PDF generado.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"  Generando reporte estático: {output_path.name}")

    with PdfPages(output_path) as pdf:
        # Página 1: portada con resumen textual
        fig_cover = _make_cover_page(df_errors, metadata, histories)
        pdf.savefig(fig_cover); plt.close(fig_cover)

        # Página 2: curvas de entrenamiento
        fig_train = plot_training_curves(histories)
        pdf.savefig(fig_train); plt.close(fig_train)

        # Página 3: distribución de error
        fig_err = plot_time_error_distribution(df_errors)
        pdf.savefig(fig_err); plt.close(fig_err)

        # Páginas adicionales: comparativo
        if df_errors_combined is not None:
            fig_cmp_loc = plot_lstm_vs_baselines_rmse(
                df_errors_combined, aggregation="by_location",
                title="RMSE por posición — LSTM vs baselines",
            )
            pdf.savefig(fig_cmp_loc); plt.close(fig_cmp_loc)

            fig_cmp_hor = plot_lstm_vs_baselines_rmse(
                df_errors_combined, aggregation="by_horizon",
                title="RMSE por horizonte — LSTM vs baselines",
            )
            pdf.savefig(fig_cmp_hor); plt.close(fig_cmp_hor)

            fig_bars = plot_lstm_vs_baselines_summary_bars(df_errors_combined)
            pdf.savefig(fig_bars); plt.close(fig_bars)

    if verbose:
        print(f"  ↳ Reporte completo: {output_path}")
    return output_path


def _make_cover_page(
    df_errors: pl.DataFrame,
    metadata:  Optional[Dict[str, Any]],
    histories: List[TrainingHistory],
) -> plt.Figure:
    """Genera una portada con resumen textual del experimento."""
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle("Resultados experimento LSTM",
                 fontsize=18, fontweight="bold", y=0.96)

    rmse = float(np.sqrt(df_errors["error_sq"].mean())) if df_errors.height else float("nan")
    mae  = float(df_errors["error_abs"].mean())          if df_errors.height else float("nan")

    lines = []
    if metadata:
        lines.append(f"  Modelo               : {metadata.get('model_type', 'sequential_lstm')}")
        lines.append(f"  Dirección            : {metadata.get('direction', '—')}")
        lines.append(f"  Viajes (n_trips)     : {metadata.get('n_trips', '—'):,}"
                     if isinstance(metadata.get('n_trips'), int) else
                     f"  Viajes (n_trips)     : {metadata.get('n_trips', '—')}")
        lines.append(f"  Paradas (n_stops)    : {metadata.get('n_stops', '—')}")
        lines.append(f"  N folds              : {metadata.get('n_folds', '—')}")
        lines.append(f"  embed_dim/hidden_dim : "
                     f"{metadata.get('embed_dim', '—')}/{metadata.get('hidden_dim', '—')}")
        lines.append(f"  Batch size           : {metadata.get('batch_size', '—')}")
        lines.append("")

    lines.append(f"  RMSE  (promedio)     : {rmse:8.2f} s")
    lines.append(f"  MAE   (promedio)     : {mae:8.2f} s")
    lines.append(f"  Predicciones totales : {df_errors.height:,}")

    if histories:
        n_epochs_avg = np.mean([len(getattr(h, 'train_loss', h.get('train_loss', [])) if isinstance(h, dict) else h.train_loss) for h in histories])
        lines.append(f"  Epochs promedio/fold : {n_epochs_avg:.1f}")

    fig.text(0.1, 0.7, "\n".join(lines), fontfamily="monospace",
             fontsize=11, verticalalignment="top")
    fig.text(0.1, 0.08,
             "Pipeline: lstm_prediction (adaptación de Sun & Kim 2021)",
             fontsize=8, style="italic", color="#666")

    return fig


# ──────────────────────────────────────────────────────────────────────
# 6) Dashboard HTML
# ──────────────────────────────────────────────────────────────────────

def build_lstm_html_dashboard(
    df_errors:          pl.DataFrame,
    histories:          List[TrainingHistory],
    output_path:        Union[str, Path],
    metadata:           Optional[Dict[str, Any]] = None,
    df_errors_combined: Optional[pl.DataFrame] = None,
    verbose:            bool = True,
) -> Path:
    """
    Genera un dashboard HTML interactivo con Chart.js (sin dependencias
    locales — usa CDN).

    Las secciones:
      1. Resumen ejecutivo (métricas)
      2. Curvas de entrenamiento
      3. CDF del error absoluto
      4. (opcional) comparativo LSTM vs baselines por posición
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Datos para el dashboard
    data = _build_dashboard_data(df_errors, histories, df_errors_combined)
    cards = _build_stats_cards(df_errors, metadata)

    html = _render_dashboard_html(cards, data, metadata or {})
    output_path.write_text(html, encoding="utf-8")

    if verbose:
        print(f"  Dashboard HTML: {output_path}")
    return output_path


def _build_dashboard_data(
    df_errors:          pl.DataFrame,
    histories:          List[TrainingHistory],
    df_errors_combined: Optional[pl.DataFrame],
) -> Dict[str, Any]:
    """Construye los payloads JSON para Chart.js."""
    data: Dict[str, Any] = {}

    # Training curves: promedio sobre folds
    if histories:
        all_train = []
        all_val   = []
        for h in histories:
            tr = h.train_loss if isinstance(h, TrainingHistory) else h.get("train_loss", [])
            va = h.val_loss   if isinstance(h, TrainingHistory) else h.get("val_loss", [])
            all_train.append(tr)
            all_val.append(va)
        max_len = max(len(x) for x in all_train) if all_train else 0
        # Promediar epoch por epoch (con padding hacia adelante)
        def avg_at(lists, i):
            vals = [l[i] for l in lists if len(l) > i]
            return float(np.mean(vals)) if vals else None
        data["training_curves"] = {
            "epochs":     list(range(1, max_len + 1)),
            "train_mean": [avg_at(all_train, i) for i in range(max_len)],
            "val_mean":   [avg_at(all_val, i)   for i in range(max_len)],
        }

    # CDF del error absoluto
    if df_errors.height > 0:
        abs_err = np.sort(df_errors["error_abs"].to_numpy())
        n = len(abs_err)
        # Submuestrear a 200 puntos para no inflar el HTML
        sample_idx = np.linspace(0, n - 1, 200, dtype=int)
        data["error_cdf"] = {
            "x": abs_err[sample_idx].tolist(),
            "y": ((sample_idx + 1) / n).tolist(),
        }

    # Comparativo
    if df_errors_combined is not None:
        summary = summarize_rmse_by_location(df_errors_combined)
        algos = summary["algorithm"].unique().sort().to_list()
        data["comparison_by_location"] = {
            "algorithms": [],
        }
        for algo in algos:
            d = summary.filter(pl.col("algorithm") == algo).sort("l_idx")
            data["comparison_by_location"]["algorithms"].append({
                "name":  algo,
                "label": ALGO_LABELS.get(algo, algo),
                "color": ALGO_COLORS.get(algo, "#666"),
                "x":     d["l_idx"].to_list(),
                "y":     d["rmse_mean"].to_list(),
            })

    return data


def _build_stats_cards(
    df_errors: pl.DataFrame,
    metadata:  Optional[Dict[str, Any]],
) -> List[Dict[str, str]]:
    """Tarjetas de estadísticas para el header del dashboard."""
    cards: List[Dict[str, str]] = []
    if df_errors.height == 0:
        return cards

    rmse = float(np.sqrt(df_errors["error_sq"].mean()))
    mae  = float(df_errors["error_abs"].mean())
    n_pred = df_errors.height

    cards.append({"label": "RMSE",          "value": f"{rmse:.1f} s"})
    cards.append({"label": "MAE",           "value": f"{mae:.1f} s"})
    cards.append({"label": "N predicciones","value": f"{n_pred:,}"})

    if metadata:
        if "n_trips" in metadata:
            cards.append({"label": "N viajes",
                          "value": f"{metadata['n_trips']:,}"})
        if "n_stops" in metadata:
            cards.append({"label": "N paradas",
                          "value": str(metadata["n_stops"])})

    return cards


def _render_dashboard_html(
    cards:    List[Dict[str, str]],
    data:     Dict[str, Any],
    metadata: Dict[str, Any],
) -> str:
    """Devuelve el string HTML completo."""
    title    = metadata.get("title", "LSTM Prediction Dashboard")
    subtitle = metadata.get("subtitle", "Sequential LSTM — Ruta 101")

    # Tarjetas
    cards_html = "".join(
        f'<div class="card"><div class="card-label">{c["label"]}</div>'
        f'<div class="card-value">{c["value"]}</div></div>'
        for c in cards
    )

    data_json = json.dumps(data, ensure_ascii=False, default=float)

    # JavaScript se mantiene fuera de las f-strings con un placeholder ASCII.
    template = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8"/>
<title>__TITLE__</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0"></script>
<style>
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    margin: 0; padding: 24px; background: #f7f9fc; color: #222;
  }
  h1 { margin: 0; font-size: 24px; }
  h2 { font-size: 16px; color: #444; margin: 24px 0 12px; }
  .subtitle { color: #666; font-size: 13px; margin-top: 4px; }
  .cards { display: flex; gap: 12px; margin: 20px 0; flex-wrap: wrap; }
  .card {
    background: white; padding: 16px 20px; border-radius: 8px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06); min-width: 130px;
  }
  .card-label { font-size: 11px; color: #888; text-transform: uppercase; }
  .card-value { font-size: 22px; font-weight: 600; margin-top: 4px; color: #1f4e79; }
  .chart-container {
    background: white; padding: 20px; border-radius: 8px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06); margin-bottom: 20px;
    height: 360px;
  }
  footer { color: #999; font-size: 11px; margin-top: 32px; }
</style>
</head>
<body>
<h1>__TITLE__</h1>
<div class="subtitle">__SUBTITLE__</div>
<div class="cards">__CARDS__</div>

<h2>Curvas de entrenamiento (promedio sobre folds)</h2>
<div class="chart-container"><canvas id="training_curves"></canvas></div>

<h2>CDF del error absoluto</h2>
<div class="chart-container"><canvas id="error_cdf"></canvas></div>

<h2>Comparativo LSTM vs baselines (RMSE por posición)</h2>
<div class="chart-container"><canvas id="comparison_by_location"></canvas></div>

<footer>
Generado por lstm_prediction · adaptación de Sun & Kim (2021)
</footer>

<script>
const data = __DATA__;

function makeLineChart(id, datasets, xLabel, yLabel, logY) {
  const el = document.getElementById(id);
  if (!el) return;
  return new Chart(el, {
    type: 'line',
    data: { datasets: datasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      animation: false,
      plugins: { legend: { position: 'top' }, tooltip: { mode: 'index' } },
      scales: {
        x: { type: 'linear', title: { display: true, text: xLabel } },
        y: { type: logY ? 'logarithmic' : 'linear',
             title: { display: true, text: yLabel } }
      }
    }
  });
}

if (data.training_curves) {
  const tc = data.training_curves;
  makeLineChart('training_curves', [
    {
      label: 'train (MSE)',
      data: tc.epochs.map((e, i) => ({ x: e, y: tc.train_mean[i] })),
      borderColor: '#888', borderDash: [4, 4], pointRadius: 0,
    },
    {
      label: 'val (MSE)',
      data: tc.epochs.map((e, i) => ({ x: e, y: tc.val_mean[i] })),
      borderColor: '#1f4e79', pointRadius: 0,
    },
  ], 'Epoch', 'Loss (s²)', true);
}

if (data.error_cdf) {
  const ec = data.error_cdf;
  makeLineChart('error_cdf', [{
    label: 'CDF',
    data: ec.x.map((v, i) => ({ x: v, y: ec.y[i] })),
    borderColor: '#1f4e79', pointRadius: 0, fill: false,
  }], 'Error absoluto (s)', 'CDF', false);
}

if (data.comparison_by_location) {
  const datasets = data.comparison_by_location.algorithms.map(a => ({
    label: a.label,
    data: a.x.map((v, i) => ({ x: v, y: a.y[i] })),
    borderColor: a.color, backgroundColor: a.color,
    pointRadius: 2, borderWidth: a.name.endsWith('_lstm') ? 3 : 1.5,
  }));
  makeLineChart('comparison_by_location', datasets,
                'Posición de corte l_idx', 'RMSE (s)', false);
}
</script>
</body>
</html>
"""
    return (template
            .replace("__TITLE__",    title)
            .replace("__SUBTITLE__", subtitle)
            .replace("__CARDS__",    cards_html)
            .replace("__DATA__",     data_json))
