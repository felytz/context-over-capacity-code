"""
_lib_style.py
=============
Estilo compartido de las figuras del proyecto (matplotlib).

Antes cada paquete de predicción definía su propio ALGO_COLORS / ALGO_LABELS /
rcParams (``arrival_prediction.plots`` y ``lstm_prediction_tf.plots``), con riesgo
de que las figuras comparativas (LSTM vs baselines, que comparten ``df_errors``)
salieran con estilos distintos. Aquí viven una sola vez.

Paleta **colorblind-safe** (Okabe & Ito, 2008): distinguible por daltónicos y en
impresión a color. Además se exponen **marcadores y estilos de línea** por
algoritmo, para que las curvas superpuestas se distingan SIN depender del color
(impresión B/N, daltonismo).
"""
from __future__ import annotations

# Paleta Okabe-Ito (segura para daltonismo).
_OKABE_ITO = {
    "orange":     "#E69F00",
    "skyblue":    "#56B4E9",
    "green":      "#009E73",
    "yellow":     "#F0E442",
    "blue":       "#0072B2",
    "vermillion": "#D55E00",
    "purple":     "#CC79A7",
    "black":      "#000000",
}

# LSTM (propuesta del proyecto) destacado en azul marino, distinguible del resto.
LSTM_COLOR = "#1f4e79"

ALGO_COLORS = {
    "delay":           _OKABE_ITO["vermillion"],
    "linreg":          _OKABE_ITO["orange"],
    "knn":             _OKABE_ITO["green"],
    "kernel":          _OKABE_ITO["blue"],
    "sequential_lstm": LSTM_COLOR,
    "hybrid_lstm":     _OKABE_ITO["purple"],
}

ALGO_LABELS = {
    "delay":           "Delay-based",
    "linreg":          "Linear Regression",
    "knn":             "K-NN (K=10)",
    "kernel":          "Kernel Regression (b=1)",
    "sequential_lstm": "Sequential LSTM (ours)",
    "hybrid_lstm":     "Hybrid LSTM",
}

# Redundancia visual: marcador + estilo de línea distintos por algoritmo.
ALGO_MARKERS = {
    "delay": "o", "linreg": "s", "knn": "^", "kernel": "D",
    "sequential_lstm": "*", "hybrid_lstm": "P",
}
ALGO_LINESTYLES = {
    "delay": "-", "linreg": "--", "knn": "-.", "kernel": ":",
    "sequential_lstm": "-", "hybrid_lstm": "--",
}

# Fallbacks para algoritmos no listados.
DEFAULT_COLOR = "#666666"
DEFAULT_MARKER = "o"
DEFAULT_LINESTYLE = "-"

# rcParams unificados (unión de los que tenían ambos paquetes).
RCPARAMS = {
    "axes.titlesize":   12,
    "axes.titleweight": "bold",
    "axes.labelsize":   10,
    "legend.fontsize":  9,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "axes.grid":        True,
    "grid.alpha":       0.25,
    "figure.dpi":       100,
    "savefig.dpi":      120,
    "savefig.bbox":     "tight",
}


def apply_style() -> None:
    """Aplica RCPARAMS al matplotlib actual (idempotente)."""
    import matplotlib.pyplot as plt
    plt.rcParams.update(RCPARAMS)


def color(algo: str) -> str:
    return ALGO_COLORS.get(algo, DEFAULT_COLOR)


def label(algo: str) -> str:
    return ALGO_LABELS.get(algo, algo)


def marker(algo: str) -> str:
    return ALGO_MARKERS.get(algo, DEFAULT_MARKER)


def linestyle(algo: str) -> str:
    return ALGO_LINESTYLES.get(algo, DEFAULT_LINESTYLE)
