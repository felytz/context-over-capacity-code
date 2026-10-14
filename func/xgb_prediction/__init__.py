"""
xgb_prediction
================
XGBoost tabular como tercer modelo del estudio comparativo, con el mismo
protocolo de evaluación que los baselines (`arrival_prediction`) y el LSTM
(`lstm_prediction_tf`).

Motivación
----------
En producción el LSTM corre con `tokenize_by="grid"` y `context_features=None`
(`lstm_prediction_tf/run_experiment.py:197`), de modo que consume exactamente la
misma información que los baselines: posición en la grid y tiempo por celda.
La paridad de MAE observada no compara "deep learning vs. métodos simples" sino
dos aproximadores de función sobre la misma señal 1-D.

Este paquete permite separar los dos efectos:

  * **XGB-A** (Tier A) — solo información de trayectoria. Aísla el efecto de la
    ARQUITECTURA con información constante.
  * **XGB-B** (Tier B) — + features exógenos (headway, dwell, velocidad
    reciente, calendario). Aísla el efecto de los FEATURES.

Componentes
-----------
    - feature_builder : Tier A (build_block, compute_prior)
    - model           : XGBRemainingTime (wrapper de XGBRegressor)
    - cross_validation: run_matched_cv (streaming, pareado, sin df_errors)
    - run_experiment  : orquestador por (ruta, leg)

Ver `docs/PLAN_XGBOOST.md`.
"""
from .feature_builder import (
    FEATURE_NAMES,
    PAIR_FEATURES,
    TRIP_FEATURES,
    build_block,
    build_trip_features,
    compute_prior,
    feature_names,
)
from .exo_features import EXO_FEATURES, HEADWAY_CAP_S, build_exo_matrix
from .model import DEFAULT_PARAMS, XGBRemainingTime
from .cross_validation import (
    APTA_EARLY_S,
    APTA_LATE_S,
    ErrorAccumulator,
    InsufficientDataError,
    make_folds,
    make_l_indices,
    run_matched_cv,
)

__all__ = [
    "FEATURE_NAMES",
    "PAIR_FEATURES",
    "TRIP_FEATURES",
    "build_block",
    "build_trip_features",
    "compute_prior",
    "feature_names",
    "EXO_FEATURES",
    "HEADWAY_CAP_S",
    "build_exo_matrix",
    "DEFAULT_PARAMS",
    "XGBRemainingTime",
    "APTA_EARLY_S",
    "APTA_LATE_S",
    "ErrorAccumulator",
    "InsufficientDataError",
    "make_folds",
    "make_l_indices",
    "run_matched_cv",
]
