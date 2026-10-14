"""
arrival_prediction
====================
Replicación del paper de predicción de tiempos de llegada
(Dublin Bus 46A) adaptada al proyecto Ruta 101 · BUSONORA.

Uso típico:

    from arrival_prediction import build_arrival_prediction_pipeline

    result = build_arrival_prediction_pipeline(
        df_final,
        direction=0,
        grid_step_m=100,
        n_folds=5,
        output_dir="../data/02_interim/routes/101/arrival/",
        verbose=True,
    )

Componentes:
    - dataset_builder.build_trajectory_matrix
    - predictors: DelayPredictor, LinearRegressionPredictor,
                  KNNPredictor, KernelRegressionPredictor
    - cross_validation.run_cross_validation
    - plots.build_static_report, plots.build_html_dashboard
    - run_experiment.build_arrival_prediction_pipeline
"""
from .dataset_builder import build_trajectory_matrix
from .predictors import (
    BasePredictor,
    DelayPredictor,
    LinearRegressionPredictor,
    KNNPredictor,
    KernelRegressionPredictor,
    default_predictors,
)
from .cross_validation import (
    run_cross_validation,
    summarize_rmse_by_location,
    summarize_error_by_horizon,
)
from .plots import (
    plot_trajectories,
    plot_rmse_by_location,
    plot_error_boxplots_by_horizon,
    plot_grid_sensitivity,
    build_static_report,
    build_html_dashboard,
)
from .run_experiment import build_arrival_prediction_pipeline

__all__ = [
    "build_trajectory_matrix",
    "BasePredictor",
    "DelayPredictor",
    "LinearRegressionPredictor",
    "KNNPredictor",
    "KernelRegressionPredictor",
    "default_predictors",
    "run_cross_validation",
    "summarize_rmse_by_location",
    "summarize_error_by_horizon",
    "plot_trajectories",
    "plot_rmse_by_location",
    "plot_error_boxplots_by_horizon",
    "plot_grid_sensitivity",
    "build_static_report",
    "build_html_dashboard",
    "build_arrival_prediction_pipeline",
]
