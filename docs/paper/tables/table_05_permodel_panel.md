**Table 2. Per-model performance across the panel (median [IQR] over routes).**

The unit is the route: each contributes its round trip, from one crossing of the route's cut point to the next, with one evaluation frame per route. Routes enter under the pre-specified criterion (trip cut not rated poor, at least 100 round trips retained by the grid); 35 qualify, and all of them form a single panel. All models are scored on identical round trips, positions, horizons and folds. *On-time (APTA)* uses the operational band `t_true − t_pred ∈ [−60, +300] s`, which is asymmetric and therefore rewards models whose residuals sit on its wide side; *on-time (centred)* uses a symmetric band of the same width with the residual centred out-of-fold, and so reflects dispersion alone. Table 4 decomposes the difference.

| Model | MAE (min) med [IQR] | On-time % (APTA) med [IQR] | On-time % (centred) med [IQR] |
|---|---|---|---|
| XGBoost (MAE, + context) | 3.29 [2.93, 3.59] | 53.5 [51.6, 57.8] | 60.9 [58.3, 66.5] |
| XGBoost (MSE, + context) | 3.31 [2.95, 3.62] | 51.0 [49.4, 55.3] | 60.1 [57.7, 66.5] |
| Delay (hourly) | 3.35 [3.02, 3.65] | 50.5 [49.1, 54.7] | 59.8 [57.7, 65.8] |
| Linear | 3.52 [3.12, 3.82] | 50.0 [48.3, 53.9] | 58.7 [56.4, 64.5] |
| XGBoost (MAE, historical) | 3.52 [3.09, 3.81] | 52.2 [50.1, 55.6] | 58.9 [56.1, 63.7] |
| XGBoost (MSE, historical) | 3.53 [3.10, 3.83] | 50.1 [48.3, 53.8] | 58.6 [55.9, 63.6] |
| Delay | 3.58 [3.10, 3.85] | 49.2 [47.3, 53.4] | 58.1 [55.5, 63.3] |
| Sequential LSTM | 3.64 [3.14, 3.88] | 49.0 [47.4, 52.5] | 57.9 [55.2, 62.9] |
| kNN | 3.66 [3.23, 3.98] | 48.3 [46.8, 52.4] | 57.0 [54.6, 62.6] |
| Kernel | 3.85 [3.33, 4.11] | 47.6 [45.7, 52.3] | 56.1 [52.8, 61.3] |
