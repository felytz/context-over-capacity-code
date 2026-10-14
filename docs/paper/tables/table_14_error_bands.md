**Table S12. Share of predictions within a symmetric error band (median over 35 routes).**

Each cell is the percentage of (trip, cut, horizon) predictions whose absolute error is at most the stated tolerance, computed per route and then taken as the median across routes. Unlike the [-1, +5] min band of Table 2, which follows the operational on-time definition and is deliberately asymmetric, these bands are symmetric and are meant to be read directly: a value of 40 at &plusmn;2 min means two predictions in five land within two minutes of the actual arrival.

| Model | &plusmn;1 min | &plusmn;2 min | &plusmn;3 min | &plusmn;5 min |
|---|---:|---:|---:|---:|
| XGBoost (MAE, + context) | 28.3 | 47.4 | 60.9 | 78.6 |
| XGBoost (MSE, + context) | 27.5 | 46.6 | 60.2 | 78.6 |
| Delay (hourly) | 27.6 | 46.2 | 59.7 | 77.9 |
| XGBoost (MAE, historical) | 27.5 | 45.7 | 58.8 | 76.3 |
| Linear | 27.3 | 45.1 | 58.6 | 76.2 |
| XGBoost (MSE, historical) | 26.8 | 45.1 | 58.5 | 76.2 |
| Delay | 26.8 | 44.6 | 58.0 | 75.7 |
| Sequential LSTM | 26.6 | 44.5 | 57.7 | 75.4 |
| kNN | 26.2 | 43.6 | 56.8 | 74.8 |
| Kernel | 25.3 | 42.8 | 56.0 | 73.8 |

*Notes.* Panel of 35 routes, one round trip each. Values are medians over routes, so rows need not be consistent with any single route.
