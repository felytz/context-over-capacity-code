**Table 4. On-time rate under three bands: operational, width-matched symmetric, and centred.**

Medians over the **35 routes** meeting the pre-specified criterion (trip cut not rated poor, at least 100 round trips retained by the grid) and having both model families computed. Each route is one unit: its round trip, from one crossing of the route's cut point to the next.


The symmetric ±180 s band has the **same width** (360 s) as the operational band `[−60, +300]`; only the centre differs, so the step between those two columns measures the reward for landing on the band's wide side rather than any gain in precision. The next step, to the centred column, removes what remained of each model's residual bias, estimated out of fold. The final column is therefore dispersion alone.


| Model | MAE (s) | Bias (s) | Operational % | ±180 s % | Δ asym. | ±180 s centred % | Δ bias |
|---|---|---|---|---|---|---|---|
| `xgb_mae_b` | 197.22 | -22.2 | 53.5 | 60.9 | +7.4 | 60.9 | +0.0 |
| `xgb_mse_b` | 198.45 | -8.9 | 51.0 | 60.2 | +9.1 | 60.1 | -0.0 |
| `delay_hour` | 200.87 | +0.7 | 50.5 | 59.7 | +9.3 | 59.8 | +0.1 |
| `xgb_mae` | 211.36 | -22.6 | 52.2 | 58.8 | +6.7 | 58.9 | +0.0 |
| `linreg` | 211.01 | +0.4 | 50.0 | 58.6 | +8.6 | 58.7 | +0.1 |
| `xgb_mse` | 211.90 | -11.1 | 50.1 | 58.5 | +8.4 | 58.6 | +0.1 |
| `delay` | 214.62 | +0.0 | 49.2 | 58.0 | +8.8 | 58.1 | +0.1 |
| `lstm_a` | 218.38 | -2.0 | 49.0 | 57.7 | +8.7 | 57.9 | +0.1 |
| `knn` | 219.52 | +4.4 | 48.3 | 56.8 | +8.5 | 57.0 | +0.2 |
| `kernel` | 230.86 | +3.3 | 47.6 | 56.0 | +8.5 | 56.1 | +0.1 |


*Notes.* *Bias* is the mean residual `t_true − t_pred`; with a right-skewed residual it is not the typical error. For `xgb_mae_b`, the most accurate model, the mean residual is -22.2 s but the median residual is +4 s (per route -1 to +6 s), and centring moves its on-time rate by less than 0.1 points. Per-route detail for all 350 (route, model) rows is available in the supplementary material.
