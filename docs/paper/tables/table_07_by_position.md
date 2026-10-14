**Table S16. Performance by cut position along the round trip (median over routes).**

Cut positions are normalized by the length of the route's grid, which spans the whole round trip, and grouped into early (first third), mid and late. Within a band, positions are pooled weighting by pair count, which is exact for MAE. Δ columns are the alternative's MAE relative to the LSTM's, so negative means the alternative is more accurate; the ratio column takes the best alternative per route. On-time here is the **uncalibrated operational band**: residual histograms are stored per fold, not per position, so the centred version of Table 4 cannot be derived at this resolution. MAE is therefore the metric to read across positions, since it does not depend on where the band is centred.

| Position | Routes | LSTM MAE (min) | Δ delay | Δ delay_hour | Δ xgb_mae_b | Ratio best/LSTM | LSTM on-time % | Δ on-time vs best (pp) |
|---|---|---|---|---|---|---|---|---|
| early | 35 | 4.00 | -1.3% | -6.1% | -7.9% | 0.92 | 45.4 | -3.8 |
| mid | 35 | 3.34 | -0.3% | -5.7% | -9.6% | 0.90 | 50.4 | -5.4 |
| late | 35 | 1.85 | -0.2% | -4.9% | -8.3% | 0.92 | 66.1 | -5.7 |

*Routes with two terminals.* On these routes the round trip passes the second terminal, and any wait there is part of the time being predicted. A prediction issued at the last cut position before that terminal has a horizon spanning the wait; one issued at the first cut position after it does not. Since error falls along the round trip on every route as the remaining horizon shortens, the step between those two cut positions (MAE after / MAE before) is read against the typical step between consecutive cut positions on the same route (the median of the others).

| Route | Intermediate terminal (km into the round trip) | Model | MAE, last cut before (min) | MAE, first cut after (min) | Step at the terminal | Typical step |
|---|---|---|---|---|---|---|
| 101 | 25.8 | Sequential LSTM | 4.32 | 3.22 | 0.74 | 0.97 |
|  |  | Delay | 4.32 | 3.20 | 0.74 | 0.97 |
|  |  | Delay (hourly) | 4.13 | 2.99 | 0.73 | 0.97 |
|  |  | XGBoost (MAE, + context) | 3.87 | 2.87 | 0.74 | 0.97 |
| 301 | 17.3 | Sequential LSTM | 3.91 | 3.34 | 0.85 | 0.96 |
|  |  | Delay | 3.89 | 3.32 | 0.85 | 0.96 |
|  |  | Delay (hourly) | 3.66 | 3.09 | 0.84 | 0.97 |
|  |  | XGBoost (MAE, + context) | 3.48 | 2.97 | 0.85 | 0.97 |
| 601 | 34.5 | Sequential LSTM | 3.74 | 3.09 | 0.83 | 0.97 |
|  |  | Delay | 3.73 | 3.08 | 0.83 | 0.97 |
|  |  | Delay (hourly) | 3.33 | 2.78 | 0.83 | 0.96 |
|  |  | XGBoost (MAE, + context) | 3.23 | 2.66 | 0.82 | 0.97 |
| 1501 | 22.4 | Sequential LSTM | 4.30 | 3.23 | 0.75 | 0.99 |
|  |  | Delay | 4.29 | 3.23 | 0.75 | 0.99 |
|  |  | Delay (hourly) | 3.86 | 2.96 | 0.77 | 0.99 |
|  |  | XGBoost (MAE, + context) | 3.67 | 2.81 | 0.76 | 0.98 |
| 1601 | 12.1 | Sequential LSTM | 3.45 | 1.98 | 0.57 | 0.97 |
|  |  | Delay | 3.43 | 1.93 | 0.56 | 0.97 |
|  |  | Delay (hourly) | 3.28 | 1.83 | 0.56 | 0.98 |
|  |  | XGBoost (MAE, + context) | 2.95 | 1.74 | 0.59 | 0.98 |
| 1602 | 12.9 | Sequential LSTM | 3.26 | 2.06 | 0.63 | 0.97 |
|  |  | Delay | 3.25 | 2.05 | 0.63 | 0.97 |
|  |  | Delay (hourly) | 3.07 | 1.91 | 0.62 | 0.97 |
|  |  | XGBoost (MAE, + context) | 2.93 | 1.85 | 0.63 | 0.97 |
| 1701 | 19.6 | Sequential LSTM | 3.89 | 2.68 | 0.69 | 0.95 |
|  |  | Delay | 3.89 | 2.67 | 0.69 | 0.96 |
|  |  | Delay (hourly) | 3.73 | 2.59 | 0.69 | 0.96 |
|  |  | XGBoost (MAE, + context) | 3.61 | 2.57 | 0.71 | 0.95 |
