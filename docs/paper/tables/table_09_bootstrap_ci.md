**Not in the article: a route-resampled check of Table 3 (lower). Bootstrap 95% confidence intervals (median over routes; route-resampled, 10,000 draws).**

Routes are resampled with replacement and the median is recomputed; intervals are percentile intervals over 10,000 draws with a fixed seed. A ratio above 1 and a positive on-time difference favour the LSTM. The *best alternative* rows select the winning model per route on the same data used to evaluate it, which biases them in the alternatives' favour; the fixed-model rows carry no such selection and are the ones to read for the main claim. Resampling that keeps the two variants of a family together is Table 3 (lower), the one the article reports.

| Quantity | Routes | Point | 95% CI | Unit |
|---|---|---|---|---|
| MAE ratio, best alternative / LSTM | 35 | 0.91 | [0.91, 0.93] | × |
| MAE, Delay (hourly) vs LSTM | 35 | -5.67 | [-6.78, -5.29] | % |
| MAE, XGBoost (MAE, + context) vs LSTM | 35 | -8.53 | [-9.28, -7.27] | % |
| On-time (centred), LSTM − best alternative | 35 | -3.08 | [-3.47, -2.88] | pp |
