**Table 3 (lower). Bootstrap 95% confidence intervals (median over routes; family-clustered, 10,000 draws).**

Each route contributes its round trip. The two variants of a route family (1101 and 1101-ALT, 1401 and 1402, 1801 and 1801-ALT, 1802 and 1802-ALT) share most of their path and their fleet, so the resampling unit is the **family**: groups are drawn with replacement and every route of a drawn family is carried with it; the other routes are groups of one. The route-resampled interval is reported alongside for comparison; it is the narrower of the two wherever the variants of a family are correlated. Point estimates are identical under both schemes and are computed on the observed routes.

| Quantity | Routes | Groups | Point | 95% CI (family) | 95% CI (route) | Unit |
|---|---|---|---|---|---|---|
| MAE ratio, best alternative / LSTM | 35 | 31 | 0.91 | [0.91, 0.93] | [0.91, 0.93] | × |
| MAE, Delay (hourly) vs LSTM | 35 | 31 | -5.67 | [-6.78, -5.29] | [-6.78, -5.29] | % |
| MAE, XGBoost (MAE, + context) vs LSTM | 35 | 31 | -8.53 | [-9.28, -7.34] | [-9.28, -7.27] | % |
| On-time (centred), LSTM − best alternative | 35 | 31 | -3.08 | [-3.52, -2.89] | [-3.47, -2.88] | pp |
