**Table S13. The hourly lookup table against the boosted models (35 routes; paired, family-clustered intervals).**

Each route contributes its round trip, and every model is scored on the same frame, so the differences are paired by construction. Differences are the boosted model minus Delay (hourly): a negative MAE difference and a positive on-time difference favour the boosted model, and *wins* counts the routes where it is better. The point is the median over routes; the interval is a percentile bootstrap that resamples route families (31 groups, 10,000 draws, as in Table 3). The *p* values are Wilcoxon signed-rank, Holm-adjusted within each metric across the two boosted models.

| Comparison | Metric | Unit | Boosted model wins | Median | 95% CI (family) | *p* (Holm) |
|---|---|---|---|---:|---|---:|
| XGBoost (MAE, + context) − Delay (hourly) | MAE | % | 35/35 | -2.22 | [-2.87, -1.81] | < 0.0001 |
| XGBoost (MAE, + context) − Delay (hourly) | MAE | s | 35/35 | -4.65 | [-5.52, -4.07] | < 0.0001 |
| XGBoost (MAE, + context) − Delay (hourly) | On-time (centred) | pp | 35/35 | +1.03 | [+0.90, +1.21] | < 0.0001 |
| XGBoost (MAE, + context) − Delay (hourly) | On-time (APTA) | pp | 35/35 | +3.10 | [+2.56, +3.35] | < 0.0001 |
| XGBoost (MAE, historical) − Delay (hourly) | MAE | % | 2/35 | +4.22 | [+3.28, +4.98] | < 0.0001 |
| XGBoost (MAE, historical) − Delay (hourly) | MAE | s | 2/35 | +8.49 | [+6.77, +10.48] | < 0.0001 |
| XGBoost (MAE, historical) − Delay (hourly) | On-time (centred) | pp | 2/35 | -1.14 | [-1.45, -0.82] | < 0.0001 |
| XGBoost (MAE, historical) − Delay (hourly) | On-time (APTA) | pp | 33/35 | +1.50 | [+1.17, +2.06] | < 0.0001 |

*Notes.* Between the plain historical mean (Delay) and the most accurate model (XGBoost (MAE, + context)), Delay (hourly) closes a median 71% of the MAE gap per route (95% CI 63–78%; more than half of it on 30 of 35 routes); read from the panel medians of Table 2, the same share is 79%.
