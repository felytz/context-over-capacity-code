**Table 3 (upper). Paired route-level tests: Sequential LSTM vs. each alternative (Wilcoxon signed-rank, Holm-adjusted).**

The route is the unit of pairing: every model is scored on identical round trips, positions, horizons and folds, so the per-route differences are paired by construction. The panel has 35 routes. Differences are LSTM minus alternative, so a negative MAE difference and a positive on-time difference both favour the LSTM. *Wins* counts routes where the LSTM is better. On-time is measured on the symmetric band with the residual centred out-of-fold, i.e. dispersion only; the operational band is decomposed in Table 4. Holm adjustment is applied within each metric, across the nine alternatives.

| Alternative | MAE: LSTM wins | MAE: median diff (min) | MAE: p (Holm) | On-time: LSTM wins | On-time: median diff (pp) | On-time: p (Holm) |
|---|---|---|---|---|---|---|
| XGBoost (MAE, + context) | 0/35 | +0.28 | < 0.0001 | 0/35 | -3.08 | < 0.0001 |
| XGBoost (MAE, historical) | 0/35 | +0.07 | < 0.0001 | 0/35 | -0.99 | < 0.0001 |
| XGBoost (MSE, + context) | 0/35 | +0.26 | < 0.0001 | 0/35 | -2.77 | < 0.0001 |
| XGBoost (MSE, historical) | 1/35 | +0.06 | < 0.0001 | 1/35 | -0.81 | < 0.0001 |
| Delay (hourly) | 0/35 | +0.21 | < 0.0001 | 0/35 | -2.12 | < 0.0001 |
| Delay | 0/35 | +0.03 | < 0.0001 | 0/35 | -0.33 | < 0.0001 |
| Kernel | 31/35 | -0.19 | < 0.0001 | 31/35 | +2.00 | < 0.0001 |
| kNN | 28/35 | -0.09 | < 0.0001 | 28/35 | +0.71 | < 0.0001 |
| Linear | 1/35 | +0.09 | < 0.0001 | 1/35 | -1.00 | < 0.0001 |

*Notes.* With paired differences this consistent, the signed-rank test detects a difference however small it is, so significance here is not the same as consequence: read each median difference against the LSTM's median MAE over the panel, 3.64 min.
