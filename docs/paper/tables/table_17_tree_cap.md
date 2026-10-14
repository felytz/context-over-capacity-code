**Table S8. Sensitivity to the cap on boosted trees (35 routes; 600, as published, against 2000).**

Everything but the cap is unchanged: the same frame, round trips, folds, seed and early stopping (30 rounds without improvement on 15% of the training trips). A fit *reached the cap* when early stopping never triggered. Δ is the median over routes of the per-route relative change in MAE from 600 to 2000 trees.

| Model | Reached 600 | Trees at cap 2000 (median) | Reached 2000 | MAE 600 (s) | MAE 2000 (s) | Δ | Routes improved |
|---|---:|---:|---:|---:|---:|---:|---:|
| XGBoost (MSE, historical) | 0/175 | 153 | 0/175 | 211.9 | 211.9 | +0.00% | 0/35 |
| XGBoost (MAE, historical) | 10/175 | 383 | 0/175 | 211.4 | 211.4 | +0.00% | 2/35 |
| XGBoost (MSE, + context) | 2/175 | 189 | 0/175 | 198.4 | 198.4 | +0.00% | 0/35 |
| XGBoost (MAE, + context) | 59/175 | 503 | 0/175 | 197.2 | 197.2 | -0.00% | 19/35 |
| XGBoost (MAE, current-trip + delay) | 55/175 | 528 | 0/175 | 219.1 | 219.1 | +0.00% | 16/35 |
| XGBoost (MAE, current-trip only) | 71/175 | 547 | 0/175 | 219.3 | 219.3 | +0.00% | 17/35 |
| Tier A + calendar | 63/175 | 538 | 0/175 | 198.2 | 198.2 | -0.01% | 24/35 |
| Tier A + headway | 15/175 | 411 | 0/175 | 209.9 | 209.9 | +0.00% | 7/35 |
| Tier A + dwell | 7/175 | 341 | 0/175 | 211.6 | 211.6 | +0.00% | 2/35 |
| Tier A + recent speed | 9/175 | 370 | 0/175 | 211.2 | 211.3 | +0.00% | 3/35 |
| Tier A + route position | 34/175 | 455 | 0/175 | 210.5 | 210.5 | +0.00% | 8/35 |

| What it could move | 600 trees | 2000 trees |
|---|---|---|
| Table 5: MAE of levels 0 / 1 / 2 / 3 (s) | 219.3 / 219.1 / 211.4 / 197.2 | 219.3 / 219.1 / 211.4 / 197.2 |
| Table 5: share of history / context | 36% / 64% | 36% / 64% |
| Table 5: level 0 → level 3 | -8.5% | -8.5% |
| Level 0 against the LSTM | better on 16/35, +0.4% (p 0.2383) | better on 16/35, +0.4% (p 0.2318) |
| XGBoost (MAE, + context) against the LSTM | better on 35/35, -8.5% | better on 35/35, -8.5% |
| XGBoost (MAE, + context) against Delay (hourly) | better on 35/35, -2.2% | better on 35/35, -2.2% |
| Table S14: Δ of each group vs tier A (cal / headway / dwell / speed / route) | -5.31% / -0.61% / -0.05% / -0.24% / -0.38% | -5.32% / -0.61% / -0.05% / -0.25% / -0.38% |

*Notes.* The sequence model and Delay (hourly) are not refitted: they have no trees. The published tables keep the pre-specified cap of 600.
