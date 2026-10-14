**Table S9 (measured). What the fits actually did (35 routes × 5 folds).**

Trees: the number the early stopping selected, on 15% of the training trips held out, out of a cap of 600; a fit *reached the cap* when the 30-round patience never triggered before tree 600. Epochs: the sequence model (honest early stopping), read from each task's log; *reached the cap* counts fits that were not stopped early. Sequence length: the cells of each route's round trip fed to the network (no window); the last column counts the routes whose attention is computed in horizon chunks (amendment E3).

| Model | Quantity | Fits | Median [IQR] | Range | Reached the cap |
|---|---|---:|---|---|---:|
| XGBoost (MSE, historical) | trees selected | 175 | 153 [126, 176] | 83–341 | 0/175 (cap 600) |
| XGBoost (MAE, historical) | trees selected | 175 | 383 [326, 451] | 151–600 | 10/175 (cap 600) |
| XGBoost (MSE, + context) | trees selected | 175 | 189 [155, 232] | 77–594 | 2/175 (cap 600) |
| XGBoost (MAE, + context) | trees selected | 175 | 503 [402, 594] | 247–600 | 59/175 (cap 600) |
| XGBoost (MAE, current-trip + delay) | trees selected | 175 | 528 [434, 590] | 278–600 | 55/175 (cap 600) |
| XGBoost (MAE, current-trip only) | trees selected | 175 | 547 [457, 595] | 239–600 | 71/175 (cap 600) |
| Sequential LSTM | epochs trained | 175 | 11 [9, 16] | 6–30 | 0/175 (cap 40) |
| Sequential LSTM | best epoch | 175 | 6 [4, 11] | 1–25 | 0/175 (cap 40) |
| Sequential LSTM | sequence length (cells) | 35 | 428 [379, 474] | 219–640 | 3 routes chunked |

**Split-gain share by block (median [IQR] over routes; the harness's `gain_frac`, summed within each block).**

| Model | Block | Median | IQR |
|---|---|---:|---|
| XGBoost (MAE, historical) | historical prior | 93.2% | [90.9, 94.2]% |
| XGBoost (MAE, historical) | current trip | 6.4% | [5.4, 8.7]% |
| XGBoost (MAE, historical) | delay vs. mean profile | 0.4% | [0.3, 0.4]% |
| XGBoost (MAE, + context) | historical prior | 88.8% | [86.7, 90.4]% |
| XGBoost (MAE, + context) | current trip | 6.7% | [4.4, 8.6]% |
| XGBoost (MAE, + context) | context: calendar | 1.9% | [1.7, 2.1]% |
| XGBoost (MAE, + context) | context: dwell | 0.6% | [0.5, 0.7]% |
| XGBoost (MAE, + context) | context: headway | 0.6% | [0.6, 0.7]% |
| XGBoost (MAE, + context) | context: recent speed | 0.6% | [0.5, 0.7]% |
| XGBoost (MAE, + context) | context: route position | 0.6% | [0.5, 0.6]% |
| XGBoost (MAE, + context) | delay vs. mean profile | 0.5% | [0.4, 0.5]% |
| XGBoost (MSE, historical) | historical prior | 97.3% | [95.2, 98.5]% |
| XGBoost (MSE, historical) | current trip | 2.6% | [1.5, 4.6]% |
| XGBoost (MSE, historical) | delay vs. mean profile | 0.1% | [0.0, 0.1]% |
| XGBoost (MSE, + context) | historical prior | 96.3% | [94.3, 97.6]% |
| XGBoost (MSE, + context) | current trip | 2.5% | [1.5, 5.2]% |
| XGBoost (MSE, + context) | context: calendar | 0.3% | [0.3, 0.4]% |
| XGBoost (MSE, + context) | context: headway | 0.1% | [0.1, 0.1]% |
| XGBoost (MSE, + context) | context: dwell | 0.1% | [0.1, 0.1]% |
| XGBoost (MSE, + context) | delay vs. mean profile | 0.1% | [0.0, 0.1]% |
| XGBoost (MSE, + context) | context: recent speed | 0.1% | [0.0, 0.1]% |
| XGBoost (MSE, + context) | context: route position | 0.0% | [0.0, 0.0]% |

*Notes.* Split gain measures how a fitted model allocates its splits among the features it was given, not what each block is worth: that is the ablation of Table 5.
