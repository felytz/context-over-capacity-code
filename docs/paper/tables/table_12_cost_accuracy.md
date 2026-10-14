**Table 7. Accuracy against computational cost (35 pre-specified routes).**

Cost is wall-clock time measured inside the cross-validation loop, summed over folds and routes, on the panel that Tables 2-4, S11, S16 and S17 use. *Training* is model fitting plus the feature construction it needs; *inference* is prediction at the evaluated cut positions. *Core-hours* multiplies by the threads the process was given and is an upper bound: only the gradient-boosted models are threaded (`n_jobs=-1` under `OMP_NUM_THREADS`), the baselines are single-threaded NumPy. The sequence model runs on a GPU and its clock covers training and evaluation together, so it has no inference column and its hours are not commensurable with the CPU ones — they are reported side by side, not summed. No model was tuned (Table S9); a hyperparameter-search budget would alter the cost comparison and was not included for any model. **On the frontier** marks models that no other model beats on both axes at once.


| Model | MAE (min) | Training | Inference | Total | Resource-hours | s/route | Device | On the frontier |
|---|---:|---:|---:|---:|---:|---:|---|---|
| XGBoost (MAE, + context) | 3.29 | 2.10 h | 0.08 h | 2.17 h | 34.79 core-h | 221.5 | CPU | **yes** |
| XGBoost (MSE, + context) | 3.31 | 0.27 h | 0.05 h | 0.32 h | 5.16 core-h | 31.5 | CPU | **yes** |
| Delay (hourly) | 3.35 | < 1 s | 13 s | 13 s | 0.00 core-h | 0.3 | CPU | **yes** |
| Linear | 3.52 | 1 s | 0.46 h | 0.46 h | 0.46 core-h | 43.0 | CPU | no |
| XGBoost (MAE, historical) | 3.52 | 1.59 h | 0.08 h | 1.66 h | 26.59 core-h | 177.3 | CPU | no |
| XGBoost (MSE, historical) | 3.53 | 0.15 h | 0.08 h | 0.23 h | 3.62 core-h | 22.9 | CPU | no |
| Delay | 3.58 | < 1 s | < 1 s | < 1 s | 0.00 core-h | 0.0 | CPU | **yes** |
| Sequential LSTM | 3.64 | — | — | 62.65 h | 62.65 GPU-h | 5363.7 | GPU | no |
| kNN | 3.66 | 0 s | 0.11 h | 0.11 h | 0.11 core-h | 11.3 | CPU | no |
| Kernel | 3.85 | 1 s | 0.10 h | 0.10 h | 0.10 core-h | 10.4 | CPU | no |

*Notes.* The hour-of-day historical mean fits no parameters: its whole cost is 13 s over the panel, 17862× less than the sequence model, and it is 8.0% more accurate than the sequence model. XGBoost (MAE, + context) costs 2.17 h against the sequence model's 62.65 h — 28.8× less observed wall-clock time on the reported hardware — and is 9.7% more accurate than it. These percentages compare panel medians of the accuracy column; the paired per-route differences, with their intervals, are in Table 3. The sequence model is not on the frontier: 7 models are both cheaper and more accurate. Accuracy here is the median over the same per-route error summaries that Table 2 aggregates — only the timings come from the instrumented execution — so the two agree to two decimals for every model. The sequence model is not bit-reproducible on GPU (Section 4.2), so re-executing its training moves its row by a second or two — enough to reorder the foot of the table, which is why the ranking there is not cited.
