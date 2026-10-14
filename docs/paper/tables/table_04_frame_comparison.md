**Table S11. Paired per-route comparison on the common evaluation frame (January–August 2024).**

All algorithms are trained and scored on identical round trips, an identical distance grid, identical cut positions and identical folds, defined once per route by the evaluation frame; the number of evaluated (trip, position, horizon) pairs is therefore identical across algorithms by construction rather than by post-hoc matching. Routes enter under a pre-specified criterion (trip cut not rated poor, at least 100 round trips retained by the grid); 35 routes qualify and have all four algorithms, each with one unit, its round trip. Δ columns are relative to the LSTM: negative means the competing model is more accurate. In the summary row, counts are totals and MAE and Δ are medians across routes. † marks routes whose trip cut is rated *review* (Table S4).

| Route | Round trips | n pairs | LSTM | delay | delay_hour | xgb_mae_b | Δ delay_hour | Δ xgb_mae_b | Best |
|---|---|---|---|---|---|---|---|---|---|
| 101 † | 2,500 | 18,447,500 | 4.21 | 4.18 | 3.93 | 3.81 | -6.6% | -9.6% | `xgb_mae_b` |
| 102 | 2,500 | 12,072,500 | 3.14 | 3.10 | 3.02 | 2.93 | -3.8% | -6.5% | `xgb_mae_b` |
| 201 | 2,500 | 21,372,500 | 4.32 | 4.29 | 4.03 | 3.96 | -6.8% | -8.3% | `xgb_mae_b` |
| 202 | 2,500 | 21,260,000 | 4.21 | 4.19 | 3.85 | 3.77 | -8.7% | -10.5% | `xgb_mae_b` |
| 301 | 2,500 | 16,910,000 | 3.66 | 3.58 | 3.35 | 3.29 | -8.6% | -10.3% | `xgb_mae_b` |
| 302 † | 2,500 | 17,435,000 | 3.87 | 3.82 | 3.49 | 3.37 | -9.9% | -12.9% | `xgb_mae_b` |
| 401 | 2,500 | 18,260,000 | 3.37 | 3.35 | 3.23 | 3.14 | -4.1% | -6.9% | `xgb_mae_b` |
| 402 | 2,500 | 18,185,000 | 3.57 | 3.50 | 3.33 | 3.28 | -6.7% | -8.1% | `xgb_mae_b` |
| 501 | 2,500 | 14,287,500 | 3.68 | 3.67 | 3.50 | 3.44 | -5.1% | -6.6% | `xgb_mae_b` |
| 601 † | 2,500 | 24,035,000 | 4.06 | 3.96 | 3.68 | 3.64 | -9.3% | -10.2% | `xgb_mae_b` |
| 801 | 2,500 | 13,197,500 | 3.41 | 3.32 | 3.23 | 3.15 | -5.3% | -7.6% | `xgb_mae_b` |
| 901 † | 2,500 | 14,285,000 | 2.92 | 2.90 | 2.76 | 2.71 | -5.4% | -7.1% | `xgb_mae_b` |
| 902 | 2,500 | 8,247,500 | 2.29 | 2.27 | 2.19 | 2.08 | -4.7% | -9.1% | `xgb_mae_b` |
| 1001 | 2,500 | 15,710,000 | 3.07 | 3.04 | 2.78 | 2.74 | -9.2% | -10.7% | `xgb_mae_b` |
| 1002 | 2,500 | 16,085,000 | 3.43 | 3.36 | 3.03 | 2.94 | -11.6% | -14.1% | `xgb_mae_b` |
| 1101 | 2,500 | 17,547,500 | 3.91 | 3.86 | 3.62 | 3.56 | -7.5% | -9.0% | `xgb_mae_b` |
| 1101-ALT (110101) | 2,500 | 17,510,000 | 3.88 | 3.85 | 3.65 | 3.63 | -5.9% | -6.5% | `xgb_mae_b` |
| 1201 | 2,500 | 18,110,000 | 3.91 | 3.84 | 3.69 | 3.67 | -5.7% | -6.1% | `xgb_mae_b` |
| 1301 | 2,500 | 14,210,000 | 3.44 | 3.41 | 3.25 | 3.22 | -5.6% | -6.5% | `xgb_mae_b` |
| 1401 | 2,500 | 15,935,000 | 4.19 | 4.18 | 4.09 | 3.97 | -2.6% | -5.4% | `xgb_mae_b` |
| 1402 | 2,500 | 15,860,000 | 4.39 | 4.29 | 4.15 | 4.06 | -5.5% | -7.6% | `xgb_mae_b` |
| 1501 | 2,500 | 16,647,500 | 3.64 | 3.61 | 3.37 | 3.30 | -7.3% | -9.2% | `xgb_mae_b` |
| 1502 | 2,500 | 9,372,500 | 2.55 | 2.53 | 2.47 | 2.36 | -3.2% | -7.6% | `xgb_mae_b` |
| 1601 | 2,500 | 9,035,000 | 2.63 | 2.58 | 2.44 | 2.31 | -7.5% | -12.2% | `xgb_mae_b` |
| 1602 | 2,500 | 9,035,000 | 2.67 | 2.64 | 2.44 | 2.36 | -8.5% | -11.6% | `xgb_mae_b` |
| 1701 | 2,500 | 14,547,500 | 3.29 | 3.26 | 3.09 | 3.01 | -6.2% | -8.5% | `xgb_mae_b` |
| 1702 | 2,500 | 12,112,500 | 2.77 | 2.75 | 2.63 | 2.51 | -5.2% | -9.3% | `xgb_mae_b` |
| 1703 | 2,500 | 15,072,500 | 3.14 | 3.09 | 2.98 | 2.82 | -5.0% | -10.2% | `xgb_mae_b` |
| 1801 | 2,500 | 16,347,500 | 3.82 | 3.80 | 3.56 | 3.49 | -6.8% | -8.7% | `xgb_mae_b` |
| 1801-ALT (180101) | 2,500 | 16,085,000 | 3.77 | 3.75 | 3.51 | 3.41 | -6.8% | -9.6% | `xgb_mae_b` |
| 1802 | 2,500 | 16,685,000 | 3.79 | 3.76 | 3.59 | 3.55 | -5.2% | -6.4% | `xgb_mae_b` |
| 1802-ALT (180201) | 2,500 | 16,460,000 | 3.67 | 3.64 | 3.51 | 3.43 | -4.5% | -6.7% | `xgb_mae_b` |
| 1901 | 2,500 | 18,372,500 | 3.52 | 3.50 | 3.35 | 3.28 | -5.0% | -6.9% | `xgb_mae_b` |
| 1902 | 2,500 | 18,035,000 | 3.87 | 3.85 | 3.67 | 3.59 | -5.3% | -7.3% | `xgb_mae_b` |
| 20000 | 2,500 | 9,072,500 | 2.85 | 2.84 | 2.69 | 2.59 | -5.7% | -9.3% | `xgb_mae_b` |
| **All (totals / medians)** | **87,500** | **545,842,500** | **3.64** | **3.58** | **3.35** | **3.29** | **-5.7%** | **-8.5%** |  |

*Notes.* MAE is in minutes. The LSTM is the best of the four in 0 of 35 routes. An hour-of-day historical mean (`delay_hour`), which requires no training and no GPU, is more accurate in 35 of 35 routes; the gradient-boosted tabular model with contextual features (`xgb_mae_b`) in 35 of 35; and the plain historical mean (`delay`) in 35. On-time percentages are deliberately absent here: under the asymmetric operational band they are confounded with residual centring, and are reported with that decomposition in Table 4.
