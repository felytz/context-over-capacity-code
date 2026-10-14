**Table S15. Rolling-origin evaluation: train on the months so far, predict the next one (35 routes).**

Each rung fits every model on all round trips up to a month and scores it on the following month; the first month is training only, so eight months give seven rungs. Within a rung every model sees the same round trips, cut positions and horizons, so the comparison is paired by construction. **Across rungs it is not**: each rung scores a different month, and months differ in difficulty, so a change in absolute MAE from one rung to the next is not degradation. The comparable quantity along the ladder is the relative difference, reported as *Δ vs Sequential LSTM*. Training size also grows by construction, so any trend mixes more data with a different month; this design does not separate the two. Rungs are calendar months: a route whose records start later enters the ladder at the first month it can be scored on. This is the sensitivity panel: it takes **every** included route, whether or not it reaches 100 training round trips at each rung, so the set does not stay constant along the ladder.


| Rung | Test month | Model | MAE (min) | Δ vs ref | Beats ref | Routes | Pairs |
|---|---|---|---:|---:|---:|---:|---:|
| 1 | 2024-02 | XGBoost (MAE, + context) | 3.63 | -6.7% | 34/34 | 34 | 67,442,778 |
|  |  | Delay (hourly) | 3.64 | -3.8% | 32/34 | 34 | 67,442,778 |
|  |  | Delay | 3.78 | -1.5% | 29/34 | 34 | 67,442,778 |
|  |  | XGBoost (MAE, historical) | 3.81 | -1.6% | 29/34 | 34 | 67,442,778 |
|  |  | Sequential LSTM | 3.88 | — | — | 34 | 67,442,778 |
| 2 | 2024-03 | XGBoost (MAE, + context) | 3.33 | -7.2% | 34/34 | 34 | 69,820,655 |
|  |  | Delay (hourly) | 3.42 | -4.4% | 33/34 | 34 | 69,820,655 |
|  |  | XGBoost (MAE, historical) | 3.52 | -2.0% | 32/34 | 34 | 69,820,655 |
|  |  | Delay | 3.54 | -1.1% | 33/34 | 34 | 69,820,655 |
|  |  | Sequential LSTM | 3.56 | — | — | 34 | 69,820,655 |
| 3 | 2024-04 | XGBoost (MAE, + context) | 3.26 | -6.9% | 34/34 | 34 | 65,101,832 |
|  |  | Delay (hourly) | 3.33 | -5.6% | 34/34 | 34 | 65,101,832 |
|  |  | Delay | 3.48 | -1.0% | 32/34 | 34 | 65,101,832 |
|  |  | XGBoost (MAE, historical) | 3.49 | -1.7% | 30/34 | 34 | 65,101,832 |
|  |  | Sequential LSTM | 3.50 | — | — | 34 | 65,101,832 |
| 4 | 2024-05 | XGBoost (MAE, + context) | 3.31 | -6.5% | 35/35 | 35 | 70,531,899 |
|  |  | Delay (hourly) | 3.36 | -5.1% | 35/35 | 35 | 70,531,899 |
|  |  | XGBoost (MAE, historical) | 3.51 | -1.1% | 29/35 | 35 | 70,531,899 |
|  |  | Delay | 3.52 | -0.9% | 34/35 | 35 | 70,531,899 |
|  |  | Sequential LSTM | 3.58 | — | — | 35 | 70,531,899 |
| 5 | 2024-06 | XGBoost (MAE, + context) | 3.22 | -7.7% | 35/35 | 35 | 71,482,625 |
|  |  | Delay (hourly) | 3.30 | -4.2% | 34/35 | 35 | 71,482,625 |
|  |  | XGBoost (MAE, historical) | 3.41 | -2.3% | 32/35 | 35 | 71,482,625 |
|  |  | Delay | 3.42 | -1.0% | 33/35 | 35 | 71,482,625 |
|  |  | Sequential LSTM | 3.54 | — | — | 35 | 71,482,625 |
| 6 | 2024-07 | XGBoost (MAE, + context) | 3.11 | -5.2% | 34/35 | 35 | 76,316,248 |
|  |  | Delay (hourly) | 3.22 | -2.8% | 31/35 | 35 | 76,316,248 |
|  |  | XGBoost (MAE, historical) | 3.30 | -2.0% | 27/35 | 35 | 76,316,248 |
|  |  | Delay | 3.36 | -1.0% | 31/35 | 35 | 76,316,248 |
|  |  | Sequential LSTM | 3.42 | — | — | 35 | 76,316,248 |
| 7 | 2024-08 | XGBoost (MAE, + context) | 3.16 | -6.9% | 34/35 | 35 | 53,188,938 |
|  |  | Delay (hourly) | 3.28 | -4.8% | 35/35 | 35 | 53,188,938 |
|  |  | XGBoost (MAE, historical) | 3.38 | -2.1% | 32/35 | 35 | 53,188,938 |
|  |  | Delay | 3.42 | -0.7% | 33/35 | 35 | 53,188,938 |
|  |  | Sequential LSTM | 3.52 | — | — | 35 | 53,188,938 |
