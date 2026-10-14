**Table 6. Rolling-origin evaluation: train on the months so far, predict the next one (33 routes).**

Each rung fits every model on all round trips up to a month and scores it on the following month; the first month is training only, so eight months give seven rungs. Within a rung every model sees the same round trips, cut positions and horizons, so the comparison is paired by construction. **Across rungs it is not**: each rung scores a different month, and months differ in difficulty, so a change in absolute MAE from one rung to the next is not degradation. The comparable quantity along the ladder is the relative difference, reported as *Δ vs Sequential LSTM*. Training size also grows by construction, so any trend mixes more data with a different month; this design does not separate the two. Rungs are calendar months: a route whose records start later enters the ladder at the first month it can be scored on. The panel is restricted to routes reaching 100 training round trips at **every** rung, which is the only set that stays constant along the ladder. 2 included routes fall short: 101 (88 training round trips at its first rung); 901 (its records start later: rungs 2024-05 to 2024-08 only).


| Rung | Test month | Model | MAE (min) | Δ vs ref | Beats ref | Routes | Pairs |
|---|---|---|---:|---:|---:|---:|---:|
| 1 | 2024-02 | XGBoost (MAE, + context) | 3.60 | -6.8% | 33/33 | 33 | 65,273,352 |
|  |  | Delay (hourly) | 3.63 | -3.9% | 31/33 | 33 | 65,273,352 |
|  |  | XGBoost (MAE, historical) | 3.77 | -1.6% | 29/33 | 33 | 65,273,352 |
|  |  | Delay | 3.78 | -1.5% | 28/33 | 33 | 65,273,352 |
|  |  | Sequential LSTM | 3.87 | — | — | 33 | 65,273,352 |
| 2 | 2024-03 | XGBoost (MAE, + context) | 3.33 | -7.2% | 33/33 | 33 | 67,186,352 |
|  |  | Delay (hourly) | 3.41 | -4.4% | 32/33 | 33 | 67,186,352 |
|  |  | XGBoost (MAE, historical) | 3.52 | -2.0% | 31/33 | 33 | 67,186,352 |
|  |  | Delay | 3.54 | -1.1% | 32/33 | 33 | 67,186,352 |
|  |  | Sequential LSTM | 3.56 | — | — | 33 | 67,186,352 |
| 3 | 2024-04 | XGBoost (MAE, + context) | 3.19 | -6.9% | 33/33 | 33 | 62,393,739 |
|  |  | Delay (hourly) | 3.30 | -5.8% | 33/33 | 33 | 62,393,739 |
|  |  | Delay | 3.48 | -1.0% | 31/33 | 33 | 62,393,739 |
|  |  | XGBoost (MAE, historical) | 3.48 | -1.8% | 29/33 | 33 | 62,393,739 |
|  |  | Sequential LSTM | 3.50 | — | — | 33 | 62,393,739 |
| 4 | 2024-05 | XGBoost (MAE, + context) | 3.31 | -6.5% | 33/33 | 33 | 64,283,510 |
|  |  | Delay (hourly) | 3.36 | -4.9% | 33/33 | 33 | 64,283,510 |
|  |  | XGBoost (MAE, historical) | 3.51 | -1.1% | 27/33 | 33 | 64,283,510 |
|  |  | Delay | 3.52 | -0.8% | 32/33 | 33 | 64,283,510 |
|  |  | Sequential LSTM | 3.58 | — | — | 33 | 64,283,510 |
| 5 | 2024-06 | XGBoost (MAE, + context) | 3.22 | -7.8% | 33/33 | 33 | 66,109,840 |
|  |  | Delay (hourly) | 3.30 | -4.2% | 32/33 | 33 | 66,109,840 |
|  |  | XGBoost (MAE, historical) | 3.41 | -2.4% | 31/33 | 33 | 66,109,840 |
|  |  | Delay | 3.42 | -1.0% | 31/33 | 33 | 66,109,840 |
|  |  | Sequential LSTM | 3.54 | — | — | 33 | 66,109,840 |
| 6 | 2024-07 | XGBoost (MAE, + context) | 3.11 | -5.2% | 32/33 | 33 | 70,390,454 |
|  |  | Delay (hourly) | 3.22 | -2.8% | 30/33 | 33 | 70,390,454 |
|  |  | XGBoost (MAE, historical) | 3.30 | -2.0% | 26/33 | 33 | 70,390,454 |
|  |  | Delay | 3.36 | -1.0% | 30/33 | 33 | 70,390,454 |
|  |  | Sequential LSTM | 3.42 | — | — | 33 | 70,390,454 |
| 7 | 2024-08 | XGBoost (MAE, + context) | 3.16 | -7.0% | 32/33 | 33 | 48,764,450 |
|  |  | Delay (hourly) | 3.28 | -4.8% | 33/33 | 33 | 48,764,450 |
|  |  | XGBoost (MAE, historical) | 3.38 | -2.1% | 31/33 | 33 | 48,764,450 |
|  |  | Delay | 3.42 | -0.8% | 31/33 | 33 | 48,764,450 |
|  |  | Sequential LSTM | 3.52 | — | — | 33 | 48,764,450 |
