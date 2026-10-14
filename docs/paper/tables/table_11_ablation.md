**Table S14. Group-wise ablation of the contextual block (35 routes; tier A plus one group at a time).**

Every variant was fitted in the same cross-validation loop of the common evaluation frame, on the same round trips, cut positions and folds, so the differences are paired by construction. Deltas are medians over routes of the per-route relative change in MAE against tier A (median 211.4 s). *Routes improved* counts routes where the group lowers MAE. Holm adjustment is across the five groups. The share column expresses each group median as a fraction of the full tier-B improvement (-6.04%). The four models this run shares with the main one reproduce it exactly (|ΔMAE| = 0 in every route).

| Group | Features | MAE median (s) | Δ vs tier A | Routes improved | p | p (Holm) | Share of tier-B gain |
|---|---|---|---|---|---|---|---|
| Calendar | hour of day (cyclical), day of week, weekend flag | 198.24 | -5.31% | 35/35 | < 0.0001 | < 0.0001 | 88% |
| Headway | headway ahead and behind (capped at 3,600 s) | 209.94 | -0.61% | 34/35 | < 0.0001 | < 0.0001 | 10% |
| Route position | segment count ahead, observed ping count | 210.51 | -0.38% | 34/35 | < 0.0001 | < 0.0001 | 6% |
| Recent speed | recent GPS speed and its dispersion | 211.24 | -0.24% | 32/35 | < 0.0001 | < 0.0001 | 4% |
| Dwell | cumulative and recent dwell | 211.61 | -0.05% | 23/35 | 0.0030 | 0.0030 | 1% |
| **Full tier B** | all twelve | 197.22 | -6.04% | 35/35 | — | — | 100% |

*Notes.* The shares need not add to 100%: the groups overlap in what they tell the model, and per-route increments are not additive.
