**Table 5. What each source of information buys (35 routes; paired within one cross-validation).**

Each level adds one source to the level above it, and all four are fitted in the same cross-validation loop on the same round trips, cut positions and folds, so the increments are paired by construction. *Gain* is the median per-route change in MAE against the level above; *share* is that level's contribution to the total improvement from level 0 to level 3, read from panel medians.

| Level | What it adds | MAE (min) | Gain | Routes improved | *p* | Share of total |
|---|---|---:|---:|---:|---:|---:|
| 0 · current-trip only | the trip's own distance–time trace on the grid, and nothing estimated from other trips | 3.655 | — | — | — | — |
| 1 · + delay vs. mean profile | one historical hook: the trip's current deviation from the training fold's mean profile | 3.652 | +0.0% | 17/35 | 0.1633 | 1% |
| 2 · + historical conditioning | the full distribution of remaining time by position and horizon | 3.523 | -2.9% | 35/35 | < 0.0001 | 35% |
| 3 · + operational context | calendar, headway ahead and behind, dwell, recent speed and route position | 3.287 | -6.0% | 35/35 | < 0.0001 | 64% |

*Notes.* Going from level 0 to level 3 improves MAE by 8.5%. Of that change, historical conditioning — levels 1 and 2 together — accounts for 36%, and operational context for the rest. The step from level 0 to level 1 is not significant: `delay_vs_mu` is the trip's elapsed time minus a constant that depends only on the cut position, and both of those are already features at level 0, so it carries almost no information a tree can use. The comparison is of information, not of model class: all four levels are the same gradient-boosted learner under the same absolute-error objective.

For reference on the same panel, the sequence model reaches 3.640 min. Level 0 — which is given no historical summary and no context at all — is more accurate than it on 16 of 35 routes, and less accurate by a median 0.4% (*p* 0.2383). The sequence model receives the full trajectory as a sequence and the tabular model seven summaries of it, so this contrast is not a difference in what was observed but in how it was represented.
