**Appendix H. Replication: the same pipeline executed again.**

Two kinds of re-execution are compared with the published run. **E6, run A**, a separate cluster job inside the published bundle, re-fitted the four boosted models of the main comparison on the published frames in every route. **The two pilots** executed the whole chain from the raw AVL records — cleaning, round-trip cut, frames and every model — on four routes, and again on route 601; their bundles are deposited with the published one. A pair is a (route, model) row of the results; it is *identical* when the number of prediction pairs and all six error metrics agree exactly. The deterministic models are reproducible given the same frame, seeds and number of threads; the sequence model is not bit-reproducible on GPU, so it is measured instead.

| Execution | From | Outputs | Routes | Frames identical (hash) | Pairs | Identical | max \|ΔMAE\| (s) |
|---|---|---|---|---|---|---|---|
| E6, run A (published bundle) | the published frames | four boosted models | 35 | same files | 140 | 140 | 0.00 |
| pilot 1 (20261007_134313) | the raw AVL records | main comparison (nine models) | 4 | 8 of 8 | 36 | 36 | 0.00 |
| pilot 1 (20261007_134313) | the raw AVL records | information ablation (Table 5) | 4 | 8 of 8 | 44 | 44 | 0.00 |
| pilot 1 (20261007_134313) | the raw AVL records | rolling origin, tabular models | 4 | 8 of 8 | 36 | 36 | 0.00 |
| pilot 2 (20261007_210609) | the raw AVL records | main comparison (nine models) | 1 | 2 of 2 | 9 | 9 | 0.00 |
| pilot 2 (20261007_210609) | the raw AVL records | information ablation (Table 5) | 1 | 2 of 2 | 11 | 11 | 0.00 |
| pilot 2 (20261007_210609) | the raw AVL records | rolling origin, tabular models | 1 | 2 of 2 | 9 | 9 | 0.00 |

| Sequence model | Route | MAE published (s) | MAE pilot (s) | ΔMAE (s) | ΔMAE (%) | Δ on-time (pp) | Rank of ten, published → pilot |
|---|---|---|---|---|---|---|---|
| main comparison | 601 | 243.43 | 241.03 | +2.39 | +0.99 | -0.38 | 9 → 8 |
| main comparison | 902 | 137.56 | 136.78 | +0.78 | +0.57 | -1.36 | 8 → 8 |
| main comparison | 1601 | 158.08 | 157.14 | +0.94 | +0.60 | +0.51 | 10 → 10 |
| main comparison | 180101 | 226.23 | 226.78 | -0.54 | -0.24 | -0.36 | 8 → 8 |
| rolling origin | 601 | 241.58 | 241.26 | +0.32 | +0.13 | +0.06 | — |
| rolling origin | 902 | 136.38 | 137.08 | -0.70 | -0.51 | +1.82 | — |
| rolling origin | 1601 | 159.28 | 158.64 | +0.64 | +0.40 | -1.38 | — |
| rolling origin | 180101 | 223.99 | 224.08 | -0.09 | -0.04 | +0.95 | — |

*Note.* Deterministic models: 285 of 285 pairs identical, every frame hash equal. Sequence model: on the same frames and the same prediction pairs its MAE moves by at most 2.4 s (1.0%) and its on-time rate by at most 1.8 percentage points; in route 601 that is enough to change its rank among the ten models of the main comparison (rank 9 in the published run, 8 in the pilot). The pilots predate amendments E3 and E4 of the protocol: E3 leaves routes whose attention fits in one chunk unchanged, and the pilot of route 601 already carried it; E4 only re-bins the breakdown by horizon, not used here.
