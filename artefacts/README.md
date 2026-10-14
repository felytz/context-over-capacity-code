# `artefacts/` — the derived artefacts

Every table and figure of the article is computed from **one bundle**: the output of
`run_pipeline.sh`, plus the outputs of amendment E6 (`run_e6.sh`: Tables S14 and S8),
already merged into it. Table S10 (the replication) also reads the two pilot bundles.
The three bundles (≈ 115 MB) are not in this repository: they are in the Zenodo record
of this software, as `artefacts_q05c100.tgz`. Extract it here, from the repository root:

```bash
tar -xzf artefacts_q05c100.tgz -C artefacts/
```

so that the tree reads

```
artefacts/
├── run_bundle_q05c100_20261007_134313/       8 MB ·   311 files
├── run_bundle_q05c100_20261007_210609/       3 MB ·   122 files
├── run_bundle_q05c100_20261009_171428/     104 MB · 3,242 files
└── MANIFEST.sha256                          (in this repository)
```

and verify it:

```bash
sha256sum -c artefacts/MANIFEST.sha256
```

A later cluster run leaves
another bundle with the same layout; extract it here as well and pass it to
`python reproduce_tables.py --run artefacts/run_bundle_q05c100_<date>`, which then also
compares it route by route with this one.

## What each package is

| Package | Run | Used for |
|---|---|---|
| `run_bundle_q05c100_20261009_171428` | the full run of the round trip, 35 routes, January–August 2024 (`run_pipeline.sh`, this code before amendment E6, 8–9 Oct 2026): the data funnel, the common evaluation frame and the nine deterministic models (`q05c100`), the sequence model with honest early stopping (`q05c100_es15`), the information ablation (`q05c100_exo_np_t0`), the rolling-origin runs (`q05c100_tw`), the per-model timings, the per-route digest, fleet concentration and grid representativity | every table and figure |
| ↳ E6, inside the bundle (`*_ablate*`, `logs/xgb_*_2078729_*`, `logs/xgb_*_2078730_*`, `logs/e6_sacct.txt`, `E6_SHA256SUMS`) | amendment E6 (pre-registration §6; `run_e6.sh`, this code, 9–10 Oct 2026, CPU only, 16 threads): run A, the group-wise ablation in the common frame (`_ablate`, 600 trees), and run B, the same with a cap of 2,000 trees (`_ablate_np_t0_n2000`) | Table S14, the context descriptives of Section 4, Table S8 |
| `run_bundle_q05c100_20261007_134313` | pilot 1 on the cluster: routes 601, 902, 1601 and 180101, January–August 2024, the whole chain from the raw records (`run_pipeline.sh`, an earlier version of this code, 7 Oct 2026, before amendments E3–E4). The sequence model of route 601 failed there (the attention tensor exceeded 2^31 elements; that is amendment E3) | Table S10 |
| `run_bundle_q05c100_20261007_210609` | pilot 2: route 601 again, with E3 (7 Oct 2026), data stage included | Table S10 |

The bundle keeps the layout the pipeline writes
(`data/02_interim/routes/<route>/{frames,xgb,lstm_frame,perf}/2024_01-08_tmin/`), the
per-route digest (`route_reports/routes/<route>.json`), fleet concentration
(`route_reports/fleet_concentration/`), grid representativity
(`docs/paper/analysis/grid_representativity.*`), every job log, and the exact
`config.env` and environment of the run (`RUNINFO_q05c100.txt`).

## Things worth knowing

* **Round trips: 317,983**, with numpy 1.26.4 and scipy 1.15.3 (pinned by TensorFlow
  2.18 in `requirements-server.txt`). The validation of the trip cut, run with numpy
  2.4.6 and scipy 1.17.1, gives 317,968 from the same code and raw records: route 101
  gains 15 round trips and route 901 splits other segments. Each environment is
  deterministic (pre-registration, §3).
* **The sequence model is not bit-reproducible on GPU.** With the same frame, folds
  and seed, its per-route MAE moved by up to 1 % between the pilot runs and this one
  (route 601, `es15`: 241.0 → 243.4 s, enough to swap its rank with kNN). Frames and
  every deterministic model came out bit-identical in the pilot routes: 285 of 285
  (route, model) rows, Table S10 (`make_appendix_h_replication.py`).
* **The appendices job also ran the August sensitivity analyses of the terminal
  detector** (`docs/paper/analysis/terminal_eps_sensitivity.*`,
  `reversal_params_sensitivity.csv` in the bundle). They do not apply to the round
  trip and no table reads them.
* **E6 depends on the thread count.** XGBoost sums histograms in parallel; with 16
  threads, as in the main run, the four models E6 shares with it (`xgb_mse`, `xgb_mae`,
  `xgb_mse_b`, `xgb_mae_b`) reproduce it exactly in all 35 routes, and
  `make_table_11_ablation.py` refuses to write Table S14 otherwise. With 8 against 12
  threads they differ by up to 0.14 s of MAE.
* No package contains vehicle positions or licence plates. Vehicles appear only as
  `bus_code`, a pseudonymous operational code. The per-route digest records the raw
  schema (column names and types), not sample rows. The only coordinates are those of
  each route's cut point (also in `cortes/`) and, in the data-stage logs, of the points
  where buses usually pause, aggregated over all round trips of the route.
* **The cluster account details were replaced before deposit.** In the logs, JSON
  files, `config.env` and `RUNINFO_*.txt` of the three bundles, as in the code, the
  cluster user name reads `usuario`, the Slurm account `pXXX` and the node names
  `nodo-cpu-NN` / `nodo-gpu-NN` (same numbers). Nothing else was edited; no Parquet
  file contains them, and `MANIFEST.sha256` is that of the edited files.
