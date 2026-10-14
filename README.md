# Code, protocol and published run for *Context over capacity*

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23382640.svg)](https://doi.org/10.5281/zenodo.23382640)

This repository accompanies the article

> F. Robles-Andrade, F. Cirett-Galán, R. Torres-Peralta. *Context over capacity:
> engineered features and operational covariates for bus arrival-time prediction
> along the route in a Mexican concessioned network.* Unpublished manuscript.

| Deposit | What | Where |
|---|---|---|
| **Software** (this repository) | code, pre-registered protocol, published tables and figures | Zenodo, https://doi.org/10.5281/zenodo.23382640 (MIT); mirror on GitHub |
| **Run bundle** | the derived artefacts of the published run (`artefacts/`) | the same Zenodo record, `artefacts_q05c100.tgz` (CC BY 4.0) |
| **Data** | AVL records, processed records, round trips, route geometry | Zenodo, *Hermosillo bus AVL records, route geometry and round trips, January–August 2024*, https://doi.org/10.5281/zenodo.23384907 (CC BY 4.0) |

It holds everything the article's Code availability statement promises: the
trip-construction stage, the common evaluation frame, the ten models, and the
analysis scripts that generate every table and figure, plus the published
outputs those scripts are checked against.

Two commands do everything:

| | Where | Command | Produces |
|---|---|---|---|
| 1 | SLURM cluster, AMD GPU | `bash run_pipeline.sh` | one bundle with every derived artefact, from the raw AVL records |
| 2 | any computer, no GPU, minutes | `python reproduce_tables.py [--run <bundle>]` | every table and figure, compared with the published ones |

Command 2 runs from the published artefacts by default, so it does not need command 1.

> **The round trip.** The article uses each route's round trip as the unit
> (`docs/PREREGISTRO_VUELTA.md`). Its generators read one cluster bundle, the full
> run of 35 routes (`PUBLISHED` in `reproduce_tables.py`,
> `artefacts/run_bundle_q05c100_20261009_171428`), and the outputs under
> `docs/paper/` are that run's. An earlier analysis (August 2026, one-way legs between
> terminals) is mentioned in some comments; it is not part of the article and is not
> reproduced here.

| | |
|---|---|
| `run_pipeline.sh`, `config.env`, `jobs/`, `func/` | the pipeline that produced the derived artefacts |
| `reproduce_tables.py`, `docs/paper/analysis/`, `docs/paper/figures/make_*.py` | the generators of every table and figure |
| `docs/paper/tables/`, `docs/paper/figures/` | the published outputs, against which every run is compared |
| `artefacts/` | the derived artefacts (from the Zenodo record, see `artefacts/README.md`) |
| `data/01_raw/` | the raw AVL records and route geometry (from the data deposit, see `data/01_raw/README.md` and `prepare_raw_from_deposit.py`) |
| `docs/PREREGISTRO_VUELTA.md`, `docs/PREREGISTRATION_HISTORY.md` | the pre-registered protocol, its amendments E1–E6, and when each was fixed |

Code comments, the pre-registration and the operational manual (`RUNBOOK.md`,
`ENVIRONMENT.md`) are in Spanish, the working language of the project. They were
written in the authors' working copy and sometimes name things outside this
repository: commit hashes, branches (`metodo-v2`, `vuelta`), the project folder
(`../test/…`, `../Plan_nueva_corrida_articulo.md`, `data raw/…`) or paths on the
cluster (`/lustre/home/…`). They are kept as written, as a record; nothing in
Sections 1 and 2 needs them. Table and figure numbers inside code comments follow
earlier drafts of the article; the article's numbers are those of the map in
Section 4 and of each output's caption. Only the cluster user name, the Slurm account and the
node names were replaced, here and in the run bundle's logs (`usuario`, `pXXX`,
`nodo-cpu-NN` / `nodo-gpu-NN`; `artefacts/README.md`). The SLURM headers of `jobs/`
(account, partitions, interpreter path) are those of the cluster the run used and
must be adapted.

## 1 · Reproduce the tables and figures (minutes, no GPU)

1. Python 3.11 with the packages of `requirements-analysis.txt` (the versions with
   which every output reproduced): `pip install -r requirements-analysis.txt`.
2. Download `artefacts_q05c100.tgz` from the Zenodo record, extract it into
   `artefacts/` and verify it:

   ```bash
   tar -xzf artefacts_q05c100.tgz -C artefacts/
   sha256sum -c artefacts/MANIFEST.sha256
   ```

3. Run:

   ```bash
   python reproduce_tables.py
   ```

Every step reads only `artefacts/` and the small CSVs in `docs/paper/analysis/`,
writes over the committed outputs, and compares. Markdown tables must come out
byte-identical; CSV files must hold the same rows (several generators collect
files with `glob`, so row order and the last digit of a float can vary between
runs); PNG figures must be pixel-identical. A faithful run ends with

```
✓ identical in content (these differ only in ordering or formatting): …
```

and no `✗` line. `python reproduce_tables.py --list` prints the steps.

The comparison uses git to find the published version of each output, so run it
in a clone of the GitHub repository. From the Zenodo zip, make the unzipped
folder a repository first (`git init && git add -A && git commit -m published`);
without git the outputs are regenerated but not compared.

## 2 · Re-run the pipeline from the raw records (cluster)

First download `avl_records.zip` and `geometry.zip` from the data deposit on Zenodo
(https://doi.org/10.5281/zenodo.23384907), extract both into one folder, and copy the raw records and the
corrected geometry into the layout the pipeline reads:

```bash
python prepare_raw_from_deposit.py path/to/that/folder
```

`bash run_pipeline.sh` queues every job the article needs at once, chained by
SLURM dependencies route by route: the data stage, the common evaluation frame
with the tabular models (plus the information ablation and the rolling-origin
runs), the two runs of the sequence model (honest early stopping, which the
headline and every breakdown read, and rolling origin), the data funnel, fleet
concentration and appendices, and a final packaging job. On the round trip (35
routes) it took 114 GPU-hours (the sequence model) and about 30 hours of CPU jobs. The resulting bundle goes into
`artefacts/`, and

```bash
python reproduce_tables.py --run artefacts/run_bundle_q05c100_<date>
```

regenerates every output from it, compares them with the published ones and
compares the run leg by leg with the published packages. `RUNBOOK.md` is the
manual.

For the August run, an independent re-run from the raw records reproduced 375 of
the 378 (leg, model) pairs of the nine deterministic models bit for bit. On the
round trip, the full run reproduced the frames and the deterministic models of the
four pilot routes bit for bit. **The sequence model is not
bit-reproducible on GPU**: with frame, folds and seeds fixed, its per-leg MAE
moves by a few seconds between runs, so tables that carry it differ in their last
digits, and the article never rests a claim on its rank.

## 3 · Data and artefacts

* **Raw AVL records, route polylines and stop inventories**: collected from the
  public web portal of UNE, Hermosillo's integrated transport system
  (https://unesonora.com), and deposited on Zenodo (https://doi.org/10.5281/zenodo.23384907)
  together with the processed records and the round trips of the published run.
  They are published **without the licence-plate column**, which the analysis does
  not use, and with only the raw columns the pipeline reads (`RAW_COLUMNS` in
  `func/steps/step_01_load_data.py`). The geometry files are byte-identical to
  those fixed in the pre-registration (§1); the record files are not, because of
  the dropped columns and because they are cut to the analysis window, but within
  that window they hold the same records and values in the columns the pipeline
  reads.
* **Derived artefacts** (evaluation frames, per-model error summaries, deviation
  histograms, by-horizon breakdowns, the data-funnel digest): they contain no
  vehicle positions and no plate numbers. Each table and figure is computed from
  them, so Section 1 needs no raw data at all.

The published copies were derived from the private originals by dropping the
plate column, and the plate field from the digest's sample rows; nothing else
was changed.

## 4 · Where each output comes from

On the round trip every table and figure is computed from **one bundle**, the
output of `run_pipeline.sh` (the nine deterministic models, the sequence model
with honest early stopping `q05c100_es15`, the information ablation, the
rolling-origin runs `q05c100_tw`, the per-model timings, the data-funnel digest),
plus the small versioned inputs under `cortes/` and `docs/paper/analysis/`.
File names keep the numbering of an earlier draft; the table above maps them to
the article, and each file's caption carries the article's number.

| Article | File in `docs/paper/` | Generator | Inputs |
|---|---|---|---|
| Table 1 | `tables/table_01_funnel_compact.md` | `make_table_01_funnel_compact.py` | `table_01_funnel.csv` |
| Table 2 | `tables/table_05_permodel_panel.md` | `make_table_05_permodel.py` | `table_04_frame_comparison.csv` |
| Table 3 | `tables/table_06_paired_tests.md` (upper), `tables/table_09b_clustered_bootstrap.md` (lower) | `make_tables_06_09_paired_stats.py`, `make_table_09b_clustered_bootstrap.py` | `table_04_frame_comparison.csv` |
| Table 4 | `tables/table_10_ontime_honest.md` | `honest_ontime.py` | bundle |
| Table 5 | `tables/table_15_information_ablation.md` | `make_table_15_information_ablation.py` | bundle |
| Table 6, Fig. 3 | `tables/table_13_temporal_ladder.md`, `figures/fig_results_7_temporal_ladder` | `make_table_13_temporal.py` | bundle |
| Table 7, Fig. 4 | `tables/table_12_cost_accuracy.md`, `figures/fig_results_6_cost_accuracy` | `make_table_12_cost.py` | bundle |
| Table S1 | `tables/appendix_e_variants.md` | `variants_separation.py` | raw records + `variants_*.csv` |
| Table S2 | `tables/table_01_funnel.md` | `make_table_01_funnel.py` | bundle (digest, frames) + `route_catalogue.csv` |
| Table S3 | `tables/appendix_a_grid_representativity.md` | `make_appendices.py` | `grid_representativity.csv` (cluster) |
| Table S4 | `tables/appendix_c_cut_points.md` | `make_appendices.py` | `cortes/` + `table_01_funnel.csv` |
| Table S5 | `tables/appendix_d_cut_invariance.md` | `make_appendices.py` | `cut_invariance_*.csv` |
| Table S6 | `tables/appendix_b_excluded_legs.md` | `make_appendices.py` | `leg_inclusion.csv` |
| Table S7 | `tables/appendix_g_geometry.md` | `make_appendix_g_geometry.py` | `geometry_stretches/` (from the trace maps of the validation, `analisis/tramos_entrada_articulo.py`) |
| Table S8 | `tables/table_17_tree_cap.md` | `make_table_17_tree_cap.py` | bundle (the E6 run, `*_ablate_np_t0_n2000*`) |
| Table S9 | written in the Online Resource; the measured part, `tables/appendix_f_model_fits.md` | `make_appendix_f_model_fits.py` | bundle (metas, importances, LSTM logs) |
| Table S10 | `tables/appendix_h_replication.md` (+ two CSV) | `make_appendix_h_replication.py` | bundle (E6 run A) + the two pilot bundles (`PILOTS` in `reproduce_tables.py`) |
| Table S11 | `tables/table_04_frame_comparison.md` | `make_table_04_frame.py` | bundle |
| Table S12 | `tables/table_14_error_bands.md` | `make_table_14_error_bands.py --all-models` | bundle |
| Table S13 | `tables/table_16_lookup_vs_best.md` | `make_table_16_lookup_vs_best.py` | `table_04_frame_comparison.csv` |
| Table S14 | `tables/table_11_ablation.md` (+ `analysis/context_descriptives.csv`) | `make_table_11_ablation.py` | bundle (the E6 run, `*_ablate*`) |
| Table S15 | `tables/table_13_temporal_ladder_all.md` | `make_table_13_temporal.py --all-legs` | bundle |
| Table S16, Fig. S2 | `tables/table_07_by_position.md`, `figures/fig_results_2_by_position` | `make_table_07_by_position.py`, `make_results_figures_frame.py` | bundle + `cortes/` |
| Table S17 | `tables/table_08_correlates.md` | `make_table_08_correlates.py` | `table_04_frame_comparison.csv` + bundle |
| Section 2.2 (cleaning) | `analysis/cleaning_stats.md` | `cleaning_stats.py` | raw records + bundle (funnels) |
| Fig. 1 | `figures/figure_01_merged` | `make_figure_01_merged.py` | `figure_01b_round_trips.{csv,json}` |
| Fig. 2, Fig. S1 | `figures/fig_results_1_dumbbell`, `figures/fig_results_5_ontime_decomposition` | `make_results_figures_frame.py` | `table_04_frame_comparison.csv` |
| Fig. S3 | `figures/fig_results_4_by_horizon` | `make_fig6_horizon_frame.py` | bundle |

`grid_representativity.csv` (behind Table S3) and the fleet concentration
(Table S2) read the intermediate dataframes and run on the cluster
(`jobs/job_appendices.slrm`, `jobs/job_fleet_concentration.slrm`); the bundle
carries their output. Three inputs are evidence copied byte for byte from the
validation of the round trip, with their sha256 in the generators:
`cut_invariance_rest_point.csv` and `cut_invariance_methods.csv` (Table S5) and
`variants_validation.csv` (Table S1). `figure_01b_round_trips.*` is written by
`docs/paper/figures/extract_figure_01b.py`, which runs the pipeline's trip cut on
one vehicle-day of a local data run; it carries neither coordinates nor vehicle
codes. Tables S3 and S4 of the August run (the sensitivity of the terminal
detector, `terminal_eps_sensitivity.py`, `reversal_params_sensitivity.py`) no
longer apply: on the round trip that detector does not cut.

## 5 · Differences from the code that produced the artefacts

The pipeline code is the one that produced the published artefacts, with these
changes and no others:

1. The dashboard step of the pipeline (step 8, HTML dashboards) is not included:
   it produces nothing the article uses, and every published run skipped it with
   `--no-dashboards`. The flag is still accepted.
2. `make_results_figures_frame.py` sorts legs before bootstrapping the intervals
   of Fig. S2. Its input came from a `group_by` without a fixed order, so the
   seeded bootstrap drew different replicates on every run; medians are unchanged.
3. Added: `reproduce_tables.py`, `jobs/job_pack.slrm` and `prepare_raw_from_deposit.py`. `run_pipeline.sh` now
   also queues the runs that were submitted by hand (information ablation,
   rolling origin, the three runs of the sequence model) and the auxiliary jobs,
   with the arguments recorded in the logs of the published bundles. The only
   addition is `--no-frames` on the ablation run, so that it does not rewrite
   the evaluation frame while the sequence model reads it; it computes the same
   frame and changes no number. `requirements-server.txt` pins xgboost and
   scikit-learn to the versions those logs record.
4. Not included, because they produce nothing the article uses and none is on
   the run path: notebooks, the test suite, diagnostic and verification scripts,
   the steps of an earlier direction-based method, the count of distinct
   licence plates, and the export scripts that derived the published copies of
   the data.
5. Step 7 keeps in `df_full` only the columns the models read (21, plus `leg`):
   the historical table (`hist_p50`, …), the interpolation diagnostics and the
   unused segment features are no longer computed or stored, and the
   post-prediction audit is gone. The loader reads only the raw columns the
   pipeline uses, so the plate field never enters it, and the per-route digest
   no longer copies sample rows of the raw records. None of this changes a
   number.

## 6 · Known limitations

* **Table S14 and Table S8 come from a second run** (amendment E6, `run_e6.sh`,
  CPU only, 9–10 Oct 2026), whose outputs were added to the published bundle
  (files with the suffix `_ablate`; nothing of the main run was touched). The
  four models it shares with the main run reproduce it exactly, which needs the
  same number of threads (16): XGBoost sums histograms in parallel, and with
  another thread count its trees change in the last digit.
* **The separation of the variant routes cannot be re-run from the published
  data.** Four AVL route codes (1101, 1400, 1801, 1802) carry two services each;
  `func/variantes/analisis/variantes_ruta.py` assigned every vehicle-day to one of
  them, producing the raw input of the eight variant routes, whose checksums the
  pre-registration fixes (§1). That script identifies vehicles by licence plate,
  which is not published (plates and unit codes are not one to one). The article
  only reads its output, and the data deposit carries, on every record of those
  routes, the assigned variant and the evidence for it (`variante`, `bloque`,
  `bloque_evidencia`), so the separation can be inspected and everything
  downstream reproduced; re-running the script itself needs the original
  database, held by F. Cirett-Galán.
* The network funnel of Section 2.1 (`network_funnel.py`) and Table S1
  (`variants_separation.py`) need the raw records in `data/01_raw/`;
  `reproduce_tables.py` runs them only when they are there.
* The error-by-horizon breakdown (Fig. S3) uses the horizon bins fixed in the
  pipeline (`func/honest_metrics.py`: 0–2, 2–5, 5–10, 10–20, 20–40, 40–60, 60–90
  and > 90 min). The first five are those of the August run; 60 and 90 were added
  by amendment E4 of the pre-registration, because a round trip lasts 60–180 min
  and "> 40 min" alone held about half of the prediction pairs. The August
  "> 40 min" bin is recovered exactly by pooling the last three (MAE, RMSE, bias
  and on-time, weighting by pair count). The figure prints each bin's share of
  pairs under it.

## License and citation

The code is released under the MIT licence (`LICENSE`); the run bundle and the
tables and figures under `docs/paper/`, under CC BY 4.0. If you use them, please
cite the software deposit (`CITATION.cff`, https://doi.org/10.5281/zenodo.23382640), the
dataset deposit (https://doi.org/10.5281/zenodo.23384907), and the article.
