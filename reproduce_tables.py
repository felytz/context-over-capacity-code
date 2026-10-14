#!/usr/bin/env python3
"""reproduce_tables.py — regenerate every table and figure of the article, then
report what does not match the committed copies.

No GPU, no cluster, no raw AVL records. One source, the bundle of a cluster run:

    python reproduce_tables.py --run artefacts/X    # a cluster run (run_pipeline.sh)
    python reproduce_tables.py                      # the published run (PUBLISHED)
    python reproduce_tables.py --list               # print the steps and exit

Outputs are written over the committed copies in `docs/paper/tables/`,
`docs/paper/figures/` and `docs/paper/analysis/`, and then compared with them
through git:

  * Markdown tables must be byte-identical.
  * CSV files must hold the same rows and columns; row order, column order and
    float formatting may differ (several generators collect files with `glob`,
    whose order depends on the file system).
  * PNG figures must be pixel-identical; PDF figures are compared as bytes,
    with SOURCE_DATE_EPOCH fixed so that matplotlib writes no wall-clock date.

The round trip (branch `vuelta`, docs/PREREGISTRO_VUELTA.md)
-----------------------------------------------------------
Every route is one unit, its round trip, and everything comes from the one bundle
that `run_pipeline.sh` leaves (extracted under `artefacts/`): the nine
deterministic models, the sequence model with honest early stopping
(`q05c100_es15`) for the headline and every breakdown, the information ablation,
the rolling-origin runs (`q05c100_tw`), the timing of the tabular models, the
data-funnel digest and the server-side inputs (fleet concentration, grid
representativity). The August run mixed four packages (July, 25-Aug, 18-Aug and the
19-Jul digest); it is reproduced from the branch `metodo-v2`.

Until the full run is committed as the reference (phase 5 of the plan),
`PUBLISHED` is empty and only `--run` works. Then the run's outputs become the
committed copies, `PUBLISHED` names its bundle, and `--run` with a later bundle is
also compared route by route with it (`compare_runs.py`): the nine deterministic
models must match; the sequence model is not bit-reproducible on GPU, so tables
that carry it differ in their last digits. `git checkout -- docs/paper` restores
the committed outputs afterwards.

`--partial` accepts a run of only some of the 35 routes (a pilot): the data-funnel
table says so at its foot instead of stopping.
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

for _s in (sys.stdout, sys.stderr):          # Windows consoles default to cp1252
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
ENV = {
    **os.environ,
    "PYTHONIOENCODING": "utf-8",   # generators print Unicode; a Windows pipe would use cp1252
    "PYTHONUTF8": "1",
    "MPLBACKEND": "Agg",
    "SOURCE_DATE_EPOCH": "1756684800",   # fixed PDF creation date → reproducible bytes
}
ANA = "docs/paper/analysis"
FIG = "docs/paper/figures"

#: The published run of the round trip: set in phase 5, when its outputs are committed
#: as the reference (the bundle under `artefacts/`, verified by `MANIFEST.sha256`).
#: The full run of 8-Oct-2026 (35 routes, code 10035cb), packaged on 9-Oct.
PUBLISHED: str | None = "artefacts/run_bundle_q05c100_20261009_171428"

# Inputs of the local steps that are computed on the cluster from the
# intermediate dataframes (not in the bundle's model outputs): they are taken from
# the run (path inside the bundle → path in the repository).
SERVER_INPUTS = [
    ("route_reports/fleet_concentration/fleet_concentration_2024_01-08_tmin.csv",
     f"{ANA}/fleet_concentration/fleet_concentration_2024_01-08_tmin.csv"),
    ("route_reports/fleet_concentration/fleet_concentration_2024_01-08_tmin.json",
     f"{ANA}/fleet_concentration/fleet_concentration_2024_01-08_tmin.json"),
    (f"{ANA}/grid_representativity.csv", f"{ANA}/grid_representativity.csv"),
    (f"{ANA}/grid_representativity.md", f"{ANA}/grid_representativity.md"),
]

# Route-by-route comparison of a run with the published one: regime tags.
COMPARISONS = [
    "q05c100",            # nine deterministic models
    "q05c100_es15",       # sequence model with honest early stopping
    "q05c100_exo_np_t0",  # information ablation
    "q05c100_tw",         # rolling origin (tabular + sequence model)
]


def steps(R: str, partial: bool = False) -> list[tuple[str, list[str]]]:
    """(article output(s), command). Order matters: later steps read earlier outputs."""
    part = ["--partial"] if partial else []
    return [
        ("route inclusion (pre-specified panel)", [f"{ANA}/leg_inclusion.py", "--root", R]),
        ("Table S2 · per-route data funnel", [f"{ANA}/make_table_01_funnel.py", "--root", R, *part]),
        ("Table 1 · data funnel by origin", [f"{ANA}/make_table_01_funnel_compact.py"]),
        ("Table 4 · on-time under three bands", [f"{ANA}/honest_ontime.py", "--root", R]),
        ("Table 2 · per-model panel", [f"{ANA}/make_table_05_permodel.py"]),
        ("Table 3 (upper) · paired tests", [f"{ANA}/make_tables_06_09_paired_stats.py"]),
        ("Table 3 (lower) · family-clustered bootstrap", [f"{ANA}/make_table_09b_clustered_bootstrap.py"]),
        ("Table S13 · the hourly lookup table against the boosted models",
         [f"{ANA}/make_table_16_lookup_vs_best.py"]),
        ("Table 5 · information ablation", [f"{ANA}/make_table_15_information_ablation.py", "--root", R]),
        ("Table 6 + Fig. 3 · rolling origin (constant panel)",
         [f"{ANA}/make_table_13_temporal.py", "--root", R, "--ref", "lstm_a"]),
        ("Table S15 · rolling origin (every included route)",
         [f"{ANA}/make_table_13_temporal.py", "--root", R, "--ref", "lstm_a", "--all-legs", "--no-figure"]),
        ("Table 7 + Fig. 4 · cost against accuracy", [f"{ANA}/make_table_12_cost.py", "--root", R]),
        ("Tables S3–S6 · appendices", [f"{ANA}/make_appendices.py"]),
        ("Table S9 (measured) · trees, epochs, sequence length, split gain",
         [f"{ANA}/make_appendix_f_model_fits.py", "--root", R]),
        ("Table S11 · per-route frame comparison", [f"{ANA}/make_table_04_frame.py", "--root", R]),
        ("Table S12 · symmetric error bands",
         [f"{ANA}/make_table_14_error_bands.py", "--root", R, "--all-models"]),
        ("Table S16 · by cut position", [f"{ANA}/make_table_07_by_position.py", "--root", R]),
        ("Table S17 · deficit correlates", [f"{ANA}/make_table_08_correlates.py", "--root", R]),
        ("Fig. 2, Figs. S1–S2 · results figures", [f"{ANA}/make_results_figures_frame.py", "--root", R]),
        ("Fig. S3 · error by horizon", [f"{ANA}/make_fig6_horizon_frame.py", "--root", R]),
        ("Fig. 1 · pipeline and the round-trip cut", [f"{FIG}/make_figure_01_merged.py"]),
        ("Table S7 · stretches of the geometry buses do not follow",
         [f"{ANA}/make_appendix_g_geometry.py"]),
    ]


# Only with the raw data in data/01_raw/: they read the AVL records or the geometry.
RAW_STEPS = [
    ("network funnel (Section 2.1)", [f"{ANA}/network_funnel.py"]),
    ("Table S1 · the variant routes", [f"{ANA}/variants_separation.py"]),
    ("cleaning, measured (Section 2.2)", [f"{ANA}/cleaning_stats.py", "--root", "<run>"]),
]
RAW_INPUTS = ["segments.geojson", "stops_routes.json", "segments_route.json",
              "polylines.geojson", "records"]

# Amendment E6 (docs/PREREGISTRO_VUELTA.md §6): the group-wise ablation (Table S14)
# and the sensitivity to the cap on trees, from `run_e6.sh`. They run only when the
# bundle carries that run's outputs (`*_ablate*`); until then Table S14 is the frozen
# August artefact — see artefacts/README.md.
E6_STEPS = [
    ("Table S14 · group-wise ablation of the contextual block (E6)",
     [f"{ANA}/make_table_11_ablation.py", "--root", "<run>"]),
    ("Table S8 · sensitivity to the cap on trees (E6)",
     [f"{ANA}/make_table_17_tree_cap.py", "--root", "<run>"]),
]


# V3 of the manuscript: the replication (Table S10). The two server pilots executed the
# chain from the raw AVL records on four routes and again on route 601; their bundles are
# deposited next to the published one (artefacts/README.md, MANIFEST.sha256). The table
# also reads E6's re-fit of the four boosted models, so it runs only with E6 and the pilots.
PILOTS = ["artefacts/run_bundle_q05c100_20261007_134313",
          "artefacts/run_bundle_q05c100_20261007_210609"]
REPLICATION_STEP = ("Table S10 · replication: E6 and the two pilots against the run",
                    [f"{ANA}/make_appendix_h_replication.py", "--root", "<run>",
                     "--pilots", *PILOTS])


def has_e6(run: Path) -> bool:
    return any(run.glob("data/02_interim/routes/*/xgb/*/*_ablate_overall.parquet"))


# ── comparison ─────────────────────────────────────────────────────────────────
def _git(*args: str, binary: bool = False):
    r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True)
    return r.stdout if binary else r.stdout.decode("utf-8", "replace")


def _csv_rows(text: str) -> tuple[list[str], list[tuple]]:
    rows = list(csv.reader(io.StringIO(text)))
    head, body = rows[0], rows[1:]
    order = sorted(range(len(head)), key=lambda i: head[i])

    def norm(v: str):
        try:
            return f"{float(v):.10g}"
        except ValueError:
            return v
    return [head[i] for i in order], sorted(tuple(norm(r[i]) for i in order) for r in body)


def _same_csv(old: str, new: str) -> bool:
    return _csv_rows(old) == _csv_rows(new)


def _same_png(old: bytes, new_path: Path) -> tuple[bool, str]:
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        return False, "cannot compare pixels (numpy/Pillow missing)"
    a = np.asarray(Image.open(io.BytesIO(old)).convert("RGBA"))
    b = np.asarray(Image.open(new_path).convert("RGBA"))
    if a.shape != b.shape:
        return False, f"size {a.shape[1]}×{a.shape[0]} → {b.shape[1]}×{b.shape[0]}"
    frac = float((a != b).any(axis=2).mean())
    return frac == 0.0, f"{100 * frac:.3f}% of pixels differ"


def compare() -> int:
    # Only outputs: the generators live in docs/paper/ too, and an edited script
    # is not a result. --diff-filter=d: the generators only ever write, so a file
    # deleted from the working tree is not an output of this run either.
    def outputs(listing: str) -> list[str]:
        return [p for p in listing.splitlines() if p and not p.endswith(".py")]
    changed = outputs(_git("diff", "--name-only", "--diff-filter=d", "--", "docs/paper"))
    new = outputs(_git("ls-files", "--others", "--exclude-standard", "--", "docs/paper"))
    real, benign = [], []
    for p in changed:
        old = _git("show", f"HEAD:{p}", binary=True)
        path = ROOT / p
        if p.endswith(".csv"):
            same = _same_csv(old.decode("utf-8"), path.read_text(encoding="utf-8"))
            (benign if same else real).append(
                (p, "same rows and columns; order or float formatting differs" if same
                 else "CSV content differs"))
        elif p.endswith(".png"):
            ok, msg = _same_png(old, path)
            (benign if ok else real).append((p, msg))
        elif p.endswith(".pdf"):
            real.append((p, "PDF bytes differ (check the PNG of the same figure)"))
        else:
            real.append((p, "text differs"))
    for p in new:
        real.append((p, "new file, not in the committed outputs"))

    if not real and not benign:
        print("✓ every regenerated output is identical to the committed one")
    if benign:
        print("✓ identical in content (these differ only in ordering or formatting):")
        for p, m in benign:
            print(f"    {p} — {m}")
    if real:
        print("✗ outputs that DIFFER from the committed copies:")
        for p, m in real:
            print(f"    {p} — {m}")
    return 1 if real else 0


# ── sources ────────────────────────────────────────────────────────────────────
def _fill_digest(src: Path) -> None:
    """make_table_01_funnel reads the per-route digest from results/digest/routes/.
    Emptied first, so that a digest from another source never mixes in."""
    dig = ROOT / "results" / "digest" / "routes"
    shutil.rmtree(dig, ignore_errors=True)
    dig.mkdir(parents=True)
    for js in src.glob("*.json"):
        shutil.copy2(js, dig / js.name)


def prepare_run(run: Path) -> list[str]:
    """Take the server-side inputs from the run. Returns what it lacked: those
    steps then read whatever copy is in the repository (reported at the end)."""
    missing = []
    for src, dst in SERVER_INPUTS:
        s = run / src
        if s.is_file():
            (ROOT / dst).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(s, ROOT / dst)
        else:
            missing.append(dst)
    digest = run / "route_reports" / "routes"
    if any(digest.glob("*.json")):
        _fill_digest(digest)
    else:
        missing.append("data-funnel digest (route_reports/routes/; Tables 1, S2)")
    return missing


def compare_with_published(run: str) -> None:
    if PUBLISHED is None:
        print("·  no published run of the round trip yet (phase 5): the route-by-route")
        print("   comparison with it is skipped.")
        return
    ref = PUBLISHED
    if not (ROOT / ref).is_dir():
        print(f"·  published run {ref} not found, comparison skipped")
        return
    panel = ROOT / ANA / "leg_inclusion.csv"
    for tag in COMPARISONS:
        r = subprocess.run(
            [sys.executable, f"{ANA}/compare_runs.py", "--a", ref, "--b", run,
             "--label-a", "paper", "--label-b", "run", "--tag", tag,
             "--panel", str(panel), "--top", "5"],
            cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
            errors="replace", env=ENV)
        print((r.stdout + r.stderr).rstrip())
        print()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default=None, metavar="DIR",
                    help="extracted bundle of a cluster run (run_pipeline.sh); "
                         "default: the published run")
    ap.add_argument("--partial", action="store_true",
                    help="a run of only some routes (a pilot): the funnel table "
                         "says so instead of stopping")
    ap.add_argument("--list", action="store_true", help="print the steps and exit")
    a = ap.parse_args()

    run = a.run or PUBLISHED
    raw_ok = all((ROOT / "data" / "01_raw" / f).exists() for f in RAW_INPUTS)
    raw = [(w, [(run or "<bundle>") if c == "<run>" else c for c in cmd])
           for w, cmd in RAW_STEPS]
    todo = steps(run or "<bundle>", a.partial) + (raw if raw_ok else [])
    if run is not None and has_e6(ROOT / run):
        todo += [(w, [run if c == "<run>" else c for c in cmd]) for w, cmd in E6_STEPS]
        if all((ROOT / p / "data" / "02_interim" / "routes").is_dir() for p in PILOTS):
            w, cmd = REPLICATION_STEP
            todo.append((w, [run if c == "<run>" else c for c in cmd]))
    if a.list:
        for i, (what, cmd) in enumerate(todo, 1):
            print(f"{i:2d}. {what}\n    python {' '.join(cmd)}")
        return 0
    if run is None:
        raise SystemExit(
            "ERROR: there is no published run of the round trip yet (phase 5 of the\n"
            "  plan sets PUBLISHED). Pass a cluster run: --run artefacts/run_bundle_<…>")
    if not (ROOT / run / "data" / "02_interim" / "routes").is_dir():
        raise SystemExit(f"ERROR: {run} does not look like an extracted run bundle "
                         f"(no data/02_interim/routes/ inside)")

    missing = prepare_run(ROOT / run)

    fallos = []
    for i, (what, cmd) in enumerate(todo, 1):
        t0 = time.time()
        r = subprocess.run([sys.executable, *cmd], cwd=ROOT, capture_output=True,
                           text=True, encoding="utf-8", errors="replace", env=ENV)
        ok = r.returncode == 0
        print(f"{'✓' if ok else '✗'} {i:2d}. {what}  ({time.time() - t0:.0f} s)")
        if not ok:
            fallos.append(what)
            print("    " + "\n    ".join((r.stdout + r.stderr).strip().splitlines()[-15:]))
    if not raw_ok:
        print("·  network funnel and Table S1 skipped: they need the raw data in data/01_raw/")

    print()
    if fallos:
        print(f"✗ {len(fallos)} step(s) failed: {fallos}")
    if missing:
        print("⚠ not in the run, so those steps read the copy in the repository:")
        for f in missing:
            print(f"    {f}")
    if not (ROOT / ".git").exists() or shutil.which("git") is None:
        print("(no git checkout: outputs regenerated but not compared)")
        return 1 if fallos else 0
    distinto = compare()          # always compare, even after a failed step

    if a.run is None:
        return 1 if (fallos or distinto) else 0

    print("\n  Tables and figures that carry the sequence model differ in their last")
    print("  digits (GPU non-determinism). The route-by-route comparison below separates")
    print("  that from a real difference: the deterministic models must match.\n")
    compare_with_published(a.run)
    print("  To restore the committed outputs:  git checkout -- docs/paper")
    print("  (it also discards any uncommitted edit to the scripts under docs/paper/)")
    return 1 if fallos else 0


if __name__ == "__main__":
    raise SystemExit(main())
