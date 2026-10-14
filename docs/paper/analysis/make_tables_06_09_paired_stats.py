"""
make_tables_06_09_paired_stats.py
=================================
Tabla 3 (arriba) del artículo y una comprobación fuera de él (archivos `table_06` y `table_09`) — inferencia pareada sobre el frame común.

  **Tabla 6** · Wilcoxon de rangos con signo, LSTM contra cada alternativa,
                corregido por Holm. → `table_06_paired_tests.{csv,md}`
  **Tabla 9** · IC bootstrap al 95% de las cantidades resumen.
                → `table_09_bootstrap_ci.{csv,md}`

Salen del mismo objeto (`table_10_ontime_honest.csv`) que las Tablas 5 y 10, así
que las cuatro no pueden discrepar entre sí.

Qué cambia respecto de las versiones del régimen anterior
---------------------------------------------------------
1. **La unidad de remuestreo y de test es la LEG**, no la ruta. Es la unidad del
   método y la del criterio de inclusión pre-especificado; además da 42 unidades
   en vez de 22, lo que importa en pruebas no paramétricas (con n=13 el Wilcoxon
   bilateral no puede bajar de p≈0.0002 aunque el efecto sea perfecto).
2. **Nueve alternativas**, no cuatro: entran XGBoost (4 variantes) y `delay_hour`.
3. **Holm bien aplicado.** La versión anterior multiplicaba por `(m − rank)` sin
   imponer monotonía, que es el paso que le falta al procedimiento escalonado:
   los p ajustados tienen que ser no decrecientes en el orden de los p crudos.
   Sin ese máximo acumulado un test podía salir con un p ajustado MENOR que otro
   más significativo (en la tabla vieja, `knn` 0.5417 frente a `delay` 0.9097).
   Aquí se usa el acumulado, así que los números no coinciden con los de antes
   y el motivo es que los de antes estaban mal.
4. **El on-time se testea CENTRADO** (banda simétrica de la misma anchura, con el
   residuo centrado fuera de fold), que es dispersión pura. Testear la banda
   operacional cruda mezclaría precisión con centrado — ver Tabla 10.

Convención de signo: `mediana(LSTM − alternativa)`. Negativo en MAE y positivo en
on-time significan **LSTM mejor**.

Con la vuelta (docs/PREREGISTRO_VUELTA.md §4–§5) cada ruta es una leg, así que la unidad
de prueba y de remuestreo es la ruta, sobre un solo panel. El bootstrap que respeta las
familias con variante es el de la Tabla 9b.

Uso
---
    python docs/paper/analysis/honest_ontime.py --root artefacts/run_bundle_<…>
    python docs/paper/analysis/make_tables_06_09_paired_stats.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import numpy as np                                              # noqa: E402
import polars as pl                                             # noqa: E402
from scipy.stats import wilcoxon                                # noqa: E402

from make_table_05_permodel import LABEL, PANELS, in_panel      # noqa: E402

REF = "lstm_a"
ALTS = ["xgb_mae_b", "xgb_mae", "xgb_mse_b", "xgb_mse",
        "delay_hour", "delay", "kernel", "knn", "linreg"]
N_BOOT = 10_000
SEED = 42


# ──────────────────────────────────────────────────────────────────

def holm(pvals: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni escalonado, CON monotonía.

    p_(i) ajustado = max sobre j<=i de (m - j) * p_(j), truncado a 1. El máximo
    acumulado es lo que hace que el procedimiento controle el FWER y que los p
    ajustados respeten el orden de los crudos; omitirlo (como hacía la versión
    anterior de esta tabla) produce ajustes no monótonos.
    """
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m, dtype=float)
    run = 0.0
    for rank, idx in enumerate(order):
        run = max(run, (m - rank) * float(pvals[idx]))
        adj[idx] = min(1.0, run)
    return adj


def fmt_p(x: float) -> str:
    """p con cuatro decimales, salvo cuando eso lo redondea a cero.

    ⚠ `p = 0.0000` no es un valor: es un redondeo que afirma imposibilidad. Con
    42 legs el Wilcoxon puede bajar de 1e-13 y `:.4f` lo imprime como cero. La
    convención de la revista —y la del dictamen de agosto-2026— es `< 0.0001`.
    """
    return "< 0.0001" if x < 0.00005 else f"{x:.4f}"


def load(src: Path) -> pl.DataFrame:
    d = pl.read_csv(src)
    return d.with_columns(pl.col("route").cast(pl.Utf8),
                          (pl.col("MAE_s") / 60.0).alias("MAE_min"))


def _paired(p: pl.DataFrame, col: str) -> Tuple[np.ndarray, List[str]]:
    """Matriz (legs × algoritmos) alineada por leg — la base de todo lo pareado."""
    w = p.pivot(values=col, index=["route", "leg"], on="algorithm")
    keys = [f"{a}/{b}" for a, b in zip(w["route"], w["leg"])]
    cols = [REF] + ALTS
    return np.column_stack([w[c].to_numpy().astype(float) for c in cols]), keys


# ──────────────────────────────────────────────────────────────────
# Tabla 6
# ──────────────────────────────────────────────────────────────────

def table_06(d: pl.DataFrame) -> pl.DataFrame:
    rows: List[Dict] = []
    for panel, keep in PANELS:
        p = in_panel(d, keep)
        M, _ = _paired(p, "MAE_min")
        O, _ = _paired(p, "pct_ontime_sym180_cal")
        n = M.shape[0]
        lstm_med = float(np.median(M[:, 0]))
        raw = {"MAE": [], "ontime": []}
        for j, _a in enumerate(ALTS, start=1):
            for metric, X in (("MAE", M), ("ontime", O)):
                diff = X[:, 0] - X[:, j]           # LSTM − alternativa
                # `zero_method="wilcox"` descarta empates exactos, que aquí no
                # los hay (son medias sobre millones de pares).
                stat, pv = wilcoxon(diff, alternative="two-sided")
                raw[metric].append((pv, stat, diff))
        adj = {m: holm(np.array([r[0] for r in raw[m]])) for m in raw}
        for j, a in enumerate(ALTS):
            r_mae, r_ot = raw["MAE"][j], raw["ontime"][j]
            rows.append({
                "panel": panel, "n_legs": n, "alternative": a,
                "model": LABEL.get(a, a),
                "lstm_wins_mae": int((r_mae[2] < 0).sum()),
                "median_diff_mae_min": float(np.median(r_mae[2])),
                "p_raw_mae": float(r_mae[0]), "p_holm_mae": float(adj["MAE"][j]),
                "lstm_wins_ontime": int((r_ot[2] > 0).sum()),
                "median_diff_ontime_pp": float(np.median(r_ot[2])),
                "p_raw_ontime": float(r_ot[0]),
                "p_holm_ontime": float(adj["ontime"][j]),
                "lstm_mae_med_min": lstm_med,
            })
    return pl.DataFrame(rows)


def md_06(t: pl.DataFrame) -> str:
    L = ["**Table 3 (upper). Paired route-level tests: Sequential LSTM vs. each "
         "alternative (Wilcoxon signed-rank, Holm-adjusted).**", ""]
    n_all = int(t["n_legs"][0])
    L.append(
        f"The route is the unit of pairing: every model is scored on identical "
        f"round trips, positions, horizons and folds, so the per-route "
        f"differences are paired by construction. The panel has {n_all} routes. "
        f"Differences are LSTM minus "
        f"alternative, so a negative MAE difference and a positive on-time "
        f"difference both favour the LSTM. *Wins* counts routes where the LSTM is "
        f"better. On-time is measured on the symmetric band with the residual "
        f"centred out-of-fold, i.e. dispersion only; the operational band is "
        f"decomposed in Table 4. Holm adjustment is applied within each metric, "
        f"across the nine alternatives.")
    L.append("")
    L.append("| Alternative | MAE: LSTM wins | MAE: median diff (min) | "
             "MAE: p (Holm) | On-time: LSTM wins | On-time: median diff (pp) | "
             "On-time: p (Holm) |")
    L.append("|---|---|---|---|---|---|---|")
    for r in t.iter_rows(named=True):
        L.append(
            f"| {r['model']} | "
            f"{r['lstm_wins_mae']}/{r['n_legs']} | "
            f"{r['median_diff_mae_min']:+.2f} | {fmt_p(r['p_holm_mae'])} | "
            f"{r['lstm_wins_ontime']}/{r['n_legs']} | "
            f"{r['median_diff_ontime_pp']:+.2f} | {fmt_p(r['p_holm_ontime'])} |")

    # Nota generada de los datos, no escrita a mano. ⚠ La de agosto afirmaba quién
    # ganaba y por cuánto («about half a percent of a typical error of three and a
    # half minutes», «beat the LSTM on almost every leg»): con otra corrida esas
    # frases quedaban falsas sin que nada fallara. Lo que se lee de los resultados
    # va al texto del manuscrito; aquí sólo la escala con que leer las diferencias.
    L.append("")
    L.append(
        f"*Notes.* With paired differences this consistent, the signed-rank test "
        f"detects a difference however small it is, so significance here is not "
        f"the same as consequence: read each median difference against the "
        f"LSTM's median MAE over the panel, "
        f"{float(t['lstm_mae_med_min'][0]):.2f} min.")
    return "\n".join(L) + "\n"


# ──────────────────────────────────────────────────────────────────
# Tabla 9
# ──────────────────────────────────────────────────────────────────

def _boot_median(x: np.ndarray, rng: np.random.Generator) -> Tuple[float, float]:
    idx = rng.integers(0, len(x), size=(N_BOOT, len(x)))
    draws = np.median(x[idx], axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def table_09(d: pl.DataFrame) -> pl.DataFrame:
    rng = np.random.default_rng(SEED)
    rows: List[Dict] = []
    for panel, keep in PANELS:
        p = in_panel(d, keep)
        M, _ = _paired(p, "MAE_min")
        O, _ = _paired(p, "pct_ontime_sym180_cal")
        n = M.shape[0]

        # (a) razón MAE mejor-alternativa / LSTM, por leg.
        # ⚠ "mejor" se elige EN LOS MISMOS DATOS en los que se evalúa, así que
        # esta cantidad está sesgada A FAVOR de las alternativas. Se reporta
        # porque es la que usaba la versión anterior de la tabla y porque es
        # conservadora en contra del LSTM; las dos filas siguientes usan modelos
        # FIJOS y no tienen ese sesgo de selección.
        ratio = M[:, 1:].min(axis=1) / M[:, 0]
        lo, hi = _boot_median(ratio, rng)
        rows.append({"panel": panel, "n_legs": n,
                     "quantity": "MAE ratio, best alternative / LSTM",
                     "point": float(np.median(ratio)), "lo": lo, "hi": hi,
                     "unit": "×"})

        # (b) diferencia relativa de MAE contra modelos fijos
        for a in ("delay_hour", "xgb_mae_b"):
            j = ALTS.index(a) + 1
            rel = (M[:, j] - M[:, 0]) / M[:, 0] * 100.0
            lo, hi = _boot_median(rel, rng)
            rows.append({"panel": panel, "n_legs": n,
                         "quantity": f"MAE, {LABEL[a]} vs LSTM",
                         "point": float(np.median(rel)), "lo": lo, "hi": hi,
                         "unit": "%"})

        # (c) on-time centrado: LSTM − mejor alternativa en on-time
        dont = O[:, 0] - O[:, 1:].max(axis=1)
        lo, hi = _boot_median(dont, rng)
        rows.append({"panel": panel, "n_legs": n,
                     "quantity": "On-time (centred), LSTM − best alternative",
                     "point": float(np.median(dont)), "lo": lo, "hi": hi,
                     "unit": "pp"})
    return pl.DataFrame(rows)


def md_09(t: pl.DataFrame) -> str:
    L = ["**Not in the article: a route-resampled check of Table 3 (lower). Bootstrap 95% "
         "confidence intervals (median over routes; route-resampled, 10,000 draws).**", ""]
    L.append(
        f"Routes are resampled with replacement and the median "
        f"is recomputed; intervals are percentile intervals over "
        f"{N_BOOT:,} draws with a fixed seed. A ratio above 1 and a positive "
        f"on-time difference favour the LSTM. The *best alternative* rows select "
        f"the winning model per route on the same data used to evaluate it, which "
        f"biases them in the alternatives' favour; the fixed-model rows carry no "
        f"such selection and are the ones to read for the main claim. Resampling "
        f"that keeps the two variants of a family together is Table 3 (lower), the one the article reports.")
    L.append("")
    L.append("| Quantity | Routes | Point | 95% CI | Unit |")
    L.append("|---|---|---|---|---|")
    for r in t.iter_rows(named=True):
        L.append(f"| {r['quantity']} | {r['n_legs']} | "
                 f"{r['point']:.2f} | [{r['lo']:.2f}, {r['hi']:.2f}] | "
                 f"{r['unit']} |")
    return "\n".join(L) + "\n"


# ──────────────────────────────────────────────────────────────────

def main() -> int:
    repo = Path(__file__).resolve().parents[3]
    ap = argparse.ArgumentParser(description="Tabla 3 — inferencia pareada")
    ap.add_argument("--src", default=str(repo / "docs" / "paper" / "tables"
                                         / "table_10_ontime_honest.csv"))
    ap.add_argument("--outdir", default=str(repo / "docs" / "paper" / "tables"))
    a = ap.parse_args()

    src = Path(a.src)
    if not src.is_file():
        raise SystemExit(f"✗ falta {src} — corre antes honest_ontime.py --root <paquete>")
    d = load(src)
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    t6 = table_06(d)
    t6.write_csv(outdir / "table_06_paired_tests.csv")
    (outdir / "table_06_paired_tests.md").write_text(md_06(t6), encoding="utf-8")

    t9 = table_09(d)
    t9.write_csv(outdir / "table_09_bootstrap_ci.csv")
    (outdir / "table_09_bootstrap_ci.md").write_text(md_09(t9), encoding="utf-8")

    print(md_06(t6))
    print(md_09(t9))
    print(f"  ✓ {outdir / 'table_06_paired_tests.md'}")
    print(f"  ✓ {outdir / 'table_09_bootstrap_ci.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
