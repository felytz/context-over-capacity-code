"""
make_table_16_lookup_vs_best.py
===============================
Tabla S13 del artículo (archivo `table_16`) — la tabla de consulta (la media por hora, `delay_hour`) contra los
modelos de árboles, pareada por ruta.

Por qué
-------
La Discusión del 9k sostenía la «complejidad proporcionada» con dos cifras sin
artefacto (plan del manuscrito, fase 0, 0.2a y 0.2b):

  * «XGBoost con contexto compra un 9 % más» sobre la media por hora;
  * «la media por hora recupera unas dos quintas partes del hueco» entre la
    media simple (`delay`) y el mejor modelo.

Las Tablas 3 y 9b sólo comparan contra el LSTM. Ésta da la comparación que falta,
con la misma unidad (la ruta), la misma prueba (Wilcoxon de rangos con signo,
Holm dentro de cada métrica) y el mismo bootstrap agrupado por familia que la
Tabla 9b (31 grupos para 35 rutas), así que las tres se leen juntas.

Qué compara
-----------
  1. XGBoost (MAE, + contexto) contra la media por hora: lo que compra el
     modelo completo sobre la tabla de consulta.
  2. XGBoost (MAE, histórico) contra la media por hora: los dos usan sólo el
     viaje y la historia; la diferencia es el calendario que la tabla de consulta
     trae dentro (la hora) y el árbol no.
  3. La fracción del hueco `delay` → `xgb_mae_b` que cierra la media por hora,
     por ruta: (delay − delay_hour) / (delay − xgb_mae_b).

Convención de signo: `mediana(modelo − media por hora)`. Negativo en MAE y
positivo en on-time significan **el modelo de árboles mejor**.

Lee `table_10_ontime_honest.csv` (el mismo objeto que las Tablas 2, 3 y 4).

Uso
---
    python docs/paper/analysis/honest_ontime.py --root artefacts/run_bundle_<…>
    python docs/paper/analysis/make_table_16_lookup_vs_best.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import numpy as np                                              # noqa: E402
import polars as pl                                             # noqa: E402
from scipy.stats import wilcoxon                                # noqa: E402

from audit_run import FAMILY                                    # noqa: E402
from make_table_05_permodel import LABEL, PANELS, in_panel      # noqa: E402
from make_table_09b_clustered_bootstrap import N_BOOT, SEED, _boot_group  # noqa: E402
from make_tables_06_09_paired_stats import fmt_p, holm          # noqa: E402

TBL = _HERE.parent / "tables"
LOOKUP = "delay_hour"
MODELS = ["xgb_mae_b", "xgb_mae"]
MEAN = "delay"
NEED = [LOOKUP, MEAN, *MODELS]


def _wide(p: pl.DataFrame, col: str) -> pl.DataFrame:
    return (p.select("route", "leg", "algorithm", col)
             .pivot(values=col, index=["route", "leg"], on="algorithm")
             .sort("route", "leg"))


def build(d: pl.DataFrame) -> pl.DataFrame:
    rng = np.random.default_rng(SEED)
    rows: List[Dict] = []
    for panel, keep in PANELS:
        p = in_panel(d, keep)
        M = _wide(p, "MAE_min")
        missing = [c for c in NEED if c not in M.columns]
        if missing:
            raise SystemExit(f"✗ faltan algoritmos en la tabla 10: {missing}")
        C = _wide(p, "pct_ontime_sym180_cal").select("route", "leg", *NEED)
        A = _wide(p, "pct_ontime_apta").select("route", "leg", *NEED)
        assert M["route"].to_list() == C["route"].to_list() == A["route"].to_list()
        groups = np.array([FAMILY.get(str(r), str(r)) for r in M["route"].to_list()])
        n, n_g = M.height, len(set(groups))
        look = M[LOOKUP].to_numpy()

        # ── 1 y 2: cada modelo de árboles contra la tabla de consulta ─────────
        raw: Dict[str, list] = {"mae": [], "ontime": [], "apta": []}
        diffs = {}
        for m in MODELS:
            rel = (M[m].to_numpy() - look) / look * 100.0
            dmin = M[m].to_numpy() - look
            dot = C[m].to_numpy() - C[LOOKUP].to_numpy()
            dap = A[m].to_numpy() - A[LOOKUP].to_numpy()
            diffs[m] = (rel, dmin, dot, dap)
            raw["mae"].append(wilcoxon(dmin, alternative="two-sided").pvalue)
            raw["ontime"].append(wilcoxon(dot, alternative="two-sided").pvalue)
            raw["apta"].append(wilcoxon(dap, alternative="two-sided").pvalue)
        adj = {k: holm(np.array(v)) for k, v in raw.items()}
        for j, m in enumerate(MODELS):
            rel, dmin, dot, dap = diffs[m]
            for metric, x, unit, wins, p_raw, p_adj in (
                    ("MAE", rel, "%", int((dmin < 0).sum()),
                     raw["mae"][j], adj["mae"][j]),
                    ("MAE", dmin * 60.0, "s", int((dmin < 0).sum()),
                     raw["mae"][j], adj["mae"][j]),
                    ("On-time (centred)", dot, "pp", int((dot > 0).sum()),
                     raw["ontime"][j], adj["ontime"][j]),
                    ("On-time (APTA)", dap, "pp", int((dap > 0).sum()),
                     raw["apta"][j], adj["apta"][j])):
                lo, hi = _boot_group(x, groups, rng)
                rows.append({
                    "panel": panel, "n_routes": n, "n_groups": n_g,
                    "comparison": f"{LABEL[m]} − {LABEL[LOOKUP]}",
                    "model": m, "metric": metric, "unit": unit,
                    "model_wins": wins, "point": float(np.median(x)),
                    "family_lo": lo, "family_hi": hi,
                    "p_raw": float(p_raw), "p_holm": float(p_adj),
                })

        # ── 3: la fracción del hueco delay → xgb_mae_b que cierra la tabla ────
        gap = M[MEAN].to_numpy() - M["xgb_mae_b"].to_numpy()
        closed = (M[MEAN].to_numpy() - look) / gap * 100.0
        if (gap <= 0).any():
            raise SystemExit("✗ alguna ruta sin hueco delay → xgb_mae_b: la "
                             "fracción no está definida")
        lo, hi = _boot_group(closed, groups, rng)
        med = {k: float(np.median(M[k].to_numpy())) for k in NEED}
        rows.append({
            "panel": panel, "n_routes": n, "n_groups": n_g,
            "comparison": (f"share of the {LABEL[MEAN]} → {LABEL['xgb_mae_b']} gap "
                           f"closed by {LABEL[LOOKUP]}"),
            "model": LOOKUP, "metric": "MAE", "unit": "% of gap",
            "model_wins": int((closed > 50).sum()), "point": float(np.median(closed)),
            "family_lo": lo, "family_hi": hi, "p_raw": float("nan"),
            "p_holm": float("nan"),
        })
        rows.append({
            "panel": panel, "n_routes": n, "n_groups": n_g,
            "comparison": "same share, from panel medians",
            "model": LOOKUP, "metric": "MAE", "unit": "% of gap",
            "model_wins": -1,
            "point": 100.0 * (med[MEAN] - med[LOOKUP]) / (med[MEAN] - med["xgb_mae_b"]),
            "family_lo": float("nan"), "family_hi": float("nan"),
            "p_raw": float("nan"), "p_holm": float("nan"),
        })
    return pl.DataFrame(rows)


def to_md(t: pl.DataFrame) -> str:
    n, n_g = int(t["n_routes"][0]), int(t["n_groups"][0])
    L = ["**Table S13. The hourly lookup table against the boosted models "
         f"({n} routes; paired, family-clustered intervals).**", ""]
    L.append(
        f"Each route contributes its round trip, and every model is scored on the "
        f"same frame, so the differences are paired by construction. Differences "
        f"are the boosted model minus {LABEL[LOOKUP]}: a negative MAE difference "
        f"and a positive on-time difference favour the boosted model, and *wins* "
        f"counts the routes where it is better. The point is the median over "
        f"routes; the interval is a percentile bootstrap that resamples route "
        f"families ({n_g} groups, {N_BOOT:,} draws, as in Table 3). The *p* values "
        f"are Wilcoxon signed-rank, Holm-adjusted within each metric across the "
        f"two boosted models.")
    L += ["", "| Comparison | Metric | Unit | Boosted model wins | Median | "
          "95% CI (family) | *p* (Holm) |", "|---|---|---|---|---:|---|---:|"]
    gap_rows = []
    for r in t.iter_rows(named=True):
        if r["unit"] == "% of gap":
            gap_rows.append(r)
            continue
        L.append(f"| {r['comparison']} | {r['metric']} | {r['unit']} | "
                 f"{r['model_wins']}/{n} | {r['point']:+.2f} | "
                 f"[{r['family_lo']:+.2f}, {r['family_hi']:+.2f}] | "
                 f"{fmt_p(r['p_holm'])} |")
    per_route, panel_med = gap_rows
    L.append("")
    L.append(
        f"*Notes.* Between the plain historical mean ({LABEL[MEAN]}) and the most "
        f"accurate model ({LABEL['xgb_mae_b']}), {LABEL[LOOKUP]} closes a median "
        f"{per_route['point']:.0f}% of the MAE gap per route (95% CI "
        f"{per_route['family_lo']:.0f}–{per_route['family_hi']:.0f}%; more than half "
        f"of it on {per_route['model_wins']} of {n} routes); read from the panel "
        f"medians of Table 2, the same share is {panel_med['point']:.0f}%.")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Tabla S13 — la tabla de consulta "
                                             "contra los modelos de árboles")
    ap.add_argument("--src", default=str(TBL / "table_10_ontime_honest.csv"))
    ap.add_argument("--out", default=str(TBL / "table_16_lookup_vs_best"))
    a = ap.parse_args()

    src = Path(a.src)
    if not src.is_file():
        raise SystemExit(f"✗ falta {src} — corre antes honest_ontime.py --root <paquete>")
    d = (pl.read_csv(src)
           .with_columns(pl.col("route").cast(pl.Utf8),
                         (pl.col("MAE_s") / 60.0).alias("MAE_min")))
    t = build(d)
    Path(a.out + ".csv").write_text(t.write_csv(), encoding="utf-8")
    md = to_md(t)
    Path(a.out + ".md").write_text(md, encoding="utf-8")
    print(md)
    print(f"→ {a.out}.{{csv,md}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
