"""
make_table_09b_clustered_bootstrap.py
=====================================
Tabla 3 (abajo) del artículo: la Tabla 9 (archivo `table_09`, fuera del artículo) con la unidad de remuestreo CORREGIDA: el grupo de rutas que no son
independientes, no la fila.

Por qué
-------
En agosto las 42 legs del panel se anidaban en 22 rutas, casi siempre por pares (los
dos sentidos del mismo corredor, con los mismos vehículos), y remuestrear legs como si
fueran independientes daba un intervalo DEMASIADO ESTRECHO. El dictamen de agosto-2026
pidió corregirlo en vez de declararlo.

Con la vuelta (docs/PREREGISTRO_VUELTA.md §5) cada ruta tiene una sola leg, pero las
dos variantes de una familia (1101 y 110101, 1401 y 1402, 1801 y 180101, 1802 y
180201) comparten el 82–93 % de la traza y la flota: el grupo es la FAMILIA
(audit_run.FAMILY). Son 31 grupos para 35 rutas.

Qué hace el bootstrap agrupado
------------------------------
Remuestrea GRUPOS con reemplazo y arrastra TODAS las rutas de cada grupo elegido.
Así la correlación dentro de la familia entra en el intervalo en vez de ignorarse.
El tamaño de la muestra remuestreada varía entre réplicas —los grupos tienen 1 o 2
rutas— y eso es correcto: es parte de la variabilidad que se está estimando.

⚠ El PUNTO no cambia, solo el intervalo. La mediana se calcula sobre las rutas
observadas en los dos casos; lo que cambia es cuánta incertidumbre se le asigna.

Uso
---
    python docs/paper/analysis/make_table_09b_clustered_bootstrap.py
    python docs/paper/analysis/make_table_09b_clustered_bootstrap.py --src <csv>
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

from audit_run import FAMILY, route_label, route_order        # noqa: E402
from make_table_05_permodel import LABEL, PANELS, in_panel      # noqa: E402

REF = "lstm_a"
ALTS = ["xgb_mae_b", "xgb_mae", "xgb_mse_b", "xgb_mse",
        "delay_hour", "delay", "kernel", "knn", "linreg"]
N_BOOT = 10_000
SEED = 42

TBL = _HERE.parent / "tables"


def load(src: Path) -> pl.DataFrame:
    d = pl.read_csv(src)
    return d.with_columns(pl.col("route").cast(pl.Utf8),
                          (pl.col("MAE_s") / 60.0).alias("MAE_min"))


def _paired(p: pl.DataFrame, col: str) -> Tuple[np.ndarray, np.ndarray]:
    """(rutas × algoritmos) y el vector de GRUPO por fila, que es lo que agrupa: la
    ruta, o su familia si es una ruta con variante (FAMILY)."""
    w = p.pivot(values=col, index=["route", "leg"], on="algorithm")
    cols = [REF] + ALTS
    M = np.column_stack([w[c].to_numpy().astype(float) for c in cols])
    grupos = np.array([FAMILY.get(str(r), str(r)) for r in w["route"].to_list()])
    return M, grupos


def _boot_unit(x: np.ndarray, rng: np.random.Generator) -> Tuple[float, float]:
    """El de la Tabla 9: cada fila (cada ruta) independiente."""
    idx = rng.integers(0, len(x), size=(N_BOOT, len(x)))
    draws = np.median(x[idx], axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def _boot_group(x: np.ndarray, groups: np.ndarray,
                rng: np.random.Generator) -> Tuple[float, float]:
    """Agrupado: se remuestrean GRUPOS y cada uno arrastra sus filas.

    No se vectoriza como el de filas porque el tamaño de cada réplica varía: una
    familia de 2 rutas aporta 2 filas y una ruta sola aporta 1, así que las réplicas
    no forman una matriz rectangular. Con 31 grupos y 10,000 réplicas el bucle es
    irrelevante frente al resto del pipeline.
    """
    by = {}
    for i, g in enumerate(groups):
        by.setdefault(g, []).append(i)
    keys = list(by)
    idx_by_group = [np.asarray(by[k]) for k in keys]
    n_g = len(keys)

    draws = np.empty(N_BOOT, dtype=float)
    for b in range(N_BOOT):
        pick = rng.integers(0, n_g, size=n_g)
        sel = np.concatenate([idx_by_group[j] for j in pick])
        draws[b] = np.median(x[sel])
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def build(d: pl.DataFrame) -> pl.DataFrame:
    rows: List[Dict] = []
    # ⚠ UN solo rng para el bootstrap por ruta, creado aquí y consumido en el
    # mismo orden que `make_tables_06_09_paired_stats.py` (panel → ratio,
    # delay_hour, xgb_mae_b, on-time). Es lo que hace que la columna «route»
    # reproduzca la Tabla 9 dígito a dígito, que es su única razón de estar aquí:
    # si no reproduce, no sirve de comparación. Sembrar por cantidad sería más
    # robusto a reordenaciones y movía los intervalos en el segundo decimal — un
    # lector que cruzara las dos tablas vería una discrepancia sin causa.
    rng_unit = np.random.default_rng(SEED)
    rng_group = np.random.default_rng(SEED)
    for panel, keep in PANELS:
        p = in_panel(d, keep)
        M, groups = _paired(p, "MAE_min")
        O, _ = _paired(p, "pct_ontime_sym180_cal")
        n, n_g = M.shape[0], len(set(groups))

        quantities = [
            ("MAE ratio, best alternative / LSTM",
             M[:, 1:].min(axis=1) / M[:, 0], "×"),
        ]
        for a in ("delay_hour", "xgb_mae_b"):
            j = ALTS.index(a) + 1
            quantities.append((f"MAE, {LABEL[a]} vs LSTM",
                               (M[:, j] - M[:, 0]) / M[:, 0] * 100.0, "%"))
        quantities.append(("On-time (centred), LSTM − best alternative",
                           O[:, 0] - O[:, 1:].max(axis=1), "pp"))

        for name, x, unit in quantities:
            lo_u, hi_u = _boot_unit(x, rng_unit)
            lo_g, hi_g = _boot_group(x, groups, rng_group)
            rows.append({
                "panel": panel, "n_routes": n, "n_groups": n_g,
                "quantity": name, "point": float(np.median(x)),
                "route_lo": lo_u, "route_hi": hi_u,
                "family_lo": lo_g, "family_hi": hi_g,
                "width_route": hi_u - lo_u, "width_family": hi_g - lo_g,
                "widening": (hi_g - lo_g) / (hi_u - lo_u) if hi_u > lo_u else float("nan"),
                "unit": unit,
            })
    return pl.DataFrame(rows)


def to_md(t: pl.DataFrame) -> str:
    fams = sorted({f for f in FAMILY.values()}, key=route_order)
    pairs = ", ".join(
        f"{route_label(f)} and " + " and ".join(
            route_label(r) for r, g in FAMILY.items() if g == f)
        for f in fams)
    L = ["**Table 3 (lower). Bootstrap 95% confidence intervals "
         "(median over routes; family-clustered, 10,000 draws).**", ""]
    L.append(
        f"Each route contributes its round trip. The two variants of a route "
        f"family ({pairs}) share most of their path and their fleet, so the "
        f"resampling unit is the **family**: groups are drawn with replacement "
        f"and every route of a drawn family is carried with it; the other routes "
        f"are groups of one. The route-resampled interval is reported alongside "
        f"for comparison; it is the narrower of the two wherever the variants of "
        f"a family are correlated. Point estimates are identical under both "
        f"schemes and are computed on the observed routes.")
    L += ["", "| Quantity | Routes | Groups | Point | 95% CI (family) "
          "| 95% CI (route) | Unit |", "|---|---|---|---|---|---|---|"]
    for r in t.iter_rows(named=True):
        L.append(
            f"| {r['quantity']} | {r['n_routes']} | "
            f"{r['n_groups']} | {r['point']:.2f} | "
            f"[{r['family_lo']:.2f}, {r['family_hi']:.2f}] | "
            f"[{r['route_lo']:.2f}, {r['route_hi']:.2f}] | {r['unit']} |")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(TBL / "table_10_ontime_honest.csv"))
    ap.add_argument("--out", default=str(TBL / "table_09b_clustered_bootstrap"))
    a = ap.parse_args()

    t = build(load(Path(a.src)))
    Path(a.out + ".csv").write_text(t.write_csv(), encoding="utf-8")
    Path(a.out + ".md").write_text(to_md(t), encoding="utf-8")

    print(f"{'cantidad':44s} {'ancho ruta':>11s} {'ancho familia':>14s} {'x':>6s}")
    for r in t.iter_rows(named=True):
        print(f"{r['quantity'][:44]:44s} "
              f"{r['width_route']:11.3f} {r['width_family']:14.3f} "
              f"{r['widening']:6.2f}")
    print(f"\n→ {a.out}.{{csv,md}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
