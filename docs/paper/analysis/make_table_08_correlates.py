"""
make_table_08_correlates.py
===========================
Tabla S17 del artículo (archivo `table_08`) — ¿con qué propiedades de la leg se asocia la (des)ventaja del
LSTM? Exploratorio.

Sustituye a `table_06_p3_correlations` (26 rutas, régimen matched). Dos cambios
que importan:

1. **La unidad es la leg** (42) y no la ruta (26): más unidades y, sobre todo, la
   unidad en la que se mide todo lo demás.
2. **Desaparece `matched coverage` como predictor.** Era la fracción de cortes
   del baseline que alcanzaba la ventana truncada del LSTM, y en el régimen
   anterior era uno de los dos únicos correlatos significativos (ρ = +0.43). Con
   el frame común vale 1.0 por construcción: aquella asociación medía el
   artefacto del protocolo, no una propiedad de la ruta. Es un resultado en sí.

Las tres medidas de ventaja se definen de modo que **negativo = el LSTM pierde**:
razón MAE (mejor alternativa / LSTM) menos 1, diferencia relativa de MAE contra
`xgb_mae_b`, y diferencia de on-time centrado contra la mejor alternativa.

Sin corrección de multiplicidad en la tabla (es exploratorio); se reporta cuántas
asociaciones sobrevivirían Bonferroni sobre el total de contrastes.

Con la vuelta la unidad es la ruta (su vuelta) y el LSTM, el `es15` (lo trae la Tabla
10). La propiedad «self-loop share» de agosto vale 100 % en todas las rutas y sale; en
su lugar entra la cobertura del corte (`coverage_pct` de la Tabla S2: qué fracción del
servicio queda en vueltas completas).

Uso
---
    python docs/paper/analysis/make_table_08_correlates.py --root test3/extracted
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import numpy as np                                              # noqa: E402
import polars as pl                                             # noqa: E402
from scipy.stats import spearmanr                               # noqa: E402

from audit_run import qtag                                      # noqa: E402
from honest_ontime import RANGE                                 # noqa: E402
from leg_inclusion import MIN_M, apply_criteria, collect_legs    # noqa: E402
from make_tables_06_09_paired_stats import ALTS, REF, _paired, load  # noqa: E402

PROPS = [
    ("n_route_m",     "Round-trip length (grid extent, m)"),
    ("M",             "Round trips analysed"),
    ("retention_pct", "Grid retention (%)"),
    ("n_trips_raw",   "Round trips modeled"),
    ("route_coverage_pct", "Trip-cut coverage (%)"),
    ("route_fleet",   "Route fleet size"),
    ("route_pct_kept", "Record retention (%)"),
]


def leg_props(root: Path, q: float, cov: float, rng: str,
              funnel_csv: Path) -> pl.DataFrame:
    tag = qtag(q, cov)
    rows: List[Dict] = []
    base = root / "data" / "02_interim" / "routes"
    for p in sorted(base.glob(f"*/frames/{rng}/*_{tag}_meta.json")):
        m = json.loads(p.read_text(encoding="utf-8"))
        rows.append({"route": str(m["route"]), "leg": str(m["leg"]),
                     "n_route_m": float(m["n_route_m"]), "M": float(m["M"]),
                     "retention_pct": float(m["retention_pct"]),
                     "n_trips_raw": float(m["n_trips_raw"])})
    d = pl.DataFrame(rows)

    if funnel_csv.is_file():
        fr = []
        with open(funnel_csv, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                def g(k):
                    v = (r.get(k) or "").strip()
                    return float(v) if v else None
                fr.append({"route": str(r["route"]),
                           "route_coverage_pct": g("coverage_pct"),
                           "route_fleet": g("fleet"),
                           "route_pct_kept": g("pct_kept")})
        d = d.join(pl.DataFrame(fr), on="route", how="left")
    else:
        print(f"  ⚠ sin {funnel_csv}: las propiedades de ruta quedan vacías")
    return d


def build(root: Path, q: float, cov: float, rng: str, min_M: int,
          src: Path, funnel_csv: Path) -> pl.DataFrame:
    # `route` viene como entero del CSV de la Tabla 10 y como texto de los
    # metas; se normaliza a texto en los dos lados antes de cruzar.
    d = load(src).with_columns(pl.col("route").cast(pl.Utf8))
    legs = apply_criteria(collect_legs(root, q, cov, rng), min_M)
    keep = legs.filter(pl.col("included")).select(
        pl.col("route").cast(pl.Utf8), pl.col("leg"))
    d = d.join(keep, on=["route", "leg"], how="inner")

    M, keys = _paired(d, "MAE_min")
    O, keys_o = _paired(d, "pct_ontime_sym180_cal")
    assert keys == keys_o
    gains = pl.DataFrame({
        "route": [k.split("/")[0] for k in keys],
        "leg":   [k.split("/", 1)[1] for k in keys],
        "gain_ratio":  M[:, 0] / M[:, 1:].min(axis=1) - 1.0,
        "gain_vs_xgb": (M[:, 0] - M[:, ALTS.index("xgb_mae_b") + 1])
                       / M[:, ALTS.index("xgb_mae_b") + 1] * -100.0,
        "gain_ontime": O[:, 0] - O[:, 1:].max(axis=1),
    })
    props = leg_props(root, q, cov, rng, funnel_csv)
    g = gains.join(props, on=["route", "leg"], how="inner")

    # `gain_ratio` NO se reporta: la mejor alternativa es `xgb_mae_b` en casi
    # todas las legs, así que es una transformación monótona de `gain_vs_xgb` y
    # sus correlaciones salen idénticas con el signo cambiado — sería el mismo
    # test contado dos veces, inflando el denominador de la corrección por
    # multiplicidad. Se conserva la de modelo FIJO, que además no arrastra el
    # sesgo de elegir al ganador en los mismos datos en que se evalúa.
    labels = {"gain_vs_xgb": "MAE advantage vs. best tabular (%)",
              "gain_ontime": "Centred on-time advantage (pp)"}
    rows: List[Dict] = []
    # Una propiedad que vale lo mismo en todas las rutas no tiene correlación (Spearman da
    # nan) y sólo inflaría el denominador de Bonferroni. Con la vuelta le pasa a M: las 35
    # rutas llegan al tope de 2,500 vueltas. Sale de la tabla y la nota lo dice.
    constantes: Dict[str, float] = {}
    for gk, glab in labels.items():
        for pk, plab in PROPS:
            if pk not in g.columns:
                continue
            x = g[pk].to_numpy().astype(float)
            y = g[gk].to_numpy().astype(float)
            ok = ~(np.isnan(x) | np.isnan(y))
            if ok.sum() < 8:
                continue
            if np.unique(x[ok]).size < 2:
                constantes[plab] = float(x[ok][0])
                continue
            rho, pv = spearmanr(x[ok], y[ok])
            rows.append({"gain": glab, "property": plab, "n": int(ok.sum()),
                         "rho": float(rho), "p": float(pv)})
    # Con < 8 rutas (un piloto) no sale ninguna correlación: tabla vacía con su
    # esquema, y el .md lo dice, en vez de reventar al filtrar una columna que no hay.
    return pl.DataFrame(rows, schema={"gain": pl.Utf8, "property": pl.Utf8,
                                      "n": pl.Int64, "rho": pl.Float64,
                                      "p": pl.Float64}), constantes


def to_markdown(t: pl.DataFrame, constantes: Dict[str, float] | None = None) -> str:
    n_tests = t.height
    n_units = int(t["n"].max()) if n_tests else 0
    sig = t.filter(pl.col("p") < 0.05)
    bonf = t.filter(pl.col("p") < 0.05 / max(n_tests, 1))
    L = ["**Table S17. Spearman correlations between the LSTM's (dis)advantage and "
         f"route properties ({n_units} routes; exploratory).**", ""]
    L.append(
        f"Advantage measures are signed so that negative means the LSTM loses. "
        f"Correlations are computed over the {n_units} routes that meet "
        f"the pre-specified criterion, one round trip each. This analysis is "
        f"exploratory and the table is not corrected for multiplicity: of the "
        f"{n_tests} tests, {sig.height} reach p < 0.05 and {bonf.height} would "
        f"survive Bonferroni correction at the same level.")
    for plab, v in (constantes or {}).items():
        L[-1] += (f" {plab} is left out: it takes the same value, {v:,.0f}, on every "
                  f"route.")
    if not n_tests:
        L.append("")
        L.append("*No correlation computed: fewer than 8 routes have every property.*")
        return "\n".join(L) + "\n"
    L.append("")
    L.append("| Advantage measure | Property | n | ρ | p |")
    L.append("|---|---|---|---|---|")
    for r in t.sort(["gain", "p"]).iter_rows(named=True):
        mark = " \\*" if r["p"] < 0.05 else ""
        L.append(f"| {r['gain']} | {r['property']} | {r['n']} | "
                 f"{r['rho']:+.2f} | {r['p']:.3f}{mark} |")
    return "\n".join(L) + "\n"


def main() -> int:
    repo = Path(__file__).resolve().parents[3]
    ap = argparse.ArgumentParser(description="Tabla S17 — correlatos exploratorios")
    ap.add_argument("--root", default=None)
    ap.add_argument("--q", type=float, default=0.05)
    ap.add_argument("--cov", type=float, default=1.0)
    ap.add_argument("--range", dest="rng", default=RANGE)
    ap.add_argument("--min-M", type=int, default=MIN_M)
    ap.add_argument("--src", default=str(repo / "docs/paper/tables"
                                         / "table_10_ontime_honest.csv"))
    ap.add_argument("--funnel", default=str(repo / "docs/paper/tables"
                                            / "table_01_funnel.csv"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    root = Path(a.root).resolve() if a.root else repo
    t, constantes = build(root, a.q, a.cov, a.rng, a.min_M, Path(a.src), Path(a.funnel))
    prefix = Path(a.out) if a.out else (repo / "docs/paper/tables"
                                        / "table_08_correlates")
    t.write_csv(prefix.with_suffix(".csv"))
    md = to_markdown(t, constantes)
    prefix.with_suffix(".md").write_text(md, encoding="utf-8")
    print(md)
    print(f"  ✓ {prefix.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
