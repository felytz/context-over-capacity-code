"""Sensibilidad del conteo de terminales al ε del DBSCAN de dwell (respalda §3.4:
"the number of detected terminals is stable across wide parameter ranges").

QUÉ ε: el conteo de terminales de producción lo fija el camino de **dwell/layover**
(`func/lib/_lib_new_trips.py:_terminal_like_cut_events`):
    discover_terminals(eps_m) → dominancia AUTO_min (_resolve_dominance_frac) →
    filtro recurrente ≥3 buses/≥2 días · on-route · dominante (n_dwells ≥ frac·max)
    [+ endpoints de polilínea si INCLUDE_ENDPOINT_TERMINALS]  →  N = nº terminales.
El DBSCAN de reversión (ε=150) parte round-trips / es fallback, NO el contador.
Este script barre el ε del dwell (default 80 m) reproduciendo esa lógica EXACTA
(reusa td.discover_terminals + nt._resolve_dominance_frac + nt._augment...), y
valida que N(ε=80) coincide con el n_terminals del digest (Tabla 1).

Datos locales: data/02_interim/routes/<r>/dataframes/df_rec_clean_2024_01-08_tmin.parquet
               + data/01_raw/polylines.geojson

Uso:
  python docs/paper/analysis/terminal_eps_sensitivity.py               # 27 rutas
  python docs/paper/analysis/terminal_eps_sensitivity.py --routes 101,902
Salidas: docs/paper/analysis/terminal_eps_sensitivity.{csv,md}
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "func" / "lib"))

import polars as pl                       # noqa: E402
import _lib_terminal_discovery as td       # noqa: E402
import _lib_new_trips as nt                # noqa: E402

GEO = ROOT / "data" / "01_raw" / "polylines.geojson"
ROUTES_DIR = ROOT / "data" / "02_interim" / "routes"
OUT = ROOT / "docs" / "paper" / "analysis"
TAG = "2024_01-08_tmin"
EPS_DEFAULT = [40, 50, 60, 70, 80, 100, 120, 150, 200]   # producción = 80
STAY_RADIUS_M = 60.0
DOMINANCE_MODE = "AUTO_min"       # el de producción (barre frac → deja 1–2)


def clean_path(route: str) -> Path:
    return ROUTES_DIR / route / "dataframes" / f"df_rec_clean_{TAG}.parquet"


def load_poly(route: str):
    try:
        return td.load_route_polyline(str(GEO), int(route) if route.isdigit() else route)
    except Exception:
        return None


def count_terminals(df, poly, cum, L, eps_m: float, layover_min: float) -> int:
    """Reproduce _terminal_like_cut_events (conteo de terminales), variando eps_m."""
    disc = td.discover_terminals(df, poly, layover_min_min=layover_min,
                                 stay_radius_m=STAY_RADIUS_M, eps_m=eps_m, dominance_frac=0.0)
    clusters = disc["clusters"]
    frac, _ = nt._resolve_dominance_frac(clusters, DOMINANCE_MODE)
    tc = []
    if clusters is not None and clusters.height:
        on = clusters.filter(pl.col("recurrent") & pl.col("on_route"))
        gmax = float(on["n_dwells"].max() or 0)
        term = on.filter(pl.col("n_dwells") >= frac * gmax)
        tc = list(zip(term["lat"].to_list(), term["lon"].to_list()))
    if nt.INCLUDE_ENDPOINT_TERMINALS:
        tc = nt._augment_with_polyline_endpoints(tc, poly, cum, L)
    return len(tc)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--routes", default="all")
    ap.add_argument("--eps", default=",".join(map(str, EPS_DEFAULT)))
    ap.add_argument("--layover", type=float, default=float(nt.BREAK_MIN),
                    help="dwell mínimo (min) para el descubrimiento; default BREAK_MIN")
    a = ap.parse_args()
    eps_grid = [float(x) for x in a.eps.split(",") if x.strip()]

    if a.routes == "all":
        routes = sorted((p.name for p in ROUTES_DIR.iterdir() if clean_path(p.name).is_file()),
                        key=lambda r: r.zfill(6))
    else:
        routes = [r.strip() for r in a.routes.split(",") if r.strip()]

    digest_N = {}
    t1 = OUT.parent / "tables" / "table_01_funnel.csv"
    if t1.is_file():
        for r in csv.DictReader(open(t1, encoding="utf-8")):
            digest_N[str(r["route"])] = int(r["n_terminals"])

    print(f"{len(routes)} rutas · ε(dwell) = {eps_grid} · layover={a.layover} min · dominancia={DOMINANCE_MODE}")
    rows = []
    for route in routes:
        cp = clean_path(route); poly = load_poly(route)
        if not cp.is_file() or not poly:
            print(f"  ! {route}: sin datos/polilínea"); continue
        df = pl.read_parquet(cp).select(["bus_code", "time", "latitude", "longitude"])
        cum = td._cumulative_distances(poly); L = float(cum[-1])
        rec = {"route": route}
        t0 = time.time()
        for eps in eps_grid:
            try:
                rec[f"N_{int(eps)}"] = count_terminals(df, poly, cum, L, eps, a.layover)
            except Exception as e:
                rec[f"N_{int(eps)}"] = None
                print(f"    {route} ε={eps}: ERROR {e}")
        vals = [rec[f"N_{int(e)}"] for e in eps_grid if rec[f"N_{int(e)}"] is not None]
        rec["N_range"] = f"{min(vals)}-{max(vals)}" if vals else "—"
        rec["stable"] = bool(vals) and (min(vals) == max(vals))
        rec["digest_N"] = digest_N.get(route)
        rec["matches_digest"] = (rec.get("N_80") == rec["digest_N"]) if rec["digest_N"] is not None else None
        rows.append(rec)
        chk = "" if rec["matches_digest"] is None else (" ✓dig" if rec["matches_digest"] else f" ✗dig({rec['digest_N']})")
        print(f"  {route}: " + " ".join(f"{int(e)}:{rec[f'N_{int(e)}']}" for e in eps_grid)
              + f"   [{'stable' if rec['stable'] else 'VARIES'}]{chk}  ({time.time()-t0:.0f}s)", flush=True)
        OUT.mkdir(parents=True, exist_ok=True)
        cols = ["route"] + [f"N_{int(e)}" for e in eps_grid] + ["N_range", "stable", "digest_N", "matches_digest"]
        with open(OUT / "terminal_eps_sensitivity.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)

    n_stable = sum(1 for r in rows if r["stable"])
    n_match = sum(1 for r in rows if r["matches_digest"])
    md = [
        "**Terminal-count sensitivity to the dwell-clustering radius ε.**",
        "N = number of terminals from the production discovery "
        "(`discover_terminals` + AUTO_min dominance selection), recomputed while "
        f"sweeping the DBSCAN radius ε around its production value of 80 m; layover "
        f"threshold {a.layover:.0f} min, all other parameters at production defaults. "
        "ε in metres.",
        "",
        "| Route | " + " | ".join(f"ε={int(e)}" for e in eps_grid) + " | stable |",
        "|---|" + "---|" * (len(eps_grid) + 1),
    ]
    for r in rows:
        md.append("| " + r["route"] + " | "
                  + " | ".join(str(r[f"N_{int(e)}"]) for e in eps_grid)
                  + f" | {'✓' if r['stable'] else '✗'} |")
    md += ["",
           f"*{n_stable} of {len(rows)} routes return an identical terminal count across "
           f"the full ε range ({int(min(eps_grid))}–{int(max(eps_grid))} m, {max(eps_grid)/min(eps_grid):.1f}× "
           f"the low end), including the production value ε = 80 m. "
           f"N(ε=80) reproduces the digest terminal count on {n_match}/{len(rows)} routes.*"]
    (OUT / "terminal_eps_sensitivity.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\n{n_stable}/{len(rows)} estables · {n_match}/{len(rows)} reproducen el digest (ε=80)")
    print(f"salidas: {OUT/'terminal_eps_sensitivity.csv'} · .md")


if __name__ == "__main__":
    main()
