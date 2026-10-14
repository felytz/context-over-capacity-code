"""
grid_representativity.py
========================
¿Los viajes que sobreviven al filtro de la matriz T son representativos de los
viajes de la ruta?

Por qué hace falta
------------------
`build_trajectory_matrix` fija la grid en el cuantil `n_route_quantile` de las
longitudes de viaje y luego exige `min_coverage` de esa grid. Con el valor
histórico q=0.95 la retención mediana es **6.5%** y no cualquier 6.5%: los
viajes MÁS LARGOS. El manuscrito afirma sus resultados sobre esa submuestra.

Hasta ahora el argumento a favor de q=0.05 era indirecto (la retención sube a
~95%; a horizonte y N igualados la muestra de q=0.95 es ~2.2× más difícil). Este
script lo mide de frente: **compara la distribución de los viajes retenidos
contra la de los descartados** en cada régimen y en cada covariable.

Es la evidencia que un revisor va a pedir para aceptar el cambio de régimen, y
la que documenta el sesgo del régimen viejo como limitación.

Método
------
Por (ruta, leg, q):
  1. Agregados por viaje: longitud, duración, hora de salida, día de semana,
     mes, bus.
  2. `n_route_m` = ceil(quantile(longitud, q) / step) · step  — igual que
     `build_trajectory_matrix`.
  3. Retenido ⇔ `n_pings >= 5` y `longitud >= min_coverage · n_route_m`.
     (Se omite el descarte por no-monotonía, que es residual; la retención
     resultante se valida contra la de las corridas reales — ver `--validate`.)
  4. **Retenidos vs POBLACIÓN COMPLETA** (no vs descartados):
       - continuas → SMD = (media_ret − media_todos) / sd_todos, y KS
       - categóricas → distancia de variación total (TVD)
     Convención habitual: |SMD| < 0.1 equilibrado · 0.1–0.25 leve ·
     > 0.25 desequilibrio material.

     ⚠ El contraste correcto es retenidos-vs-TODOS, no retenidos-vs-descartados.
     Con q=0.05 los descartados son el ~5% de viajes más cortos, así que
     retenidos-vs-descartados da diferencias enormes en TODOS los regímenes y no
     mide nada: la pregunta es si la muestra ANALIZADA representa a la ruta, no
     si el filtro separa lo que descarta. (La primera versión de este script
     media lo segundo y daba SMD ~2.2-2.7 no monótono en q, sin sentido.)

Con la vuelta (rama `vuelta`)
-----------------------------
Recorre las 35 rutas de `audit_run.ROUTES`. ⚠ Hasta la fase 1b recorría las 26 de los
estratos de agosto, así que en el piloto del servidor (7-oct) sólo salieron la 601 y la
1601: la 902 y la 1801-ALT se saltaban sin aviso. La «leg» es la vuelta (`T0->T0`) y
`cum_dist_m` ya es el avance sobre la traza (`GRID_AXIS=traza`), así que la longitud de
cada vuelta es la que usa la malla.

Uso
---
    python docs/paper/analysis/grid_representativity.py
    python docs/paper/analysis/grid_representativity.py --routes 101,1601 --validate
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import polars as pl

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT / "func"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_run import ROUTES, route_label                       # noqa: E402

OUT_DIR = _ROOT / "docs" / "paper" / "analysis"
RANGE   = "2024_01-08_tmin"
GRID_M  = 100

ALL_ROUTES = list(ROUTES)          # las 35 de la vuelta, en el orden de los jobs

CONT = ["trip_len_m", "duration_s", "hour_dep"]
CATS = ["day_of_week", "month"]


# ──────────────────────────────────────────────────────────────────

def smd_vs_pop(sample: np.ndarray, pop: np.ndarray) -> float:
    """
    Desplazamiento de la MUESTRA respecto de la POBLACIÓN, en unidades de
    desviación típica de la población: (media_muestra − media_pob) / sd_pob.

    Se estandariza con la sd de la población (no combinada) porque la muestra es
    un subconjunto de ella: la referencia es la ruta, no un grupo de control.
    """
    if len(sample) < 2 or len(pop) < 2:
        return float("nan")
    sd = float(np.std(pop, ddof=1))
    if sd <= 0:
        return 0.0 if abs(np.mean(sample) - np.mean(pop)) < 1e-12 else float("inf")
    return float((np.mean(sample) - np.mean(pop)) / sd)


def ks(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    try:
        from scipy.stats import ks_2samp
        r = ks_2samp(a, b)
        return float(r.statistic), float(r.pvalue)
    except Exception:
        return float("nan"), float("nan")


def tvd(a: np.ndarray, b: np.ndarray) -> float:
    """Distancia de variación total entre dos distribuciones categóricas
    (= máxima diferencia de proporciones sobre los niveles)."""
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    levels = np.union1d(np.unique(a), np.unique(b))
    pa = np.array([(a == l).mean() for l in levels])
    pb = np.array([(b == l).mean() for l in levels])
    return float(0.5 * np.abs(pa - pb).sum())


# ──────────────────────────────────────────────────────────────────

def trip_table(df: pl.DataFrame) -> pl.DataFrame:
    """Un registro por viaje válido con las covariables de interés."""
    has_hour = "hour_of_day" in df.columns
    has_dow  = "day_of_week" in df.columns

    aggs = [
        pl.col("cum_dist_m").max().alias("trip_len_m"),
        pl.len().alias("n_pings"),
        pl.col("time").min().alias("t0"),
        pl.col("time").max().alias("t1"),
        pl.col("bus_code").first().alias("bus_code"),
        pl.col("leg").first().alias("leg"),
    ]
    if has_hour:
        aggs.append(pl.col("hour_of_day").first().alias("hour_dep"))
    if has_dow:
        aggs.append(pl.col("day_of_week").first().alias("day_of_week"))

    t = (df.filter(pl.col("trip_valid"))
         .group_by("trip_id").agg(aggs)
         .with_columns(
             ((pl.col("t1").cast(pl.Int64) - pl.col("t0").cast(pl.Int64))
              / 1_000_000).cast(pl.Float64).alias("duration_s"),
             pl.col("t0").dt.month().alias("month"),
         ))
    if not has_hour:
        t = t.with_columns(pl.col("t0").dt.hour().cast(pl.Float64).alias("hour_dep"))
    if not has_dow:
        t = t.with_columns(pl.col("t0").dt.weekday().cast(pl.Int32).alias("day_of_week"))
    return t


def analyse_leg(t: pl.DataFrame, route: str, leg: str, qs: List[float],
                min_coverage: float) -> List[Dict[str, Any]]:
    sub = t.filter(pl.col("leg") == leg)
    n_raw = sub.height
    if n_raw < 10:
        return []
    length = sub["trip_len_m"].to_numpy().astype(float)
    n_ping = sub["n_pings"].to_numpy()

    rows: List[Dict[str, Any]] = []
    for q in qs:
        n_route_m = float(np.quantile(length, q))
        n_route_m = int(np.ceil(n_route_m / GRID_M) * GRID_M)
        keep = (n_ping >= 5) & (length >= min_coverage * n_route_m)
        n_keep = int(keep.sum())
        if n_keep < 2 or n_keep == n_raw:
            # Sin descartados no hay comparación posible; se registra igual la
            # retención porque es el dato central.
            rows.append({
                "route": route, "name": route_label(route), "leg": leg,
                "q": q, "min_coverage": min_coverage, "n_route_m": n_route_m,
                "n_raw": n_raw, "n_keep": n_keep,
                "retention_pct": round(100.0 * n_keep / n_raw, 2),
                "comparable": False,
            })
            continue

        row: Dict[str, Any] = {
            "route": route, "name": route_label(route), "leg": leg,
            "q": q, "min_coverage": min_coverage, "n_route_m": n_route_m,
            "n_raw": n_raw, "n_keep": n_keep,
            "retention_pct": round(100.0 * n_keep / n_raw, 2),
            "comparable": True,
            "len_p50_keep":  float(np.median(length[keep])),
            "len_p50_drop":  float(np.median(length[~keep])),
            "len_ratio_keep_all": round(
                float(np.median(length[keep]) / np.median(length)), 4),
        }
        # Muestra analizada (retenidos) contra la POBLACIÓN de la leg.
        for c in CONT:
            if c not in sub.columns:
                continue
            v = sub[c].to_numpy().astype(float)
            ok = ~np.isnan(v)
            samp, pop = v[keep & ok], v[ok]
            row[f"smd_{c}"] = round(smd_vs_pop(samp, pop), 4)
            d, p = ks(samp, pop)
            row[f"ks_{c}"] = round(d, 4)
            row[f"ksp_{c}"] = p
        for c in CATS:
            if c not in sub.columns:
                continue
            v = sub[c].to_numpy()
            row[f"tvd_{c}"] = round(tvd(v[keep], v), 4)

        # Flota: ¿el filtro se queda con los buses de siempre?
        bus = sub["bus_code"].to_numpy()
        row["n_bus_all"]  = int(len(np.unique(bus)))
        row["n_bus_keep"] = int(len(np.unique(bus[keep])))
        row["bus_kept_frac"] = round(row["n_bus_keep"] / max(row["n_bus_all"], 1), 4)
        rows.append(row)
    return rows


# ──────────────────────────────────────────────────────────────────

def validate(route: str, leg: str, q: float, min_coverage: float,
             approx_keep: int) -> None:
    """Contrasta la retención aproximada contra `build_trajectory_matrix`.

    ⚠ Se lee el parquet por RUTA EXPLÍCITA, no vía `config.PATHS`: `PATHS` fija
    la ruta al importarse y devolvería siempre la primera, dando validaciones
    falsas (pasó en la primera versión de este script)."""
    # Import por archivo: el `__init__` de `arrival_prediction` arrastra plots →
    # `_lib_style`, que necesita func/lib en sys.path. El builder solo usa
    # numpy/polars.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_ap_dsb", _ROOT / "func" / "arrival_prediction" / "dataset_builder.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)                                  # type: ignore
    build_trajectory_matrix = mod.build_trajectory_matrix

    p = (_ROOT / "data" / "02_interim" / "routes" / route / "dataframes"
         / f"df_rec_full_{RANGE}.parquet")
    df = pl.read_parquet(p)
    ds = build_trajectory_matrix(df, group_col="leg", group_value=leg,
                                 grid_step_m=GRID_M, n_route_quantile=q,
                                 min_coverage=min_coverage, verbose=False)
    real = ds["T"].shape[0]
    delta = approx_keep - real
    print(f"    validación {route}/{leg} q={q} cov={min_coverage}: "
          f"aprox={approx_keep:,} real={real:,} (Δ={delta:+,}, "
          f"{100*abs(delta)/max(real,1):.2f}%)")


def main() -> int:
    ap = argparse.ArgumentParser(description="Representatividad de la grid")
    ap.add_argument("--routes", default="all")
    ap.add_argument("--qs", default="0.05,0.10,0.50,0.95")
    ap.add_argument("--min-coverage", type=float, default=1.0)
    ap.add_argument("--also-cov", default="0.99",
                    help="coberturas extra solo para la tabla de retención")
    ap.add_argument("--validate", action="store_true",
                    help="contrasta la retención aproximada contra "
                         "build_trajectory_matrix en la primera leg de cada ruta")
    ap.add_argument("--out-stem", default="grid_representativity")
    a = ap.parse_args()

    routes = (ALL_ROUTES if a.routes.strip().lower() == "all"
              else [r.strip() for r in a.routes.split(",") if r.strip()])
    qs = [float(x) for x in a.qs.split(",") if x.strip()]
    covs = [a.min_coverage] + [float(x) for x in a.also_cov.split(",")
                               if x.strip()]

    print("=" * 74)
    print(f"  REPRESENTATIVIDAD DE LA GRID · {len(routes)} rutas · q={qs} · "
          f"cov={covs}")
    print("=" * 74)

    need = ["bus_code", "trip_id", "leg", "trip_valid", "cum_dist_m", "time",
            "hour_of_day", "day_of_week"]
    all_rows: List[Dict[str, Any]] = []
    for i, route in enumerate(routes, 1):
        p = (_ROOT / "data" / "02_interim" / "routes" / route / "dataframes"
             / f"df_rec_full_{RANGE}.parquet")
        if not p.is_file():
            print(f"  [{i:2d}/{len(routes)}] {route:>6}: sin df_full — saltada")
            continue
        cols = set(pl.read_parquet_schema(p))
        df = pl.read_parquet(p, columns=[c for c in need if c in cols])
        if "leg" not in df.columns:
            print(f"  [{i:2d}/{len(routes)}] {route:>6}: sin columna `leg` "
                  f"(fuera del dominio) — saltada")
            continue
        t = trip_table(df)
        legs = (t.group_by("leg").agg(pl.len().alias("n"))
                .sort("n", descending=True)["leg"].to_list())
        n_before = len(all_rows)
        for cov in covs:
            for leg in legs:
                all_rows += analyse_leg(t, route, leg, qs, cov)
        print(f"  [{i:2d}/{len(routes)}] {route:>6} ({route_label(route):9s}): "
              f"{len(legs)} legs · {t.height:,} viajes válidos · "
              f"{len(all_rows)-n_before} filas")

        if a.validate and legs:
            r0 = [r for r in all_rows
                  if r["route"] == route and r["leg"] == legs[0]
                  and r["q"] == qs[0] and r["min_coverage"] == covs[0]]
            if r0:
                validate(route, legs[0], qs[0], covs[0], r0[0]["n_keep"])

    if not all_rows:
        print("\n  ✗ sin datos")
        return 1

    df_out = pl.DataFrame(all_rows, infer_schema_length=None)
    csv_p = OUT_DIR / f"{a.out_stem}.csv"
    df_out.write_csv(csv_p)
    print(f"\n  ✓ {df_out.height:,} filas → {csv_p.relative_to(_ROOT)}")

    write_markdown(df_out, OUT_DIR / f"{a.out_stem}.md", covs[0])
    return 0


def write_markdown(df: pl.DataFrame, path: Path, cov_main: float) -> None:
    """Resumen legible: retención por régimen y desequilibrio de covariables."""
    lines: List[str] = []
    add = lines.append
    add("# Representatividad de la grid — retenidos vs descartados\n")
    add("Generado por `docs/paper/analysis/grid_representativity.py`. Un viaje se")
    add("retiene si tiene ≥5 pings y cubre `min_coverage · n_route_m`, donde")
    add("`n_route_m` es el cuantil `q` de las longitudes de viaje de la leg.\n")
    add("Todas las comparaciones son **muestra analizada (retenidos) vs población")
    add("completa de la leg**, no retenidos vs descartados: la pregunta es si lo")
    add("que se analiza representa a la ruta.\n")
    add("SMD = (media_retenidos − media_población) / sd_población.")
    add("Convención: |SMD| < 0.1 equilibrado · 0.1–0.25 leve · **> 0.25")
    add("desplazamiento material**. TVD = distancia de variación total; KS =")
    add("estadístico de Kolmogórov-Smirnov contra la población.\n")

    # ── Retención por (q, cov) ──
    add("## 1. Retención por régimen\n")
    add("| q | min_cov | legs | retención mediana | p25 | p75 | legs < 20% |")
    add("|---|---|---|---|---|---|---|")
    for (q, cov), g in sorted(df.group_by(["q", "min_coverage"]).__iter__(),
                              key=lambda kv: (-kv[0][0], -kv[0][1])):
        r = g["retention_pct"].to_numpy()
        add(f"| {q:.2f} | {cov:g} | {len(r)} | **{np.median(r):.1f}%** | "
            f"{np.percentile(r, 25):.1f}% | {np.percentile(r, 75):.1f}% | "
            f"{int((r < 20).sum())} |")
    add("")

    # ── Desequilibrio por q, a cov principal ──
    d = df.filter((pl.col("min_coverage") == cov_main) & pl.col("comparable"))
    add(f"## 2. Desplazamiento de covariables (min_coverage = {cov_main:g})\n")
    add("Mediana sobre legs del |SMD| / TVD de la muestra analizada respecto de")
    add("la población de la leg.\n")
    metrics = [c for c in d.columns if c.startswith(("smd_", "tvd_"))]
    add("| q | legs | " + " | ".join(m.replace("smd_", "|SMD| ")
                                     .replace("tvd_", "TVD ") for m in metrics)
        + " | legs con \\|SMD\\| longitud > 0.25 |")
    add("|---" * (len(metrics) + 3) + "|")
    for q in sorted(d["q"].unique().to_list(), reverse=True):
        g = d.filter(pl.col("q") == q)
        cells = []
        for m in metrics:
            v = np.abs(g[m].to_numpy().astype(float))
            v = v[np.isfinite(v)]
            cells.append(f"{np.median(v):.3f}" if len(v) else "—")
        sl = np.abs(g["smd_trip_len_m"].to_numpy().astype(float))
        sl = sl[np.isfinite(sl)]
        add(f"| {q:.2f} | {g.height} | " + " | ".join(cells)
            + f" | {int((sl > 0.25).sum())}/{len(sl)} |")
    add("")

    # ── Longitud: el sesgo directo ──
    add("## 3. Sesgo de longitud (el mecanismo)\n")
    add("`len_ratio_keep_all` = longitud mediana de los retenidos / longitud")
    add("mediana de TODOS los viajes de la leg. 1.00 = sin sesgo.\n")
    add("| q | legs | ratio mediano | p25 | p75 | máx |")
    add("|---|---|---|---|---|---|")
    for q in sorted(d["q"].unique().to_list(), reverse=True):
        g = d.filter(pl.col("q") == q)
        r = g["len_ratio_keep_all"].to_numpy().astype(float)
        r = r[np.isfinite(r)]
        if not len(r):
            continue
        add(f"| {q:.2f} | {len(r)} | **{np.median(r):.3f}** | "
            f"{np.percentile(r,25):.3f} | {np.percentile(r,75):.3f} | "
            f"{r.max():.3f} |")
    add("")

    # ── Flota ──
    add("## 4. Flota\n")
    add("`bus_kept_frac` = buses presentes entre los retenidos / buses de la leg.")
    add("Si baja mucho, el filtro además selecciona vehículos.\n")
    add("| q | legs | fracción mediana de buses conservados | p25 |")
    add("|---|---|---|---|")
    for q in sorted(d["q"].unique().to_list(), reverse=True):
        g = d.filter(pl.col("q") == q)
        r = g["bus_kept_frac"].to_numpy().astype(float)
        r = r[np.isfinite(r)]
        if not len(r):
            continue
        add(f"| {q:.2f} | {len(r)} | **{np.median(r):.3f}** | "
            f"{np.percentile(r,25):.3f} |")
    add("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  ✓ resumen → {path.relative_to(_ROOT)}")


if __name__ == "__main__":
    raise SystemExit(main())
