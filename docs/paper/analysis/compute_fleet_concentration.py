"""Fleet concentration (Pareto) per route — ¿cuántos camiones hacen los viajes?

El "Fleet" de la Tabla 1 (funnel.buses) cuenta CADA bus_code que apareció en la
ruta, aunque haya hecho 2 viajes en 8 meses. Este script mide la concentración
real: de N camiones, cuántos concentran el grueso de los viajes (Pareto).

Fuente (server-side): por cada ruta
  <root>/02_interim/routes/<ruta>/dataframes/df_final_new_trips_<range>_<tag>.parquet
que tiene una fila por ping con columnas bus_code, trip_id (y, si existen,
trip_valid / direction / leg). Un "viaje" = un par distinto (bus_code, trip_id).

Métricas por ruta:
  n_trips            viajes distintos (denominador de las cuotas)
  fleet_total        camiones distintos (≈ el "Fleet" de la Tabla 1)
  trips_per_bus_*    mediana / media / p90 / max de viajes por camión
  gini, hhi          desigualdad (0 = equitativo) / Herfindahl (1/fleet..1)
  share_top1/5/10    % de viajes que hacen los 1/5/10 camiones más activos
  n_buses_50/80/90   camiones MÍNIMOS que cubren el 50/80/90% de los viajes
  eff_fleet_90       n_buses_90 / fleet_total  (la "flota efectiva")

Si el df_final trae trip_valid, se computa también sobre los viajes válidos
(sufijo _valid) — la flota que de verdad entra al modelado.

Salidas (bajo --out, por defecto route_reports/fleet_concentration/):
  fleet_concentration_<range>_<tag>.csv    una fila por ruta (todas las cols)
  fleet_concentration_<range>_<tag>.json   idem, + metadatos de corrida
Escribe conforme avanza; un timeout deja parciales.

Uso (server):
  sbatch jobs/job_fleet_concentration.slrm
  # o directo en nodo de login (es liviano, CPU, minutos):
  /lustre/home/usuario/.conda/envs/tf-local/bin/python \\
      docs/paper/analysis/compute_fleet_concentration.py \\
      --root /lustre/home/usuario/db/data --routes all
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import polars as pl


def gini(counts: list[int]) -> float | None:
    """Gini sobre los viajes-por-camión (0 = todos iguales, →1 = uno acapara)."""
    xs = sorted(int(c) for c in counts)
    n = len(xs)
    s = sum(xs)
    if n == 0 or s == 0:
        return None
    # G = (2*sum(i*x_i) / (n*sum)) - (n+1)/n , i = 1..n
    cum = sum((i + 1) * x for i, x in enumerate(xs))
    return round((2.0 * cum) / (n * s) - (n + 1.0) / n, 4)


def concentration(per_bus: list[int]) -> dict:
    """per_bus = lista de #viajes por camión (uno por camión)."""
    fleet = len(per_bus)
    total = sum(per_bus)
    xs = sorted(per_bus, reverse=True)                    # más activos primero
    out: dict[str, float | int | None] = {
        "fleet_total": fleet,
        "n_trips": total,
    }
    if fleet == 0 or total == 0:
        for k in ("trips_per_bus_med", "trips_per_bus_mean", "trips_per_bus_p90",
                  "trips_per_bus_max", "gini", "hhi", "share_top1_pct",
                  "share_top5_pct", "share_top10_pct", "n_buses_50", "n_buses_80",
                  "n_buses_90", "eff_fleet_90_pct"):
            out[k] = None
        return out

    srt = sorted(per_bus)
    out["trips_per_bus_med"] = srt[fleet // 2] if fleet % 2 else (srt[fleet // 2 - 1] + srt[fleet // 2]) / 2
    out["trips_per_bus_mean"] = round(total / fleet, 2)
    out["trips_per_bus_p90"] = srt[min(fleet - 1, int(round(0.9 * (fleet - 1))))]
    out["trips_per_bus_max"] = xs[0]
    out["gini"] = gini(per_bus)
    out["hhi"] = round(sum((c / total) ** 2 for c in per_bus), 4)     # Herfindahl

    def top_share(k: int) -> float:
        return round(100.0 * sum(xs[:k]) / total, 1)
    out["share_top1_pct"] = top_share(1)
    out["share_top5_pct"] = top_share(5)
    out["share_top10_pct"] = top_share(10)

    def min_buses_for(frac: float) -> int:
        acc = 0
        for i, c in enumerate(xs, start=1):          # acumula del más activo
            acc += c
            if acc >= frac * total:
                return i
        return len(xs)
    out["n_buses_50"] = min_buses_for(0.50)
    out["n_buses_80"] = min_buses_for(0.80)
    out["n_buses_90"] = min_buses_for(0.90)
    out["eff_fleet_90_pct"] = round(100.0 * out["n_buses_90"] / fleet, 1)
    return out


def per_bus_frame(lf: pl.LazyFrame) -> pl.DataFrame:
    """#viajes (trip_id distinto) por camión: DataFrame [bus_code, n]."""
    return (
        lf.select("bus_code", "trip_id")
        .unique()                                   # un (bus, trip) por viaje
        .group_by("bus_code")
        .agg(pl.len().alias("n"))
        .collect(engine="streaming")
    )


def route_of(path: Path) -> str:
    # <root>/02_interim/routes/<ruta>/dataframes/df_final_...parquet
    return path.parent.parent.name


def find_finals(root: Path, range_base: str, tag: str, routes: str) -> list[Path]:
    base = root / "02_interim" / "routes"
    fname = f"df_final_new_trips_{range_base}_{tag}.parquet"
    if routes == "all":
        found = sorted(base.glob(f"*/dataframes/{fname}"))
    else:
        found = []
        for r in (x.strip() for x in routes.split(",") if x.strip()):
            p = base / r / "dataframes" / fname
            if p.is_file():
                found.append(p)
            else:
                print(f"  ! aviso: sin {p} (se salta)")
    return found


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="/lustre/home/usuario/db/data",
                    help="raíz con 02_interim/routes/")
    ap.add_argument("--out", default="route_reports/fleet_concentration",
                    help="dir de salida (server-side)")
    ap.add_argument("--range", dest="range_base", default="2024_01-08")
    ap.add_argument("--tag", default="tmin")
    ap.add_argument("--routes", default="all", help='"all" o "101,501,..."')
    a = ap.parse_args()

    root = Path(a.root)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    finals = find_finals(root, a.range_base, a.tag, a.routes)
    if not finals:
        raise SystemExit(f"ERROR: no se encontró ningún df_final en {root}/02_interim/routes")
    print(f"{len(finals)} rutas con df_final ({a.range_base}_{a.tag})")

    rows: list[dict] = []
    net_buses: set = set()          # unión de bus_code de TODAS las rutas (red)
    for p in finals:
        route = route_of(p)
        lf = pl.scan_parquet(p)
        cols = lf.collect_schema().names()
        if "bus_code" not in cols or "trip_id" not in cols:
            print(f"  ! {route}: faltan bus_code/trip_id (cols={cols[:8]}...) — se salta")
            continue
        rec: dict = {"route": route}
        frame = per_bus_frame(lf)
        rec.update(concentration(frame["n"].to_list()))
        net_buses.update(str(b) for b in frame["bus_code"].to_list())   # unión de red
        # variante solo-válidos, si la columna existe
        if "trip_valid" in cols:
            v = concentration(per_bus_frame(lf.filter(pl.col("trip_valid")))["n"].to_list())
            for k, val in v.items():
                rec[f"{k}_valid"] = val
        rows.append(rec)
        print(f"  {route}: fleet={rec['fleet_total']}  n_trips={rec['n_trips']}  "
              f"top10={rec['share_top10_pct']}%  n_buses_90={rec['n_buses_90']}  "
              f"eff_fleet_90={rec['eff_fleet_90_pct']}%", flush=True)

        # escribir incremental (CSV completo cada vuelta → un corte deja parciales)
        fields = list(dict.fromkeys(k for r in rows for k in r))
        with open(out_dir / f"fleet_concentration_{a.range_base}_{a.tag}.csv",
                  "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)

    sum_fleet = sum(r["fleet_total"] for r in rows if r.get("fleet_total"))
    net_codes = len(net_buses)
    payload = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "root": str(root), "range": a.range_base, "tag": a.tag,
        "n_routes": len(rows),
        # bus_code (= raw `code`) es el ID de unidad del vehículo; los buses se
        # reasignan entre rutas, así que la SUMA per-ruta sobrecuenta y la UNIÓN
        # es la flota física real de la red (dominio: confirmado por el autor).
        "network_distinct_vehicles": net_codes,
        "sum_per_route_fleet": sum_fleet,
        "cross_route_overlap": sum_fleet - net_codes,
        "note": "bus_code (=`code`) = ID de unidad del vehículo; los buses se "
                "reasignan entre rutas → la suma per-ruta sobrecuenta y la UNIÓN "
                "(network_distinct_vehicles) es la flota física de la red. Dentro de "
                "una ruta code es 1:1 con la matrícula (plates_per_code_max=1); "
                "plate_number no sirve para contar (50-99% nula, ver "
                "compute_unique_vehicles.py).",
        "definition": "un viaje = par distinto (bus_code, trip_id) en df_final; "
                      "share_topK = % de viajes de los K camiones más activos; "
                      "n_buses_P = camiones mínimos que cubren el P% de los viajes.",
        "routes": rows,
    }
    (out_dir / f"fleet_concentration_{a.range_base}_{a.tag}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nNETWORK distinct vehicles (union of bus_code) = {net_codes}  "
          f"(sum per-route = {sum_fleet}; cross-route overlap = {sum_fleet - net_codes})")
    print(f"OK: {len(rows)} rutas -> {out_dir}/fleet_concentration_{a.range_base}_{a.tag}.[csv,json]")


if __name__ == "__main__":
    main()
