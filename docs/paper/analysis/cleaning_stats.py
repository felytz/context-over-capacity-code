"""
cleaning_stats.py
=================
Lo que §2.2 dice de la limpieza, medido sobre el crudo de las 35 rutas (plan del
manuscrito, 0.2e). El 9k citaba cifras de los datos de agosto («~30.6 % de registros
quietos», «unimodal entre vehículos, 25–40 %», «el hueco de 7 min es el P90 de los
intervalos») y un filtro de 50 m/s que la vuelta ya no aplica: un salto imposible
(> 25 m/s·dt + 300 m sobre la traza) parte el tramo de servicio, no borra el registro.

La limpieza es la de `dc_records` (`func/lib/_lib_data.py`) más la ventana de análisis
(`func/config.py`), replicada aquí sin el resto del pipeline:
  1. duplicados por (code, utc_datetime): se queda el de menor `id_registro`;
  2. `disabled == 0`;
  3. hora local = UTC − 7 h, y sólo entre las 05:00 y las 22:00 (exclusivo);
  4. enero–agosto 2024 (`[2024-01-01, 2024-09-01)`).
Se comprueba contra el `pings_clean` del funnel de cada ruta (`--root`, el bundle):
tiene que dar el mismo número de registros, o el script se detiene.

Mide, por ruta y en la red:
  * la fracción de registros quietos (`speed <= 0`), que la limpieza conserva, y su
    reparto entre vehículos (cada par vehículo–ruta con ≥ 1,000 registros: un vehículo
    que sirve dos rutas cuenta dos veces);
  * el intervalo entre registros consecutivos del mismo vehículo el mismo día: mediana,
    P90 y P99, y la fracción de intervalos de más de 5 y de 20 min (los umbrales del
    cruce inferido y del corte del tramo de servicio).

Uso
---
    python docs/paper/analysis/cleaning_stats.py --root artefacts/run_bundle_<…>
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import numpy as np                                              # noqa: E402
import polars as pl                                             # noqa: E402

from audit_run import ROUTES, route_label                       # noqa: E402

RAW = _HERE.parents[2] / "data" / "01_raw" / "records"
START, END = datetime(2024, 1, 1), datetime(2024, 9, 1)
MIN_PINGS_VEHICLE = 1000


def clean(route: str) -> pl.DataFrame:
    files = sorted((RAW / route).glob(f"df_rec_raw_{route}_*.parquet"))
    if not files:
        raise SystemExit(f"✗ sin crudo de la ruta {route} en {RAW / route}")
    # el pipeline toma el más grande (config.env §5, RAW_DATE_RANGE vacío)
    f = max(files, key=lambda p: p.stat().st_size)
    d = pl.read_parquet(f, columns=["id_registro", "code", "utc_datetime", "speed",
                                    "disabled"])
    return (d.lazy()
             .sort(["code", "utc_datetime", "id_registro"])
             .unique(subset=["code", "utc_datetime"], keep="first")
             .filter(pl.col("disabled") == 0)
             .with_columns((pl.col("utc_datetime") - pl.duration(hours=7)).alias("time"))
             .filter((pl.col("time").dt.time() > time(5, 0, 0))
                     & (pl.col("time").dt.time() < time(22, 0, 0)))
             .filter((pl.col("time") >= START) & (pl.col("time") < END))
             .select("code", "time", "speed")
             .collect())


def per_route(route: str, d: pl.DataFrame) -> tuple[dict, np.ndarray, pl.DataFrame]:
    gaps = (d.sort("code", "time")
             .with_columns(pl.col("time").dt.date().alias("day"))
             .with_columns(pl.col("time").diff().over("code", "day")
                           .dt.total_milliseconds().truediv(1000).alias("dt"))
             ["dt"].drop_nulls().to_numpy())
    veh = (d.group_by("code").agg(pl.len().alias("n"),
                                  (pl.col("speed") <= 0).mean().alias("stationary"))
            .filter(pl.col("n") >= MIN_PINGS_VEHICLE))
    q = np.percentile(gaps, [50, 90, 99])
    return ({"route": route, "pings_clean": d.height,
             "stationary_pct": 100 * float((d["speed"] <= 0).mean()),
             "gap_median_s": float(q[0]), "gap_p90_s": float(q[1]), "gap_p99_s": float(q[2]),
             "gap_over_5min_pct": 100 * float((gaps > 300).mean()),
             "gap_over_20min_pct": 100 * float((gaps > 1200).mean()),
             "vehicles": veh.height}, gaps, veh)


def main() -> int:
    ap = argparse.ArgumentParser(description="la limpieza de §2.2, medida")
    ap.add_argument("--root", required=True, help="el bundle (para el pings_clean del funnel)")
    ap.add_argument("--out", default=str(_HERE / "cleaning_stats"))
    a = ap.parse_args()

    rows, all_gaps, vehs = [], [], []
    stat_n = stat_tot = 0
    for r in ROUTES:
        d = clean(r)
        fn = json.loads((Path(a.root) / "data" / "02_interim" / "routes" / r / "perf"
                         / "funnel_2024_01-08_tmin.json").read_text(encoding="utf-8"))
        if d.height != int(fn["pings_clean"]):
            raise SystemExit(f"✗ ruta {r}: {d.height:,} registros limpios aquí y "
                             f"{int(fn['pings_clean']):,} en el funnel: la réplica no es fiel")
        row, g, v = per_route(r, d)
        rows.append(row)
        all_gaps.append(g)
        vehs.append(v.with_columns(pl.lit(r).alias("route")))
        stat_n += int((d["speed"] <= 0).sum())
        stat_tot += d.height
        print(f"  {r:>7}: {d.height:>9,} registros · quietos {row['stationary_pct']:.1f} % · "
              f"intervalo mediano {row['gap_median_s']:.0f} s, P90 {row['gap_p90_s']:.0f} s")
    t = pl.DataFrame(rows)
    g = np.concatenate(all_gaps)
    v = pl.concat(vehs)
    q = np.percentile(g, [50, 90, 99])
    vq = np.percentile(v["stationary"].to_numpy() * 100, [10, 50, 90])
    net = {
        "routes": t.height, "pings_clean": int(t["pings_clean"].sum()),
        "stationary_pct": 100 * stat_n / stat_tot,
        "stationary_route_min": float(t["stationary_pct"].min()),
        "stationary_route_max": float(t["stationary_pct"].max()),
        "vehicles": v.height, "stationary_vehicle_p10": vq[0],
        "stationary_vehicle_p50": vq[1], "stationary_vehicle_p90": vq[2],
        "gap_median_s": q[0], "gap_p90_s": q[1], "gap_p99_s": q[2],
        "gap_over_5min_pct": 100 * float((g > 300).mean()),
        "gap_over_20min_pct": 100 * float((g > 1200).mean()),
    }
    t.write_csv(a.out + ".csv")
    L = [f"**Cleaning, measured ({net['routes']} routes, January–August 2024, "
         f"05:00–22:00).**", "",
         f"After removing duplicate reports and those outside the operating window, "
         f"{net['pings_clean']:,} position reports remain. Stationary reports (reported "
         f"speed ≤ 0) are kept: they are {net['stationary_pct']:.1f}% of the total, "
         f"{net['stationary_route_min']:.1f}–{net['stationary_route_max']:.1f}% by route, and "
         f"{net['stationary_vehicle_p10']:.1f}–{net['stationary_vehicle_p90']:.1f}% for the "
         f"middle 80% of the {net['vehicles']:,} vehicle–route pairs with at least "
         f"{MIN_PINGS_VEHICLE:,} reports (median {net['stationary_vehicle_p50']:.1f}%). "
         f"Between consecutive reports of the same vehicle on the same day the median "
         f"interval is {net['gap_median_s']:.0f} s, the 90th percentile "
         f"{net['gap_p90_s']:.0f} s and the 99th {net['gap_p99_s']:.0f} s; "
         f"{net['gap_over_5min_pct']:.2f}% of intervals exceed 5 min and "
         f"{net['gap_over_20min_pct']:.2f}% exceed 20 min.", "",
         "| Route | Reports | Stationary % | Interval median (s) | P90 (s) | P99 (s) "
         "| > 5 min % | > 20 min % |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in t.iter_rows(named=True):
        L.append(f"| {route_label(r['route'])} | {r['pings_clean']:,} | "
                 f"{r['stationary_pct']:.1f} | {r['gap_median_s']:.0f} | "
                 f"{r['gap_p90_s']:.0f} | {r['gap_p99_s']:.0f} | "
                 f"{r['gap_over_5min_pct']:.2f} | {r['gap_over_20min_pct']:.2f} |")
    Path(a.out + ".md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n" + L[2])
    print(f"→ {a.out}.{{csv,md}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
