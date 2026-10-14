"""Tabla S1 del artículo — la separación de las rutas con variante (P1 del plan; pre-registro §1).

Cuatro códigos de ruta del AVL mezclan dos servicios que comparten casi todo el
recorrido: 1101 / 1101-ALT, 1400 (1401 y 1402), 1801 / 1801-ALT y 1802 / 1802-ALT. Se
separaron bus por bus y día por día con `func/variantes/analisis/variantes_ruta.py`, y el
crudo separado es la entrada del artículo. Esta tabla dice cómo quedó la separación.

De dónde sale cada columna
--------------------------
* **La zona distinta y el recorrido compartido**: de la geometría de `data/01_raw/`, con
  `geometria()` del mismo script que separó (los puntos de una traza que la otra no
  recorre en el mismo sentido a ≤ 50 m). «Mismas calles al revés» si la mayor parte de la
  zona es una calle que la otra variante recorre en sentido contrario.
* **Los registros de cada variante, su reparto por mes y los bus-día con las dos**: del
  crudo separado, `data/01_raw/records/<n>/` (lo que se publica).
* **Los indeterminados** (registros que no se asignaron; no están en el crudo publicado) y
  **el total del AVL**: de `variants_unassigned.csv`, sacado de los `resumen.md` del test
  (`test/hermosillo-bus-test/reports/variantes/<familia>/`). Se comprueba que principal +
  alterna + indeterminados = total, y que los registros del crudo separado son los que dice
  el resumen.
* **La validación**: `variants_validation.csv`, copia byte a byte de `pruebas.csv` del test
  (sha256 `082cb3e3182ddc8bf22fa6127655011061b5fd3c70727ab3a7c8d216ed1a0b39`): rutas reales
  que comparten calles, mezcladas sin su número y clasificadas con sus trazas.

`data/` está gitignorado, así que este script sólo corre donde esté el crudo —igual que
`network_funnel.py`—. Lo que se versiona es su salida.

Salidas:
  docs/paper/tables/appendix_e_variants.md
  docs/paper/analysis/variants_separation.csv

Run:  python docs/paper/analysis/variants_separation.py
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

import polars as pl

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parents[2]
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(ROOT / "func" / "variantes" / "analisis"))

from audit_run import route_label                              # noqa: E402

RAW = ROOT / "data" / "01_raw"
UNASSIGNED = _HERE / "variants_unassigned.csv"
VALIDATION = _HERE / "variants_validation.csv"
OUT_MD = ROOT / "docs" / "paper" / "tables" / "appendix_e_variants.md"
OUT_CSV = _HERE / "variants_separation.csv"
MONTHS = range(1, 9)                     # la ventana del análisis: enero–agosto de 2024


def zones(main: str, alt: str, geom: Path) -> dict:
    """La zona distinta de cada variante, con la función que separó."""
    import variantes_ruta as vr                      # func/variantes/analisis/
    T = vr.geometria((main, alt), geom)
    out = {}
    for v in (main, alt):
        zona_m = float(T[v]["excl"].sum() * vr.mt.PASO_M)
        out[v] = {"zone_km": zona_m / 1000.0,
                  "shared_pct": 100.0 * (1.0 - float(T[v]["excl"].mean())),
                  "reversed_km": T[v]["contrario_m"] / 1000.0,
                  "loop_km": T[v]["L"] / 1000.0}
    return out


def records(route: str) -> pl.DataFrame:
    files = sorted((RAW / "records" / route).glob(f"df_rec_raw_{route}_*.parquet"))
    if not files:
        raise SystemExit(f"✗ sin crudo de la {route} en {RAW / 'records' / route}")
    return pl.read_parquet(files[-1], columns=["id", "time"]).with_columns(
        pl.col("time").dt.date().alias("day"), pl.col("time").dt.month().alias("month"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", default=str(RAW))
    a = ap.parse_args()
    geom = Path(a.raw)

    with open(UNASSIGNED, encoding="utf-8") as fh:
        fams = list(csv.DictReader(fh))
    rows = []
    for f in fams:
        m, al = f["main"], f["alternate"]
        z = zones(m, al, geom)
        dm, da = records(m), records(al)
        n_m, n_a = dm.height, da.height
        if (n_m, n_a) != (int(f["main_records"]), int(f["alternate_records"])):
            raise SystemExit(f"✗ {f['family']}: el crudo separado tiene {n_m:,} / {n_a:,} "
                             f"registros y el resumen de la separación {f['main_records']} / "
                             f"{f['alternate_records']}: no es el crudo que salió de ella.")
        unassigned = int(f["unassigned_records"])
        total = int(f["avl_records"])
        if n_m + n_a + unassigned != total:
            raise SystemExit(f"✗ {f['family']}: {n_m} + {n_a} + {unassigned} ≠ {total}")
        # el reparto por mes, en la ventana del análisis
        pm = dm.filter(pl.col("month").is_in(list(MONTHS))).group_by("month").len()
        pa = da.filter(pl.col("month").is_in(list(MONTHS))).group_by("month").len()
        mon = pm.join(pa, on="month", suffix="_alt").with_columns(
            (100 * pl.col("len_alt") / (pl.col("len") + pl.col("len_alt"))).alias("alt_pct"))
        # bus-día: los de una variante, los de la otra y los que tienen las dos
        bm = set(dm.select("id", "day").unique().iter_rows())
        ba = set(da.select("id", "day").unique().iter_rows())
        reversed_share = (z[m]["reversed_km"] + z[al]["reversed_km"]) / max(
            z[m]["zone_km"] + z[al]["zone_km"], 1e-9)
        rows.append({
            "family": f["family"], "avl_group_id": f["avl_group_id"],
            "main": m, "alternate": al,
            "loop_km_main": round(z[m]["loop_km"], 1), "loop_km_alt": round(z[al]["loop_km"], 1),
            "shared_pct_main": round(z[m]["shared_pct"], 1),
            "shared_pct_alt": round(z[al]["shared_pct"], 1),
            "zone_km_main": round(z[m]["zone_km"], 1), "zone_km_alt": round(z[al]["zone_km"], 1),
            "reversed_km_main": round(z[m]["reversed_km"], 1),
            "reversed_km_alt": round(z[al]["reversed_km"], 1),
            "zone_type": ("same streets, opposite direction" if reversed_share >= 0.5
                          else "different streets"),
            "avl_records": total,
            "main_pct": round(100 * n_m / total, 1), "alt_pct": round(100 * n_a / total, 1),
            "unassigned_pct": round(100 * unassigned / total, 1),
            "alt_month_min": round(float(mon["alt_pct"].min()), 1),
            "alt_month_max": round(float(mon["alt_pct"].max()), 1),
            "busdays_assigned": len(bm | ba), "busdays_both": len(bm & ba),
        })

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    with open(VALIDATION, encoding="utf-8-sig") as fh:
        val = list(csv.DictReader(fh))
    # acierto_reg: de los registros que el método clasificó, cuántos casan con el código
    # de ruta que traían; clasif_reg: cuántos clasificó (el resto, indeterminados).
    # (variantes_ruta.metricas_prueba, «reales»: sin los empalmes sintéticos.)
    acc = [100 * float(v["acierto_reg"]) for v in val]
    cls = [100 * float(v["clasif_reg"]) for v in val]

    L = ["**Table S1. The four route codes that carry two services, separated vehicle by "
         "vehicle.**", ""]
    L.append(
        "The AVL records file both services of each family under one route code (the "
        "1401 and the 1402 under 1400, which is neither). Every record was assigned to one "
        "of the two by matching it to both polylines in the direction of travel: a record "
        "is evidence for a service when it lies on a stretch that only that service "
        "drives in that direction, and each vehicle-day is split into blocks where the "
        "evidence changes service. *Shared path* is the share of each polyline that the "
        "other one also drives in the same direction; the *distinct zone* is the rest. "
        "Records that no block could assign (*unassigned*, mostly vehicle-days of a few "
        "minutes on the route) are left out of both. Month shares are over January–August "
        "2024.")
    L += ["", "| Family (AVL code) | Main / alternate | Shared path (%) | Distinct zone (km) "
          "| Zone | AVL records | Main (%) | Alternate (%) | Unassigned (%) "
          "| Alternate share by month (%) | Vehicle-days with both |",
          "|---|---|---|---|---|---:|---:|---:|---:|---|---|"]
    for r in rows:
        L.append(
            f"| {r['family']} ({r['avl_group_id']}) | {route_label(r['main'])} / "
            f"{route_label(r['alternate'], number=True)} | "
            f"{r['shared_pct_main']:.0f} / {r['shared_pct_alt']:.0f} | "
            f"{r['zone_km_main']:.1f} / {r['zone_km_alt']:.1f} | {r['zone_type']} | "
            f"{r['avl_records']:,} | {r['main_pct']:.1f} | {r['alt_pct']:.1f} | "
            f"{r['unassigned_pct']:.1f} | {r['alt_month_min']:.0f}–{r['alt_month_max']:.0f} | "
            f"{r['busdays_both']:,} of {r['busdays_assigned']:,} |")
    L += ["", (
        f"*Notes.* The two services of every family run side by side throughout the window "
        f"(the alternate takes {min(r['alt_month_min'] for r in rows):.0f}–"
        f"{max(r['alt_month_max'] for r in rows):.0f}% of the records in every month), and "
        f"a vehicle runs one service per day: at most "
        f"{max(100 * r['busdays_both'] / r['busdays_assigned'] for r in rows):.1f}% of the "
        f"vehicle-days carry both. Where the distinct zone is the same streets driven in "
        f"opposite directions, position alone cannot tell the services apart and the "
        f"vehicle's heading does. The method was validated on {len(val)} mixes of two real "
        f"routes that share streets, with their route codes removed: it classified "
        f"{min(cls):.1f}–{max(cls):.1f}% of the records, and {min(acc):.1f}–{max(acc):.1f}% "
        f"of those matched the route code they carried. The variants share most of their path and "
        f"their fleet, so the route-clustered bootstrap treats each family as one group "
        f"(Table 3).")]
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\n✓ {OUT_MD}\n✓ {OUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
