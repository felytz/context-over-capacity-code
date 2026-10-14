"""Tabla 1 compacta — el funnel agregado, para el cuerpo del paper.

El dictamen externo (punto 9) pide que la Tabla 1 completa por ruta salga del
cuerpo: 35 filas × 15 columnas es visualmente pesada y ninguna afirmación del
manuscrito se lee de una fila concreta. La versión por ruta va en Supplementary
(Tabla S2) y el cuerpo se queda con esta.

Con la vuelta (docs/PREREGISTRO_VUELTA.md) ya no hay estratos de fiabilidad (venían del
descubrimiento de terminales). Las filas son el **origen del crudo**, que es el reparto
que §2.1 tiene que contar (P1):
  · las 27 rutas del crudo público (las 26 de agosto y la 902);
  · las 8 variantes, separadas bus por bus de cuatro códigos de ruta del AVL que
    mezclan dos servicios (`origin` en route_catalogue.csv).

⚠ **Lee el CSV PUBLICADO, no las fuentes.** `make_table_01_funnel.py` reconstruye
el funnel desde el digest y los metadatos de los frames, que son artefactos del
servidor. Esta tabla se deriva de `docs/paper/tables/table_01_funnel.csv`, así que
**no puede divergir** de la tabla por ruta: si la completa se regenera, esta se
regenera de la misma fuente.

Agregación:
  · conteos (registros, tramos, vueltas)  → suma
  · Kept (%)                              → media ponderada por registros crudos
  · Coverage (%)                          → mediana de las coberturas por ruta,
                                            etiquetada como tal
  · Grid kept (%)                         → recalculado de las sumas, NO promediado
  · Median round trip (min)               → mediana de las medianas por ruta,
                                            etiquetada como tal

Salidas:
  docs/paper/tables/table_01_funnel_compact.md
  docs/paper/tables/table_01_funnel_compact.csv

Run:  python docs/paper/analysis/make_table_01_funnel_compact.py
"""
from __future__ import annotations

import argparse
import csv
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TBL = ROOT / "docs" / "paper" / "tables"
SRC = TBL / "table_01_funnel.csv"

# El orden del funnel: primero el crudo público, luego lo que se rescató.
GROUPS = ["public", "variant"]
LABEL = {
    "public": "Routes in the public AVL release",
    "variant": "Variant routes, separated from four shared route codes",
}


def num(row: dict, key: str) -> float:
    v = row.get(key, "")
    return 0.0 if v in ("", "nan", None) else float(v)


def aggregate(rows: list[dict]) -> dict:
    raw = sum(num(r, "pings_raw") for r in rows)
    modeled = sum(num(r, "trips_modeled") for r in rows)
    grid = sum(num(r, "grid_trips") for r in rows)
    return {
        "routes": len(rows),
        "pings_raw": int(raw),
        # ponderada: una ruta con 2 M de registros no pesa lo mismo que una con 1 M
        "pct_kept": (sum(num(r, "pings_raw") * num(r, "pct_kept") for r in rows) / raw
                     if raw else 0.0),
        "stretches": int(sum(num(r, "stretches") for r in rows)),
        "coverage_med": st.median([num(r, "coverage_pct") for r in rows]),
        "trips_valid": int(sum(num(r, "trips_valid") for r in rows)),
        "trips_modeled": int(modeled),
        "grid_trips": int(grid),
        # recalculado de las sumas: promediar porcentajes por ruta daría otra cosa
        "grid_kept_pct": 100.0 * grid / modeled if modeled else 0.0,
        "dur_med_min": st.median([num(r, "dur_med_min") for r in rows]),
    }


CAPTION = (
    "**Table 1. Data funnel (January–August 2024 analysis window).**"
)

NOTE = """Routes are grouped by where their AVL records come from; the per-route
funnel is Table S2 in the Supplementary Information. Raw AVL records and cleaning
retention are computed over the full downloaded range. A *service stretch* is a
vehicle's continuous operation on the route; *coverage* is the share of a route's
records that fall inside valid round trips; a *round trip* runs from one crossing of
the route's cut point to the next within a stretch; *modeled* round trips exclude
those with a shortcut and reach the per-route modeling dataset; **grid** round trips
are those retained by the trajectory matrix and therefore actually analyzed, and are
the denominator of every model comparison in this paper. Counts are sums over the
routes of each group; *kept* is weighted by raw records, *grid kept* is recomputed
from the sums, and *coverage* and *median round trip* are medians of the per-route
values. The variant routes are the two services of four route codes that the AVL
records do not distinguish, separated vehicle by vehicle (Table S1)."""

COLS = [
    ("Group", "{}", "l"),
    ("Routes", "{:,}", "r"),
    ("Raw records", "{:,}", "r"),
    ("Kept (%)", "{:.1f}", "r"),
    ("Service stretches", "{:,}", "r"),
    ("Coverage (%), median", "{:.1f}", "r"),
    ("Round trips", "{:,}", "r"),
    ("Modeled", "{:,}", "r"),
    ("Grid", "{:,}", "r"),
    ("Grid kept (%)", "{:.1f}", "r"),
    ("Median round trip (min)", "{:.1f}", "r"),
]
KEYS = ["label", "routes", "pings_raw", "pct_kept", "stretches", "coverage_med",
        "trips_valid", "trips_modeled", "grid_trips", "grid_kept_pct", "dur_med_min"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--out", default=str(TBL / "table_01_funnel_compact"))
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.src, encoding="utf-8")))
    if not rows:
        raise SystemExit(f"ERROR: {a.src} está vacío.")
    if "origin" not in rows[0]:
        raise SystemExit(f"ERROR: {a.src} no tiene la columna `origin`: es el funnel de "
                         f"agosto (por estratos). Regenera antes make_table_01_funnel.py.")

    seen = {r["origin"] for r in rows}
    unknown = seen - set(GROUPS)
    if unknown:
        raise SystemExit(
            f"\nERROR: origen(es) desconocido(s) en {a.src}: {sorted(unknown)}.\n"
            f"  Añádelos a GROUPS y a LABEL, o la fila se perdería en silencio.")

    out_rows = []
    for g in GROUPS:
        sub = [r for r in rows if r["origin"] == g]
        if not sub:
            continue
        out_rows.append({"label": LABEL[g], **aggregate(sub)})
    out_rows.append({"label": "**All routes**", **aggregate(rows)})

    # control: los agregados tienen que cuadrar con las sumas crudas del CSV
    tot = out_rows[-1]
    assert tot["routes"] == len(rows)
    assert tot["grid_trips"] == int(sum(num(r, "grid_trips") for r in rows))

    md = [CAPTION, "", NOTE, "",
          "| " + " | ".join(c for c, _, _ in COLS) + " |",
          "|" + "|".join("---" if al == "l" else "---:" for _, _, al in COLS) + "|"]
    for r in out_rows:
        md.append("| " + " | ".join(
            fmt.format(r[k]) for (_, fmt, _), k in zip(COLS, KEYS)) + " |")

    out = Path(a.out)
    out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    with open(out.with_suffix(".csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=KEYS)
        w.writeheader()
        w.writerows(out_rows)
    print(f"OK: {out.with_suffix('.md')} · {len(out_rows)} filas "
          f"({tot['routes']} rutas)")


if __name__ == "__main__":
    main()
