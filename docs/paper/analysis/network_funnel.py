"""El embudo de la RED: de los identificadores de ruta del operador a las 35 procesadas.

Punto 8 del dictamen externo: «Falta explicar por qué se analizan 27 de las
aproximadamente 37 rutas de Hermosillo. ¿Las diez restantes no tenían AVL,
geometría, stop inventory, operación suficiente o no formaban parte del
operador?».

⚠ **El handoff daba este dato por perdido** —«no está en el código ni en los
artefactos: solo lo tiene el autor»— y era falso para la parte que importa. Las
tres fuentes secundarias que §3.3 declara están en `data/01_raw/`, y cada una
lleva dentro su propio universo de rutas:

    stops_routes.json     inventario de paradas por ruta   → 45 identificadores
    segments_route.json   geometría por ruta               → 42 identificadores
    records/<ruta>/       registros AVL compilados         → 27 rutas

y los tres conjuntos están **anidados**: records ⊆ segments ⊆ stops. O sea que el
embudo de la red se lee de los datos, no de la memoria.

⚠ **No hubo entrega ni convenio** (el autor, 29-sep): las tres fuentes se
recolectaron del portal web público de UNE (https://unesonora.com). Hasta ese
día este script decía que el operador las había «entregado», y los manuscritos
también.

La causa del corte la pone el autor: las doce rutas sin AVL son **servicios no
convencionales** —atípicos por frecuencia o por longitud, o registrados bajo un
código de ruta compartido— para los que la cadena no está diseñada. La
**longitud sí se puede comprobar** contra la geometría (`segments.geojson`), y
este script la publica; la frecuencia no, sin su AVL.

⚠ **Los identificadores no son rutas uno a uno.** Seis son variantes de tres bases
(1101, 1801 y 1802): las `-UI` (variante universitaria, sólo visual en el portal) y las
`-ALT` (ramal alternativo). Al plegarlos, el inventario tiene **39** rutas base. El
manuscrito decía «roughly 37», que es una cifra sin fuente en el repo: la discrepancia
se declara, no se esconde.

Con la geometría corregida (`data raw/new-01_raw/`, rama `vuelta`)
-----------------------------------------------------------------
* **Las -ALT llevan número entero** (110101, 180101, 180201: el de la principal con
  «01»), porque el pipeline sólo carga geometría con clave entera. Se reconocen por el
  catálogo de rutas (`route_catalogue.csv`, la etiqueta `-ALT`), no por el sufijo:
  con el sufijo solo, «ya no las reconoce como variantes» (plan, fase 1b).
* **La 2201 es la 20000**: la misma traza (70 de 72 tramos; la de UNE trae dos paradas
  más), con el número que le da el catálogo de UNE. Se pliega con una regla que sale del
  catálogo y no a mano: un identificador del portal que es el número de UNE de una ruta
  del AVL y que no es él mismo una ruta del AVL. Así 101↔102, 1001↔1002 y 1601↔1602
  (números cruzados entre el AVL y UNE, pero las dos son rutas) no se pliegan.
* **El AVL ya tiene las 35**: las 27 del crudo público y las 8 variantes que se separaron
  bus por bus de cuatro códigos del AVL (1101, 1400, 1801 y 1802). La 1401 y la 1402 son
  dos rutas del portal (no hay sufijo que plegar), así que con AVL quedan 32 rutas base;
  en el bootstrap la familia 1400 cuenta como un grupo (pre-registro §5).

`data/` está gitignorado, así que este script solo corre en la máquina del autor
—igual que `make_table_01_funnel.py` con el digest del servidor—. Lo que se
versiona es su salida.

Salidas:
  docs/paper/analysis/network_funnel.csv
  docs/paper/analysis/network_funnel.md

Run:  python docs/paper/analysis/network_funnel.py
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_run import CATALOGUE                                  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "01_raw"
OUT = ROOT / "docs" / "paper" / "analysis" / "network_funnel"

#: sufijos que marcan una variante de una ruta base, no una ruta distinta
VARIANT = re.compile(r"-(ALT|UI)$")
#: las -ALT con número entero (110101 → 1101): las del catálogo con etiqueta `-ALT`
ALT_NUMBERED = {r: c["family"] for r, c in CATALOGUE.items()
                if c["label"].endswith("-ALT")}
#: identificadores del portal que son otra ruta del AVL con su número de UNE (2201 →
#: 20000). Sólo si el número de UNE no es él mismo una ruta del AVL: 101↔102 no.
ALIAS = {c["une_number"]: r for r, c in CATALOGUE.items()
         if c["une_number"] not in CATALOGUE and not c["une_number"].endswith("-ALT")}


def base(route: str) -> str:
    """La ruta base de un identificador: sin sufijo de variante, la -ALT numerada con
    su principal y la 2201 con la 20000."""
    r = VARIANT.sub("", route)
    r = ALT_NUMBERED.get(r, r)
    return ALIAS.get(r, r)


def is_variant(route: str) -> bool:
    return bool(VARIANT.search(route)) or route in ALT_NUMBERED or route in ALIAS


def _haversine_km(a: list[float], b: list[float]) -> float:
    lon1, lat1, lon2, lat2 = map(math.radians, (*a, *b))
    h = (math.sin((lat2 - lat1) / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2)
    return 2 * 6371.0088 * math.asin(math.sqrt(h))


def route_lengths_km(raw: Path) -> dict[str, float]:
    """Longitud de la polilínea de cada identificador: suma de sus segmentos."""
    geo = json.loads((raw / "segments.geojson").read_text(encoding="utf-8"))
    km: dict[str, float] = {}
    for f in geo["features"]:
        c = f["geometry"]["coordinates"]
        linea = f["properties"]["line"]
        km[linea] = km.get(linea, 0.0) + sum(_haversine_km(c[i], c[i + 1])
                                              for i in range(len(c) - 1))
    return km


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default=str(RAW))
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    raw = Path(a.raw)

    missing = [f for f in ("stops_routes.json", "segments_route.json", "segments.geojson",
                           "records")
               if not (raw / f).exists()]
    if missing:
        raise SystemExit(
            f"\nERROR: falta(n) {missing} en {raw}.\n"
            f"  `data/` está gitignorado: este script solo corre donde estén los\n"
            f"  datos crudos. La salida versionada es network_funnel.{{csv,md}}.")

    stops = set(json.loads((raw / "stops_routes.json").read_text(encoding="utf-8")))
    segs = set(json.loads((raw / "segments_route.json").read_text(encoding="utf-8")))
    recs = {d.name for d in (raw / "records").iterdir() if d.is_dir()}

    # ── control: el anidamiento es lo que hace legible el embudo ────────────
    for name, sub, sup in (("records ⊆ segments", recs, segs),
                           ("segments ⊆ stops", segs, stops),
                           ("records ⊆ stops", recs, stops)):
        extra = sub - sup
        if extra:
            raise SystemExit(
                f"\nERROR: {name} NO se cumple; sobran {sorted(extra)}.\n"
                f"  El embudo deja de ser un embudo y la redacción de §3.2 no\n"
                f"  vale: hay que mirar de dónde salen esas rutas antes de nada.")

    # ── control: las rutas con AVL tienen que traer archivos ────────────────
    empty = sorted(d.name for d in (raw / "records").iterdir()
                   if d.is_dir() and not any(d.iterdir()))
    if empty:
        raise SystemExit(f"\nERROR: rutas con carpeta de AVL vacía: {empty}.")

    variants = sorted(r for r in stops if is_variant(r))
    rows = [
        dict(stage="Route identifiers in the portal's stop inventory",
             ids=len(stops), base_routes=len({base(r) for r in stops}),
             source="stops_routes.json",
             note="ordered stop sequence per route; the widest universe the "
                  "portal publishes"),
        dict(stage="…with route geometry",
             ids=len(segs), base_routes=len({base(r) for r in segs}),
             source="segments_route.json",
             note="the three without geometry are university variants: "
                  + ", ".join(sorted(stops - segs))),
        dict(stage="…with AVL records compiled for the analysis window",
             ids=len(recs), base_routes=len({base(r) for r in recs}),
             source="records/<route>/",
             note="the one cut; no identifier with AVL lacks geometry or "
                  "stops. "
                  + (f"{sum(1 for r in recs if CATALOGUE.get(r, {}).get('origin') == 'variant')} "
                     "of them are variants separated vehicle by vehicle from "
                     f"{len({CATALOGUE[r]['avl_group_id'] for r in recs if CATALOGUE.get(r, {}).get('origin') == 'variant'})} "
                     "route codes under which the AVL records both services"
                     if any(CATALOGUE.get(r, {}).get("origin") == "variant" for r in recs)
                     else "")),
        dict(stage="Processed end-to-end by the pipeline",
             ids=len(recs), base_routes=len({base(r) for r in recs}),
             source="Table 1", note="every route with AVL was processed"),
    ]

    hdr = ["stage", "ids", "base_routes", "source", "note"]
    with open(Path(a.out).with_suffix(".csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=hdr)
        w.writeheader()
        w.writerows(rows)

    md = [f"# Network funnel — from the portal's route catalogue to the {len(recs)} "
          "processed",
          "",
          "Derived from the three secondary sources of Section 3.3, collected from the",
          "public web portal of UNE, Hermosillo's integrated transport system",
          "(https://unesonora.com). Route identifiers are not routes one to one: "
          f"{len(variants)} are",
          "variants or duplicates of a base route — a variant suffix (`-ALT`, `-UI`), an",
          "alternative branch numbered after its main route ("
          + ", ".join(f"{r} for {CATALOGUE[r]['label']}" for r in sorted(ALT_NUMBERED))
          + "), or the catalogue number of a route recorded under another"
          + (" (" + ", ".join(f"{a} for {r}" for a, r in sorted(ALIAS.items())) + ")"
             if ALIAS else "")
          + " — over "
          f"{len({base(v) for v in variants})} base routes "
          f"({', '.join(sorted({base(v) for v in variants}))}), so the",
          "*base routes* column folds them in.",
          "",
          "| Stage | Identifiers | Base routes | Source |",
          "|---|---:|---:|---|"]
    for r in rows:
        md.append(f"| {r['stage']} | {r['ids']} | {r['base_routes']} | "
                  f"`{r['source']}` |")
    md += ["",
           "*Notes.* The three sets are strictly nested: every identifier with AVL",
           "records also has geometry and a stop sequence, so the funnel narrows at",
           "exactly one place — **which routes AVL records were compiled for**. The",
           "other base routes are, in the authors' account, non-conventional services —",
           "atypical in frequency or length, or recorded under a route code shared with",
           "other services — that the trip-construction pipeline is not designed to",
           "handle. Length can be checked against the geometry (below); frequency",
           "cannot without their AVL.",
           ""]
    for r in rows:
        md.append(f"- **{r['stage']}** — {r['note']}.")

    # ── la parte comprobable de la causa: longitud contra geometría ─────────
    km = route_lengths_km(raw)
    con_avl = [km[r] for r in sorted(recs) if r in km]
    if len(con_avl) != len(recs):
        raise SystemExit(f"\nERROR: {len(recs) - len(con_avl)} ruta(s) con AVL sin "
                         f"geometría en segments.geojson; el anidamiento dice que no puede ser.")
    lo, hi = min(con_avl), max(con_avl)
    sin_avl = sorted({base(r) for r in stops} - {base(r) for r in recs},
                     key=lambda b: km.get(b, float("inf")))
    con_variantes = {base(v) for v in variants}
    md += ["",
           "## The base routes without AVL, against the geometry",
           "",
           f"Polyline length of the {len(con_avl)} routes with AVL: median "
           f"{statistics.median(con_avl):.1f} km, range {lo:.1f}–{hi:.1f} km.",
           "",
           "| Base route | Length (km) | Shorter than every analysed route | "
           "Variants under the same code |",
           "|---|---:|:---:|:---:|"]
    for b in sin_avl:
        largo = km.get(b)
        md.append(f"| {b} | {largo:.1f} | {'yes' if largo < lo else ''} | "
                  f"{'yes' if b in con_variantes else ''} |" if largo is not None else
                  f"| {b} | — | | {'yes' if b in con_variantes else ''} |")
    corta = {b for b in sin_avl if km.get(b, lo) < lo}
    resto = [b for b in sin_avl if b not in corta and b not in con_variantes]
    n_var = len(con_variantes & set(sin_avl))
    md += ["",
           f"{len(corta)} of the {len(sin_avl)} are shorter than any analysed route"
           + (f" and {n_var} carry variants under the same code" if n_var else "")
           + f"; the remaining {len(resto)} ({', '.join(resto)}) "
           + ("is" if len(resto) == 1 else "are")
           + " of ordinary length, and whether frequency or registration sets "
           + ("it" if len(resto) == 1 else "them")
           + " apart is not recoverable from the files."]
    Path(a.out).with_suffix(".md").write_text("\n".join(md) + "\n",
                                              encoding="utf-8")

    print(f"OK: {Path(a.out).with_suffix('.md')}")
    for r in rows:
        print(f"  {r['ids']:>3} ids / {r['base_routes']:>3} base   {r['stage']}")
    print(f"\n  variantes plegadas: {variants}")
    print(f"  sin geometría     : {sorted(stops - segs)}")
    print(f"  sin AVL           : {len(stops - recs)} identificadores")


if __name__ == "__main__":
    main()
