"""
make_appendix_g_geometry.py
===========================
Tabla S7 del artículo — los tramos de la geometría corregida a mano por los que los buses no
pasan como dice la traza, en ≥ 3 de los 8 meses de enero–agosto 2024 (plan del
manuscrito, 0.7 y decisión V9: el aviso de la geometría).

Es el port de `test/hermosillo-bus-test/analisis/tramos_poco_fiables.py` (27 rutas,
5-oct-2026) a las 35 rutas: la misma lógica y las mismas causas revisadas a mano.

La entrada (`geometry_stretches/`) la escribe el test desde los mapas del crudo contra
la polilínea (`mapas_traza.py`, tramos de 100 m, mes a mes):
`analisis/tramos_entrada_articulo.py`. Trae sólo los tramos con algún mes con problema
y el largo de cada traza. Para las 27 rutas de siempre esos mapas dan lo mismo que los
de agosto, estado por estado, y esta tabla reproduce la de agosto (la comprobación
está abajo: `--check-27`).

Estados de un tramo en un mes (los de `mapas_traza.py`): `saltada` (los buses lo
cruzan sin que se les vea: van por otra calle), `contrasentido`, `desplazada` (la
traza a un lado de por donde pasan) y `poco recorrida`.

Uso
---
    python docs/paper/analysis/make_appendix_g_geometry.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import polars as pl                                             # noqa: E402

from audit_run import CATALOGUE, route_label                    # noqa: E402

TBL = _HERE.parent / "tables"
SRC = _HERE / "geometry_stretches"
MIN_MONTHS = 3

TYPE = {"saltada": "skipped", "contrasentido": "wrong way", "desplazada": "offset",
        "poco recorrida": "rarely travelled"}

#: Desvíos temporales revisados a mano (5-oct-2026): entran con ≥ 1 mes.
TEMPORARY = {
    (101, 19.3, 21.2): "temporary detour, 1 January to 12 March 2024: buses cut along Av. José "
                       "María Mendoza instead of Blvd. Perimetral and Blvd. Agustín Gómez del "
                       "Campo (two stops unserved); from 13 March they follow the path",
    (101, 28.1, 30.2): "temporary detour, 1 January to 12 March 2024 (return direction): along "
                       "Av. José María Mendoza instead of Blvd. Agustín Gómez del Campo and Blvd. "
                       "Perimetral (four stops unserved); from 13 March they follow the path",
}
#: Causas revisadas a mano (4–5 oct 2026): (ruta, km aproximado) → causa.
CAUSE = {
    (401, 43.5): "change of path mid-period: buses switch streets around June 2024 (no single "
                 "path fits January–August)",
    (1301, 14.3): "change of path mid-period: the loop around the block is dropped in May–June "
                  "2024 and one stop stops being served",
    (1501, 22.7): "a loop the raw records do not show (clipped to ~50 m of another path); the "
                  "time across the gap matches it",
    (1501, 5.9): "two variants: about 46% of buses turn one street earlier",
    (301, 2.7): "path drawn on one lane of Blvd. Luis Donaldo Colosio; buses use the next lane "
                "or the stop bay (15–20 m)",
    (301, 3.0): "path drawn on one lane of Blvd. Luis Donaldo Colosio; buses use the next lane "
                "or the stop bay (15–20 m)",
    (301, 6.8): "corner drawn about 30 m from where buses turn",
    (1502, 15.9): "a loop only some buses make",
    (501, 37.7): "two variants: about 44% of buses skip that loop",
    (1001, 37.7): "dead-end branch where buses turn at different points (100–320 m)",
    (601, 21.4): "a loop only some buses make",
    # Revisados a mano el 10-oct-2026 (V9 del manuscrito), con los registros de ene–ago.
    # Ahí la traza de la 1401 va por una calle paralela, a 110–150 m de la de la 1402.
    # - En la de la 1401 no hay registros. En la de la 1402 hay miles de buses de la 1402
    #   por cada 100 m, pero de buses de la 1401 sólo unos 40 y unos 200 en 8 meses.
    # - Los buses de la 1401 cruzan los huecos de ~1.2 km sin registro intermedio (5,756 y
    #   5,526 veces) en una mediana de 238 y 181 s, a 20–26 km/h: el tiempo de recorrerlos.
    # Es el caso de la 1501: el crudo (group_id 1400) viene recortado a ~50 m de otra traza.
    # - Confirmado con la ventana sin recorte de la BD (27-jun a 16-jul-2025, `disabled = 1`).
    #   Sobre la calle de la 1401 hay 153 y 60 registros (31 y 22 buses en 14 y 13 días),
    #   todos con `disabled = 1`: los que el recorte quita.
    (1401, 25.1): "a stretch the raw records do not show (clipped to ~50 m of another path; the "
                  "parallel street of route 1402 is recorded, and these buses are not on it); the "
                  "time across the gap matches it",
    (1401, 33.7): "a stretch the raw records do not show (clipped to ~50 m of another path; the "
                  "parallel street of route 1402 is recorded, and these buses are not on it); the "
                  "time across the gap matches it",
}


def cause(r: int, k0: float, typ: str, k1: float, L: float) -> str:
    for (cr, a, b), txt in TEMPORARY.items():
        if cr == r and a - 0.05 <= k0 < b:
            return txt
    for (cr, ck), txt in CAUSE.items():
        if cr == r and abs(ck - k0) <= 0.25:
            return txt
    if k0 < 0.5 or k1 > L - 0.5:
        return "manoeuvre at the terminal (next to km 0)"
    if typ == "contrasentido":
        return "short stretch driven the other way (U-turn, return or one-way street)"
    if typ == "desplazada":
        return "path drawn 15–30 m from where buses run"
    return "not reviewed"


def build(st: pl.DataFrame, L: pl.DataFrame) -> pl.DataFrame:
    x = st.with_columns(pl.col("estados").str.split(";")).explode("estados").with_columns(
        pl.col("estados").str.slice(0, 2).alias("mes"),
        pl.col("estados").str.slice(3).alias("estado")).drop("estados")
    temporal = pl.lit(False)
    for (r, a, b) in TEMPORARY:
        temporal = temporal | ((pl.col("ruta") == r) & (pl.col("km_ini") >= a - 1e-6)
                               & (pl.col("km_ini") < b - 1e-6))
    bad = pl.col("estado") != "ok"
    x = (x.group_by("ruta", "tramo", "km_ini", "lat", "lon")
          .agg(bad.sum().alias("meses_mal"),
               pl.col("estado").filter(bad).mode().sort().first().alias("tipo"),
               pl.col("mes").filter(bad).sort().str.join(" ").alias("meses"))
          .filter((pl.col("meses_mal") >= MIN_MONTHS) | (temporal & (pl.col("meses_mal") >= 1)))
          .sort("ruta", "tramo")
          .with_columns(criterio=pl.when(temporal).then(pl.lit("temporary detour"))
                        .otherwise(pl.lit(f"≥ {MIN_MONTHS} of 8 months"))))
    x = x.with_columns(g=((pl.col("ruta") != pl.col("ruta").shift())
                          | (pl.col("tramo") - pl.col("tramo").shift() != 1)
                          | (pl.col("criterio") != pl.col("criterio").shift()))
                       .fill_null(True).cum_sum())
    z = (x.group_by("g", maintain_order=True)
          .agg(pl.col("ruta").first(), pl.col("criterio").first(),
               pl.col("km_ini").min().alias("km_ini"),
               (pl.col("km_ini").max() + 0.1).round(1).alias("km_fin"),
               pl.len().alias("n"), pl.col("tipo").mode().sort().first(),
               pl.col("meses_mal").max(),
               pl.col("meses").sort_by("meses_mal", descending=True).first(),
               # el punto medio del tramo seguido: el del tramo de 100 m del medio
               pl.col("lat").get(pl.len() // 2), pl.col("lon").get(pl.len() // 2))
          .join(L, on="ruta")
          .with_columns(largo_m=pl.col("n") * 100,
                        pct=(100 * pl.col("n") * 0.1 / pl.col("L_km")).round(2))
          .sort("ruta", "km_ini").drop("g", "n"))
    z = z.with_columns(pl.struct("ruta", "km_ini", "tipo", "km_fin", "L_km").map_elements(
        lambda s: cause(s["ruta"], s["km_ini"], s["tipo"], s["km_fin"], s["L_km"]),
        return_dtype=pl.Utf8).alias("causa"))
    return z.select(
        pl.col("ruta").cast(pl.Utf8).alias("route"),
        pl.col("ruta").cast(pl.Utf8).replace_strict(
            {r: c["une_number"] for r, c in CATALOGUE.items()}, default=None)
          .alias("une_number"),
        pl.col("km_ini").alias("km_from"), pl.col("km_fin").alias("km_to"),
        pl.col("largo_m").alias("length_m"), pl.col("pct").alias("pct_of_path"),
        pl.col("criterio").alias("criterion"),
        pl.col("tipo").replace_strict(TYPE).alias("type"),
        pl.col("meses_mal").alias("months_with_problem"),
        pl.col("meses").alias("which_months"), pl.col("causa").alias("cause"),
        "lat", "lon")


def summary(t: pl.DataFrame, L: pl.DataFrame) -> dict:
    per = t.filter(pl.col("criterion") != "temporary detour")
    tem = t.filter(pl.col("criterion") == "temporary detour")
    by = (L.with_columns(pl.col("ruta").cast(pl.Utf8).alias("route"))
           .join(per.group_by("route").agg(pl.col("pct_of_path").sum()), on="route",
                 how="left").fill_null(0))
    return {"n": per.height, "km": per["length_m"].sum() / 1000, "L": L["L_km"].sum(),
            "pct": 100 * per["length_m"].sum() / 1000 / L["L_km"].sum(),
            "lo": by["pct_of_path"].min(), "hi": by["pct_of_path"].max(),
            "med": by["pct_of_path"].median(), "routes": L.height,
            "none": sorted(set(by["route"]) - set(per["route"]), key=int),
            "tn": tem.height, "tkm": tem["length_m"].sum() / 1000,
            "troutes": sorted(set(tem["route"]), key=int)}


def to_md(t: pl.DataFrame, s: dict) -> str:
    L = [f"**Table S7. Stretches of the hand-corrected route geometry that buses do not "
         f"follow ({s['routes']} routes, January–August 2024).**", ""]
    L.append(
        f"Each route's path is cut into 100 m stretches and compared, month by month, with "
        f"the raw position reports: a stretch is *skipped* when buses cross it without being "
        f"seen (they run along another street), *wrong way* when most reports near it head "
        f"the other way, *offset* when the path is drawn to one side of where buses run, and "
        f"*rarely travelled* when far fewer buses pass than on the rest of the route. Listed "
        f"are runs of consecutive stretches with a problem in at least {MIN_MONTHS} of the 8 "
        f"months, and temporary detours reviewed by hand. Persistent stretches add up to "
        f"{s['km']:.1f} km of {s['L']:,.0f} km of path ({s['pct']:.2f}%); by route, "
        f"{s['lo']:.1f}–{s['hi']:.1f}% (median {s['med']:.1f}%), and {len(s['none'])} routes "
        f"have none. Temporary detours: {s['tkm']:.1f} km on route "
        f"{', '.join(route_label(r) for r in s['troutes'])}.")
    L += ["", "| Route | km | Length (m) | % of path | Criterion | Type | Months | Cause |",
          "|---|---|---:|---:|---|---|---|---|"]
    for r in t.iter_rows(named=True):
        L.append(f"| {route_label(r['route'])} | {r['km_from']:.1f}–{r['km_to']:.1f} | "
                 f"{r['length_m']} | {r['pct_of_path']:.2f} | {r['criterion']} | "
                 f"{r['type']} | {r['which_months']} | {r['cause']} |")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Tabla S7 — tramos poco fiables")
    ap.add_argument("--out", default=str(TBL / "appendix_g_geometry"))
    ap.add_argument("--check-27", action="store_true",
                    help="resumen de las 27 rutas de agosto, para compararlo con la tabla "
                         "del test (50 tramos, 9.4 km de 1,125.2, 0.84 %)")
    a = ap.parse_args()
    st = pl.read_csv(SRC / "stretch_states.csv")
    L = pl.read_csv(SRC / "route_lengths.csv")
    t = build(st, L)
    s = summary(t, L)
    t.write_csv(a.out + ".csv")
    md = to_md(t, s)
    Path(a.out + ".md").write_text(md, encoding="utf-8")
    print(f"persistentes: {s['n']} tramos · {s['km']:.1f} km de {s['L']:,.1f} "
          f"({s['pct']:.2f} %) · por ruta {s['lo']:.1f}–{s['hi']:.1f} % (mediana "
          f"{s['med']:.1f} %) · sin ninguno: {s['none']}")
    print(f"desvíos temporales: {s['tn']} tramos · {s['tkm']:.1f} km · rutas {s['troutes']}")
    print(f"sin revisar a mano: {t.filter(pl.col('cause') == 'not reviewed').height} tramos: "
          f"{t.filter(pl.col('cause') == 'not reviewed').select('route', 'km_from', 'type').rows()}")
    if a.check_27:
        old = [r for r, c in CATALOGUE.items() if c.get("origin") == "public"]
        s27 = summary(t.filter(pl.col("route").is_in(old)),
                      L.filter(pl.col("ruta").cast(pl.Utf8).is_in(old)))
        print(f"las 27: {s27['n']} tramos · {s27['km']:.1f} km de {s27['L']:,.1f} "
              f"({s27['pct']:.2f} %) · sin ninguno: {s27['none']}")
    print(f"→ {a.out}.{{csv,md}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
