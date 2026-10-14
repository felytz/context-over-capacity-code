"""Tabla S2 del paper — el funnel de datos por ruta, con la vuelta.

Fuentes:
  results/digest/routes/<ruta>.json        (el digest de la corrida: `trips_summary.funnel`
                                            con su bloque `cruce`)
  <root>/data/02_interim/routes/<r>/frames/<rango>/*_<tag>_meta.json   (la etapa de la malla)
  docs/paper/analysis/fleet_concentration/…csv                         (Active 90 %)
  docs/paper/analysis/route_catalogue.csv  (número y nombre de UNE, -ALT; vía audit_run)

Con la vuelta (TRIP_CUT=cruce; docs/PREREGISTRO_VUELTA.md)
---------------------------------------------------------
La unidad es la vuelta: de un cruce del punto de corte de la ruta al siguiente, dentro
de un mismo **tramo de servicio** (los registros seguidos de un bus en un día; el tramo se
parte al cambiar de día, en un hueco de más de 20 min y en un salto imposible que no sea
un atajo). Las columnas de agosto que salían del descubrimiento de terminales —estrato,
número de terminales, viajes direccionales, % self-loop— ya no aplican: con la vuelta
salían en 0 o vacías sin error (riesgo 3 del plan). Las que entran:

  Service stretches  tramos de servicio (`cruce.tramos`); la nota da cuántos se parten
                     por día, por hueco y por salto
  Coverage (%)       registros que caen en vueltas válidas / registros de la ruta
                     (`cruce.cobertura_pct`)
  Round trips        vueltas válidas (`cruce.viajes_validos`)
  With shortcut (%)  vueltas válidas con un atajo (un salto hacia adelante de ≤ 8 km que
                     no parte el tramo); salen del frame (P3)
  Modeled            vueltas válidas sin atajo (`trips_valid_models`): las que llegan a la
                     malla
  Grid / Grid kept   la etapa de la malla, de los metas del frame (como en agosto)
  Cut quality        ok / review / poor, con los umbrales del pre-registro §3
                     (`leg_inclusion.cut_quality`, la misma función que decide C1)

El punto de corte de cada ruta, su validación contra la duración de UNE y los cruces
inferidos van en la Tabla S4 (`make_appendices.py`).

⚠ LA ETAPA DE LA MALLA (2026-07-25). Hasta la fase 5 el funnel terminaba en
"modeled trips" y NUNCA mostraba cuántos de esos viajes entraban de verdad al
modelado: con q=0.95 la matriz retenía la mediana del 7 % por leg. Desde entonces se
reporta explícitamente (*Grid*, *Grid kept*), tomada de los metadatos del frame de
evaluación, que es el objeto que define la muestra realmente analizada.

Salidas:
  docs/paper/tables/table_01_funnel.md   — tabla lista para el manuscrito
  docs/paper/tables/table_01_funnel.csv  — versión completa (columnas extra)

Run:  python docs/paper/analysis/make_table_01_funnel.py --root artefacts/run_bundle_<…>
      (`--partial` para un piloto: sólo las rutas que corrieron)
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
import sys
from pathlib import Path

# El `print(md)` final escupe «≤», «—» y «⚠», que una consola cp1252 de Windows
# no sabe codificar: la corrida moría con UnicodeEncodeError DESPUÉS de haber
# escrito los dos ficheros, y parecía fallida cuando había salido bien.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_run import CATALOGUE, ROUTES, route_label                # noqa: E402
from leg_inclusion import cut_quality                              # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
DIGEST_ROUTES = ROOT / "results" / "digest" / "routes"
OUT_DIR = ROOT / "docs" / "paper" / "tables"
RANGE = "2024_01-08_tmin"
TAG = "q05c100"          # régimen primario: q=0.05, min_coverage=1.0
FLEET_CSV = (ROOT / "docs" / "paper" / "analysis" / "fleet_concentration"
             / "fleet_concentration_2024_01-08_tmin.csv")
#: la calidad del corte, en inglés para la tabla
QUALITY_EN = {"ok": "ok", "revisar": "review", "mal": "poor", "": "—"}


def load_fleet() -> dict[str, dict]:
    """Concentración de flota por ruta (salida de compute_fleet_concentration.py).
    Active (90%) = camiones que hacen el 90% de los viajes modelados."""
    if not FLEET_CSV.is_file():
        print(f"⚠ sin {FLEET_CSV} — columna Active(90%) vacía "
              f"(corre jobs/job_fleet_concentration.slrm y baja el CSV)")
        return {}
    out: dict[str, dict] = {}
    with open(FLEET_CSV, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            g = lambda k: (row.get(k) or "").strip()
            out[str(row["route"])] = {
                "n_buses_90": int(g("n_buses_90")) if g("n_buses_90") else None,
                "share_top10_pct": float(g("share_top10_pct")) if g("share_top10_pct") else None,
                "eff_fleet_90_pct": float(g("eff_fleet_90_pct")) if g("eff_fleet_90_pct") else None,
            }
    return out


def load_frames(root: Path, tag: str, rng: str) -> tuple[dict[str, dict], list[dict]]:
    """Etapa de la matriz de trayectorias, por ruta, desde los metas del frame.

    El frame es la definición de la muestra realmente analizada (misma para
    baselines, XGBoost y LSTM), así que es la fuente correcta de esta etapa —
    no una reconstrucción. Su `n_trips_raw` es el `trips_valid_models` del funnel
    (las vueltas válidas sin atajo), lo que cierra el funnel de punta a punta.
    """
    out: dict[str, dict] = {}
    legs: list[dict] = []
    base = root / "data" / "02_interim" / "routes"
    for p in sorted(base.glob(f"*/frames/{rng}/*_{tag}_meta.json")):
        try:
            m = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        legs.append(m)
        a = out.setdefault(str(m["route"]),
                           {"n_legs": 0, "trips_in": 0, "grid_trips": 0,
                            "cv_trips": 0, "capped": 0})
        a["n_legs"]     += 1
        a["trips_in"]   += int(m["n_trips_raw"])
        a["grid_trips"] += int(m["M_retained"])
        a["cv_trips"]   += int(m["M"])
        a["capped"]     += 1 if m.get("subsampled") else 0
    for a in out.values():
        a["grid_kept_pct"] = (100.0 * a["grid_trips"] / a["trips_in"]
                              if a["trips_in"] else None)
    return out, legs


def _digest(route: str) -> dict:
    p = DIGEST_ROUTES / f"{route}.json"
    if not p.is_file():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def load_rows(frames: dict[str, dict], partial: bool) -> tuple[list[dict], list[str]]:
    """Una fila por ruta de ROUTES, en su orden. Devuelve también las que no tienen
    funnel de la vuelta (sólo se toleran con `partial`)."""
    fleet = load_fleet()
    rows, missing = [], []
    for route in ROUTES:
        d = _digest(route)
        ts = d.get("trips_summary") or {}
        f = ts.get("funnel") or {}
        cr = f.get("cruce") or {}
        if f.get("trip_cut") != "cruce" or not cr:
            missing.append(route)
            continue
        t = ts.get("trips") or {}
        pp = cr.get("partidos_por") or {}
        q, why = cut_quality(cr.get("cobertura_pct"), cr.get("cruces_inferidos_pct"),
                             cr.get("raras_pct"), cr.get("viajes_con_atajo_pct"))
        c = CATALOGUE[route]
        fr = frames.get(route, {})
        rows.append({
            "route": route,
            "name": route_label(route),
            "une_number": c["une_number"],
            "une_name": c["une_name"],
            "origin": c["origin"],
            "family": c["family"],
            "pings_raw": f["pings_raw"],
            "pct_kept": f["pct_pings_kept"],
            "fleet": f["buses"],
            "fleet_active90": fleet.get(route, {}).get("n_buses_90"),
            "fleet_eff90_pct": fleet.get(route, {}).get("eff_fleet_90_pct"),
            "fleet_top10_pct": fleet.get(route, {}).get("share_top10_pct"),
            "loop_km": cr.get("lazo_km"),
            "cut_km": (cr.get("cortes_km") or [None])[0],
            "stretches": cr.get("tramos"),
            "split_day": pp.get("dia"),
            "split_gap": pp.get("hueco"),
            "split_jump": pp.get("salto"),
            "coverage_pct": cr.get("cobertura_pct"),
            "trips_complete": cr.get("viajes_completos"),
            "lost_crossing": cr.get("cruce_perdido"),
            "trips_valid": cr.get("viajes_validos"),
            "shortcut_pct": cr.get("viajes_con_atajo_pct"),
            "shortcut_out": f.get("trips_atajo_out_of_models"),
            "inferred_pct": cr.get("cruces_inferidos_pct"),
            "rare_pct": cr.get("raras_pct"),
            "trips_modeled": f.get("trips_valid_models"),
            # ── etapa de la matriz de trayectorias (frame) ──
            "grid_trips": fr.get("grid_trips"),
            "grid_kept_pct": fr.get("grid_kept_pct"),
            "cv_trips": fr.get("cv_trips"),
            "n_legs_frame": fr.get("n_legs"),
            "legs_capped": fr.get("capped"),
            "dur_med_min": cr.get("dur_med_min"),
            "dur_p10_min": cr.get("dur_p10_min"),
            "dur_p90_min": cr.get("dur_p90_min"),
            "median_trip_km": (t.get("dist_km") or {}).get("p50"),
            "quality": q,
            "quality_reasons": why,
        })
    return rows, missing


def fmt_int(x) -> str:
    return f"{x:,}" if x is not None else "—"


def fmt_f(x, nd: int = 1) -> str:
    return f"{x:.{nd}f}" if x is not None else "—"


def write_md(rows: list[dict], legs: list[dict], missing: list[str]) -> str:
    lines = [
        "**Table S2. Per-route data funnel (January–August 2024 analysis window).**",
        "Raw AVL records and cleaning retention are computed over the full downloaded",
        "range. A *service stretch* is a vehicle's continuous operation on the route;",
        "*coverage* is the share of the route's records that fall inside valid round",
        "trips; *round trips* run from one crossing of the route's cut point to the",
        "next within a stretch; *with shortcut* are those containing a forward jump",
        "along the route that the vehicle could not have driven, which stay out of",
        "the models; *modeled* round trips reach the per-route modeling dataset;",
        "**grid** round trips are those retained by the trajectory matrix and",
        "therefore actually analyzed. *Cut quality* applies the pre-specified",
        "thresholds (Table S4).",
        "",
        "| Route | UNE no. | Raw records | Kept (%) | Fleet | Active (90%) | Loop (km) | Service stretches | Coverage (%) | Round trips | With shortcut (%) | Modeled | Grid | Grid kept (%) | Cut quality |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    tot = {k: 0 for k in ("pings_raw", "stretches", "trips_valid", "trips_modeled",
                          "grid_trips", "cv_trips", "split_day", "split_gap",
                          "split_jump", "shortcut_out", "lost_crossing")}
    for r in rows:
        for k in tot:
            tot[k] += r[k] or 0
        act = str(r["fleet_active90"]) if r.get("fleet_active90") is not None else "—"
        lines.append(
            f"| {route_label(r['route'], number=True)} | {r['une_number']} | "
            f"{fmt_int(r['pings_raw'])} | {r['pct_kept']:.1f} | {r['fleet']} | {act} | "
            f"{fmt_f(r['loop_km'])} | {fmt_int(r['stretches'])} | "
            f"{fmt_f(r['coverage_pct'])} | {fmt_int(r['trips_valid'])} | "
            f"{fmt_f(r['shortcut_pct'])} | {fmt_int(r['trips_modeled'])} | "
            f"{fmt_int(r.get('grid_trips'))} | {fmt_f(r.get('grid_kept_pct'))} | "
            f"{QUALITY_EN.get(r['quality'], r['quality'])} |")
    _gk_tot = (100.0 * tot["grid_trips"] / tot["trips_modeled"]
               if tot["trips_modeled"] else 0.0)
    # El conteo se DERIVA de las filas, no se escribe a mano (hasta 2026-09-06 decía
    # `Total (27)` como constante, y un día imprimió «Total (27)» sobre cero rutas).
    lines.append(
        f"| **Total ({len(rows)})** |  | **{fmt_int(tot['pings_raw'])}** |  |  |  |  | "
        f"**{fmt_int(tot['stretches'])}** |  | **{fmt_int(tot['trips_valid'])}** |  | "
        f"**{fmt_int(tot['trips_modeled'])}** | **{fmt_int(tot['grid_trips'])}** | "
        f"**{_gk_tot:.1f}** |  |")

    # ⚠ La razón de flota: `eff_fleet_90_pct` YA es n_buses_90 / fleet_total, de la
    # misma fuente (paso 54: dividir `fleet_active90` entre `fleet` del digest mezclaba
    # dos poblaciones). Un decimal, como la fuente.
    _ratios = [r["fleet_eff90_pct"] for r in rows if r.get("fleet_eff90_pct")]
    _med = round(st.median(_ratios), 1) if _ratios else None
    _conc = (f"a median of {_med}% of the vehicles that operate a route"
             if _med else "far below the observed fleet")
    # Cuántas tienen otro número en el catálogo de UNE: se cuenta, no se escribe.
    _renum = sum(1 for r in rows if r["une_number"] not in (r["route"], r["name"]))
    _cov = [r["coverage_pct"] for r in rows if r.get("coverage_pct") is not None]
    _rev = [r for r in rows if r["quality"] == "revisar"]
    _poor = [r for r in rows if r["quality"] == "mal"]

    # Cifras de la etapa de la matriz, derivadas de los metas de los frames
    # (no se codifican a mano: si cambia el régimen, la nota cambia sola).
    _n_legs   = len(legs)
    _gk_med   = st.median([m["retention_pct"] for m in legs]) if legs else 0.0
    _Q        = legs[0]["n_route_quantile"] if legs else 0.05
    _MAXTRIPS = legs[0].get("max_trips") if legs else None
    _capped   = sum(1 for m in legs if m.get("subsampled"))
    _cv_tot   = tot["cv_trips"]
    lines += [
        "",
        "*Notes.* Route numbers are those of the AVL records; *UNE no.* is the",
        "number under which the operator's current catalogue lists the route, which",
        f"differs for {_renum} of them. The variant routes are the two services of four",
        "route codes that the AVL records do not distinguish, separated vehicle by",
        "vehicle (Table S1). *Fleet* is the number of distinct vehicles observed on",
        "the route; *Active (90%)* is the number of vehicles that perform 90% of the",
        f"route's modeled round trips — {_conc}.",
        f"Stretches end at the end of the service day ({fmt_int(tot['split_day'])}),",
        f"at a gap of more than 20 minutes between records ({fmt_int(tot['split_gap'])})",
        f"or at a jump along the route that cannot have been driven"
        f" ({fmt_int(tot['split_jump'])}); only round trips completed within one stretch",
        "are kept, which is why coverage falls short of 100%"
        + (f" (median {st.median(_cov):.1f}%, range {min(_cov):.1f}–{max(_cov):.1f}%)."
           if _cov else "."),
        f"{fmt_int(tot['lost_crossing'])} round trips that travelled more than 1.5 times",
        "the loop (a missed crossing) are not counted as valid, and"
        f" {fmt_int(tot['shortcut_out'])} valid round trips with a shortcut stay out of",
        "the models: the grid cells they skip have no observed time.",
        (f"{len(_rev)} route{'s are' if len(_rev) != 1 else ' is'} rated *review* "
         f"({', '.join(route_label(r['route']) for r in _rev)}) and kept under the "
         "pre-specified criterion" if _rev else "No route is rated *review*")
        + (f"; {len(_poor)} rated *poor* "
           f"({', '.join(route_label(r['route']) for r in _poor)}) "
           "fail the criterion." if _poor else "; none is rated *poor*."),
        "",
        "*Grid* is the last stage of the funnel and the one that defines what",
        "is actually analyzed: the trajectory matrix places a common distance grid,",
        "laid on the progress along the route polyline from the cut point, at the",
        f"{int(_Q * 100)}th percentile of round-trip length and keeps the round trips",
        "that cover it completely, truncating the longer ones, which discards distance",
        "beyond the grid but no round trips. It",
        f"retains {_gk_tot:.1f}% of modeled round trips overall ({_gk_med:.1f}% median "
        f"across the {_n_legs} routes). Grid round trips are the denominator of every",
        "model comparison in this paper.",
    ]
    if _MAXTRIPS:
        lines += [
            "",
            f"For tractability the cross-validation is capped at {_MAXTRIPS:,} round "
            f"trips per route (a uniform random subsample applied identically to every "
            f"algorithm, so the comparison remains paired); {_capped} of the {_n_legs} "
            f"routes hit the cap, leaving {fmt_int(_cv_tot)} round trips in the "
            f"cross-validation. The cap bounds the cost of the nearest-neighbour and "
            f"kernel baselines, which are quadratic in the number of round trips.",
        ]
    if missing:
        lines += ["", f"⚠ *Partial run: {len(rows)} of {len(ROUTES)} routes. Without a "
                  f"round-trip funnel: {', '.join(missing)}.*"]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Tabla S2 — funnel por ruta (la vuelta)")
    ap.add_argument("--root", default=None,
                    help="raíz del árbol de datos con los frames (default: la "
                         "del repo). Úsalo para leer un paquete descargado.")
    ap.add_argument("--tag", default=TAG, help="régimen (default q05c100)")
    ap.add_argument("--range", dest="rng", default=RANGE)
    ap.add_argument("--partial", action="store_true",
                    help="tolera rutas sin funnel de la vuelta (un piloto); la tabla "
                         "lo dice al pie")
    a = ap.parse_args()

    root = Path(a.root).resolve() if a.root else ROOT

    # ⚠ GUARD DEL DIGEST (2026-09-06). Sin él, correr sin `results/digest/routes/`
    # NO falla: la tabla sale con todas las etapas del funnel en CERO. Y un digest
    # parcial es igual de silencioso y más difícil de ver, porque la tabla sale bien
    # formada con las rutas que encontró. Por eso se exige el censo completo de ROUTES,
    # cada una con su bloque `cruce`, salvo con `--partial`.
    frames, legs = load_frames(root, a.tag, a.rng)
    rows, missing = load_rows(frames, a.partial)
    if missing and not a.partial:
        raise SystemExit(
            f"✗ {len(missing)} de {len(ROUTES)} rutas sin funnel de la vuelta en el "
            f"digest: {', '.join(missing)}.\n"
            f"  Buscado en: {DIGEST_ROUTES}\n"
            f"  Sin el censo completo la tabla sale con rutas de menos y no se nota al\n"
            f"  leerla. `reproduce_tables.py --run <bundle>` copia ahí el digest del\n"
            f"  paquete (route_reports/routes/). Para un piloto: --partial.")
    if not rows:
        raise SystemExit(f"✗ ninguna ruta con funnel de la vuelta en {DIGEST_ROUTES}")
    if not frames:
        raise SystemExit(
            f"✗ no hay frames del régimen {a.tag} bajo {root}.\n"
            f"  La etapa 'grid' es obligatoria: sin ella la tabla vuelve a\n"
            f"  ocultar sobre qué fracción de las vueltas se mide. Pasa --root al\n"
            f"  paquete descargado.")
    print(f"  frames: {len(legs)} rutas · régimen {a.tag}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    md = write_md(rows, legs, missing)
    (OUT_DIR / "table_01_funnel.md").write_text(md, encoding="utf-8")
    with open(OUT_DIR / "table_01_funnel.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(md)
    print("total rutas:", len(rows), "| sin funnel:", missing or "ninguna")
    print("salidas:", OUT_DIR / "table_01_funnel.md", "·", OUT_DIR / "table_01_funnel.csv")


if __name__ == "__main__":
    main()
