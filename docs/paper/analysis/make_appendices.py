"""
make_appendices.py
==================
Los apéndices del Online Resource que salen de CSV pequeños, en inglés y listos para
insertar.

  **A** · `appendix_a_grid_representativity.md` (Tabla S3) — cuánto y a quién retiene
        la colocación de la malla, por régimen. De `grid_representativity.csv`, que se
        calcula en el servidor (`jobs/job_appendices.slrm`).
  **B** · `appendix_b_excluded_legs.md` (Tabla S6) — las rutas que el criterio
        pre-registrado deja fuera, con su M y su motivo, y las que entran marcadas.
  **C** · `appendix_c_cut_points.md` (Tabla S4) — el punto de corte de cada ruta, la
        regla que lo ubicó y la vuelta medida contra la duración que declara UNE.
  **D** · `appendix_d_cut_invariance.md` (Tabla S5) — la duración de la vuelta no
        depende del punto de corte.

Con la vuelta (rama `vuelta`)
-----------------------------
Las Tablas S3 y S4 de agosto medían el detector de terminales (la sensibilidad del
conteo al radio ε del DBSCAN y la de los parámetros del detector de reversión). Con la
vuelta ese detector ya no corta: lo que hay que mostrar es dónde se corta y que la
duración no depende de eso (decisión P2 del plan). Los scripts de agosto
(`terminal_eps_sensitivity.py`, `reversal_params_sensitivity.py`) se quedan en el repo
para esa corrida, pero ya no entran a la tabla. La Tabla S1 (variantes) la escribe
`variants_separation.py`, porque necesita el crudo.

Fuentes de C y D
----------------
* C: `cortes/cortes_vuelta.csv` y `cortes/vueltas_une.csv` (las entradas del corte, con su
  sha256 en el pre-registro) y `docs/paper/tables/table_01_funnel.csv` (la vuelta medida
  en ESTA corrida, del bloque `cruce` del funnel).
* D: `cut_invariance_rest_point.csv` y `cut_invariance_methods.csv`, copias byte a byte de
  `p2_descanso_vs_hoy.csv` y `p2_invariancia_metodos.csv` del test
  (`test/hermosillo-bus-test/reports/plan_articulo_2026-10-06/`, la evidencia de P2):
    - el primero, 15 rutas corridas con el pipeline (el mismo código del corte que el
      artículo) en enero–agosto, con su punto de hoy y con el de mayor descanso
      (sha256 `673695256c10a76a71f7aaac13614a70eb2b4b5fcd0fb3cfe046782dfc8c66c8`);
    - el segundo, el comparativo de métodos sobre las 27 rutas del crudo público
      (sha256 `87afc14540c25b4c80e4484364895a8071e2b65a7c388cd8c1d194db34378140`).

Uso
---
    python docs/paper/analysis/make_appendices.py
"""
from __future__ import annotations

import csv
import statistics
import sys
from pathlib import Path
from typing import Dict, List

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import polars as pl                                             # noqa: E402

from audit_run import ROUTES, route_label, route_order          # noqa: E402
from leg_inclusion import VUELTA_MAL, VUELTA_REVISAR            # noqa: E402

REPO = _HERE.parents[2]
ANA = REPO / "docs" / "paper" / "analysis"
OUT = REPO / "docs" / "paper" / "tables"
CORTES = REPO / "cortes"
QUALITY_EN = {"ok": "ok", "revisar": "review", "mal": "poor", "": "—"}
TYPE_EN = {"una": "one terminal", "dos": "two terminals", "variante": "variant"}
#: la regla que ubicó cada punto (cortes_vuelta.csv, `metodo`; comparativo de métodos §8)
RULE_EN = {
    "km0": "km 0 of the portal polyline",
    "descanso1": "main layover point",
    "descanso2": "layover points (two)",
    "consenso1": "consensus of layover, start of service and lowest occupancy",
    "intervalos1": "dispatch point (where headways become regular)",
    "intervalos": "dispatch points (two)",
    "ocupacion": "lowest occupancy (two)",
    "explorador": "terminals with most complete cycles",
    "dwell_art": "layover-cluster terminals",
    "geo_extremos": "geometric extremes of the polyline",
}


def _csv(path: Path) -> List[Dict[str, str]]:
    with open(path, encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def _f(x) -> float | None:
    return None if x in (None, "", "nan") else float(x)


# ── A · la malla ────────────────────────────────────────────────────────────

def appendix_a() -> str:
    # Texto: `name` trae las -ALT (1101-ALT…) más allá de las filas con que polars infiere
    d = pl.read_csv(ANA / "grid_representativity.csv",
                    schema_overrides={"route": pl.Utf8, "name": pl.Utf8})
    n_routes = d["route"].n_unique()
    g = (d.group_by(["q", "min_coverage"])
         .agg(pl.len().alias("legs"),
              pl.col("retention_pct").median().alias("ret"),
              pl.col("smd_trip_len_m").abs().median().alias("smd_len"),
              pl.col("smd_duration_s").abs().median().alias("smd_dur"),
              pl.col("smd_hour_dep").abs().median().alias("smd_hour"),
              pl.col("tvd_month").median().alias("tvd_month"),
              (pl.col("smd_trip_len_m").abs() > 0.25).sum().alias("biased"),
              (pl.col("bus_kept_frac") * 100).median().alias("bus"))
         .sort(["min_coverage", "q"], descending=[False, True]))

    def _v(x, fmt):
        return "—" if x is None else format(x, fmt)

    L: List[str] = []
    L.append("**Appendix A. What the grid placement retains, and from whom.**")
    L.append("")
    L.append(
        "The trajectory matrix places a common distance grid, laid on the progress along "
        "the route polyline from the cut point, at the `q`-quantile of round-trip length "
        "within each route and keeps round trips covering at least `min_coverage` of it. "
        f"The table sweeps both parameters over the {n_routes} routes. Standardized mean "
        "differences compare the **retained sample against the route's full population** "
        "of round trips — not against the discarded ones, which is the contrast that "
        "answers whether the analysed sample represents the route. `|SMD| > 0.25` is the "
        "conventional threshold for a material shift. The last column is the share of the "
        "route's vehicles that appear at all in the retained sample. The row in bold is the "
        "regime used in this paper (q = 0.05, coverage 1.0). A dash marks a regime that "
        "retains too few round trips to compare.")
    L.append("")
    L.append("| q | min. coverage | routes | median retention | \\|SMD\\| length | "
             "\\|SMD\\| duration | \\|SMD\\| departure hour | TVD month | "
             "routes with \\|SMD\\| > 0.25 | vehicles retained |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in g.iter_rows(named=True):
        key = abs(r["q"] - 0.05) < 1e-9 and abs(r["min_coverage"] - 1.0) < 1e-9
        b = "**" if key else ""
        L.append(
            f"| {b}{r['q']:.2f}{b} | {b}{r['min_coverage']:.2f}{b} | {r['legs']} | "
            f"{b}{r['ret']:.1f}%{b} | {b}{_v(r['smd_len'], '.3f')}{b} | "
            f"{_v(r['smd_dur'], '.3f')} | {_v(r['smd_hour'], '.3f')} | "
            f"{_v(r['tvd_month'], '.3f')} | "
            f"{b}{r['biased']}/{r['legs']}{b} | {b}{_v(r['bus'], '.1f')}%{b} |")
    L.append("")

    # La nota se escribe con los números de la tabla. ⚠ La de agosto decía «The 0.113
    # residual length shift at q = 0.05» y describía q = 0.95 como «draws on under a
    # fifth of the vehicles»: cifras de esa corrida, escritas a mano.
    def _row(q, cov):
        x = g.filter(((pl.col("q") - q).abs() < 1e-9)
                     & ((pl.col("min_coverage") - cov).abs() < 1e-9))
        return x.row(0, named=True) if x.height else None
    # V2 del manuscrito (10-oct): ni la tabla ni la nota mencionan la versión anterior del
    # análisis, cuyo código no se publica. La nota dice qué pasa con la malla en un cuantil
    # alto, con las cifras de esta misma tabla.
    a, b, c = _row(0.05, 1.0), _row(0.95, 0.99), _row(0.95, 1.0)
    note = []
    if a and a["smd_len"] is not None:
        note.append(
            f"*Note.* At q = 0.05 the retained round trips are shifted by a median "
            f"{a['smd_len']:.3f} sd in length, {a['biased']} of {a['legs']} routes beyond the "
            "0.25 convention: requiring full coverage drops the shortest round trips, which "
            "is the unavoidable cost of a common grid.")
    if b and b["smd_len"] is not None:
        note.append(
            f"Placing the grid at a high quantile instead shifts the analysed sample: at "
            f"q = 0.95 with coverage 0.99 it keeps a median {b['ret']:.1f}% of round trips "
            f"and {b['bus']:.1f}% of the vehicles, shifted by {b['smd_len']:.3f} sd in "
            f"length, with {b['biased']} of {b['legs']} routes beyond the convention")
        if c is not None:
            note[-1] += (f"; with full coverage a median {c['ret']:.1f}% of round trips "
                         "survives.")
        else:
            note[-1] += "."
    if note:
        L.append(" ".join(note))
    return "\n".join(L) + "\n"


# ── B · lo que el criterio deja fuera ───────────────────────────────────────

def appendix_b() -> str:
    d = pl.read_csv(ANA / "leg_inclusion.csv",
                    schema_overrides={"route": pl.Utf8, "reason": pl.Utf8,
                                      "flag": pl.Utf8})
    exc = d.filter(~pl.col("included"))
    inc = d.filter(pl.col("included"))
    rev = inc.filter(pl.col("flag").fill_null("") != "")

    def why(reason: str) -> str:
        if reason.startswith("C3"):
            return "not computable (fewer round trips than folds)"
        if reason.startswith("C1"):
            return "trip cut rated poor"
        return "below the round-trip threshold"

    L: List[str] = []
    L.append("**Table S6. Routes excluded by the pre-specified criterion.**")
    L.append("")
    L.append(
        f"A route enters the comparison if its trip cut is not rated poor (coverage at "
        f"least {VUELTA_MAL['cobertura_min']:g}%, at most {VUELTA_MAL['inferidos_max']:g}% "
        f"of crossings inferred across a long gap and at most "
        f"{VUELTA_MAL['raras_max']:g}% of round trips outside 0.6–1.6 times the median "
        f"duration), retains at least 100 round trips on the grid, and has more round "
        f"trips than cross-validation folds. Of the {d.height} routes with a frame, "
        f"**{inc.height} qualify and {exc.height} do not**. The criterion was fixed before "
        f"any model was run; excluded routes are listed here in full rather than dropped "
        f"silently.")
    L.append("")
    if exc.is_empty():
        L.append("*No route is excluded.*")
    else:
        L.append("| Route | Round trips retained | Reason |")
        L.append("|---|---|---|")
        for r in exc.iter_rows(named=True):
            L.append(f"| {route_label(r['route'], number=True)} | {r['M']:,} | "
                     f"{why(r['reason'])} |")
    L.append("")
    if rev.height:
        L.append(
            f"*Routes kept and marked.* {rev.height} route"
            f"{'s are' if rev.height != 1 else ' is'} rated *review* — coverage below "
            f"{VUELTA_REVISAR['cobertura_min']:g}%, more than "
            f"{VUELTA_REVISAR['inferidos_max']:g}% inferred crossings, more than "
            f"{VUELTA_REVISAR['raras_max']:g}% rare round trips or more than "
            f"{VUELTA_REVISAR['atajo_max']:g}% of round trips with a shortcut — and kept, as "
            "the criterion specifies; round trips with a shortcut are out of the frame on "
            "every route.")
        L.append("")
        L.append("| Route | Why it is marked |")
        L.append("|---|---|")
        for r in rev.iter_rows(named=True):
            L.append(f"| {route_label(r['route'], number=True)} | "
                     f"{_flag_en(r['flag'])} |")
    return "\n".join(L) + "\n"


def _flag_en(flag: str) -> str:
    """El motivo de `revisar` de leg_inclusion, en inglés."""
    out = []
    for part in (flag or "").split("; "):
        p = part.replace(" % de vueltas con atajo (fuera del frame)",
                         "% of round trips with a shortcut")
        p = p.replace(" % de cruces inferidos", "% inferred crossings")
        p = p.replace(" % de vueltas raras", "% rare round trips")
        p = p.replace("cobertura ", "coverage ").replace(" %", "%")
        out.append(p)
    return "; ".join(out)


# ── C · dónde se corta ──────────────────────────────────────────────────────

def appendix_c() -> str:
    cut = {r["ruta"]: r for r in _csv(CORTES / "cortes_vuelta.csv")}
    une = {r["ruta"]: _f(r["vuelta_une_min"]) for r in _csv(CORTES / "vueltas_une.csv")}
    fun = {r["route"]: r for r in _csv(REPO / "docs" / "paper" / "tables"
                                         / "table_01_funnel.csv")}
    routes = [r for r in ROUTES if r in fun]
    L: List[str] = []
    L.append("**Table S4. Where each route's round trip is cut, and the round trip "
             "measured against the operator's declared duration.**")
    L.append("")
    L.append(
        "Each route is cut at one point of its polyline, fixed before any model was run "
        "(pre-registration, `cortes/cortes_vuelta.json`); a round trip runs from one "
        "crossing of that point to the next. On one-terminal routes the point is the "
        "location chosen by the rule shown; on two-terminal routes, of the two terminals, "
        "the one that leaves more of the service in complete round trips; on the variant "
        "routes, a single-cut rule run on their own records. *km* is the position of the "
        "point along the polyline and the loop length; *stop* is the nearest stop and its "
        "distance. *Median* is the median duration of valid round trips in this run "
        "(10th–90th percentile), against the round-trip duration the operator's portal "
        "declares (*UNE*). *Inferred* is the share of crossings whose time is "
        "interpolated across a gap of more than five minutes between records; *rare*, the "
        "share of round trips outside 0.6–1.6 times the median.")
    L.append("")
    L.append("| Route | Type | Cut point (stop, distance m) | km / loop | Located by "
             "| Median (min) | UNE (min) | Δ vs UNE (%) | Inferred (%) | Rare (%) "
             "| Cut quality |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    deltas = []
    for r in routes:
        c, f = cut.get(r, {}), fun[r]
        med, p10, p90 = _f(f.get("dur_med_min")), _f(f.get("dur_p10_min")), _f(f.get("dur_p90_min"))
        u = une.get(r)
        dlt = 100.0 * (med - u) / u if (med is not None and u) else None
        if dlt is not None:
            deltas.append((r, dlt))
        stop = (c.get("parada") or "").strip() or "—"
        dist = c.get("m_parada") or "—"
        rng = (f" ({p10:.0f}–{p90:.0f})" if p10 is not None and p90 is not None else "")
        L.append(
            f"| {route_label(r, number=True)} | {TYPE_EN.get(c.get('tipo'), c.get('tipo', '—'))} | "
            f"{stop} ({dist}) | {_f(c.get('km')):.2f} / {_f(f.get('loop_km')):.1f} | "
            f"{RULE_EN.get(c.get('metodo'), c.get('metodo', '—'))} | "
            f"{'—' if med is None else f'{med:.1f}'}{rng} | "
            f"{'—' if u is None else f'{u:.0f}'} | "
            f"{'—' if dlt is None else f'{dlt:+.1f}'} | "
            f"{_f(f.get('inferred_pct')) or 0:.1f} | {_f(f.get('rare_pct')) or 0:.2f} | "
            f"{QUALITY_EN.get(f.get('quality', ''), f.get('quality', ''))} |")
    L.append("")
    if deltas:
        ab = sorted(abs(x) for _, x in deltas)
        within = lambda k: sum(1 for x in ab if x <= k)                      # noqa: E731
        worst = max(deltas, key=lambda t: abs(t[1]))
        n_type = {t: sum(1 for r in routes if cut.get(r, {}).get("tipo") == t)
                  for t in TYPE_EN}
        L.append(
            f"*Notes.* {n_type['una']} routes have one terminal, {n_type['dos']} two and "
            f"{n_type['variante']} are variants. The measured round trip is within 2% of the "
            f"declared duration on {within(2)} of {len(ab)} routes and within 5% on "
            f"{within(5)} (median absolute difference {statistics.median(ab):.1f}%; largest "
            f"{route_label(worst[0])}, {worst[1]:+.1f}%). Two-terminal routes wait at the "
            "second terminal within the round trip, and that wait is part of the time "
            "predicted (Table S16). The duration hardly depends on where the round trip is "
            "cut (Table S5).")
    return "\n".join(L) + "\n"


# ── D · la invariancia al punto de corte ────────────────────────────────────

def appendix_d() -> str:
    rest = _csv(ANA / "cut_invariance_rest_point.csv")
    meth = {r["ruta"]: r for r in _csv(ANA / "cut_invariance_methods.csv")}
    rows = []
    for r in rest:
        L_km = _f(meth.get(r["ruta"], {}).get("L_km"))
        a, b = _f(r["cortes_km_hoy"]), _f(r["cortes_km_descanso"])
        moved = None
        if L_km and a is not None and b is not None:
            d = abs(a - b) % L_km
            moved = min(d, L_km - d)
        rows.append({**r, "L_km": L_km, "moved_km": moved})
    rows.sort(key=lambda r: route_order(r["ruta"]))

    L: List[str] = []
    L.append("**Table S5. The round trip does not depend on where it is cut.**")
    L.append("")
    L.append(
        "Each row re-runs trip construction on January–August 2024 with the route's cut "
        "point moved from the one used in this paper (*used*) to the point where vehicles "
        "spend most time at rest (*rest*), a single rule applied to every route; only the "
        "routes where that rule picks a different point are shown. *Moved* is the "
        "distance between the two points along the loop. The trip-construction code is "
        "the one of this paper; the runs predate the models.")
    L.append("")
    L.append("| Route | Cut used / rest (km) | Moved (km) | Median round trip, used / rest "
             "(min) | Δ (min) | Coverage, used / rest (%) | Inferred, used / rest (%) "
             "| Quality, used / rest |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        # Sin f-strings anidadas con las mismas comillas: el python del servidor es 3.10.
        moved = "—" if r["moved_km"] is None else "{:.1f}".format(r["moved_km"])
        L.append(
            f"| {route_label(r['ruta'], number=True)} | {_f(r['cortes_km_hoy']):.2f} / "
            f"{_f(r['cortes_km_descanso']):.2f} | {moved} | "
            f"{_f(r['dur_med_min_hoy']):.1f} / {_f(r['dur_med_min_descanso']):.1f} | "
            f"{_f(r['d_dur_min']):+.1f} | "
            f"{_f(r['cobertura_pct_hoy']):.1f} / {_f(r['cobertura_pct_descanso']):.1f} | "
            f"{_f(r['cruces_inferidos_pct_hoy']):.1f} / "
            f"{_f(r['cruces_inferidos_pct_descanso']):.1f} | "
            f"{QUALITY_EN.get(r['veredicto_hoy'], r['veredicto_hoy'])} / "
            f"{QUALITY_EN.get(r['veredicto_descanso'], r['veredicto_descanso'])} |")
    L.append("")
    dd = [abs(_f(r["d_dur_min"])) for r in rows]
    mv = [r["moved_km"] for r in rows if r["moved_km"] is not None]
    worse = [r for r in rows if r["veredicto_descanso"] != r["veredicto_hoy"]]
    txt = (f"*Notes.* Moving the cut point by up to {max(mv):.1f} km changes the median "
           f"round trip by at most {max(dd):.1f} min (median {statistics.median(dd):.1f}). "
           "What the point does change is how much of the service ends up in complete "
           "round trips and where crossings are timed")
    if worse:
        txt += (": at the rest point " + "; ".join(
            f"{route_label(r['ruta'])} would be rated "
            f"{QUALITY_EN.get(r['veredicto_descanso'])}, its inferred crossings rising from "
            f"{_f(r['cruces_inferidos_pct_hoy']):.1f} to "
            f"{_f(r['cruces_inferidos_pct_descanso']):.1f}%" for r in worse)
            + " — vehicles park at the rest point and stop reporting, so the crossing "
              "falls inside a gap.")
    else:
        txt += "."
    L.append(txt)

    # El comparativo de métodos: cuatro reglas de un corte sobre las 27 rutas públicas.
    m = list(meth.values())
    far = [r for r in m if _f(r["sep_max_km"]) >= 5.0]
    if far:
        dm = [_f(r["d_dur_min"]) for r in far]
        L.append("")
        L.append(
            f"The same holds across four single-cut rules (km 0 of the polyline, main "
            f"layover point, consensus, dispatch point) compared on the {len(m)} routes of "
            f"the public release: on the {len(far)} routes where the rules' points lie at "
            f"least 5 km apart (up to {max(_f(r['sep_max_km']) for r in far):.1f} km), the "
            f"median round trip differs across rules by at most {max(dm):.1f} min (median "
            f"{statistics.median(dm):.1f}), while coverage differs by up to "
            f"{max(_f(r['d_cob']) for r in far):.1f} points.")
    return "\n".join(L) + "\n"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in (("appendix_a_grid_representativity", appendix_a),
                     ("appendix_b_excluded_legs", appendix_b),
                     ("appendix_c_cut_points", appendix_c),
                     ("appendix_d_cut_invariance", appendix_d)):
        (OUT / f"{name}.md").write_text(fn(), encoding="utf-8")
        print(f"  ✓ {OUT / (name + '.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
