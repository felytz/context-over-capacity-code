"""
audit_run.py
============
Auditoría de la re-corrida en régimen representativo (q=0.05, cov=1.0).

Responde dos preguntas, en este orden:

  1. ¿CORRIÓ TODO?      cobertura de frames / XGB / LSTM por ruta y leg
  2. ¿ES VÁLIDO?        régimen correcto, mismo frame en las tres familias,
                        cortes pareados, n_pairs idéntico, cordura de resultados

y solo entonces muestra 3. LOS RESULTADOS.

Se puede correr en el server contra el árbol vivo, o en local tras bajar los
agregados (son KB).

Uso
---
    python docs/paper/analysis/audit_run.py
    python docs/paper/analysis/audit_run.py --q 0.05 --strict
    python docs/paper/analysis/audit_run.py --routes 101,102 --verbose
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "func"))

import numpy as np
import polars as pl

import parquet_guard as pg                                     # noqa: E402

_ROOT = Path(__file__).resolve().parents[3]
RANGE = "2024_01-08_tmin"


def set_root(p: "str | Path") -> None:
    """Raíz alternativa del árbol de datos.

    Permite auditar un paquete descargado del server (`tar -xf … -C algun/dir`)
    SIN extraerlo encima de `data/`, que mezclaría la procedencia de los
    artefactos locales con los del cluster."""
    global _ROOT
    _ROOT = Path(p).resolve()

# Orden del array de jobs/job_lstm_frame.slrm (índices 0-34): las 35 rutas de la
# vuelta (docs/PREREGISTRO_VUELTA.md), por número de línea y cada variante junto a su
# principal. Es el orden de todas las tablas por ruta.
#
# ⚠ Con la vuelta NO hay estratos. Los de agosto (validated / group B / N=1 / excluded)
# venían del descubrimiento de terminales, que ya no corta: la fiabilidad del corte la
# dice ahora su calidad (`leg_inclusion.cut_quality`: ok / revisar / mal) y todas las
# rutas incluidas van en un solo panel (pre-registro §4). El STRATUM de agosto vive en
# `metodo-v2`, con el resto de la corrida de agosto.
ROUTES = ["101", "102", "201", "202", "301", "302", "401", "402", "501", "601",
          "801", "901", "902", "1001", "1002", "1101", "110101", "1201", "1301",
          "1401", "1402", "1501", "1502", "1601", "1602", "1701", "1702", "1703",
          "1801", "180101", "1802", "180201", "1901", "1902", "20000"]

# El catálogo de las 35 (docs/paper/analysis/route_catalogue.csv): el número del AVL,
# cómo se escribe en el texto (las -ALT como 1101-ALT, P6), la familia, el `group_id`
# del AVL (la 1401 y la 1402 vienen como 1400) y el número y el nombre de UNE (catálogo
# del portal, 11-sep-2026: 101↔102, 1001↔1002 y 1601↔1602 están cruzados, y la 20000
# es la 2201).
CATALOGUE_CSV = Path(__file__).resolve().parent / "route_catalogue.csv"


def _load_catalogue() -> Dict[str, Dict[str, str]]:
    import csv
    with open(CATALOGUE_CSV, encoding="utf-8") as fh:
        cat = {r["route"]: r for r in csv.DictReader(fh)}
    if list(cat) != ROUTES:
        raise SystemExit(f"✗ {CATALOGUE_CSV.name} no lista las 35 rutas de ROUTES, en su "
                         f"orden: el catálogo y los arrays de los jobs tienen que casar.")
    return cat


CATALOGUE = _load_catalogue()
# Las familias con variante comparten traza y flota: en el bootstrap agrupado son un
# solo grupo (pre-registro §5). Ruta → grupo; las que no están son su propio grupo.
FAMILY = {r: c["family"] for r, c in CATALOGUE.items() if c["family"] != r}
assert FAMILY == {"110101": "1101", "1402": "1401", "180101": "1801",
                  "180201": "1802"}, FAMILY      # el pre-registro §5 las nombra así


def route_label(route: "str | int", number: bool = False) -> str:
    """Cómo se escribe la ruta (P6): ``1101-ALT`` para la 110101; con ``number``,
    ``1101-ALT (110101)``, que es lo que piden las tablas. Las demás, su número."""
    r = str(route)
    lab = CATALOGUE.get(r, {}).get("label", r)
    return f"{lab} ({r})" if number and lab != r else lab


def route_order(route: "str | int") -> int:
    """Posición de la ruta en ROUTES (las desconocidas, al final)."""
    try:
        return ROUTES.index(str(route))
    except ValueError:
        return len(ROUTES)

OK, BAD, WARN = "✔", "✗", "⚠"


def qtag(q: float, cov: float = 0.99, split: str = "random") -> str:
    """Régimen en nombres de archivo: ``q05``, ``q05c100`` o ``q05c100_tw``.

    Delega en `eval_frame.q_tag`, que es quien nombra los artefactos al
    escribirlos. Antes había aquí una copia con el comentario "debe coincidir
    con eval_frame.q_tag": una regla de nombres duplicada a mano en el que
    escribe y en el que lee es la forma exacta de que un análisis deje de
    encontrar los ficheros sin que nada falle — solo salen menos legs.
    """
    from eval_frame import q_tag as _q_tag
    return _q_tag(q, cov, split)


#: El LSTM que se reporta (pre-registro §5, P5): `es15` en la corrida aleatoria y `tw`
#: en la escalera. `es0` —el `{tag}_overall` a secas de agosto— ya no se corre.
LSTM_SUFFIX = {"random": "_es15", "expanding": ""}


def lstm_qtag(q: float, cov: float = 0.99, split: str = "random") -> str:
    """Régimen en el nombre de los artefactos del LSTM: ``q05c100_es15`` en la corrida
    aleatoria (el titular y todos los desgloses) y ``q05c100_tw`` en la escalera.

    ⚠ Antes cada lector buscaba el LSTM en ``{tag}_overall``, que con la vuelta no
    existe: la leg salía «sin LSTM» y caía de todas las tablas sin error."""
    return qtag(q, cov, split) + LSTM_SUFFIX[split]


def is_directional(leg: str) -> bool:
    parts = str(leg).split("->")
    return len(parts) == 2 and parts[0] != parts[1]


# ──────────────────────────────────────────────────────────────────
# Recolección
# ──────────────────────────────────────────────────────────────────

def collect(routes: List[str], q: float, rng: str, cov: float,
            split: str = "random") -> Dict[str, Any]:
    tag = qtag(q, cov, split)
    ltag = lstm_qtag(q, cov, split)
    rows: List[Dict[str, Any]] = []
    for r in routes:
        base = _ROOT / "data" / "02_interim" / "routes" / r
        fdir = base / "frames" / rng
        xdir = base / "xgb" / rng
        ldir = base / "lstm_frame" / rng

        # ⚠ Las legs se enumeran desde la UNIÓN de las tres fuentes, no solo
        # desde los frames. Con la versión anterior, una ruta sin frames nunca
        # se miraba en xgb/ y el informe decía "xgb: 0" habiendo 1025 archivos
        # en disco — justo el caso que había que diagnosticar (2026-07-24).
        legs: Dict[str, Dict[str, Any]] = {}
        for p in sorted(fdir.glob(f"*_{tag}_meta.json")):
            label = p.name[: -len(f"_{tag}_meta.json")]
            try:
                legs[label] = {"frame": json.loads(p.read_text(encoding="utf-8"))}
            except Exception as e:
                legs[label] = {"frame": None, "frame_error": str(e)}
        for pat, d_ in ((f"*_{tag}_exo_overall.parquet", xdir),
                        (f"*_{tag}_overall.parquet", xdir),
                        (f"*_{ltag}_overall.parquet", ldir)):
            n_suf = len(pat) - 1
            for p in sorted(d_.glob(pat)) if d_.is_dir() else []:
                legs.setdefault(p.name[:-n_suf], {"frame": None})

        for label, d in legs.items():
            fm = d.get("frame") or {}
            row: Dict[str, Any] = {
                "route": r, "name": route_label(r), "label": label,
                "leg": fm.get("leg"),
                "frame": bool(fm),
                "frame_hash": fm.get("frame_hash"),
                "q": fm.get("n_route_quantile"),
                "cov": fm.get("min_coverage"),
                "max_trips": fm.get("max_trips"),
                "M": fm.get("M"),
                "n_cells": fm.get("n_cells"),
                "retention_pct": fm.get("retention_pct"),
            }

            # XGB: con --exo el sufijo es _exo
            xo = xdir / f"{label}_{tag}_exo_overall.parquet"
            xm = xdir / f"{label}_{tag}_exo_meta.json"
            if not xo.is_file():                       # fallback: sin --exo
                xo = xdir / f"{label}_{tag}_overall.parquet"
                xm = xdir / f"{label}_{tag}_meta.json"
            row["xgb"] = pg.parquet_ok(xo)
            if row["xgb"]:
                try:
                    dfx = pl.read_parquet(xo)
                    row["xgb_algos"] = dfx.height
                    row["xgb_n_pairs"] = int(dfx["n_pairs"].max())
                    row["_xgb"] = dfx
                except Exception as e:
                    row["xgb"] = False
                    row["xgb_error"] = str(e)
            if xm.is_file():
                try:
                    mx = json.loads(xm.read_text(encoding="utf-8"))
                    row["xgb_frame_hash"] = mx.get("frame_hash")
                    row["xgb_q"] = mx.get("n_route_quantile")
                    row["xgb_cov"] = mx.get("min_coverage")
                    row["xgb_max_trips"] = mx.get("max_trips")
                except Exception:
                    pass

            # Histograma de `dev` (fase 5): sin él no hay banda simétrica ni
            # on-time calibrado para esa leg, y no se puede recuperar a
            # posteriori — los residuos no se persisten.
            # ⚠ `parquet_ok`, no `is_file()`: un devhist truncado dejaba la leg
            # marcada como completa y la corrida se certificaba con FRESCURA OK
            # sobre ficheros que nadie podía leer (2026-07-25, 11 artefactos).
            row["xgb_devhist"] = (
                pg.parquet_ok(xdir / f"{label}_{tag}_exo_devhist.parquet")
                or pg.parquet_ok(xdir / f"{label}_{tag}_devhist.parquet"))

            lo = ldir / f"{label}_{ltag}_overall.parquet"
            row["lstm"] = pg.parquet_ok(lo)
            row["lstm_devhist"] = pg.parquet_ok(
                ldir / f"{label}_{ltag}_devhist.parquet")
            if row["lstm"]:
                try:
                    dfl = pl.read_parquet(lo)
                    row["lstm_n_pairs"] = int(dfl["n_pairs"][0])
                    row["lstm_frame_hash"] = (dfl["frame_hash"][0]
                                              if "frame_hash" in dfl.columns else None)
                    row["cuts_match"] = (bool(dfl["cuts_match_frame"][0])
                                         if "cuts_match_frame" in dfl.columns else None)
                    row["_lstm"] = dfl
                except Exception as e:
                    row["lstm"] = False
                    row["lstm_error"] = str(e)
            rows.append(row)

        if not legs:
            rows.append({"route": r, "name": route_label(r),
                         "label": None, "leg": None, "frame": False,
                         "xgb": False, "lstm": False})
    return {"rows": rows, "tag": tag}


# ──────────────────────────────────────────────────────────────────
# 1 · Cobertura
# ──────────────────────────────────────────────────────────────────

def report_coverage(rows: List[Dict[str, Any]], q: float,
                    routes: List[str], cov: float,
                    n_folds: int = 5, split: str = "random") -> Dict[str, Any]:
    print("\n" + "=" * 78)
    print(f"  1 · ¿CORRIÓ TODO?  (régimen {qtag(q, cov, split)})")
    print("=" * 78)
    print(f"  {'ruta':>6} {'nombre':10} {'legs':>5} {'frames':>7} {'xgb':>5} "
          f"{'lstm':>5}   estado")
    print("  " + "-" * 74)

    by_route: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_route.setdefault(r["route"], []).append(r)

    missing_frames, missing_xgb, missing_lstm = [], [], []
    for route in [r for r in ROUTES if r in set(routes)]:
        rs = [x for x in by_route.get(route, []) if x.get("label")]
        n_leg = len(rs)
        # Una leg con M < n_folds NO puede correr (InsufficientDataError en
        # run_matched_cv, y el LSTM la salta igual): tiene frame pero nunca
        # tendrá agregados. Contarla como "falta" es una alarma falsa
        # permanente — p. ej. 20000/T0->T0 con M=4 (2026-07-25).
        rs = [x for x in rs if (x.get("M") is None or x["M"] >= n_folds)]
        n_f = sum(1 for x in rs if x.get("frame"))
        n_x = sum(1 for x in rs if x.get("xgb"))
        n_l = sum(1 for x in rs if x.get("lstm"))
        if n_f == 0 and n_x > 0:
            # Hay agregados XGB pero ningún frame: la fase CPU produjo salida sin
            # escribir frames. Pasó el 2026-07-24 porque el guard de `cached` no
            # miraba la cobertura y las rutas se saltaron contra la corrida vieja.
            estado = f"{n_x} XGB pero SIN FRAMES — ¿corrida vieja o saltada?"
            mark = BAD
            missing_frames.append(route)
        elif n_f == 0:
            estado, mark = "sin frames — la fase CPU no pasó por esta ruta", BAD
            missing_frames.append(route)
        elif n_x < n_f:
            estado, mark = f"faltan {n_f - n_x} XGB", WARN
            missing_xgb.append(route)
        elif n_l == 0:
            estado, mark = "sin LSTM — falta la fase GPU", BAD
            missing_lstm.append(route)
        elif n_l < n_f:
            estado, mark = f"faltan {n_f - n_l} LSTM", WARN
            missing_lstm.append(route)
        else:
            estado, mark = "completa", OK
        st = route_label(route)
        print(f"  {mark} {route:>4} {st:10} {n_leg:>5} {n_f:>7} {n_x:>5} "
              f"{n_l:>5}   {estado}")

    print()
    if missing_frames:
        print(f"  {BAD} SIN FRAMES ({len(missing_frames)}): {' '.join(missing_frames)}")
        print(f"      → la fase CPU no terminó (o falló) en esas rutas. El LSTM con")
        print(f"        --require-frame las SALTA en silencio y el job sale con rc=0.")
    if missing_lstm:
        print(f"  {WARN} LSTM incompleto ({len(missing_lstm)}): {' '.join(missing_lstm)}")
        print(f"      → re-envía el array de GPU con el MISMO comando: es reanudable,")
        print(f"        salta lo hecho y solo corre lo que falta.")
    if not missing_frames and not missing_lstm and not missing_xgb:
        print(f"  {OK} Cobertura completa en las {len(routes)} rutas auditadas.")
    return {"missing_frames": missing_frames, "missing_xgb": missing_xgb,
            "missing_lstm": missing_lstm}


# ──────────────────────────────────────────────────────────────────
# 2 · Validez
# ──────────────────────────────────────────────────────────────────

def report_validity(rows: List[Dict[str, Any]], q: float, cov: float,
                    max_trips: Optional[int], sanity_min_M: int = 100) -> bool:
    print("\n" + "=" * 78)
    print("  2 · ¿ES VÁLIDO?  (si algo aquí sale en rojo, los números NO sirven)")
    print("=" * 78)

    live = [r for r in rows if r.get("frame")]
    ok_all = True

    def check(name: str, bad: List[str], n_applicable: int,
              detail: str = "") -> None:
        """`n_applicable` es el nº de legs a las que el chequeo PUEDE aplicarse.

        Sin ese denominador, un chequeo sobre cero legs imprimía un verde
        engañoso (p. ej. '✔ 4/4' cuando todavía no hay ni un LSTM), que es
        exactamente la falsa tranquilidad que este auditor debe evitar.
        """
        nonlocal ok_all
        if n_applicable == 0:
            print(f"  {WARN} {name}: sin datos todavía (0 legs aplicables)"
                  + (f"  ·  {detail}" if detail else ""))
            return
        good = not bad
        ok_all &= good
        mark = OK if good else BAD
        print(f"  {mark} {name}: {n_applicable - len(bad)}/{n_applicable}"
              + (f"  ·  {detail}" if detail else ""))
        if bad:
            for b in bad[:8]:
                print(f"        {b}")
            if len(bad) > 8:
                print(f"        … y {len(bad) - 8} más")

    # a) régimen del frame
    bad = [f"{r['route']}/{r['label']}: q={r['q']} cov={r['cov']} "
           f"max_trips={r['max_trips']}"
           for r in live
           if r["q"] != q or r["cov"] != cov
           or (max_trips is not None and r["max_trips"] != max_trips)]
    check(f"régimen del frame (q={q}, cov={cov}, max_trips={max_trips})",
          bad, len(live))

    # b) el frame del XGB es el mismo objeto
    withx = [r for r in live if r.get("xgb")]
    bad = [f"{r['route']}/{r['label']}: xgb={r.get('xgb_frame_hash')} "
           f"frame={r['frame_hash']}"
           for r in withx
           if r.get("xgb_frame_hash") and r["xgb_frame_hash"] != r["frame_hash"]]
    check("XGB corrió sobre el frame en disco (mismo hash)", bad, len(withx),
          f"{len(withx)} legs con XGB")

    # c) el LSTM usó el mismo frame
    withl = [r for r in live if r.get("lstm")]
    bad = [f"{r['route']}/{r['label']}: lstm={r.get('lstm_frame_hash')} "
           f"frame={r['frame_hash']}"
           for r in withl
           if r.get("lstm_frame_hash") and r["lstm_frame_hash"] != r["frame_hash"]]
    check("LSTM corrió sobre el MISMO frame que XGB/baselines", bad,
          len(withl), f"{len(withl)} legs con LSTM")

    # d) cortes pareados
    bad = [f"{r['route']}/{r['label']}" for r in withl
           if r.get("cuts_match") is False]
    check("cortes del LSTM == cortes del frame (cuts_match_frame)", bad,
          len(withl))

    # e) n_pairs idéntico — LA prueba de que la comparación es matched
    both = [r for r in live if r.get("xgb") and r.get("lstm")]
    bad = [f"{r['route']}/{r['label']}: xgb={r.get('xgb_n_pairs'):,} "
           f"lstm={r.get('lstm_n_pairs'):,}"
           for r in both
           if r.get("xgb_n_pairs") != r.get("lstm_n_pairs")]
    check("n_pairs IDÉNTICO entre LSTM y baselines/XGB", bad, len(both),
          f"{len(both)} legs comparables")

    # f) cordura: cada variante XGB no debe perder contra `delay` EN SU PROPIA
    # MÉTRICA. Es un superconjunto estricto del baseline (recibe μ_{l+h}−μ_l como
    # feature), así que perder indica bug de entrenamiento.
    #
    # ⚠ La métrica tiene que emparejarse con la pérdida. `delay` predice la MEDIA
    # histórica del incremento, o sea es el óptimo-MSE entre los predictores
    # constantes por (l,h). `xgb_mse` persigue esa media MEJOR, pero el óptimo en
    # MAE es la MEDIANA, y con cola derecha pesada (RMSE/MAE ≈ 3.8 medido) media y
    # mediana divergen: `xgb_mse` puede ganar en RMSE y perder en MAE justamente
    # por hacer bien su trabajo. Comparar `xgb_mse` en MAE (como hacía la primera
    # versión de este auditor) marca rojo 33/67 legs sin que haya nada roto.
    #   xgb_mse → RMSE      ·      xgb_mae → MAE
    #
    # Solo con muestra suficiente: en legs de 13-25 viajes con CV de 5 folds el
    # ajuste es ruido y el chequeo daría rojo siempre, desensibilizando.
    # El criterio se evalúa sobre la MEDIANA de la razón algo/delay, no leg a
    # leg. Razón: XGBoost es un superconjunto de features de `delay`, pero NO se
    # ajusta con los mismos datos — su prior se calcula sobre el ~85% de los
    # viajes de train (train menos el holdout de early stopping, ver
    # cross_validation.run_matched_cv) frente al 100% de los baselines, y además
    # lleva regularización. Esa desventaja deliberada produce déficits de 2-5% en
    # legs sueltas que NO son bugs. Un bug de entrenamiento se vería como un
    # desplazamiento SISTEMÁTICO, que es lo que mide la mediana.
    pairs = [("xgb_mse", "RMSE_s", "RMSE"), ("xgb_mae", "MAE_s", "MAE"),
             ("xgb_mse_b", "RMSE_s", "RMSE"), ("xgb_mae_b", "MAE_s", "MAE")]
    for algo, col, mname in pairs:
        ratios, worst, small = [], [], 0
        for r in withx:
            d = r.get("_xgb")
            if d is None or col not in d.columns:
                continue
            m = {a: v for a, v in zip(d["algorithm"], d[col])}
            if algo not in m or "delay" not in m or not m["delay"]:
                continue
            if (r.get("M") or 0) < sanity_min_M:
                small += 1
                continue
            ratio = m[algo] / m["delay"]
            ratios.append(ratio)
            if ratio > 1.10:
                worst.append(f"{r['route']}/{r['label']} (M={r.get('M')}): "
                             f"{algo} {mname}={m[algo]:.1f} vs delay="
                             f"{m['delay']:.1f}  ({100*(ratio-1):+.1f}%)")
        if not ratios and not small:
            continue
        med = float(np.median(ratios)) if ratios else float("nan")
        n_worse = sum(1 for x in ratios if x > 1.0)
        # Falla solo si el desplazamiento es sistemático (mediana > 2%).
        bad = worst if med > 1.02 else []
        check(f"cordura: {algo} vs delay en {mname} (su propia pérdida)",
              bad, len(ratios),
              f"razón mediana {med:.3f} · peor que delay en {n_worse}/{len(ratios)}"
              f" · {small} legs con M < {sanity_min_M} excluidas")

    # g) histograma de `dev` — fase 5. NO invalida el MAE, así que no toca
    # `ok_all`; lo que bloquea es el reporte de on-time (banda simétrica y
    # calibración por sesgo), y no se puede arreglar a posteriori porque los
    # residuos no se persisten en ningún sitio.
    n_dx = sum(1 for r in withx if r.get("xgb_devhist"))
    n_dl = sum(1 for r in withl if r.get("lstm_devhist"))
    falta = (len(withx) - n_dx) + (len(withl) - n_dl)
    mark = OK if falta == 0 else WARN
    print(f"  {mark} histograma de dev (fase 5): XGB {n_dx}/{len(withx)} · "
          f"LSTM {n_dl}/{len(withl)}")
    if falta:
        print(f"        → esas legs no tienen histograma utilizable: su on-time")
        print(f"          no se puede desasnar del sesgo. Re-córrelas y luego:")
        print(f"          python docs/paper/analysis/honest_ontime.py --root <dir>")

    # h) integridad — distingue AUSENTE de CORRUPTO, que piden cosas distintas:
    # lo ausente se recalcula solo, lo corrupto hay que BORRARLO antes o la
    # reanudación lo da por hecho y lo salta.
    roto = pg.scan(_ROOT, "*.parquet")
    n_pq = sum(1 for _ in _ROOT.rglob("*.parquet"))
    if roto:
        print(f"  {BAD} integridad: {len(roto)} de {n_pq} parquet CORRUPTO(S) — "
              f"arriba se han contado como AUSENTES")
        for p, why in roto[:10]:
            try:
                shown = p.relative_to(_ROOT)
            except ValueError:
                shown = p
            print(f"        ✗ {shown}: {why}")
        if len(roto) > 10:
            print(f"        … y {len(roto) - 10} más")
        print(f"        → BÓRRALOS en el árbol vivo antes de reenviar el array:")
        print(f"          mientras existan, la reanudación los da por hechos.")
    else:
        print(f"  {OK} integridad: los {n_pq} parquet del árbol se leen")

    print()
    if not ok_all:
        print("  ❌ HAY FALLOS DE VALIDEZ — no reportes estos números todavía")
    elif roto:
        # El pareado sigue en pie: lo corrupto son agregados secundarios, no los
        # `overall` de los que salen MAE y n_pairs. Pero el árbol NO está sano y
        # decirlo en verde es justo el error que dejó pasar lo del 2026-07-25.
        print("  ⚠ PAREADO VÁLIDO, PERO EL ÁRBOL NO ESTÁ SANO — la comparación de")
        print("    MAE se sostiene; el on-time honesto va sobre menos legs hasta")
        print("    que rehagas las corruptas.")
    else:
        print("  ✅ TODO VÁLIDO — la comparación es pareada por construcción")
    return ok_all


# ──────────────────────────────────────────────────────────────────
# 3 · Resultados
# ──────────────────────────────────────────────────────────────────

def report_results(rows: List[Dict[str, Any]],
                   min_M: int) -> Optional[pl.DataFrame]:
    print("\n" + "=" * 78)
    print("  3 · RESULTADOS")
    print("=" * 78)

    frames: List[pl.DataFrame] = []
    for r in rows:
        keep_cols = ["route", "leg", "algorithm", "n_pairs", "MAE_s", "RMSE_s",
                     "MedAE_s", "P90AE_s", "bias_s", "pct_ontime_dev"]
        for key in ("_xgb", "_lstm"):
            d = r.get(key)
            if d is None:
                continue
            cols = [c for c in keep_cols if c in d.columns]
            frames.append(d.select(cols).with_columns(
                pl.lit(r["route"]).alias("route"),
                pl.lit(str(r.get("leg") or r["label"])).alias("leg"),
                pl.lit(r["name"]).alias("name"),
                pl.lit(int(r.get("M") or 0)).alias("M"),
            ))
    if not frames:
        print("  (sin resultados todavía)")
        return None

    df = pl.concat(frames, how="diagonal_relaxed")
    n_all = df.select(["route", "leg"]).unique().height
    # ⚠ Hasta la vuelta el filtro por defecto era «solo direccionales» (`Ta->Tb`): con
    # la vuelta toda leg es `T0->T0` y la sección salía vacía, en verde. Lo que entra
    # al análisis lo decide `leg_inclusion.py`; aquí sólo se exige M.
    if min_M > 0:
        df = df.filter(pl.col("M") >= min_M)
    n_use = df.select(["route", "leg"]).unique().height
    print(f"  legs: {n_use} de {n_all}"
          + (f"  (M ≥ {min_M})" if min_M > 0 else "  (todas)"))

    piv = (df.group_by("algorithm")
           .agg(pl.col("MAE_s").median().alias("MAE med"),
                pl.col("pct_ontime_dev").median().alias("on-time med"),
                pl.col("bias_s").median().alias("bias med"),
                pl.len().alias("legs"))
           .sort("MAE med"))

    # ¿cuántas legs gana cada algoritmo?
    best = (df.sort("MAE_s").group_by(["route", "leg"]).first()
            .group_by("algorithm").agg(pl.len().alias("gana MAE")))
    piv = piv.join(best, on="algorithm", how="left").with_columns(
        pl.col("gana MAE").fill_null(0))

    # % vs delay, pareado por leg
    base = df.filter(pl.col("algorithm") == "delay").select(
        ["route", "leg", pl.col("MAE_s").alias("mae_delay")])
    rel = (df.join(base, on=["route", "leg"], how="inner")
           .with_columns(((pl.col("MAE_s") - pl.col("mae_delay"))
                          / pl.col("mae_delay") * 100).alias("d"))
           .group_by("algorithm")
           .agg(pl.col("d").median().alias("vs delay %"),
                (pl.col("d") < 0).sum().alias("mejor que delay")))
    piv = piv.join(rel, on="algorithm", how="left").sort("MAE med")

    print()
    print(f"  {'algoritmo':14} {'MAE med':>9} {'on-time':>8} {'bias':>8} "
          f"{'gana':>6} {'vs delay':>9} {'mejor/n':>9}")
    print("  " + "-" * 72)
    # Sin f-strings anidadas: el python del server es 3.10 y las comillas
    # reutilizadas dentro de una f-string solo son válidas desde 3.12 (PEP 701).
    for row in piv.iter_rows(named=True):
        vd = row.get("vs delay %")
        mb = row.get("mejor que delay")
        s_vd = "—" if vd is None else "{:+.1f}%".format(vd)
        s_mb = "—" if mb is None else "{}/{}".format(int(mb), int(row["legs"]))
        print("  {:14} {:9.2f} {:8.2f} {:8.1f} {:6d} {:>9} {:>9}".format(
            row["algorithm"], row["MAE med"], row["on-time med"],
            row["bias med"], int(row["gana MAE"]), s_vd, s_mb))

    print()
    print("  Lectura: `bias` negativo = subestima el tiempo restante ⇒ cae del lado")
    print("  TARDE de la banda [−60, +300], que es el ancho. Un on-time alto con")
    print("  bias muy negativo está inflado por la asimetría, no es precisión.")
    return df


# ──────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description="Auditoría de la re-corrida")
    ap.add_argument("--routes", default="all")
    ap.add_argument("--q", type=float, default=0.05)
    ap.add_argument("--cov", type=float, default=1.0)
    ap.add_argument("--max-trips", type=int, default=2500)
    ap.add_argument("--range", dest="rng", default=RANGE)
    ap.add_argument("--min-M", type=int, default=100,
                    help="excluye legs con M < este valor (0 = sin filtro)")
    ap.add_argument("--split", default="random",
                    choices=["random", "expanding"],
                    help="'expanding' audita los artefactos `_tw` de la "
                         "corrida con ventana temporal")
    ap.add_argument("--csv", default=None, help="vuelca los resultados a CSV")
    ap.add_argument("--root", default=None,
                    help="raíz del árbol de datos (default: la del repo). Úsalo "
                         "para auditar un paquete descargado sin extraerlo "
                         "sobre data/.")
    a = ap.parse_args()

    if a.root:
        set_root(a.root)

    routes = (ROUTES if a.routes.strip().lower() == "all"
              else [r.strip() for r in a.routes.split(",") if r.strip()])

    print("=" * 78)
    print(f"  AUDITORÍA DE LA RE-CORRIDA · {a.rng} · {qtag(a.q, a.cov, a.split)}")
    print(f"  raíz: {_ROOT}")
    print("=" * 78)

    got = collect(routes, a.q, a.rng, a.cov, a.split)
    rows = got["rows"]

    cov_info = report_coverage(rows, a.q, routes, a.cov, split=a.split)
    valid = report_validity(rows, a.q, a.cov, a.max_trips,
                            sanity_min_M=max(a.min_M, 1))
    df = report_results(rows, a.min_M)

    if df is not None and a.csv:
        out = Path(a.csv)
        df.write_csv(out)
        print(f"\n  ✓ resultados → {out}")

    print("\n" + "=" * 78)
    blockers = (cov_info["missing_frames"] or cov_info["missing_lstm"]
                or not valid)
    print("  ✅ CORRIDA LIMPIA" if not blockers else
          "  ⚠ CORRIDA INCOMPLETA O INVÁLIDA — revisa las secciones 1 y 2")
    print("=" * 78)
    return 0 if not blockers else 1


if __name__ == "__main__":
    raise SystemExit(main())
