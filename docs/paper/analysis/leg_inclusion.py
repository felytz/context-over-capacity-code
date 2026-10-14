"""
leg_inclusion.py
================
Criterio de inclusión **pre-registrado** y la lista de excluidas, que se publica.

Con la vuelta (TRIP_CUT=cruce; docs/PREREGISTRO_VUELTA.md §3, fijado el 6-oct-2026)
-----------------------------------------------------------------------------------
Cada ruta tiene una sola leg, ``T0->T0``: la vuelta, de un cruce del punto de corte al
siguiente. Una ruta entra en el análisis si cumple las tres:

  **C1 · el corte no sale mal** — con el bloque ``cruce`` del funnel de la ruta:
  cobertura ≥ 50 %, cruces inferidos ≤ 10 % y vueltas raras ≤ 5 % (los umbrales de
  ``mal`` de ``calidad_corte.py`` del test).

  **C2 · ``M ≥ 100``** — vueltas retenidas por el frame. Con 5 folds, M=20 deja 4 de
  validación; cualquier diferencia entre algoritmos a ese tamaño es ruido de partición.

  **C3 · computable** — ``M ≥ n_folds``. Por debajo, `run_matched_cv` lanza
  `InsufficientDataError` y el LSTM salta la ruta: no es una exclusión de criterio sino
  una imposibilidad.

Las rutas que saldrían ``revisar`` (cobertura < 65 %, > 5 % de cruces inferidos, > 2 % de
vueltas raras o > 10 % de vueltas con atajo) entran, marcadas en ``flag``. Lo que entra
por vuelta (sin atajo, sin filtro por duración) lo decide el df_full (``GRID_AXIS=traza``
en run_new_pipeline.py), no este script.

Las excluidas NO se borran: van a un apéndice con su M y su motivo. Ese es el punto de
pre-registrar — que la lista sea auditable y no una selección hecha después de ver quién
ganaba.

Antes de la vuelta (agosto)
---------------------------
C1 era «direccional» (``Ta->Tb`` con ``a ≠ b``): las self-loop eran el residuo del
descubrimiento de terminales. Ese criterio, con los estratos, está en `metodo-v2`; aquí ya
no se aplica: un funnel sin bloque ``cruce`` es una ruta que no corrió con la vuelta, y se
excluye con ese motivo.

Uso
---
    python docs/paper/analysis/leg_inclusion.py --root artefacts/run_bundle_<…>
    python docs/paper/analysis/leg_inclusion.py --root artefacts/run_bundle_<…> \\
           --out docs/paper/analysis/leg_inclusion
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import polars as pl                                            # noqa: E402

from audit_run import ROUTES, lstm_qtag, qtag, route_label     # noqa: E402
from audit_run import pg                                       # noqa: E402

RANGE = "2024_01-08_tmin"
MIN_M = 100

# La vuelta: los umbrales del pre-registro (§3), los de calidad_corte.py del test.
VUELTA_MAL = {"cobertura_min": 50.0, "inferidos_max": 10.0, "raras_max": 5.0}
VUELTA_REVISAR = {"cobertura_min": 65.0, "inferidos_max": 5.0, "raras_max": 2.0,
                  "atajo_max": 10.0}


def read_funnel(root: Path, route: str, rng: str) -> Dict[str, Any]:
    """Lo que el criterio de la vuelta necesita del funnel de la ruta (vacío si no hay)."""
    p = root / "data" / "02_interim" / "routes" / route / "perf" / f"funnel_{rng}.json"
    try:
        fn = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        fn = {}
    cr = fn.get("cruce") or {}
    return {
        "trip_cut": fn.get("trip_cut") or "",
        "cobertura_pct": cr.get("cobertura_pct"),
        "inferidos_pct": cr.get("cruces_inferidos_pct"),
        "raras_pct": cr.get("raras_pct"),
        "atajo_pct": cr.get("viajes_con_atajo_pct"),
        "atajo_fuera": fn.get("trips_atajo_out_of_models"),
    }


def cut_quality(cobertura_pct: Optional[float], inferidos_pct: Optional[float],
                raras_pct: Optional[float],
                atajo_pct: Optional[float]) -> Tuple[str, str]:
    """La calidad del corte de una ruta, con los umbrales del pre-registro (§3):
    ``("mal" | "revisar" | "ok", motivos)``. Sin cobertura, ``("", "sin bloque cruce")``.

    `mal` decide C1; `revisar` sólo marca. La usan este script y la tabla del funnel,
    para que las dos no puedan clasificar distinto la misma ruta."""
    if cobertura_pct is None:
        return "", "sin bloque `cruce` en el funnel"
    u, mal = VUELTA_MAL, []
    if cobertura_pct < u["cobertura_min"]:
        mal.append(f"cobertura {cobertura_pct} %")
    if inferidos_pct is not None and inferidos_pct > u["inferidos_max"]:
        mal.append(f"{inferidos_pct} % de cruces inferidos")
    if raras_pct is not None and raras_pct > u["raras_max"]:
        mal.append(f"{raras_pct} % de vueltas raras")
    if mal:
        return "mal", "; ".join(mal)
    u, rev = VUELTA_REVISAR, []
    if cobertura_pct < u["cobertura_min"]:
        rev.append(f"cobertura {cobertura_pct} %")
    if (inferidos_pct or 0) > u["inferidos_max"]:
        rev.append(f"{inferidos_pct} % de cruces inferidos")
    if (raras_pct or 0) > u["raras_max"]:
        rev.append(f"{raras_pct} % de vueltas raras")
    if (atajo_pct or 0) > u["atajo_max"]:
        rev.append(f"{atajo_pct} % de vueltas con atajo (fuera del frame)")
    return ("revisar", "; ".join(rev)) if rev else ("ok", "")


def collect_legs(root: Path, q: float, cov: float, rng: str,
                 routes: Optional[List[str]] = None,
                 split: str = "random") -> pl.DataFrame:
    """Una fila por ruta (su leg ``T0->T0``), con su M y qué familias tienen agregados.

    La enumeración sale de los frames porque el frame es la definición del universo:
    existe para toda leg que el régimen admite, la haya corrido o no alguna familia.
    Una ruta con funnel de la vuelta y sin frame no desaparece: es una exclusión (sin
    frame no hay M), y el pre-registro pide listarla.
    """
    tag = qtag(q, cov, split)
    ltag = lstm_qtag(q, cov, split)
    rows: List[Dict[str, Any]] = []
    for r in (routes or ROUTES):
        base = root / "data" / "02_interim" / "routes" / r
        fdir = base / "frames" / rng
        xdir = base / "xgb" / rng
        ldir = base / "lstm_frame" / rng
        fn = read_funnel(root, r, rng)
        metas = sorted(fdir.glob(f"*_{tag}_meta.json"))
        if not metas and fn.get("trip_cut") == "cruce":
            rows.append({"route": r, "name": route_label(r), "label": "",
                         "leg": "T0->T0", "M": 0, "split": split, "n_rungs": 0,
                         "min_n_train": 0, "min_n_test": 0, "n_trips_raw": 0,
                         "retention_pct": 0.0, "n_route_m": 0, "n_cells": 0,
                         "has_xgb": False, "has_lstm": False,
                         "has_devhist": False, "has_devhist_lstm": False, **fn})
            continue
        for p in metas:
            label = p.name[: -len(f"_{tag}_meta.json")]
            try:
                fm = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            leg = str(fm.get("leg") or label)
            # Escalones del frame temporal. `min_n_train` es la columna que
            # decide el panel de la escalera: una ruta solo es comparable a lo
            # largo de los escalones si cumple el umbral en TODOS, y el primero
            # es el que se queda corto. En un frame aleatorio salen 0.
            rungs = fm.get("rungs") or []
            rows.append({
                "route": r,
                "name": route_label(r),
                "label": label,
                "leg": leg,
                "M": int(fm.get("M") or 0),
                "split": str(fm.get("split") or "random"),
                "n_rungs": len(rungs),
                "min_n_train": int(min((x["n_train"] for x in rungs), default=0)),
                "min_n_test": int(min((x["n_test"] for x in rungs), default=0)),
                "n_trips_raw": int(fm.get("n_trips_raw") or 0),
                "retention_pct": float(fm.get("retention_pct") or 0.0),
                "n_route_m": int(fm.get("n_route_m") or 0),
                "n_cells": int(fm.get("n_cells") or 0),
                # `parquet_ok` y no `is_file()`: estas cuatro columnas viajan a
                # las tablas del paper como "esta ruta tiene métricas". Un fichero
                # truncado las ponía en True (2026-07-25: 11 artefactos del LSTM
                # con la cola en ceros pasaron las tres auditorías).
                # ⚠ El LSTM es el de `lstm_qtag` (`_es15`; `_tw` en la escalera): el
                # de `{tag}` a secas era el `es0` de agosto, que ya no se corre, y con
                # él toda ruta salía «sin LSTM» y caía de las tablas sin error.
                "has_xgb": pg.parquet_ok(xdir / f"{label}_{tag}_exo_overall.parquet")
                           or pg.parquet_ok(xdir / f"{label}_{tag}_overall.parquet"),
                "has_lstm": pg.parquet_ok(ldir / f"{label}_{ltag}_overall.parquet"),
                "has_devhist": any(
                    pg.parquet_ok(d) for d in (
                        xdir / f"{label}_{tag}_exo_devhist.parquet",
                        xdir / f"{label}_{tag}_devhist.parquet")),
                "has_devhist_lstm": pg.parquet_ok(
                    ldir / f"{label}_{ltag}_devhist.parquet"),
                # del funnel: el tipo de corte y su calidad
                **fn,
            })
    return pl.DataFrame(rows, infer_schema_length=None)


def apply_criteria(df: pl.DataFrame, min_M: int = MIN_M,
                   n_folds: int = 5) -> pl.DataFrame:
    """Añade ``quality``, ``reason``, ``included`` y ``flag``. El orden de los motivos
    importa: se reporta el PRIMERO que aplica (una ruta con M=4 y el corte mal se
    reporta como no computable, que es lo que manda)."""
    rows = list(df.iter_rows(named=True))
    qual = [cut_quality(r.get("cobertura_pct"), r.get("inferidos_pct"),
                        r.get("raras_pct"), r.get("atajo_pct")) for r in rows]

    def _reason(row: Dict[str, Any], q: Tuple[str, str]) -> str:
        if row["M"] < n_folds:
            return f"C3 no computable (M={row['M']} < n_folds={n_folds})"
        if q[0] in ("", "mal"):
            return f"C1 el corte sale mal ({q[1]})"
        if row["M"] < min_M:
            return f"C2 M={row['M']} < {min_M}"
        return ""

    reasons = [_reason(r, q) for r, q in zip(rows, qual)]
    return df.with_columns(
        pl.Series("quality", [q[0] for q in qual], dtype=pl.Utf8),
        pl.Series("reason", reasons, dtype=pl.Utf8),
        pl.Series("included", [x == "" for x in reasons], dtype=pl.Boolean),
        pl.Series("flag", [q[1] if q[0] == "revisar" else "" for q in qual],
                  dtype=pl.Utf8),
    )


def to_markdown(df: pl.DataFrame, min_M: int) -> str:
    """El informe del criterio (pre-registro §3): por ruta, sin estratos."""
    inc = df.filter(pl.col("included"))
    exc = df.filter(~pl.col("included"))
    n3 = inc.filter(pl.col("has_xgb") & pl.col("has_lstm")).height
    out: List[str] = []
    out.append("# Criterio de inclusión — la vuelta (pre-registrado)\n")
    out.append("Fijado el 6-oct-2026, antes de correr modelos: "
               "`docs/PREREGISTRO_VUELTA.md` §3. La justificación está en el docstring de "
               "`docs/paper/analysis/leg_inclusion.py`.\n")
    out.append(f"- **C1** el corte no sale mal: cobertura ≥ {VUELTA_MAL['cobertura_min']:g} %, "
               f"cruces inferidos ≤ {VUELTA_MAL['inferidos_max']:g} %, vueltas raras ≤ "
               f"{VUELTA_MAL['raras_max']:g} %\n"
               f"- **C2** `M >= {min_M}` vueltas en el frame\n"
               f"- **C3** computable (`M >= n_folds`)\n")
    out.append(f"\n**{inc.height} de {df.height} rutas entran** ({exc.height} excluidas). "
               f"De las incluidas, **{n3}** tienen ya las dos familias de modelos.\n")
    out.append("\n## Rutas excluidas\n")
    if exc.is_empty():
        out.append("Ninguna.\n")
    else:
        out.append("| ruta | leg | M | motivo |")
        out.append("|---|---|---|---|")
        for r in exc.iter_rows(named=True):
            out.append(f"| {route_label(r['route'], number=True)} | `{r['leg']}` | {r['M']} "
                       f"| {r['reason']} |")
    out.append("\n## Rutas incluidas\n")
    out.append("Las marcadas saldrían `revisar` en la calidad del corte: entran, como fija el "
               "pre-registro. Sus vueltas con atajo ya están fuera del frame.\n")
    out.append("| ruta | M | retención % | n_route (m) | celdas | cobertura % | con atajo % "
               "| fuera por atajo | marca | XGB | LSTM |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in inc.iter_rows(named=True):
        def _v(x, fmt="{}"):
            return "—" if x is None else fmt.format(x)
        out.append(f"| {route_label(r['route'], number=True)} | {r['M']} | "
                   f"{r['retention_pct']:.1f} | "
                   f"{r['n_route_m']:,} | {r['n_cells']} | {_v(r.get('cobertura_pct'))} | "
                   f"{_v(r.get('atajo_pct'))} | {_v(r.get('atajo_fuera'))} | "
                   f"{r.get('flag') or '—'} | {'✔' if r['has_xgb'] else '—'} | "
                   f"{'✔' if r['has_lstm'] else '—'} |")
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Criterio de inclusión (la vuelta)")
    ap.add_argument("--root", default=None,
                    help="raíz del árbol de datos (default: la del repo)")
    ap.add_argument("--q", type=float, default=0.05)
    ap.add_argument("--cov", type=float, default=1.0)
    ap.add_argument("--range", dest="rng", default=RANGE)
    ap.add_argument("--min-M", type=int, default=MIN_M)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--split", default="random",
                    choices=["random", "expanding"],
                    help="'expanding' lee los frames `_tw` y añade las columnas "
                         "de la escalera (n_rungs, min_n_train, min_n_test)")
    ap.add_argument("--out", default=None,
                    help="prefijo de salida (escribe .csv y .md). Default: "
                         "docs/paper/analysis/leg_inclusion")
    a = ap.parse_args()

    repo = Path(__file__).resolve().parents[3]
    root = Path(a.root).resolve() if a.root else repo
    df = collect_legs(root, a.q, a.cov, a.rng, split=a.split)
    if df.is_empty():
        print(f"  ✗ no hay frames {qtag(a.q, a.cov, a.split)} bajo {root}")
        return 1
    df = apply_criteria(df, a.min_M, a.n_folds)
    # el orden de los jobs (por línea, cada variante junto a su principal)
    orden = {r: i for i, r in enumerate(ROUTES)}
    df = (df.with_columns(pl.col("route").replace_strict(orden, default=len(orden))
                          .alias("_o"))
            .sort(["_o", "leg"]).drop("_o"))

    inc = df.filter(pl.col("included"))
    print("=" * 74)
    print(f"  CRITERIO DE INCLUSIÓN PRE-REGISTRADO · "
          f"{qtag(a.q, a.cov, a.split)} · {a.rng} · la vuelta (docs/PREREGISTRO_VUELTA.md)")
    print(f"  C1 el corte no sale mal · C2 M >= {a.min_M} · "
          f"C3 M >= n_folds={a.n_folds}")
    print("=" * 74)
    print(f"  rutas con frame       : {df.height}")
    print(f"  INCLUIDAS             : {inc.height}")
    print(f"  excluidas             : {df.height - inc.height}")
    for reason, n in (df.filter(~pl.col("included"))
                      .group_by(pl.col("reason").str.split(" ").list.first())
                      .agg(pl.len()).sort("reason").iter_rows()):
        print(f"      {reason}: {n}")
    print(f"  incluidas con XGB     : {inc['has_xgb'].sum()}")
    print(f"  incluidas con LSTM    : {inc['has_lstm'].sum()}  "
          f"({lstm_qtag(a.q, a.cov, a.split)})")
    print(f"  incluidas con devhist : {inc['has_devhist'].sum()} XGB · "
          f"{inc['has_devhist_lstm'].sum()} LSTM")
    marcadas = inc.filter(pl.col("flag") != "")
    print(f"  marcadas (revisar)    : {marcadas.height}"
          + "".join(f"\n      {r['route']}: {r['flag']}"
                    for r in marcadas.iter_rows(named=True)))

    prefix = Path(a.out) if a.out else (repo / "docs" / "paper" / "analysis"
                                        / "leg_inclusion")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(prefix.with_suffix(".csv"))
    prefix.with_suffix(".md").write_text(to_markdown(df, a.min_M), encoding="utf-8")
    print(f"\n  ✓ {prefix.with_suffix('.csv')}")
    print(f"  ✓ {prefix.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
