"""
xgb_prediction.run_experiment
===============================
Orquestador por (ruta, leg): construye T, corre la CV pareada con baselines +
XGB-A y persiste los agregados.

Salidas (en ``data/02_interim/routes/{ruta}/xgb/{ANALYSIS_RANGE}/``)
--------------------------------------------------------------------
    {leg}_overall.parquet      — una fila por algoritmo (equivalente a *_fullgrid)
    {leg}_by_fold.parquet      — métricas por (algoritmo, fold) → IC empírico
    {leg}_by_position.parquet  — métricas por (algoritmo, l_idx) → figura de posición
    {leg}_by_horizon.parquet   — métricas por (algoritmo, bin de tiempo restante)
    {leg}_devhist.parquet      — histograma de dev por (algoritmo, fold), bins 1 s
    {leg}_importance.parquet   — ganancia normalizada por feature
    {leg}_meta.json            — particiones, hiperparámetros, versiones

El `devhist` es lo que hace reportable el on-time (fase 5): de él salen la banda
simétrica ±180 s y el on-time calibrado por sesgo sin volver a correr. Ver
`func/honest_metrics.py`.

Uso
---
    BUS_ROUTE=101 BUS_ANALYSIS_RANGE=2024_01-08_tmin \
      python func/xgb_prediction/run_experiment.py --legs "T0->T1,T1->T0"
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import polars as pl

from config import PATHS, INT_ROUTE, ANALYSIS_RANGE          # noqa: E402
from arrival_prediction import build_trajectory_matrix, default_predictors  # noqa: E402
from xgb_prediction.cross_validation import (                 # noqa: E402
    InsufficientDataError,
    make_l_indices,
    run_matched_cv,
)
from xgb_prediction.model import (                            # noqa: E402
    DEFAULT_PARAMS,
    VARIANTS,
    XGBRemainingTime,
)


def _make_factories(params: Optional[Dict[str, Any]] = None,
                    with_exo: bool = False,
                    ablate: bool = False,
                    no_prior: bool = False,
                    trip_only: bool = False) -> Dict[str, Any]:
    """
    Variantes a evaluar. Cada entrada es (factory, exo_cols); exo_cols=None es
    Tier A. Todas corren en el MISMO loop de CV y sobre las MISMAS filas, así
    que las comparaciones entre variantes son pareadas por construcción.

      * siempre       xgb_mse, xgb_mae                    (Tier A)
      * --exo         + xgb_mse_b, xgb_mae_b              (Tier B completo)
      * --exo-ablate  + xgb_mae_<grupo> por cada grupo    (Tier A + UN grupo)
      * --no-prior    + xgb_mae_np                        (Tier A SIN el prior)
      * --trip-only   + xgb_mae_t0                        (además sin delay_vs_mu)

    La ablación usa solo el objetivo MAE: es el que se reporta, y duplicarla
    con MSE multiplicaría el coste sin aportar a la atribución.

    `no_prior` completa la escalera de INFORMACIÓN de tres niveles que pide la
    revisión de agosto-2026, y que hasta ahora no se podía correr porque el
    nivel 1 no existía: el prior histórico se calcula siempre y lo usaban todas
    las variantes, Tier A incluida. Los tres niveles quedan:

      0. `xgb_mae_t0`  trayectoria del viaje en curso, y NADA histórico
      1. `xgb_mae_np`  + `delay_vs_mu` (t_l − μ_l)
      2. `xgb_mae`     + el bloque prior_* completo
      3. `xgb_mae_b`   + bloque contextual derivado del AVL

    ⚠ El nivel 0 existe porque el 1 no era lo que su nombre decía: `delay_vs_mu`
    se calcula contra la media del fold de entrenamiento, así que el "nivel
    trayectoria" llevaba un gancho histórico dentro y la atribución salía
    sesgada a la baja.

    Los tres en el mismo loop y sobre las mismas filas, así que la comparación
    entre niveles es pareada igual que las demás.
    """
    from xgb_prediction.cross_validation import NO_PRIOR, TRIP_ONLY
    from xgb_prediction.exo_features import EXO_GROUPS, group_cols

    def _mk(name: str, extra: Dict[str, Any]):
        return lambda: XGBRemainingTime(
            params={**extra, **(params or {})}, name=name,
        )
    out: Dict[str, Any] = {
        name: (_mk(name, extra), None) for name, extra in VARIANTS.items()
    }
    if no_prior:
        out["xgb_mae_np"] = (_mk("xgb_mae_np", VARIANTS["xgb_mae"]), NO_PRIOR)
    if trip_only:
        out["xgb_mae_t0"] = (_mk("xgb_mae_t0", VARIANTS["xgb_mae"]), TRIP_ONLY)
    if with_exo:
        all_cols = group_cols(["all"])
        for name, extra in VARIANTS.items():
            nb = f"{name}_b"
            out[nb] = (_mk(nb, extra), all_cols)
        if ablate:
            extra = VARIANTS["xgb_mae"]
            for g in EXO_GROUPS:
                nb = f"xgb_mae_{g}"
                out[nb] = (_mk(nb, extra), group_cols([g]))
    return out


def _safe_label(group_value) -> str:
    """Etiqueta de carpeta/archivo — misma convención que step_10 / step_11."""
    if isinstance(group_value, (int, np.integer)) or (
            isinstance(group_value, str) and str(group_value).isdigit()):
        return f"dir{group_value}"
    s = str(group_value).replace("->", "_").replace(" ", "")
    s = "".join(c if (c.isalnum() or c in "_-") else "_" for c in s)
    return f"leg{s}"


def _out_dir() -> Path:
    d = PATHS.interim_route_dir / "xgb" / ANALYSIS_RANGE
    d.mkdir(parents=True, exist_ok=True)
    return d


def exo_descriptives(exo: np.ndarray) -> List[Dict[str, Any]]:
    """Cuantiles de cada feature de Tier B sobre todas las celdas (viaje × corte)
    con dato. No entra a ningún modelo: es descriptivo, y es determinista."""
    from xgb_prediction.exo_features import EXO_FEATURES, HEADWAY_CAP_S
    out: List[Dict[str, Any]] = []
    for k, name in enumerate(EXO_FEATURES):
        x = exo[:, :, k].astype(np.float64).ravel()
        x = x[np.isfinite(x)]
        if x.size == 0:
            continue
        q = np.percentile(x, [10, 25, 50, 75, 90])
        out.append({
            "feature": name, "n_cells": int(x.size), "mean": float(x.mean()),
            "p10": float(q[0]), "p25": float(q[1]), "p50": float(q[2]),
            "p75": float(q[3]), "p90": float(q[4]),
            "share_zero": float((x == 0).mean()),
            "share_at_cap": (float((x >= HEADWAY_CAP_S).mean())
                             if name.startswith("headway") else float("nan")),
        })
    return out


# ──────────────────────────────────────────────────────────────────

def run_one_leg(
    df_full:     pl.DataFrame,
    leg:         str,
    grid_step_m: int   = 100,
    n_folds:     int   = 5,
    seed:        int   = 42,
    n_h_train:   int   = 20,
    n_route_quantile: float = 0.95,
    min_coverage: float = 0.99,
    max_trips:   Optional[int] = None,
    params:      Optional[Dict[str, Any]] = None,
    with_baselines: bool = True,
    with_exo:    bool = False,
    ablate:      bool = False,
    no_prior:    bool = False,
    trip_only:   bool = False,
    write_frame: bool = True,
    split:       str  = "random",
    period:      str  = "month",
    verbose:     bool  = True,
) -> Optional[Dict[str, Any]]:
    """Corre la CV pareada para una leg. Devuelve None si se salta."""
    label = _safe_label(leg)
    t0 = time.time()

    if verbose:
        print(f"\n  {'─' * 60}")
        print(f"  leg {leg}  ({label})")
        print(f"  {'─' * 60}", flush=True)

    n_trips = (
        df_full.filter(pl.col("leg") == leg, pl.col("trip_valid"))
        ["trip_id"].n_unique()
    )
    if n_trips < n_folds:
        print(f"  ⚠ SALTADA: {n_trips} viaje(s) < n_folds={n_folds}")
        return None

    try:
        ds = build_trajectory_matrix(
            df_full, group_col="leg", group_value=leg,
            grid_step_m=grid_step_m, n_route_quantile=n_route_quantile,
            min_coverage=min_coverage,
            verbose=False,
        )
    except ValueError as e:
        print(f"  ⚠ SALTADA (dataset_builder): {e}")
        return None

    T, l_grid = ds["T"], ds["l_grid"]
    M_full = T.shape[0]
    trip_ids = list(ds["trip_ids"])

    # Tope de viajes. KNN y kernel son O(M_train · M_val · l) por posición
    # (cdist en predictors.py:152 y :196), así que el coste de los baselines
    # crece CUADRÁTICAMENTE con M. Con q=0.05 la retención pasa de ~6% a ~95%
    # y eso multiplica su coste por ~200×. El submuestreo se aplica ANTES de la
    # CV y por igual a todos los algoritmos, así que la comparación sigue siendo
    # pareada; solo reduce el tamaño del estudio.
    subsampled = False
    if max_trips is not None and M_full > max_trips:
        keep_idx = np.sort(np.random.default_rng(seed).choice(
            M_full, size=max_trips, replace=False))
        T = T[keep_idx]
        # trip_ids DEBE seguir el mismo submuestreo: es la clave con la que se
        # alinean los exógenos de Tier B con las filas de T.
        trip_ids = [trip_ids[i] for i in keep_idx]
        subsampled = True

    if verbose:
        keep = 100.0 * M_full / max(n_trips, 1)
        extra = (f"  ·  submuestreado a {T.shape[0]:,}" if subsampled else "")
        print(f"  T = {T.shape}  ·  n_route = {ds['n_route']:,} m  "
              f"(q={n_route_quantile}, cov={min_coverage})  "
              f"·  viajes crudos = {n_trips:,}  "
              f"·  retención = {keep:.1f}%{extra}", flush=True)

    # ── Frame de evaluación: se persiste LO QUE ESTA CORRIDA USA ──
    # No es una reconstrucción: son las mismas `trip_ids` y los mismos folds que
    # `run_matched_cv` recorre abajo. El LSTM lo lee y queda pareado por
    # construcción (`func/eval_frame.py`, `lstm_prediction_tf/run_frame.py`).
    # Las particiones se calculan UNA vez y se usan para el frame y para la CV.
    # Antes cada uno las derivaba de (M, n_folds, seed) por su cuenta, lo cual
    # funciona mientras la regla sea determinista; con la ventana expansiva ya
    # no lo es —depende de la fecha de cada viaje— así que pasan a viajar como
    # dato. Si volvieran a derivarse dos veces, un cambio en una de las dos
    # rutas rompería el pareo sin que nada avisara.
    import eval_frame as ef
    from xgb_prediction.cross_validation import make_folds
    n_cells_grid = T.shape[1] - 1
    l_idx_frame = [int(l) for l in make_l_indices(n_cells_grid)
                   if l < n_cells_grid]
    period_labels: list = []
    if split == ef.SPLIT_EXPANDING:
        try:
            periods = ef.trip_periods(df_full, trip_ids, "leg", leg, period)
            folds_here, period_labels = ef.make_folds_expanding(periods)
        except ValueError as e:
            print(f"  ⚠ SALTADA (ventana expansiva): {e}")
            return None
        n_folds = len(period_labels)
    else:
        folds_here = make_folds(T.shape[0], n_folds, seed)
    fold_by_row = np.empty(T.shape[0], dtype=np.int32)
    for fid, idx in enumerate(folds_here):
        fold_by_row[idx] = fid
    rungs = (ef.describe_rungs(folds_here, period_labels)
             if split == ef.SPLIT_EXPANDING else [])
    if verbose and rungs:
        print(f"  escalera de {len(rungs)} escalones: " +
              " · ".join(f"→{r['test_period']} ({r['n_train']}/{r['n_test']})"
                         for r in rungs), flush=True)

    frame_hash = None
    if write_frame:
        fr = ef.frame_from_arrays(
            trip_ids, fold_by_row, int(ds["n_route"]), grid_step_m, l_idx_frame,
            route=str(INT_ROUTE), analysis_range=ANALYSIS_RANGE,
            group_col="leg", leg=str(leg), label=label,
            n_route_quantile=n_route_quantile, min_coverage=min_coverage,
            n_grid_pts=int(T.shape[1]), n_cells=n_cells_grid,
            n_folds=n_folds, seed=seed, n_trips_raw=int(n_trips),
            M_retained=int(M_full),
            retention_pct=round(100.0 * M_full / max(n_trips, 1), 2),
            max_trips=max_trips, subsampled=subsampled,
            fold_sizes=[int(len(f)) for f in folds_here],
            split=split, period=(period if split == ef.SPLIT_EXPANDING else None),
            periods=period_labels, rungs=rungs,
        )
        ef.save_frame(fr, route=str(INT_ROUTE),
                      analysis_range=ANALYSIS_RANGE, verbose=verbose)
        frame_hash = fr["meta"]["frame_hash"]

    # ── Tier B: exógenos muestreados en cada corte (nunca más allá) ──
    # `hours` alimenta al baseline estratificado; se extrae en la misma pasada.
    exo = hours = None
    if with_exo:
        # `make_l_indices` ya está importado a nivel de módulo: re-importarlo aquí
        # lo convertía en local de la función y rompía su uso anterior (frame).
        from xgb_prediction.exo_features import build_exo_matrix
        n = T.shape[1] - 1
        l_indices = [int(l) for l in make_l_indices(n) if l < n]
        exo, hours = build_exo_matrix(
            df_full, group_col="leg", group_value=leg, trip_ids=trip_ids,
            l_grid=l_grid, l_indices=l_indices, verbose=verbose,
            return_hours=True,
        )

    bl = default_predictors() if with_baselines else {}
    if with_baselines and hours is not None:
        # El baseline honesto que faltaba: mismo `delay`, pero con la media
        # histórica condicionada a la hora del día. Si captura buena parte de
        # la ganancia de Tier B, la conclusión no es "hace falta XGBoost" sino
        # "hace falta un prior por franja horaria".
        from xgb_prediction.stratified_baselines import HourStratifiedDelay
        bl["delay_hour"] = HourStratifiedDelay()

    try:
        res = run_matched_cv(
            T=T, l_grid=l_grid,
            baselines=bl,
            xgb_models=_make_factories(params, with_exo=with_exo, ablate=ablate,
                                       no_prior=no_prior, trip_only=trip_only),
            exo=exo, strata=hours,
            n_folds=n_folds, seed=seed, n_h_train=n_h_train,
            folds=folds_here, split=split,
            verbose=verbose,
        )
    except InsufficientDataError as e:
        print(f"  ⚠ SALTADA (folds vacíos): {e}")
        return None

    if exo is not None:
        res["exo_descriptives"] = exo_descriptives(exo)

    res["metadata"].update({
        "route": str(INT_ROUTE), "leg": leg, "label": label,
        "analysis_range": ANALYSIS_RANGE, "grid_step_m": grid_step_m,
        "n_route_m": int(ds["n_route"]),
        "n_route_quantile": n_route_quantile,
        "min_coverage": min_coverage,
        "frame_hash": frame_hash,
        "n_trips_raw": int(n_trips),
        "M_retained": int(M_full),
        "retention_pct": round(100.0 * M_full / max(n_trips, 1), 2),
        "max_trips": max_trips,
        "subsampled": subsampled,
        "with_baselines": with_baselines,
        "with_exo": with_exo,
        "ablate": ablate,
        "no_prior": no_prior,
        "trip_only": trip_only,
        "split": split,
        "period": (period if split == "expanding" else None),
        "periods": period_labels,
        "rungs": rungs,
        "xgb_params": {**DEFAULT_PARAMS, **(params or {})},
        "elapsed_s": round(time.time() - t0, 1),
    })
    return res


# ──────────────────────────────────────────────────────────────────

def persist(res: Dict[str, Any], out_dir: Path, verbose: bool = True) -> None:
    md    = res["metadata"]
    route = md["route"]
    leg   = md["leg"]
    # El régimen COMPLETO (cuantil + cobertura) entra en el nombre: corridas con
    # distinta definición de grid NO deben pisarse ni darse por hechas entre sí.
    # ⚠ Antes solo llevaba el cuantil, y eso hizo que la corrida del 2026-07-24
    # con --min-coverage 1.0 encontrara los agregados del 23-jul (cov=0.99) y se
    # saltara las 26 rutas como `cached`, sin escribir un solo frame.
    import eval_frame as _ef
    label = (f"{md['label']}_"
             f"{_ef.q_tag(md['n_route_quantile'], md.get('min_coverage', 0.99), md.get('split', _ef.SPLIT_RANDOM))}")
    # ⚠ La ablación TAMBIÉN tiene with_exo=True. Sin este sufijo propio, una
    # corrida --exo-ablate ve los archivos de una --exo previa y se salta todo
    # como "cached" (pasó en la corrida del 2026-07-23).
    if md.get("ablate"):
        label += "_ablate"
    elif md.get("with_exo"):
        label += "_exo"
    # Mismo motivo, tercer eje: una corrida --no-prior añade una variante que
    # las anteriores no tenían, así que sus agregados NO son los de una --exo
    # previa. Va después del sufijo de exógenos, no en su lugar: las dos
    # dimensiones son independientes y la del nivel 1 no anula la de Tier B.
    if md.get("no_prior"):
        label += "_np"
    if md.get("trip_only"):
        label += "_t0"
    # Cuarto eje: el tope de árboles. Sólo cuando no es el de siempre (600), así
    # que los nombres de todas las corridas anteriores no cambian. Una corrida con
    # otro tope NO es la publicada y no debe darse por hecha contra ella.
    n_est = int((md.get("xgb_params") or {}).get("n_estimators",
                                                   DEFAULT_PARAMS["n_estimators"]))
    if n_est != DEFAULT_PARAMS["n_estimators"]:
        label += f"_n{n_est}"

    overall = pl.DataFrame([
        {"route": route, "leg": leg, "algorithm": a, **m}
        for a, m in res["overall"].items()
    ]).sort("MAE_s")
    overall.write_parquet(out_dir / f"{label}_overall.parquet")

    by_fold = pl.DataFrame([
        {"route": route, "leg": leg, "algorithm": a, **m}
        for a, rows in res["by_fold"].items() for m in rows
    ]).sort(["algorithm", "fold"])
    by_fold.write_parquet(out_dir / f"{label}_by_fold.parquet")

    # Coste computacional por (fold, algoritmo) — la mitad CPU de lo que §3.9
    # promete. Va como artefacto HERMANO y no como columna de `_overall`:
    # el packer (`job_frame_pack.slrm`, rc=4) y las auditorías comprueban el
    # trío overall/devhist/by_horizon, y ampliar el esquema de `_overall`
    # obligaría a tocarlas. Se escribe solo si existe, así que los paquetes
    # anteriores —que no lo llevan— se siguen leyendo sin cambios.
    tm = res.get("timing") or []
    if tm:
        pl.DataFrame(
            [{"route": route, "leg": leg, **r} for r in tm]
        ).sort(["algorithm", "fold"]).write_parquet(
            out_dir / f"{label}_timing.parquet"
        )

    by_pos = pl.DataFrame([
        {"route": route, "leg": leg, "algorithm": a, "l_idx": l, **m}
        for a, d in res["by_position"].items() for l, m in d.items()
    ]).sort(["algorithm", "l_idx"])
    by_pos.write_parquet(out_dir / f"{label}_by_position.parquet")

    # Rendimiento por horizonte (§4.5): tiempo restante REAL, eje ortogonal al
    # de posición. `by_position` agrega sobre horizontes, así que esto no se
    # puede derivar de él.
    bh = res.get("by_horizon") or {}
    if bh:
        pl.DataFrame([
            {"route": route, "leg": leg, "algorithm": a, "horizon": h, **m}
            for a, d in bh.items() for h, m in d.items()
        ]).write_parquet(out_dir / f"{label}_by_horizon.parquet")

    # Histograma con signo de `dev` por (algoritmo, fold) — fase 5. Es lo que
    # permite derivar a posteriori cualquier banda de on-time y el on-time
    # calibrado por sesgo SIN volver a correr (`func/honest_metrics.py`). Solo
    # se guardan los bins no vacíos: son KB.
    dh = res.get("dev_hist") or {}
    if dh:
        parts = []
        for a, folds in dh.items():
            for fold_id, centers, counts in folds:
                if centers.size == 0:
                    continue
                parts.append(pl.DataFrame({
                    "route":     pl.Series([route] * centers.size, dtype=pl.Utf8),
                    "leg":       pl.Series([leg] * centers.size, dtype=pl.Utf8),
                    "algorithm": pl.Series([a] * centers.size, dtype=pl.Utf8),
                    "fold":      pl.Series(np.full(centers.size, fold_id),
                                           dtype=pl.Int32),
                    # bin_s = 1 s ⇒ los centros son enteros exactos.
                    "dev_s":     pl.Series(centers.astype(np.int32),
                                           dtype=pl.Int32),
                    "count":     pl.Series(counts, dtype=pl.Int64),
                }))
        if parts:
            pl.concat(parts).write_parquet(out_dir / f"{label}_devhist.parquet")

    # Descriptivos del bloque de contexto EN LOS CORTES, tal como lo ve el modelo
    # (headway ya saturado a HEADWAY_CAP_S). Para la Discusión («dwell median 0 s,
    # headway median 15.7 min» eran de los legs de agosto). Sólo con Tier B.
    ed = res.get("exo_descriptives") or []
    if ed:
        pl.DataFrame([{"route": route, "leg": leg, **r} for r in ed]).write_parquet(
            out_dir / f"{label}_exodesc.parquet")

    if res["importance"]:
        pl.DataFrame([
            {"route": route, "leg": leg, "algorithm": a, "feature": f,
             "gain_frac": g}
            for a, d in res["importance"].items() for f, g in d.items()
        ]).write_parquet(out_dir / f"{label}_importance.parquet")

    (out_dir / f"{label}_meta.json").write_text(
        json.dumps(md, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    if verbose:
        print(f"  ✔ agregados → {out_dir / (label + '_*.parquet')}")


def report(res: Dict[str, Any]) -> None:
    """Tabla de consola: métricas globales y top de importancias."""
    md = res["metadata"]
    print(f"\n  RESULTADOS · ruta {md['route']} · leg {md['leg']} "
          f"· q={md.get('n_route_quantile')} · M={md['M']} "
          f"({md.get('retention_pct')}% de {md.get('n_trips_raw'):,}) "
          f"· {md['n_folds']} folds · {md['elapsed_s']}s")
    print(f"  {'algoritmo':<12} {'n_pairs':>12} {'MAE_s':>10} {'RMSE_s':>10} "
          f"{'MedAE_s':>10} {'P90AE_s':>10} {'bias_s':>10} {'on-time %':>10}")
    print("  " + "─" * 90)

    rows = sorted(res["overall"].items(), key=lambda kv: kv[1]["MAE_s"])
    best_mae = rows[0][0]
    best_ot  = max(res["overall"].items(), key=lambda kv: kv[1]["pct_ontime_dev"])[0]
    for a, m in rows:
        mark = ""
        if a == best_mae:
            mark += " ←MAE"
        if a == best_ot:
            mark += " ←on-time"
        print(f"  {a:<12} {m['n_pairs']:>12,} {m['MAE_s']:>10.2f} "
              f"{m['RMSE_s']:>10.2f} {m['MedAE_s']:>10.2f} "
              f"{m['P90AE_s']:>10.2f} {m['bias_s']:>10.2f} "
              f"{m['pct_ontime_dev']:>10.2f}{mark}")

    for algo, imp in res["importance"].items():
        print(f"\n  Importancia por ganancia · {algo} (top 8, media de folds):")
        for i, (f, g) in enumerate(list(imp.items())[:8], 1):
            bar = "█" * max(1, int(round(g * 40)))
            print(f"    {i:2d}. {f:<18} {g*100:5.1f}%  {bar}")

    bi = res["metadata"].get("best_iterations") or {}
    if bi:
        txt = " · ".join(f"{a}={np.mean(v):.0f}" for a, v in bi.items())
        print(f"\n  Árboles elegidos por early stopping (media): {txt}")


# ──────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description="XGB-A por leg (fase 1)")
    ap.add_argument("--legs", default=None,
                    help="lista separada por comas; default = todas las del df")
    ap.add_argument("--grid-step-m", type=int, default=100)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-h-train", type=int, default=20)
    ap.add_argument("--n-route-quantile", type=float, default=0.95,
                    help="cuantil de longitud de viaje que fija la extensión de "
                         "la grid. 0.95 = valor histórico (retiene ~6%% de los "
                         "viajes); 0.05-0.10 = soporte común (~95%%)")
    ap.add_argument("--min-coverage", type=float, default=0.99,
                    help="fracción de la grid que un viaje debe cubrir. 1.0 "
                         "elimina toda saturación de np.interp (target "
                         "fabricado) y hace que n_pairs cuadre EXACTO con el "
                         "LSTM; 0.99 (histórico) deja fabricado hasta el 1%% "
                         "final de la grid.")
    ap.add_argument("--max-trips", type=int, default=None,
                    help="tope de viajes por leg (submuestreo con seed fijo, "
                         "igual para todos los algoritmos). Acota el coste "
                         "cuadrático de KNN/kernel.")
    ap.add_argument("--no-frames", action="store_true",
                    help="no escribir el frame de evaluación (por defecto SÍ se "
                         "escribe: es lo que consume el LSTM para quedar pareado)")
    ap.add_argument("--split", default="random",
                    choices=["random", "expanding"],
                    help="'random' = los 5 folds de siempre (¿cuánta señal hay "
                         "en la trayectoria?). 'expanding' = ventana expansiva "
                         "por periodo: entrena con los meses anteriores y "
                         "evalúa el siguiente (¿sirve en despliegue?). Escribe "
                         "en artefactos con sufijo `_tw`, así que NO pisa la "
                         "corrida aleatoria.")
    ap.add_argument("--period", default="month",
                    choices=["month", "week", "day"],
                    help="grano de la escalera temporal (default: mes)")
    ap.add_argument("--no-baselines", action="store_true")
    ap.add_argument("--exo-ablate", action="store_true", dest="ablate",
                    help="añade una variante por grupo de Tier B (cal/headway/"
                         "dwell/speed/route) para atribuir la ganancia. Implica --exo")
    ap.add_argument("--exo", action="store_true",
                    help="añade Tier B (headway, dwell, velocidad reciente, "
                         "calendario) como variantes xgb_*_b")
    ap.add_argument("--trip-only", action="store_true", dest="trip_only",
                    help="añade xgb_mae_t0: nivel 0, sin prior_* NI delay_vs_mu. "
                         "Es el unico verdaderamente trajectory-only.")
    ap.add_argument("--no-prior", action="store_true", dest="no_prior",
                    help="añade xgb_mae_np: Tier A SIN el bloque de prior "
                         "histórico. Es el nivel 1 de la ablación de "
                         "información; combínalo con --exo para tener los tres "
                         "niveles pareados en una sola corrida.")
    ap.add_argument("--n-estimators", type=int, default=None,
                    help="tope de árboles de TODAS las variantes XGBoost (default "
                         f"{DEFAULT_PARAMS['n_estimators']}, el publicado). Otro valor "
                         "escribe con sufijo `_n<N>` y no pisa la corrida publicada.")
    a = ap.parse_args()
    params = ({"n_estimators": int(a.n_estimators)}
              if a.n_estimators is not None else None)

    print("=" * 72)
    print(f"  XGB-A · ruta {INT_ROUTE} · {ANALYSIS_RANGE}")
    print("=" * 72)

    df_full = pl.read_parquet(PATHS.df_full)
    print(f"  df_full: {df_full.height:,} filas · {len(df_full.columns)} columnas")

    if a.legs:
        legs: List[str] = [s.strip() for s in a.legs.split(",") if s.strip()]
    else:
        legs = (
            df_full.filter(pl.col("trip_valid"))
            .group_by("leg").agg(pl.col("trip_id").n_unique().alias("m"))
            .sort("m", descending=True)["leg"].to_list()
        )
    print(f"  legs: {legs}")

    out_dir = _out_dir()
    n_ok = 0
    for leg in legs:
        res = run_one_leg(
            df_full, leg,
            grid_step_m=a.grid_step_m, n_folds=a.n_folds, seed=a.seed,
            n_h_train=a.n_h_train, n_route_quantile=a.n_route_quantile,
            min_coverage=a.min_coverage,
            max_trips=a.max_trips, with_baselines=not a.no_baselines,
            with_exo=a.exo or a.ablate, ablate=a.ablate,
            no_prior=a.no_prior, trip_only=a.trip_only,
            write_frame=not a.no_frames,
            split=a.split, period=a.period, params=params,
        )
        if res is None:
            continue
        report(res)
        persist(res, out_dir)
        n_ok += 1

    print(f"\n  ✅ {n_ok}/{len(legs)} legs completadas → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
