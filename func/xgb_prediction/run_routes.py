"""
xgb_prediction.run_routes
===========================
Driver multi-ruta para el server. Corre `run_experiment.py` como SUBPROCESO
independiente por ruta.

Por qué subproceso y no un bucle en el mismo intérprete
-------------------------------------------------------
Una ruta que se queda sin memoria (KNN/kernel son O(M²), ver más abajo) o que
revienta con SIGSEGV mata el intérprete entero. Aislando por subproceso, una
ruta caída no tumba las otras 26 — mismo patrón que `run_new_pipeline.py`.

Casos que maneja
----------------
  1. `df_full` inexistente para una ruta          → SKIP (status: no_df_full)
  2. ruta en la lista de exclusión (902)          → SKIP (status: excluded)
  3. ruta ya calculada                            → SKIP (status: cached) salvo --force
  4. subproceso caído (OOM / SIGSEGV / timeout)   → FAIL registrado, sigue con la siguiente
  5. leg con menos viajes que folds               → lo salta `run_experiment`
  6. reanudación                                  → el estado se persiste tras CADA ruta

Estado en `<out>/xgb_status_<rango>_q<NN>.json`, reescrito tras cada ruta: un
timeout de SLURM deja el avance registrado, no se pierde.

Uso
---
    python func/xgb_prediction/run_routes.py --routes all --n-route-quantile 0.05
    python func/xgb_prediction/run_routes.py --routes 101,102 --no-baselines
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))

# Las 35 rutas de la corrida con la vuelta (docs/PREREGISTRO_VUELTA.md): las 26 de
# agosto, la 902 y las 8 de las familias con variante (1101-ALT = 110101,
# 1801-ALT = 180101, 1802-ALT = 180201; la 1401 y la 1402 son el group_id 1400 del
# AVL, separado). Ya no se excluye ninguna: en agosto la 902 quedaba fuera porque
# sin descansos en terminal no se partían sus viajes de ida y vuelta, y con la
# vuelta eso no importa. Mismo orden que los ROUTES de jobs/*.slrm.
ALL_ROUTES: List[str] = [
    "101", "102", "201", "202", "301", "302", "401", "402", "501", "601",
    "801", "901", "902", "1001", "1002", "1101", "110101", "1201", "1301",
    "1401", "1402", "1501", "1502", "1601", "1602", "1701", "1702", "1703",
    "1801", "180101", "1802", "180201", "1901", "1902", "20000",
]
EXCLUDED_ROUTES: set = set()


def _repo_root() -> Path:
    return Path(os.environ.get("BUSONORA_ROOT", _HERE.parent.parent.parent)).resolve()


def _df_full_path(route: str, analysis_range: str) -> Path:
    return (_repo_root() / "data" / "02_interim" / "routes" / route
            / "dataframes" / f"df_rec_full_{analysis_range}.parquet")


def _out_dir(route: str, analysis_range: str) -> Path:
    return (_repo_root() / "data" / "02_interim" / "routes" / route
            / "xgb" / analysis_range)


def _mode_suffix(exo: bool, ablate: bool, no_prior: bool = False,
                 trip_only: bool = False, n_estimators: int | None = None) -> str:
    """Sufijo de archivo por modo. Los modos de exógenos son EXCLUYENTES entre
    sí: si comparten sufijo, una corrida se salta a otra como 'cached'.

    ⚠ `no_prior` es ORTOGONAL a los otros dos y por eso se concatena en vez de
    sustituir — y en el MISMO orden que `run_experiment.py` pone en el label de
    los artefactos. Si los dos órdenes se separan, `_already_done` busca con un
    glob que no existe: la ruta se recalcula entera cada vez y nadie se entera,
    que es el fallo silencioso simétrico al de saltarse el trabajo.
    """
    suf = "_ablate" if ablate else ("_exo" if exo else "")
    if no_prior:
        suf += "_np"
    if trip_only:
        suf += "_t0"
    # El tope de árboles, sólo si no es el publicado (600): mismo orden y misma
    # regla que `persist` en run_experiment.py.
    if n_estimators is not None and int(n_estimators) != 600:
        suf += f"_n{int(n_estimators)}"
    return suf


def _already_done(route: str, analysis_range: str, rtag: str,
                  exo: bool, ablate: bool,
                  need_devhist: bool = True, no_prior: bool = False,
                  trip_only: bool = False,
                  n_estimators: int | None = None) -> bool:
    """¿Esta ruta ya está calculada EN ESTE RÉGIMEN?

    ⚠ `rtag` debe incluir cuantil Y cobertura. Cuando solo llevaba el cuantil,
    una corrida con `--min-coverage 1.0` daba por hechas las 26 rutas contra los
    agregados de una corrida previa con 0.99 y no hacía nada (2026-07-24).

    Desde la fase 5 exige también el `devhist`: un agregado sin él es de una
    corrida vieja y NO permite derivar el on-time honesto (banda simétrica y
    calibración por sesgo). Igual que arriba, darlo por hecho sería un fallo
    silencioso — solo que esta vez el que se pierde es el reporte de on-time.
    """
    d = _out_dir(route, analysis_range)
    suf = _mode_suffix(exo, ablate, no_prior, trip_only, n_estimators)
    if not (d.is_dir() and any(d.glob(f"leg*_{rtag}{suf}_overall.parquet"))):
        return False
    if need_devhist and not any(d.glob(f"leg*_{rtag}{suf}_devhist.parquet")):
        return False
    # Igual para el desglose por horizonte (§4.5): sin él la sección no se
    # puede escribir, y tampoco se recupera después.
    if need_devhist and not any(d.glob(f"leg*_{rtag}{suf}_by_horizon.parquet")):
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="XGB por ruta (server)")
    ap.add_argument("--routes", default="all",
                    help="'all' o lista separada por comas")
    ap.add_argument("--exclude", default="",
                    help="rutas extra a excluir, separadas por comas")
    ap.add_argument("--analysis-range", default=None,
                    help="default: $BUS_ANALYSIS_RANGE o 2024_01-08_tmin")
    ap.add_argument("--n-route-quantile", type=float, default=0.95)
    ap.add_argument("--min-coverage", type=float, default=0.99,
                    help="1.0 = sin saturación de np.interp y n_pairs exacto "
                         "contra el LSTM (ver docs/PLAN_REPRESENTATIVIDAD.md)")
    ap.add_argument("--split", default="random",
                    choices=["random", "expanding"],
                    help="'expanding' = ventana expansiva por periodo (entrena "
                         "con los meses anteriores, evalúa el siguiente). "
                         "Escribe con sufijo `_tw` y no pisa la corrida "
                         "aleatoria ni se salta contra ella.")
    ap.add_argument("--period", default="month",
                    choices=["month", "week", "day"])
    ap.add_argument("--no-frames", action="store_true",
                    help="no escribir los frames de evaluación")
    ap.add_argument("--max-trips", type=int, default=None)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--n-h-train", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-baselines", action="store_true")
    ap.add_argument("--exo", action="store_true",
                    help="Tier B: añade xgb_*_b (headway, dwell, velocidad "
                         "reciente, calendario) y el baseline `delay_hour`")
    ap.add_argument("--trip-only", action="store_true", dest="trip_only",
                    help="anade xgb_mae_t0 (nivel 0: sin prior_* ni "
                         "delay_vs_mu). Combinalo con --no-prior y --exo para "
                         "los cuatro niveles pareados en una sola corrida.")
    ap.add_argument("--no-prior", action="store_true", dest="no_prior",
                    help="añade xgb_mae_np (Tier A sin el prior histórico): el "
                         "nivel 1 de la ablación de información. Con --exo, los "
                         "tres niveles salen pareados en una sola corrida.")
    ap.add_argument("--exo-ablate", action="store_true", dest="ablate",
                    help="ablación por grupo (cal/headway/dwell/speed/route). "
                         "Implica --exo. CARA: 14 algoritmos por leg.")
    ap.add_argument("--n-estimators", type=int, default=None,
                    help="tope de árboles de XGBoost (default 600, el publicado); "
                         "otro valor escribe con sufijo `_n<N>`")
    ap.add_argument("--force", action="store_true",
                    help="recalcula aunque ya existan resultados")
    ap.add_argument("--accept-missing-devhist", action="store_true",
                    help="da por hecha una ruta aunque sus agregados sean de "
                         "antes de la fase 5 (sin histograma de dev). Esas legs "
                         "quedan SIN banda simétrica ni on-time calibrado.")
    ap.add_argument("--timeout-min", type=int, default=240,
                    help="tope por ruta; al superarlo se mata y se sigue")
    ap.add_argument("--out", default=None,
                    help="carpeta del JSON de estado (default: cwd/route_reports/xgb)")
    a = ap.parse_args()

    analysis_range = (a.analysis_range
                      or os.environ.get("BUS_ANALYSIS_RANGE")
                      or "2024_01-08_tmin")
    # Etiqueta de régimen COMPLETA (cuantil + cobertura), compartida con
    # eval_frame y con la persistencia de run_experiment.
    sys.path.insert(0, str(_HERE.parent))
    import eval_frame as _ef                                       # noqa: E402
    # El sufijo de partición entra en `rtag`, así que atraviesa a la vez el
    # nombre del status, el de los artefactos y `_already_done`. Sin eso una
    # corrida temporal daría por hechas las rutas contra los agregados de la
    # aleatoria y no escribiría nada — exactamente lo que pasó el 2026-07-24
    # cuando el tag no llevaba la cobertura.
    rtag = _ef.q_tag(a.n_route_quantile, a.min_coverage, a.split)

    routes = (list(ALL_ROUTES) if a.routes.strip().lower() == "all"
              else [r.strip() for r in a.routes.split(",") if r.strip()])
    extra_excl = {r.strip() for r in a.exclude.split(",") if r.strip()}

    out_dir = Path(a.out) if a.out else (Path.cwd() / "route_reports" / "xgb")
    out_dir.mkdir(parents=True, exist_ok=True)
    # El nombre lleva tier Y task id: en modo array cada tarea escribe el suyo,
    # si no se pisan entre ellas y solo sobrevive la última (bug de la 1ª corrida).
    suf = _mode_suffix(a.exo, a.ablate, a.no_prior, a.trip_only, a.n_estimators)
    tid = os.environ.get("SLURM_ARRAY_TASK_ID")
    tag = f"_t{tid}" if tid is not None else ""
    status_path = out_dir / f"xgb_status_{analysis_range}_{rtag}{suf}{tag}.json"

    py = sys.executable
    script = _HERE.parent / "run_experiment.py"

    print("=" * 76)
    print(f"  XGB multi-ruta · {analysis_range} · {rtag} "
          f"(q={a.n_route_quantile}, cov={a.min_coverage}) "
          f"· baselines={'no' if a.no_baselines else 'sí'} "
          f"· tier={'A+B+ablación' if a.ablate else ('A+B' if a.exo else 'A')}")
    print(f"  rutas: {len(routes)}  ·  repo root: {_repo_root()}")
    print(f"  estado → {status_path}")
    print("=" * 76, flush=True)

    status: Dict[str, Any] = {}
    t_all = time.time()

    for i, route in enumerate(routes, 1):
        head = f"[{i}/{len(routes)}] ruta {route}"

        if route in EXCLUDED_ROUTES or route in extra_excl:
            print(f"{head}: EXCLUIDA", flush=True)
            status[route] = {"status": "excluded"}
        elif not _df_full_path(route, analysis_range).is_file():
            print(f"{head}: SIN df_full → {_df_full_path(route, analysis_range)}",
                  flush=True)
            status[route] = {"status": "no_df_full",
                             "path": str(_df_full_path(route, analysis_range))}
        elif _already_done(route, analysis_range, rtag, a.exo, a.ablate,
                           need_devhist=not a.accept_missing_devhist,
                           no_prior=a.no_prior, trip_only=a.trip_only,
                           n_estimators=a.n_estimators
                           ) and not a.force:
            print(f"{head}: ya calculada (usa --force para rehacer)", flush=True)
            status[route] = {"status": "cached"}
        else:
            cmd = [py, "-u", str(script),
                   "--n-route-quantile", str(a.n_route_quantile),
                   "--min-coverage", str(a.min_coverage),
                   "--n-folds", str(a.n_folds),
                   "--n-h-train", str(a.n_h_train),
                   "--seed", str(a.seed)]
            if a.max_trips:
                cmd += ["--max-trips", str(a.max_trips)]
            if a.no_frames:
                cmd += ["--no-frames"]
            if a.no_baselines:
                cmd += ["--no-baselines"]
            if a.ablate:
                cmd += ["--exo-ablate"]
            elif a.exo:
                cmd += ["--exo"]
            if a.no_prior:
                cmd += ["--no-prior"]
            if a.trip_only:
                cmd += ["--trip-only"]
            if a.n_estimators is not None:
                cmd += ["--n-estimators", str(a.n_estimators)]
            if a.split != "random":
                cmd += ["--split", a.split, "--period", a.period]

            env = {**os.environ,
                   "BUS_ROUTE": route,
                   "BUS_ANALYSIS_RANGE": analysis_range,
                   "PYTHONIOENCODING": "utf-8",
                   "PYTHONUNBUFFERED": "1",
                   "MPLBACKEND": "Agg"}

            print(f"{head}: corriendo …", flush=True)
            t0 = time.time()
            try:
                rc = subprocess.run(cmd, env=env,
                                    timeout=a.timeout_min * 60).returncode
                took = round((time.time() - t0) / 60, 2)
                if rc == 0:
                    print(f"{head}: ✔ OK ({took} min)", flush=True)
                    status[route] = {"status": "ok", "elapsed_min": took}
                else:
                    # rc negativo = señal (p.ej. -11 = SIGSEGV, -9 = OOM-killer)
                    print(f"{head}: ✗ FALLÓ rc={rc} ({took} min)", flush=True)
                    status[route] = {"status": "failed", "returncode": rc,
                                     "elapsed_min": took,
                                     "hint": ("señal (OOM/SIGSEGV): baja "
                                              "--max-trips o sube --mem"
                                              if rc and rc < 0 else None)}
            except subprocess.TimeoutExpired:
                took = round((time.time() - t0) / 60, 2)
                print(f"{head}: ✗ TIMEOUT tras {took} min", flush=True)
                status[route] = {"status": "timeout", "elapsed_min": took}

        # Persistir tras CADA ruta: un timeout de SLURM no borra el avance.
        status_path.write_text(
            json.dumps({"analysis_range": analysis_range,
                        "n_route_quantile": a.n_route_quantile,
                        # La cobertura es parte de la identidad del régimen
                        # (define la tarea, no solo la muestra): sin ella en el
                        # estado no se puede saber a qué corrida corresponde.
                        "min_coverage": a.min_coverage,
                        "regime_tag": rtag,
                        "max_trips": a.max_trips,
                        "with_baselines": not a.no_baselines,
                        "write_frames": not a.no_frames,
                        "routes": status}, indent=2, ensure_ascii=False),
            encoding="utf-8")

    n = {k: sum(1 for v in status.values() if v["status"] == k)
         for k in ("ok", "cached", "skipped", "failed", "timeout",
                   "excluded", "no_df_full")}
    print("\n" + "=" * 76)
    print(f"  RESUMEN ({round((time.time() - t_all) / 60, 1)} min): "
          + " · ".join(f"{k}={v}" for k, v in n.items() if v))
    print(f"  estado → {status_path}")
    print("=" * 76)
    return 0 if (n["failed"] + n["timeout"]) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
