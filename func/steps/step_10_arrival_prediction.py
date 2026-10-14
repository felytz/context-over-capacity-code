"""
step_10_arrival_prediction.py
==============================
PASO 10 · Predicción de tiempos de llegada (paper Dublin Bus 46A).

Aplica el paquete `arrival_prediction` sobre el `df_full` producido por
el paso 07 para replicar el experimento del paper:

    Zhao, J. et al. (2005) / Bus Arrival Time Prediction Using Support Vector
    Machines, y la formulación vectorizada de trayectorias de
    Cathey & Dailey (2003) adaptada a la Ruta 101.

Los 4 algoritmos evaluados (siguiendo el paper):
    1. Delay-based        — baseline naive
    2. Linear Regression  — OLS con ridge para estabilidad
    3. K-Nearest Neighbors (K=10)
    4. Kernel Regression  (Gaussian, b=1.0)

Sub-tareas
----------
    10a  Dirección 0 — construye matriz T, corre 5-fold CV, genera outputs.
    10b  Dirección 1 — ídem para la dirección de regreso.
    10c  (Opcional) Grid de sensibilidad a 500 m para comparar con Fig. 6 del paper.

Inputs
------
    df_full : pl.DataFrame
        Salida del paso 07b — incluye `cum_dist_m`, `trip_id`, `direction`,
        `trip_valid`, y todas las features calculadas.

Outputs
-------
    Por dirección, en ``data/02_interim/routes/{route}/arrival/{dir}/``:
        - df_errors_{dir}.parquet    — errores por (posición, horizonte, fold, algoritmo)
        - dashboard_{dir}.html       — dashboard interactivo (Chart.js)
        - static/                    — Figuras 2, 4, 5, 6 del paper en PNG/PDF
        - T_{dir}.npy                — matriz de trayectorias interpoladas

    Resumen consolidado (ambas direcciones):
        - arrival_summary.parquet    — RMSE promedio por algoritmo y dirección
        - arrival_report.html        — reporte HTML comparativo

Dependencia del paquete
-----------------------
    El paquete `arrival_prediction` debe estar en sys.path. La estructura
    esperada (en src/arrival_prediction/) es la entregada en la conversación
    anterior. Si no está disponible, el paso falla con un ImportError claro.

Usage
-----
    from step_10_arrival_prediction import run_arrival_prediction_step

    result = run_arrival_prediction_step(df_full, verbose=True)
    # result["dir0"], result["dir1"], result["summary_path"]

    # También cacheable: si los parquets existen, no recalcula.

Notas de integración
--------------------
    - `cum_dist_m` debe existir en df_full (lo produce camino1_interpolation
      en el paso 07b). Si no existe, el paso la calcula internamente con una
      proyección snap simple.
    - El split TRAIN/TEST de las features históricas NO afecta este paso:
      la predicción de llegada opera sobre trayectorias completas de viajes
      válidos, sin usar las features históricas como inputs directos.
    - `trip_valid=True` se filtra aquí; el paso 10 no modifica los datos
      anteriores.
"""

from __future__ import annotations

# Ejecutable directo: registra func/ en el path para hallar config (que a su vez registra lib/, steps/ y audits/).
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl

from config import PATHS, INT_ROUTE, ANALYSIS_RANGE


# ──────────────────────────────────────────────────────────────────
# Rutas de salida para este paso
# ──────────────────────────────────────────────────────────────────

def _safe_label(group_value) -> str:
    """Etiqueta segura para nombre de carpeta a partir del valor de grupo.
    Para direction 0/1 → 'dir0'/'dir1'; para una leg 'T0->T1' → 'legT0_T1'.
    Idéntico al de step_11 para que ambos pasos compartan carpeta por grupo."""
    if isinstance(group_value, (int, np.integer)) or (
            isinstance(group_value, str) and group_value.isdigit()):
        return f"dir{group_value}"
    s = str(group_value).replace("->", "_").replace(" ", "")
    s = "".join(c if (c.isalnum() or c in "_-") else "_" for c in s)
    return f"leg{s}"


#: Régimen histórico del paper original (P95 de longitud, 99% de cobertura).
#: Cualquier otra combinación define una TAREA DISTINTA, no la misma remedida.
_LEGACY_REGIME = (0.95, 0.99)


def regime_tag(n_route_quantile: float = 0.95,
               min_coverage: float = 0.99) -> str:
    """
    Sufijo que identifica el régimen de la grid en rutas de disco.

    ⚠ Sin esto el caché es una trampa: `df_errors_<leg>_<rango>.parquet` no
    depende de `n_route_quantile` ni de `min_coverage`, así que re-correr con
    q=0.05 cargaría en silencio los errores de la corrida q=0.95 y el cambio de
    régimen no se notaría en ningún número. Se devuelve cadena vacía para el
    régimen histórico, de modo que los artefactos ya existentes siguen siendo
    válidos y solo las combinaciones nuevas estrenan carpeta.
    """
    if (round(float(n_route_quantile), 4),
            round(float(min_coverage), 4)) == _LEGACY_REGIME:
        return ""
    q = int(round(float(n_route_quantile) * 100))
    c = int(round(float(min_coverage) * 100))
    return f"_q{q:02d}c{c:03d}"


def _arrival_dir(group_value, n_route_quantile: float = 0.95,
                 min_coverage: float = 0.99) -> Path:
    """Directorio de outputs para un grupo concreto (direction o leg).

    Anidado bajo ``ANALYSIS_RANGE`` (que ya incluye la etiqueta de terminales,
    p. ej. ``…_t2``) para que dashboards/PDF por leg de configuraciones distintas
    NO se pisen — los nombres de leg (legT0_T0) colisionan entre variantes.
    El régimen de la grid añade su propio sufijo (ver ``regime_tag``)."""
    base = (PATHS.interim_route_dir / "arrival"
            / (ANALYSIS_RANGE + regime_tag(n_route_quantile, min_coverage))
            / _safe_label(group_value))
    base.mkdir(parents=True, exist_ok=True)
    return base


def _summary_path(n_route_quantile: float = 0.95,
                  min_coverage: float = 0.99) -> Path:
    """Parquet consolidado con RMSE por algoritmo × dirección."""
    p = PATHS.interim_route_dir / "arrival"
    p.mkdir(parents=True, exist_ok=True)
    tag = regime_tag(n_route_quantile, min_coverage)
    return p / f"arrival_summary_{ANALYSIS_RANGE}{tag}.parquet"


def _report_path(n_route_quantile: float = 0.95,
                 min_coverage: float = 0.99) -> Path:
    """HTML comparativo de ambas direcciones."""
    tag = regime_tag(n_route_quantile, min_coverage)
    return (PATHS.interim_route_dir / "arrival"
            / f"arrival_report_{ANALYSIS_RANGE}{tag}.html")


# ──────────────────────────────────────────────────────────────────
# Guardia de importación del paquete arrival_prediction
# ──────────────────────────────────────────────────────────────────

def _import_arrival_prediction():
    """
    Intenta importar el paquete `arrival_prediction`.

    Busca en:
        1. sys.path ya configurado (el más común si se pone en src/).
        2. ../src/  relativo al directorio de este archivo.
        3. src/     en la raíz del proyecto (PATHS.interim_route_dir.parent.parent.parent).

    Raises
    ------
    ImportError con instrucciones claras si no se encuentra.
    """
    try:
        import arrival_prediction as _ap
        return _ap
    except ModuleNotFoundError:
        pass

    # Intentar añadir rutas candidatas
    candidates = [
        Path(__file__).resolve().parent.parent / "src",
        Path(__file__).resolve().parent / "src",
        PATHS.interim_route_dir.parents[2] / "src",
    ]
    for c in candidates:
        if c.is_dir() and (c / "arrival_prediction").is_dir():
            sys.path.insert(0, str(c))
            try:
                import arrival_prediction as _ap
                return _ap
            except ModuleNotFoundError:
                sys.path.pop(0)

    raise ImportError(
        "\n"
        "  ❌  No se encontró el paquete `arrival_prediction`.\n"
        "\n"
        "  Solución: copia la carpeta `arrival_prediction/` (con su __init__.py)\n"
        "  dentro de `src/` en la raíz del proyecto, de modo que la estructura\n"
        "  sea:\n"
        "\n"
        "      src/\n"
        "      └── arrival_prediction/\n"
        "          ├── __init__.py\n"
        "          ├── dataset_builder.py\n"
        "          ├── predictors.py\n"
        "          ├── cross_validation.py\n"
        "          ├── plots.py\n"
        "          └── run_experiment.py\n"
        "\n"
        "  Y añade al notebook / script llamador:\n"
        "      import sys; sys.path.insert(0, '../src')\n"
    )


# ──────────────────────────────────────────────────────────────────
# Validación de columnas mínimas requeridas
# ──────────────────────────────────────────────────────────────────

_REQUIRED_COLS = {
    "trip_id":    "identificador de viaje (string o int)",
    "trip_valid": "flag de viaje válido (bool)",
    "time":       "timestamp del registro",
    "cum_dist_m": "distancia acumulada desde el inicio del viaje (metros)",
}


def _validate_df_full(df: pl.DataFrame, verbose: bool, group_col: str = "direction") -> None:
    """Verifica que df_full tiene las columnas mínimas; imprime aviso si falta algo.

    `group_col` ('direction' o 'leg') también debe existir en el df."""
    required = {**_REQUIRED_COLS,
                group_col: f"columna de agrupación ('{group_col}': direction 0/1 o leg terminal→terminal)"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        detail = "\n".join(f"    • {c}: {required[c]}" for c in missing)
        raise ValueError(
            f"\n  ❌  df_full no tiene las columnas requeridas para el paso 10:\n"
            f"{detail}\n\n"
            f"  Asegúrate de correr el paso 07 completo (con camino1_interpolation)\n"
            f"  antes de llamar al paso 10."
        )
    if verbose:
        n_valid = df.filter(pl.col("trip_valid")).height
        n_total = df.height
        print(f"  df_full: {n_total:,} registros · {n_valid:,} en viajes válidos")


# ──────────────────────────────────────────────────────────────────
# Función interna por dirección
# ──────────────────────────────────────────────────────────────────

def _run_one_direction(
    df_full:         pl.DataFrame,
    direction:       int,
    ap,                              # módulo arrival_prediction importado
    grid_step_m:     int   = 100,
    n_folds:         int   = 5,
    K:               int   = 10,
    b:               float = 1.0,
    n_route_quantile:float = 0.95,
    min_coverage:    float = 0.99,
    include_grid_500:bool  = True,
    force_rerun:     bool  = False,
    verbose:         bool  = True,
    group_col:       str   = "direction",
    group_value:     object = None,
) -> dict:
    """
    Corre el pipeline de predicción para UN grupo (direction 0/1 o leg
    terminal→terminal del método nuevo de viajes).

    Cachea el parquet de errores; si ya existe y force_rerun=False, lo carga.

    Returns
    -------
    dict con claves:  T, l_grid, df_errors, df_errors_500, metadata, paths
    """
    gval      = group_value if group_value is not None else direction
    out_dir   = _arrival_dir(gval, n_route_quantile, min_coverage)
    # El nombre del parquet lleva TAMBIÉN el régimen: la carpeta ya lo separa,
    # pero el archivo viaja solo (se copia a route_reports, se descarga del
    # server) y sin el sufijo dos regímenes serían indistinguibles.
    _rtag     = regime_tag(n_route_quantile, min_coverage)
    cache_pq  = (out_dir /
                 f"df_errors_{_safe_label(gval)}_{ANALYSIS_RANGE}{_rtag}.parquet")

    # Cache hit solo si TAMBIÉN existe el dashboard: el pipeline escribe
    # df_errors ANTES de las figuras (etapa D), así que un crash en medio
    # dejaría un caché "válido" sin visualizaciones que nunca se regenerarían.
    _dash_ok = (out_dir / "dashboard.html").is_file()
    if cache_pq.is_file() and not _dash_ok and not force_rerun and verbose:
        print(f"  ℹ {group_col} {gval}: df_errors cacheado pero falta "
              f"dashboard.html → se recalcula el grupo completo")
    if cache_pq.is_file() and _dash_ok and not force_rerun:
        if verbose:
            print(f"  ✔ {group_col} {gval}: cargando errores de cache → {cache_pq.name}")
        df_errors = pl.read_parquet(cache_pq)
        # Metadata mínima desde el parquet cacheado
        return {
            "T":             None,
            "l_grid":        None,
            "df_errors":     df_errors,
            "df_errors_500": None,
            "metadata":      {"cached": True, "direction": direction, "group_value": gval},
            "paths":         {"parquet": str(cache_pq)},
        }

    if verbose:
        print(f"\n  {'─'*55}")
        print(f"  {group_col} {gval} — corriendo pipeline arrival_prediction …")
        print(f"  {'─'*55}")

    # Guard: con menos viajes que folds, run_cross_validation deja folds vacíos
    # y los predictores revientan con un ValueError de numpy poco claro (zero-size
    # array). En vez de tumbar el step completo, se salta el grupo con un aviso.
    n_trips = (
        df_full.filter(pl.col(group_col) == gval, pl.col("trip_valid"))
        ["trip_id"].n_unique()
    )
    if n_trips < n_folds:
        if verbose:
            print(f"  ⚠ {group_col} {gval}: solo {n_trips} viaje(s) válido(s) < "
                  f"n_folds={n_folds} → SALTADO (CV no es posible con tan pocos datos).")
        return {
            "T": None, "l_grid": None, "df_errors": None, "df_errors_500": None,
            "metadata": {"skipped": True, "reason": "insufficient_trips",
                         "n_trips": n_trips, "n_folds": n_folds,
                         "direction": direction, "group_value": gval},
            "paths": {},
        }

    # El guard de arriba usa el conteo CRUDO de trip_id; el dataset_builder
    # descarta además los viajes "demasiado cortos" (poca cobertura de la
    # polilínea), por lo que el M efectivo puede caer por debajo de n_folds
    # incluso pasando el guard crudo (visto con legs reales: 7 viajes crudos
    # → 1 tras filtrar). InsufficientDataError es la señal definitiva.
    from arrival_prediction.cross_validation import InsufficientDataError
    try:
        result = ap.build_arrival_prediction_pipeline(
            df_full,
            direction        = direction,
            group_col        = group_col,
            group_value       = gval,
            grid_step_m      = grid_step_m,
            n_folds          = n_folds,
            K                = K,
            b                = b,
            n_route_quantile = n_route_quantile,
            min_coverage     = min_coverage,
            include_grid_500 = include_grid_500,
            output_dir       = str(out_dir) + "/",
            verbose          = verbose,
        )
    except InsufficientDataError as e:
        if verbose:
            print(f"  ⚠ {group_col} {gval}: SALTADO tras filtrar viajes cortos → {e}")
        return {
            "T": None, "l_grid": None, "df_errors": None, "df_errors_500": None,
            "metadata": {"skipped": True, "reason": "insufficient_data_post_filter",
                         "error": str(e), "direction": direction, "group_value": gval},
            "paths": {},
        }

    # Renombrar parquet de errores con sufijo de análisis para trazabilidad
    default_pq = Path(result["paths"].get("parquet", ""))
    if default_pq.is_file() and default_pq != cache_pq:
        import shutil
        shutil.copy(default_pq, cache_pq)
        if verbose:
            print(f"  ✔ Errores guardados en: {cache_pq.name}")

    result["paths"]["parquet_step10"] = str(cache_pq)
    return result


# ──────────────────────────────────────────────────────────────────
# Resumen consolidado (ambas direcciones)
# ──────────────────────────────────────────────────────────────────

def _build_summary(
    results: Dict[object, dict],
    ap,
    verbose: bool,
    n_route_quantile: float = 0.95,
    min_coverage: float = 0.99,
) -> pl.DataFrame:
    """
    Concatena los df_errors de todos los grupos (directions o legs) y
    calcula RMSE promedio por (algoritmo, grupo). Persiste en
    arrival_summary.parquet.
    """
    from arrival_prediction.cross_validation import summarize_rmse_by_location

    frames = []
    for group_value, res in results.items():
        df_err = res.get("df_errors")
        if df_err is None or df_err.is_empty():
            continue
        rmse = summarize_rmse_by_location(df_err)
        rmse = rmse.with_columns(pl.lit(str(group_value)).alias("group"))
        frames.append(rmse)

    if not frames:
        return pl.DataFrame()

    df_summary = pl.concat(frames, how="diagonal_relaxed")

    # Guardar
    sp = _summary_path(n_route_quantile, min_coverage)
    from _lib_data import save_df
    save_df(df_summary, sp)
    if verbose:
        print(f"\n  ✔ Resumen consolidado guardado: {sp.name}")

    # Tabla bonita en consola
    if verbose:
        print("\n  ── RMSE promedio por algoritmo y grupo (leg/dirección) ──")
        agg = (
            df_summary
            .group_by(["group", "algorithm"])
            .agg(pl.col("rmse_mean").mean().alias("rmse_avg"))
            .sort(["group", "rmse_avg"])
        )
        for row in agg.iter_rows(named=True):
            print(f"    {row['group']:10s}  {row['algorithm']:7s}"
                  f"  →  RMSE prom = {row['rmse_avg']:6.1f} s")

    return df_summary


# ──────────────────────────────────────────────────────────────────
# Métricas completas por grupo (Tabla del paper: % APTA + IC por folds)
# ──────────────────────────────────────────────────────────────────

# Banda operacional APTA [-1, +5] min — MISMA convención que el paso 12
# (integrity_audit): dev = t_true − t_pred; + = el bus llega DESPUÉS de lo
# predicho. NO invertir (ver docs/historial_paper/11_pre_resultados_auditoria).
_APTA_EARLY_S = -60.0
_APTA_LATE_S  = 300.0


def _build_metrics(results: Dict[object, dict], verbose: bool,
                   n_route_quantile: float = 0.95,
                   min_coverage: float = 0.99) -> pl.DataFrame:
    """Métricas por (grupo, algoritmo): n, MAE, RMSE, MedAE, P90AE y % APTA
    (banda dev ∈ [-60, +300] s), globales Y por fold (fold=-1 = global; los
    folds dan el IC empírico del CV). Persiste arrival_metrics_<rango>.parquet.
    Complementa a arrival_summary (RMSE por posición) para la tabla del panel."""
    frames = []
    for group_value, res in results.items():
        df_err = res.get("df_errors")
        if df_err is None or df_err.is_empty():
            continue
        if not {"t_true", "t_pred"} <= set(df_err.columns):
            continue
        base = df_err.with_columns(
            (pl.col("t_pred") - pl.col("t_true")).alias("_res"),
            (pl.col("t_true") - pl.col("t_pred")).alias("_dev"),
        )

        def _agg(d: pl.DataFrame, fold: int) -> pl.DataFrame:
            return (d.group_by("algorithm").agg(
                        pl.len().alias("n"),
                        pl.col("_res").abs().mean().round(2).alias("mae_s"),
                        (pl.col("_res") ** 2).mean().sqrt().round(2).alias("rmse_s"),
                        pl.col("_res").abs().median().round(2).alias("medae_s"),
                        pl.col("_res").abs().quantile(0.9).round(2).alias("p90ae_s"),
                        pl.col("_dev").is_between(_APTA_EARLY_S, _APTA_LATE_S)
                          .mean().mul(100).round(2).alias("pct_apta_band"))
                    .with_columns(pl.lit(fold).alias("fold"),
                                  pl.lit(str(group_value)).alias("group")))

        frames.append(_agg(base, -1))                       # global del grupo
        if "fold" in base.columns:
            for (f,), d in base.group_by(["fold"]):
                frames.append(_agg(d, int(f)))

    if not frames:
        return pl.DataFrame()
    df_metrics = pl.concat(frames, how="diagonal_relaxed").sort(
        ["group", "algorithm", "fold"])

    mp = (PATHS.interim_route_dir / "arrival"
          / (f"arrival_metrics_{ANALYSIS_RANGE}"
             f"{regime_tag(n_route_quantile, min_coverage)}.parquet"))
    mp.parent.mkdir(parents=True, exist_ok=True)
    from _lib_data import save_df
    save_df(df_metrics, mp)
    if verbose:
        print(f"  ✔ Métricas por grupo (MAE/RMSE/%APTA, global+folds): {mp.name}")
    return df_metrics


def _dump_group_status(results: Dict[object, dict], verbose: bool,
                       n_route_quantile: float = 0.95,
                       min_coverage: float = 0.99) -> None:
    """Persiste el estado por grupo (ok / cached / skipped+razón) en
    arrival_skips_<rango>.json — diagnóstico de por qué una leg/ruta no tiene
    predicciones (dato para el funnel del paper, claim '26 rutas')."""
    import json
    status = {}
    for gv, res in results.items():
        meta = res.get("metadata") or {}
        if meta.get("skipped"):
            status[str(gv)] = {"status": "skipped",
                               "reason": meta.get("reason"),
                               "n_trips": meta.get("n_trips"),
                               "error": meta.get("error")}
        elif meta.get("cached"):
            status[str(gv)] = {"status": "cached"}
        else:
            status[str(gv)] = {"status": "ok"}
    sp = (PATHS.interim_route_dir / "arrival"
          / (f"arrival_skips_{ANALYSIS_RANGE}"
             f"{regime_tag(n_route_quantile, min_coverage)}.json"))
    sp.parent.mkdir(parents=True, exist_ok=True)
    sp.write_text(json.dumps(status, indent=2, ensure_ascii=False),
                  encoding="utf-8")
    if verbose:
        n_skip = sum(1 for v in status.values() if v["status"] == "skipped")
        print(f"  ✔ Estado por grupo → {sp.name} ({n_skip} saltado(s))")


# ──────────────────────────────────────────────────────────────────
# Reporte HTML comparativo
# ──────────────────────────────────────────────────────────────────

def _build_html_report(
    df_summary: pl.DataFrame,
    results:    Dict[object, dict],
    verbose:    bool,
    n_route_quantile: float = 0.95,
    min_coverage: float = 0.99,
) -> str:
    """Genera un HTML ligero con la tabla de RMSE y links a los dashboards."""

    if df_summary.is_empty():
        return ""

    agg = (
        df_summary
        .group_by(["group", "algorithm"])
        .agg(pl.col("rmse_mean").mean().alias("rmse_avg"))
        .sort(["group", "rmse_avg"])
    )

    rows_html = ""
    for row in agg.iter_rows(named=True):
        rows_html += (
            f"<tr><td>{row['group']}</td>"
            f"<td>{row['algorithm']}</td>"
            f"<td>{row['rmse_avg']:.1f}</td></tr>\n"
        )

    # Links a dashboards individuales
    dash_links = ""
    for d, res in results.items():
        dash = res.get("paths", {}).get("dashboard", "")
        if dash:
            dash_links += (
                f'<p><a href="{dash}" target="_blank">'
                f'Dashboard {d}</a></p>\n'
            )

    html = f"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>Arrival Prediction · Ruta {INT_ROUTE}</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, sans-serif;
         background: #1b1b1b; color: #ddd; margin: 0; padding: 24px; }}
  h1   {{ color: #fff; }}
  h2   {{ color: #aaa; font-size: 1rem; font-weight: normal; margin-top: 0; }}
  table {{ border-collapse: collapse; min-width: 400px; margin-top: 16px; }}
  th, td {{ padding: 8px 14px; text-align: left;
            border-bottom: 1px solid #333; font-size: 14px; }}
  th {{ color: #aaa; font-weight: normal; background: #222; }}
  tr:hover {{ background: #222; }}
  a {{ color: #4da6ff; }}
  .section {{ background: #262626; border-radius: 6px;
              padding: 20px; margin-bottom: 20px;
              border: 1px solid #333; }}
</style>
</head>
<body>
<h1>Predicción de tiempos de llegada · Ruta {INT_ROUTE}</h1>
<h2>Replicación del paper Dublin Bus 46A — Análisis {ANALYSIS_RANGE}</h2>

<div class="section">
  <h3 style="color:#fff;margin-top:0">RMSE promedio (s) por algoritmo y grupo (leg/dirección)</h3>
  <table>
    <thead>
      <tr><th>Grupo</th><th>Algoritmo</th><th>RMSE prom (s)</th></tr>
    </thead>
    <tbody>
{rows_html}
    </tbody>
  </table>
</div>

<div class="section">
  <h3 style="color:#fff;margin-top:0">Dashboards individuales</h3>
  {dash_links if dash_links else '<p>No generados (sin datos).</p>'}
</div>

</body>
</html>"""

    rp = _report_path(n_route_quantile, min_coverage)
    rp.write_text(html, encoding="utf-8")
    if verbose:
        print(f"  ✔ Reporte HTML: {rp.name}")
    return str(rp)


# ──────────────────────────────────────────────────────────────────
# Función pública principal
# ──────────────────────────────────────────────────────────────────

def run_arrival_prediction_step(
    df_full:          pl.DataFrame,
    directions:       list[int]   = [0, 1],
    # Método nuevo de viajes: agrupar por leg (terminal→terminal) en vez de
    # direction 0/1. Si `legs` se pasa, sustituye a `directions` y group_col="leg".
    legs:             Optional[List[str]] = None,
    group_col:        str         = "direction",
    grid_step_m:      int         = 100,
    n_folds:          int         = 5,
    K:                int         = 10,
    b:                float       = 1.0,
    n_route_quantile: float       = 0.95,
    min_coverage:     float       = 0.99,
    include_grid_500: bool        = True,
    force_rerun:      bool        = False,
    verbose:          bool        = True,
) -> Dict[str, object]:
    """
    Paso 10 completo: predicción de tiempos de llegada para la Ruta 1701.

    Parámetros
    ----------
    df_full : pl.DataFrame
        Salida del paso 07b. Debe contener las columnas:
        trip_id, direction (o leg), trip_valid, time, cum_dist_m.

    directions : list[int]
        Direcciones a procesar. Default [0, 1]. Ignorado si se pasa `legs`.

    legs : list[str] | None
        Legs (terminal→terminal) a procesar — método nuevo de viajes. Si se
        pasa, sustituye a `directions` y agrupa por la columna `leg`.

    group_col : str
        Columna de agrupación ('direction' o 'leg'). Se fuerza a 'leg' si
        `legs` no es None.

    grid_step_m : int
        Resolución de la cuadrícula de posiciones en metros (paper: 100 m).

    n_folds : int
        Número de folds para la validación cruzada (paper: 5).

    K : int
        Número de vecinos para KNN (paper: 10).

    b : float
        Ancho de banda para Kernel Regression (paper: 1.0).

    n_route_quantile : float
        Cuantil de cum_dist_m que define el final de la grid. 0.95 = régimen
        histórico del paper original, que retiene solo el ~5% de viajes MÁS
        LARGOS; 0.05 = soporte común (~95%), el régimen del estudio desde
        2026-07-24. Ver `docs/PLAN_REPRESENTATIVIDAD.md`.

    min_coverage : float
        Fracción de la grid que un viaje debe cubrir para entrar. 1.0 elimina la
        saturación de `np.interp` (que fabrica tiempo restante en el tramo no
        cubierto) y da paridad exacta de `n_pairs` con el LSTM.

    include_grid_500 : bool
        Si True, repite el CV con grid de 500 m (replica Figura 6 del paper).

    force_rerun : bool
        Si True, ignora caches existentes y recalcula todo.

    verbose : bool

    Retorna
    -------
    dict con:
        "dir{n}" / "leg{...}" : resultado del paquete arrival_prediction por grupo
        "df_summary"          : RMSE consolidado (pl.DataFrame)
        "summary_path"        : ruta al parquet del resumen
        "report_path"         : ruta al HTML comparativo
    """
    # Grupos a procesar: legs (método nuevo) o direcciones (clásico).
    if legs is not None:
        group_col = "leg"
        group_values: List = list(legs)
    else:
        group_values = list(directions)

    if verbose:
        print("=" * 60)
        print("  PASO 10 · Predicción de tiempos de llegada")
        print(f"           Ruta {INT_ROUTE} — {ANALYSIS_RANGE}")
        print(f"           Agrupación: {group_col} = {group_values}")
        print(f"           Grid: {grid_step_m} m · {n_folds}-fold CV")
        _rt = regime_tag(n_route_quantile, min_coverage)
        print(f"           Régimen: q={n_route_quantile} cov={min_coverage}"
              f"{'  (histórico)' if not _rt else '  → sufijo ' + _rt}")
        print("=" * 60)

    # 1. Importar el paquete
    ap = _import_arrival_prediction()

    # 2. Validar columnas
    _validate_df_full(df_full, verbose=verbose, group_col=group_col)

    # 3. Correr por grupo (leg o direction)
    results: Dict[object, dict] = {}
    for gv in group_values:
        results[gv] = _run_one_direction(
            df_full          = df_full,
            direction        = gv if group_col == "direction" else 0,
            group_col        = group_col,
            group_value      = gv,
            ap               = ap,
            grid_step_m      = grid_step_m,
            n_folds          = n_folds,
            K                = K,
            b                = b,
            n_route_quantile = n_route_quantile,
            min_coverage     = min_coverage,
            include_grid_500 = include_grid_500,
            force_rerun      = force_rerun,
            verbose          = verbose,
        )

    # 4. Resumen consolidado
    df_summary = _build_summary(results, ap, verbose=verbose,
                                n_route_quantile=n_route_quantile,
                                min_coverage=min_coverage)

    # 4b. Métricas completas por grupo (%APTA + per-fold) + estado por grupo
    df_metrics = _build_metrics(results, verbose=verbose,
                                n_route_quantile=n_route_quantile,
                                min_coverage=min_coverage)
    _dump_group_status(results, verbose=verbose,
                       n_route_quantile=n_route_quantile,
                       min_coverage=min_coverage)

    # 5. Reporte HTML comparativo
    report_path = _build_html_report(df_summary, results, verbose=verbose,
                                     n_route_quantile=n_route_quantile,
                                     min_coverage=min_coverage)

    if verbose:
        print(f"\n  ✅ PASO 10 completo → {PATHS.interim_route_dir / 'arrival'}\n")

    out = {_safe_label(gv): results[gv] for gv in group_values}
    out["df_summary"]   = df_summary
    out["df_metrics"]   = df_metrics
    out["summary_path"] = str(_summary_path(n_route_quantile, min_coverage))
    out["report_path"]  = report_path
    return out


# ──────────────────────────────────────────────────────────────────
# CLI / smoke-test
# ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from _lib_data import load_df
    print("Cargando df_full desde cache …")
    _df = load_df(PATHS.df_full)   # valida que el cache no esté vacío (0 filas / 0 bytes)
    result = run_arrival_prediction_step(_df, directions=[0, 1], verbose=True)
    print("Listo. Resumen:")
    print(result["df_summary"])
