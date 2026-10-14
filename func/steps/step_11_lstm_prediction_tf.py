"""
step_11_lstm_prediction_tf.py
================================
PASO 11 · Predicción de tiempos de llegada con Sequential LSTM
        (adaptación de Sun & Kim 2021) — versión TensorFlow.

Aplica el paquete `lstm_prediction_tf` sobre el `df_full` producido por
el paso 07 para extender el experimento del paso 10 con un quinto
algoritmo basado en deep learning, usando TensorFlow / Keras 3.

Diferencia clave con el paso 10
-------------------------------
El paso 10 implementa los 4 algoritmos clásicos del paper Dublin Bus 46A
(delay, linreg, KNN, kernel). Este paso 11 los compara contra un modelo
basado en LSTM compartida + self-attention con tokens espacio-temporales
conjuntos (`(stop_code, link_travel_time)`).

Sub-tareas
----------
    11a  Dirección 0 — entrena SequentialLSTM, evalúa con 5-fold CV.
    11b  Dirección 1 — ídem dirección de regreso.
    11c  Reporte comparativo LSTM vs. 4 baselines (si df_errors del paso 10
         está disponible en disco).

Reproducibilidad
----------------
El parámetro `seed` controla las particiones de fold y debe coincidir con
el `seed` del paso 10 para que la comparación sea válida. Por defecto
ambos pasos usan seed=42.

Inputs
------
    df_full : pl.DataFrame
        Salida del paso 07 — incluye `next_stop`, `link_travel_time`,
        `trip_id`, `direction`, `trip_valid`, `time`.

Outputs
-------
    Por dirección, en ``data/02_interim/routes/{route}/lstm_tf/dir{d}/``:
        - df_errors.parquet         — errores compatibles con paso 10
        - vocab.json                — vocabulario stop_code → int
        - training_history.json     — curvas de loss/lr por fold
        - fold_assignments.json     — mapeo trip_id → fold (clave para paridad)
        - metadata.json             — config completa (incluye tf_version)
        - dashboard.html            — dashboard interactivo
        - static/report.pdf         — reporte estático

    Comparativo entre direcciones y vs. paso 10:
        - lstm_tf_summary_{range}.parquet  — RMSE por algoritmo × dirección
        - lstm_tf_report_{range}.html      — HTML comparativo
        - lstm_tf_vs_baselines_{range}.parquet — df_errors combinado; SUPERSET
          (filas y columnas) de todos los df_errors por-leg del LSTM, con o sin
          baseline del paso 10 → el empaquetado puede omitir los per-leg

NOTA: Este step coexiste con `step_11_lstm_prediction.py` (versión
PyTorch). Outputs se guardan en `lstm_tf/` (no `lstm/`) para evitar
sobrescritura. Si solo tienes acceso a TensorFlow, este es el único step
que debes usar.

Usage
-----
    from step_11_lstm_prediction_tf import run_lstm_prediction_step

    result = run_lstm_prediction_step(df_full, verbose=True)
    # result["dir0"], result["dir1"], result["summary_path"], result["report_path"]

Dependencia del paquete
-----------------------
El paquete `lstm_prediction_tf` debe estar en `sys.path`, idealmente en
`src/lstm_prediction_tf/`. Si no se encuentra, el paso falla con un
ImportError explícito con instrucciones.

Notas de integración
--------------------
- A diferencia del paso 10 (~minutos en CPU), el paso 11 puede tomar de
  decenas de minutos a horas dependiendo del tamaño del dataset y del
  hardware. En GPU el entrenamiento baja a varios minutos.
- Si el paso 10 ya generó un `df_errors_*.parquet` para esta dirección,
  se carga automáticamente y se concatena para el comparativo de la
  subtarea 11c.
- El paso es cacheable: si los outputs principales existen y
  `force_rerun=False`, no recalcula.
"""

from __future__ import annotations

# Ejecutable directo: registra func/ en el path para hallar config (que a su vez registra lib/, steps/ y audits/).
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import polars as pl

from config import PATHS, INT_ROUTE, ANALYSIS_RANGE


# ──────────────────────────────────────────────────────────────────
# Rutas de salida para este paso
# ──────────────────────────────────────────────────────────────────

def _safe_label(group_value) -> str:
    """Etiqueta segura para nombre de carpeta a partir del valor de grupo.
    Para direction 0/1 → 'dir0'/'dir1'; para una leg 'T0->T1' → 'legT0_T1'."""
    if isinstance(group_value, (int, np.integer)) or (
            isinstance(group_value, str) and group_value.isdigit()):
        return f"dir{group_value}"
    s = str(group_value).replace("->", "_").replace(" ", "")
    s = "".join(c if (c.isalnum() or c in "_-") else "_" for c in s)
    return f"leg{s}"


def _lstm_dir(group_value) -> Path:
    """Directorio de outputs para un grupo concreto (direction o leg).

    Anidado bajo ``ANALYSIS_RANGE`` (incluye la etiqueta de terminales) para que
    dashboards/PDF por leg de configuraciones distintas NO se pisen — igual que
    ``step_10._arrival_dir``."""
    base = PATHS.interim_route_dir / "lstm_tf" / ANALYSIS_RANGE / _safe_label(group_value)
    base.mkdir(parents=True, exist_ok=True)
    return base


def _summary_path() -> Path:
    """Parquet consolidado con RMSE por algoritmo × dirección."""
    p = PATHS.interim_route_dir / "lstm_tf"
    p.mkdir(parents=True, exist_ok=True)
    return p / f"lstm_tf_summary_{ANALYSIS_RANGE}.parquet"


def _report_path() -> Path:
    """HTML comparativo de ambas direcciones y vs. baselines."""
    return PATHS.interim_route_dir / "lstm_tf" / f"lstm_tf_report_{ANALYSIS_RANGE}.html"


def _combined_errors_path() -> Path:
    """Parquet con df_errors combinado (LSTM + baselines del paso 10)."""
    return PATHS.interim_route_dir / "lstm_tf" / f"lstm_tf_vs_baselines_{ANALYSIS_RANGE}.parquet"


def _lstm_cache_path(group_value) -> Path:
    """Cache RANGE-AWARE del df_errors LSTM de un grupo (misma convención que
    el paso 10). El `df_errors.parquet` plano que escribe el paquete NO lleva
    la ventana en la clave: usarlo como cache hacía que al cambiar de ventana
    se reutilizara silenciosamente el modelo de otra (detectado: la corrida
    2024_01-08 de la 1602 reutilizó el LSTM de enero, metadata n_trips=464)."""
    return _lstm_dir(group_value) / (
        f"df_errors_{_safe_label(group_value)}_{ANALYSIS_RANGE}.parquet")


def _baseline_errors_path(group_value) -> Path:
    """Path del df_errors del paso 10 para un grupo, PREFERIENDO la copia
    range-aware (`df_errors_<grupo>_<rango>.parquet`). El `df_errors.parquet`
    plano es simplemente el de la última corrida, sea cual sea su ventana."""
    d = PATHS.interim_route_dir / "arrival" / ANALYSIS_RANGE / _safe_label(group_value)
    tagged = d / f"df_errors_{_safe_label(group_value)}_{ANALYSIS_RANGE}.parquet"
    return tagged if tagged.exists() else d / "df_errors.parquet"


# ──────────────────────────────────────────────────────────────────
# Guardia de importación del paquete lstm_prediction
# ──────────────────────────────────────────────────────────────────

def _import_lstm_prediction():
    """
    Intenta importar el paquete `lstm_prediction_tf`.

    Busca en:
        1. sys.path ya configurado (el más común si se pone en src/).
        2. ../src/  relativo al directorio de este archivo.
        3. src/     en la raíz del proyecto.

    Raises
    ------
    ImportError con instrucciones claras si no se encuentra.
    """
    try:
        import lstm_prediction_tf as _lp
        return _lp
    except ModuleNotFoundError:
        pass

    candidates = [
        Path(__file__).resolve().parent.parent / "src",
        Path(__file__).resolve().parent / "src",
        PATHS.interim_route_dir.parents[2] / "src",
    ]

    # Diagnóstico: registrar QUÉ se intentó y qué se encontró en cada ruta.
    diag_lines = []
    for i, c in enumerate(candidates, 1):
        exists_dir = c.is_dir()
        pkg        = c / "lstm_prediction_tf"
        has_pkg    = pkg.is_dir()
        has_init   = (pkg / "__init__.py").is_file() if has_pkg else False
        estado = (
            "OK · aquí está el paquete" if (has_pkg and has_init)
            else "falta __init__.py" if has_pkg
            else "no hay carpeta lstm_prediction_tf/" if exists_dir
            else "la carpeta src/ no existe"
        )
        diag_lines.append(f"     [{i}] {c}\n         → {estado}")

        if has_pkg and has_init:
            sys.path.insert(0, str(c))
            try:
                import lstm_prediction_tf as _lp
                return _lp
            except ModuleNotFoundError:
                sys.path.pop(0)

    raise ImportError(
        "\n"
        "  ❌  No se encontró el paquete `lstm_prediction_tf`.\n"
        "\n"
        "  Contexto de búsqueda:\n"
        f"     • CWD actual            : {Path.cwd()}\n"
        f"     • Ubicación de step_11  : {Path(__file__).resolve()}\n"
        f"     • interim_route_dir     : {PATHS.interim_route_dir}\n"
        "\n"
        "  Rutas src/ que se probaron (ninguna sirvió):\n"
        + "\n".join(diag_lines) + "\n"
        "\n"
        "  Primeras entradas de sys.path:\n"
        + "\n".join(f"     - {p}" for p in sys.path[:8]) + "\n"
        "\n"
        "  Solución: coloca la carpeta `lstm_prediction_tf/` (con su __init__.py)\n"
        "  en CUALQUIERA de las rutas src/ listadas arriba, con esta estructura:\n"
        "\n"
        "      src/\n"
        "      └── lstm_prediction_tf/\n"
        "          ├── __init__.py\n"
        "          ├── dataset_builder.py\n"
        "          ├── vocab.py\n"
        "          ├── models.py\n"
        "          ├── data_loader.py\n"
        "          ├── trainer.py\n"
        "          ├── cross_validation.py\n"
        "          ├── plots.py\n"
        "          └── run_experiment.py\n"
        "\n"
        "  Alternativa sin mover nada: exporta PYTHONPATH apuntando a la carpeta\n"
        "  que CONTIENE a lstm_prediction_tf/, p. ej.:\n"
        "      export PYTHONPATH=/ruta/a/esa/carpeta:$PYTHONPATH\n"
        "\n"
        "  Adicional: este paquete requiere `tensorflow >= 2.16`.\n"
    )


# ──────────────────────────────────────────────────────────────────
# Validación de columnas mínimas requeridas
# ──────────────────────────────────────────────────────────────────

_REQUIRED_COLS = {
    "trip_id":          "identificador de viaje (string o int)",
    "trip_valid":       "flag de viaje válido (bool)",
    "time":             "timestamp del registro",
}
# Columnas extra según el modo de tokenización:
_TOKENIZE_COLS = {
    "stop": {
        "next_stop":        "código de la próxima parada (string)",
        "link_travel_time": "tiempo de viaje desde la parada anterior (segundos, float)",
    },
    "grid": {
        "cum_dist_m":       "distancia acumulada del viaje (m) — la produce §3 / paso 07b",
    },
}


def _validate_df_full(df: pl.DataFrame, verbose: bool, group_col: str = "direction",
                      tokenize_by: str = "grid") -> None:
    """Verifica que df_full tiene las columnas mínimas para LSTM.

    En modo ``grid`` (el del método nuevo de viajes) NO se exigen next_stop ni
    link_travel_time: la tokenización por rejilla sólo usa cum_dist_m + time.
    """
    required = {**_REQUIRED_COLS,
                **_TOKENIZE_COLS.get(tokenize_by, _TOKENIZE_COLS["stop"]),
                group_col: f"columna de agrupación ('{group_col}': direction 0/1 o leg terminal→terminal)"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        detail = "\n".join(f"    • {c}: {required[c]}" for c in missing)
        raise ValueError(
            f"\n  ❌  df_full no tiene las columnas requeridas para el paso 11 "
            f"(tokenize_by={tokenize_by!r}):\n"
            f"{detail}\n\n"
            f"  En modo 'stop' corre el paso 07c (next_stop + link_travel_time);\n"
            f"  en modo 'grid' basta con cum_dist_m (lo produce §3 del método nuevo)."
        )
    if verbose:
        n_valid = df.filter(pl.col("trip_valid")).height
        n_total = df.height
        print(f"  df_full: {n_total:,} registros · {n_valid:,} en viajes válidos")
        if "link_travel_time" in df.columns:
            n_with_link = df.filter(pl.col("link_travel_time").is_not_null()).height
            print(f"           {n_with_link:,} con link_travel_time no nulo")


# ──────────────────────────────────────────────────────────────────
# Carga del df_errors del paso 10 (opcional)
# ──────────────────────────────────────────────────────────────────

def _load_baseline_errors(group_value, verbose: bool) -> Optional[pl.DataFrame]:
    """
    Si existe el df_errors del paso 10 para este grupo (direction/leg), lo carga.
    Esto permite que el paso 11c (comparativo) corra automáticamente.

    Retorna None si no existe, sin levantar error.
    """
    p = _baseline_errors_path(group_value)
    if not p.exists():
        if verbose:
            print(f"  ⚠ df_errors del paso 10 no encontrado para {group_value}.")
            print(f"     Buscado en: {p}")
            print(f"     El comparativo LSTM vs baselines NO se generará para este grupo.")
        return None

    df = pl.read_parquet(p)
    if verbose:
        algos = sorted(df["algorithm"].unique().to_list())
        print(f"  ✔ Baseline cargado: {df.height:,} predicciones, "
              f"algoritmos: {algos}")
    return df


# ──────────────────────────────────────────────────────────────────
# Función interna por dirección
# ──────────────────────────────────────────────────────────────────

def _run_one_direction(
    df_full:         pl.DataFrame,
    direction:       int,
    model_type:      str,
    embed_dim:       int,
    hidden_dim:      int,
    n_epochs:        int,
    batch_size:      int,
    n_folds:         int,
    patience:        int,
    lr:              float,
    dropout:         float,
    seed:            int,
    force_rerun:     bool,
    verbose:         bool,
    tokenize_by:     str = "grid",
    grid_step_m:     int = 100,
    l_strategy:      str = "uniform",
    val_l_strategy:  Optional[str] = "all",
    group_col:       str = "direction",
    group_value:     object = None,
) -> Dict:
    """
    Ejecuta el experimento LSTM para un grupo dado (direction 0/1 o leg
    terminal→terminal del método nuevo de viajes).

    Si ya existen los outputs principales en disco y force_rerun=False,
    los carga sin recalcular. En otro caso, llama al orquestador del
    paquete.
    """
    lp = _import_lstm_prediction()

    gval = group_value if group_value is not None else direction
    out_dir = _lstm_dir(gval)
    parquet_path = out_dir / "df_errors.parquet"
    cache_path   = _lstm_cache_path(gval)

    # ── Cache hit? — SOLO la copia range-aware cuenta como cache, y además
    # deben existir el dashboard y la metadata: un crash entre el parquet y
    # las figuras dejaría un caché "válido" sin reporte que nunca se
    # regeneraría (los checkpoints por fold hacen barato el re-run).
    _arts_ok = ((out_dir / "dashboard.html").is_file()
                and (out_dir / "metadata.json").is_file())
    if cache_path.exists() and not _arts_ok and not force_rerun and verbose:
        print(f"  ℹ {group_col} {gval}: df_errors cacheado pero faltan "
              f"dashboard/metadata → se re-corre el grupo (reanuda de checkpoints)")
    if cache_path.exists() and _arts_ok and not force_rerun:
        if verbose:
            print(f"\n  ── {group_col} {gval} (cached) ──")
            print(f"     Cargando df_errors desde {cache_path.name}")
        df_errors = pl.read_parquet(cache_path)

        # Cargar baseline si existe → combined
        df_baseline = _load_baseline_errors(gval, verbose=False)
        df_combined: Optional[pl.DataFrame] = None
        if df_baseline is not None:
            df_combined = pl.concat(
                [df_errors, df_baseline], how="diagonal_relaxed"
            )

        return {
            "direction":          direction,
            "group_value":        gval,
            "df_errors":          df_errors,
            "df_errors_combined": df_combined,
            "paths": {
                "parquet":          cache_path,
                "dashboard":        out_dir / "dashboard.html",
                "pdf_report":       out_dir / "static" / "report.pdf",
                "vocab":            out_dir / "vocab.json",
                "training_history": out_dir / "training_history.json",
                "fold_assignments": out_dir / "fold_assignments.json",
                "metadata":         out_dir / "metadata.json",
            },
            "cached": True,
        }

    # ── Cache miss: correr el pipeline completo ──────────────────
    if verbose:
        print(f"\n  ── {group_col} {gval} ──")
        print(f"     Salida: {out_dir}")

    # force_rerun = desde cero: sin esto, los checkpoints por fold de una
    # corrida interrumpida se REANUDARÍAN aunque se pidió re-entrenar.
    if force_rerun:
        for p in out_dir.glob("cv_ckpt_fold*"):
            p.unlink(missing_ok=True)

    # Guard: con menos viajes que folds el CV no es posible (mismo problema que
    # en step_10) y el entrenamiento revienta con errores poco claros más abajo
    # en la pila. Se salta el grupo con un aviso en vez de tumbar el step.
    n_trips = (
        df_full.filter(pl.col(group_col) == gval, pl.col("trip_valid"))
        ["trip_id"].n_unique()
    )
    if n_trips < n_folds:
        if verbose:
            print(f"  ⚠ {group_col} {gval}: solo {n_trips} viaje(s) válido(s) < "
                  f"n_folds={n_folds} → SALTADO (CV no es posible con tan pocos datos).")
        return {
            "direction": direction, "group_value": gval,
            "df_errors": None, "df_errors_combined": None,
            "paths": {}, "cached": False, "skipped": True,
        }

    df_baseline = _load_baseline_errors(gval, verbose=verbose)

    # El guard de arriba usa el conteo CRUDO de trip_id; el dataset_builder del
    # paquete (tokenización + filtros) puede dejar menos viajes utilizables que
    # eso (mismo problema visto en step_10). InsufficientDataError es la señal
    # definitiva — se captura para saltar el grupo en vez de tumbar el run.
    from lstm_prediction_tf.cross_validation import InsufficientDataError
    try:
        result = lp.build_lstm_prediction_pipeline(
            df_full            = df_full,
            direction          = direction,
            group_col          = group_col,
            group_value        = gval,
            model_type         = model_type,
            embed_dim          = embed_dim,
            hidden_dim         = hidden_dim,
            dropout            = dropout,
            n_epochs           = n_epochs,
            batch_size         = batch_size,
            lr                 = lr,
            patience           = patience,
            n_folds            = n_folds,
            # Robustez: "grid" tokeniza por tramos de distancia fija (como las
            # baselines del paso 10), sin depender de la ubicación de paradas.
            tokenize_by        = tokenize_by,
            grid_step_m        = grid_step_m,
            # Memoria: "uniform" en train acota las muestras a ≤30 cortes/viaje
            # (en vez de una por parada → O(n²)). Eval queda en "all" para no
            # alterar el RMSE por posición ni la comparación con el paso 10.
            l_strategy         = l_strategy,
            val_l_strategy     = val_l_strategy,
            output_dir         = out_dir,
            seed               = seed,
            df_errors_baseline = df_baseline,
            verbose            = verbose,
        )
    except InsufficientDataError as e:
        if verbose:
            print(f"  ⚠ {group_col} {gval}: SALTADO tras filtrar viajes cortos → {e}")
        return {
            "direction": direction, "group_value": gval,
            "df_errors": None, "df_errors_combined": None,
            "paths": {}, "cached": False, "skipped": True,
        }
    result["direction"]   = direction
    result["group_value"] = gval
    result["cached"]      = False

    # Copia range-aware del df_errors: es la que valida el cache-hit de arriba.
    # (El df_errors.parquet plano del paquete queda como "última corrida".)
    df_err = result.get("df_errors")
    if df_err is not None and not df_err.is_empty():
        df_err.write_parquet(cache_path)
    elif parquet_path.exists():
        import shutil
        shutil.copy2(parquet_path, cache_path)
    return result


# ──────────────────────────────────────────────────────────────────
# Resumen consolidado (ambas direcciones)
# ──────────────────────────────────────────────────────────────────

def _build_summary(
    results: Dict[int, dict],
    verbose: bool,
) -> pl.DataFrame:
    """
    Construye una tabla resumida (algorithm × direction × l_idx) con
    RMSE_mean ± RMSE_std a partir de los df_errors de cada dirección.

    Incluye TANTO LSTM como baselines (si están cargados en df_errors_combined).
    """
    lp = _import_lstm_prediction()
    frames: List[pl.DataFrame] = []

    for d, res in results.items():
        # Preferir el combined si está; si no, solo LSTM
        df = res.get("df_errors_combined")
        if df is None:
            df = res.get("df_errors")
        if df is None or df.is_empty():
            continue
        summary_d = lp.summarize_rmse_by_location(df).with_columns(
            pl.lit(str(d)).alias("group"),
        )
        frames.append(summary_d)

    if not frames:
        if verbose:
            print("  ⚠ Sin resultados; no se genera summary.")
        return pl.DataFrame()

    df_summary = pl.concat(frames, how="diagonal_relaxed")

    sp = _summary_path()
    from _lib_data import save_df
    save_df(df_summary, sp)
    if verbose:
        print(f"\n  ✔ Resumen consolidado guardado: {sp.name}")

    if verbose:
        print("\n  ── RMSE promedio por algoritmo y grupo (leg/direction) ──")
        agg = (
            df_summary
            .group_by(["group", "algorithm"])
            .agg(pl.col("rmse_mean").mean().alias("rmse_avg"))
            .sort(["group", "rmse_avg"])
        )
        for row in agg.iter_rows(named=True):
            marker = "★" if str(row['algorithm']).endswith("_lstm") else " "
            print(f"  {marker} {row['group']:10s}  "
                  f"{row['algorithm']:22s}"
                  f"  →  RMSE prom = {row['rmse_avg']:6.1f} s")

    return df_summary


# ──────────────────────────────────────────────────────────────────
# Reporte HTML comparativo
# ──────────────────────────────────────────────────────────────────

def _build_html_report(
    df_summary: pl.DataFrame,
    results:    Dict[int, dict],
    verbose:    bool,
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
        is_lstm = str(row['algorithm']).endswith("_lstm")
        row_class = ' class="lstm-row"' if is_lstm else ""
        marker = "★ " if is_lstm else ""
        rows_html += (
            f"<tr{row_class}><td>{row['group']}</td>"
            f"<td>{marker}{row['algorithm']}</td>"
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
<title>LSTM Prediction · Ruta {INT_ROUTE}</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, sans-serif;
         background: #1b1b1b; color: #ddd; margin: 0; padding: 24px; }}
  h1   {{ color: #fff; }}
  h2   {{ color: #aaa; font-size: 1rem; font-weight: normal; margin-top: 0; }}
  table {{ border-collapse: collapse; min-width: 500px; margin-top: 16px; }}
  th, td {{ padding: 8px 14px; text-align: left;
            border-bottom: 1px solid #333; font-size: 14px; }}
  th {{ color: #aaa; font-weight: normal; background: #222; }}
  tr:hover {{ background: #222; }}
  tr.lstm-row {{ background: #1f3a4d; }}
  tr.lstm-row:hover {{ background: #264a63; }}
  a {{ color: #4da6ff; }}
  .section {{ background: #262626; border-radius: 6px;
              padding: 20px; margin-bottom: 20px;
              border: 1px solid #333; }}
  .note {{ font-size: 12px; color: #888; margin-top: 8px; }}
</style>
</head>
<body>
<h1>Predicción de tiempos de llegada con LSTM · Ruta {INT_ROUTE}</h1>
<h2>Adaptación de Sun &amp; Kim (2021) — Análisis {ANALYSIS_RANGE}</h2>

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
  <p class="note">★ = modelo basado en LSTM (este trabajo). Resto = baselines del paso 10.</p>
</div>

<div class="section">
  <h3 style="color:#fff;margin-top:0">Dashboards individuales</h3>
  {dash_links if dash_links else '<p>No generados (sin datos).</p>'}
</div>

</body>
</html>"""

    rp = _report_path()
    rp.write_text(html, encoding="utf-8")
    if verbose:
        print(f"  ✔ Reporte HTML: {rp.name}")
    return str(rp)


# ──────────────────────────────────────────────────────────────────
# Función pública principal
# ──────────────────────────────────────────────────────────────────

def run_lstm_prediction_step(
    df_full:      pl.DataFrame,
    directions:   List[int] = [0, 1],
    # Método nuevo de viajes: agrupar por leg (terminal→terminal) en vez de
    # direction 0/1. Si `legs` se pasa, sustituye a `directions` y group_col="leg".
    legs:         Optional[List[str]] = None,
    group_col:    str       = "direction",
    model_type:   str       = "sequential",
    # Hiperparámetros del modelo
    embed_dim:    int       = 64,
    hidden_dim:   int       = 128,
    dropout:      float     = 0.1,
    # Hiperparámetros del entrenamiento
    n_epochs:     int       = 50,
    batch_size:   int       = 64,
    lr:           float     = 1e-3,
    patience:     int       = 5,
    n_folds:      int       = 5,
    # Tokenización (robustez): "grid" = tramos fijos como baselines; "stop" = paradas
    tokenize_by:  str       = "grid",
    grid_step_m:  int       = 100,
    # Estrategia de cortes (memoria)
    l_strategy:   str             = "uniform",
    val_l_strategy: Optional[str] = "all",
    # Otros
    seed:         int       = 42,
    force_rerun:  bool      = False,
    verbose:      bool      = True,
) -> Dict:
    """
    Ejecuta el paso 11 completo: 11a + 11b + 11c.

    Parameters
    ----------
    df_full : pl.DataFrame
        Salida del paso 07 enriquecido (con next_stop, link_travel_time).
    directions : list[int], default=[0, 1]
        Direcciones a procesar.
    model_type : "sequential" | "hybrid", default="sequential"
        Arquitectura del modelo. Sequential = ganador del paper Sun & Kim.

    embed_dim, hidden_dim, dropout : hiperparámetros del modelo.
        Defaults razonables para Ruta 101 (~99 paradas). Para datasets más
        grandes considera embed_dim=128, hidden_dim=256.

    n_epochs, batch_size, lr, patience : hiperparámetros del entrenamiento.

    n_folds : int, default=5
        Mantén 5 para paridad con el paso 10.

    seed : int, default=42
        Debe coincidir con el seed del paso 10 para que la comparación
        LSTM vs baselines sea válida.

    force_rerun : bool, default=False
        Si False y los outputs existen en disco, no recalcula.

    verbose : bool

    Returns
    -------
    dict con claves:
        - "dir0"            : dict con resultados de la dirección 0
        - "dir1"            : dict con resultados de la dirección 1
        - "summary_path"    : Path del parquet de resumen
        - "report_path"     : Path del HTML comparativo
        - "combined_path"   : Path del parquet df_errors combinado
    """
    # Grupos a procesar: legs (método nuevo) o direcciones (clásico).
    if legs is not None:
        group_col = "leg"
        group_values: List = list(legs)
    else:
        group_values = list(directions)

    if verbose:
        print("\n" + "═" * 60)
        print(f"  PASO 11 · LSTM Prediction (TF) Ruta {INT_ROUTE}  {ANALYSIS_RANGE}")
        print("═" * 60)
        print(f"  Modelo               : {model_type}")
        print(f"  Agrupación           : {group_col} = {group_values}")
        print(f"  N folds              : {n_folds}")
        print(f"  Embed/Hidden         : {embed_dim}/{hidden_dim}")
        print(f"  N epochs (máx)       : {n_epochs}")
        print(f"  Batch size           : {batch_size}")
        print(f"  Tokenización         : {tokenize_by}"
              + (f" ({grid_step_m}m)" if tokenize_by == "grid" else ""))
        print(f"  l_strategy           : {l_strategy} (train) / "
              f"{val_l_strategy or l_strategy} (val)")
        print(f"  Seed                 : {seed}")
        print(f"  Force rerun          : {force_rerun}")

    # ── Validación de columnas ────────────────────────────────────
    _validate_df_full(df_full, verbose=verbose, group_col=group_col, tokenize_by=tokenize_by)

    # ── Verificar que el paquete importa antes de hacer trabajo ───
    _import_lstm_prediction()

    # ── Ejecución por grupo (leg o direction) ─────────────────────
    results: Dict = {}
    for gv in group_values:
        results[gv] = _run_one_direction(
            df_full     = df_full,
            direction   = gv if group_col == "direction" else 0,
            group_col   = group_col,
            group_value = gv,
            model_type  = model_type,
            embed_dim   = embed_dim,
            hidden_dim  = hidden_dim,
            n_epochs    = n_epochs,
            batch_size  = batch_size,
            n_folds     = n_folds,
            patience    = patience,
            lr          = lr,
            dropout     = dropout,
            seed        = seed,
            force_rerun = force_rerun,
            verbose     = verbose,
            tokenize_by    = tokenize_by,
            grid_step_m    = grid_step_m,
            l_strategy     = l_strategy,
            val_l_strategy = val_l_strategy,
        )

    # ── Estado por grupo (ok/cached/skipped) → JSON de diagnóstico ─
    # Mismo espíritu que arrival_skips del paso 10: deja en disco POR QUÉ una
    # leg no tiene LSTM (funnel del paper; diagnóstico de rutas sin summary).
    try:
        import json as _json
        _status = {}
        for gv, res in results.items():
            if res.get("skipped"):
                _status[str(gv)] = {"status": "skipped",
                                    "reason": "insufficient_trips_or_data"}
            elif res.get("cached"):
                _status[str(gv)] = {"status": "cached"}
            else:
                _status[str(gv)] = {"status": "ok"}
        _sp = (PATHS.interim_route_dir / "lstm_tf"
               / f"lstm_tf_skips_{ANALYSIS_RANGE}.json")
        _sp.parent.mkdir(parents=True, exist_ok=True)
        _sp.write_text(_json.dumps(_status, indent=2, ensure_ascii=False),
                       encoding="utf-8")
        if verbose:
            _n = sum(1 for v in _status.values() if v["status"] == "skipped")
            print(f"  ✔ Estado por grupo → {_sp.name} ({_n} saltado(s))")
    except Exception as _e:
        print(f"  ⚠ no pude escribir lstm_tf_skips: {_e}")

    # ── Resumen consolidado ───────────────────────────────────────
    df_summary = _build_summary(results, verbose=verbose)

    # ── Persistir df_errors combinado (LSTM + baselines) ──────────
    # INVARIANTE para el empaquetado: el combinado debe contener las filas de
    # CADA grupo (leg/direction), tenga o no baseline del paso 10. Si un grupo
    # no tiene baseline, se usa su df_errors del LSTM solo. Como el concat es
    # `diagonal_relaxed` (une columnas), vs_baselines queda como SUPERSET —filas
    # y columnas, incluidas las `*_seg`— de todos los df_errors por-leg; así el
    # empaquetado puede omitir esos por-leg sin perder nada. (Antes, un grupo
    # sin baseline se excluía por completo del combinado.)
    combined_path: Optional[Path] = None
    combined_frames: List[pl.DataFrame] = []
    _grupos_solo_lstm: List = []
    for gv, res in results.items():
        dfc = res.get("df_errors_combined")
        if dfc is None:
            dfc = res.get("df_errors")          # sin baseline: al menos el LSTM
            if dfc is not None and not dfc.is_empty():
                _grupos_solo_lstm.append(gv)
        if dfc is not None and not dfc.is_empty():
            combined_frames.append(dfc)
    if combined_frames:
        df_combined_all = pl.concat(combined_frames, how="diagonal_relaxed")
        combined_path = _combined_errors_path()
        from _lib_data import save_df
        save_df(df_combined_all, combined_path)
        if verbose:
            print(f"\n  ✔ df_errors combinado (LSTM + baselines) guardado:")
            print(f"     {combined_path.name}  ({df_combined_all.height:,} filas)")
            if _grupos_solo_lstm:
                print(f"     (sin baseline del paso 10, solo LSTM: "
                      f"{_grupos_solo_lstm})")

    # ── Reporte HTML ──────────────────────────────────────────────
    report_str = _build_html_report(df_summary, results, verbose=verbose)

    if verbose:
        print("\n" + "═" * 60)
        print(f"  ✅ PASO 11 completo")
        print(f"     summary : {_summary_path().name}")
        print(f"     report  : {_report_path().name}")
        if combined_path:
            print(f"     combined: {combined_path.name}")
        print("═" * 60 + "\n")

    return {
        _safe_label(gv): results[gv] for gv in group_values
    } | {
        "results":       results,
        "summary_path":  str(_summary_path()) if not df_summary.is_empty() else None,
        "report_path":   report_str or None,
        "combined_path": str(combined_path) if combined_path else None,
    }


# ──────────────────────────────────────────────────────────────────
# CLI / smoke-test
# ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from _lib_data import load_df
    print("Cargando df_full desde cache …")
    _df = load_df(PATHS.df_full)   # valida que el cache no esté vacío (0 filas / 0 bytes)
    result = run_lstm_prediction_step(_df, directions=[0, 1], verbose=True)
    print("Listo. Reporte:")
    print(result["report_path"])