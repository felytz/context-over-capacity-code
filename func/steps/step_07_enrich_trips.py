"""
step_07_enrich_trips.py
=======================
PASO 7 · Enriquecimiento de viajes + feature engineering completo.

Este paso fusiona los antiguos pasos 06 (enrich_trips) y 10
(features_extended) en un solo paso. Tiene dos sub-tareas:

    7a. Enriquecimiento de viajes
    -----------------------------
    Aplica `_lib_features.enrich_trip_segments` sobre los viajes válidos
    (trip_valid=True) para añadir las columnas de referencia espacial:

        - next_stop    (str)   : código completo "101-FQAE"
        - m_next_stop  (float) : metros restantes hasta esa parada
        - s_next_stop  (float) : segundos hasta el cambio de next_stop

    7b. Feature engineering (camino1 + camino2 + ext)
    -------------------------------------------------
    Aplica `_lib_features.compute_all_features` sobre el df enriquecido y
    se queda con DF_FULL_COLUMNS: las columnas de 7a más las features que
    leen los modelos.

v3 (mayo 2026) — CACHÉ POR MES
-------------------------------
Cuando la ventana de análisis cubre varios meses, este paso se tarda
horas. Esta versión rompe el procesamiento en *shards mensuales* y
guarda cada uno en disco al terminar:

    PATHS.df_dir / "monthly" / "df_rec_enrich" / "2024-01.parquet"
                                                 "2024-02.parquet" ...
    PATHS.df_dir / "monthly" / "df_rec_full"    / "2024-01.parquet"
                                                 "2024-02.parquet" ...

Beneficios:
    • Si crashea (OOM, kernel, lo que sea) en el mes 5 de 8, los meses
      1-4 sobreviven y la próxima corrida solo procesa el 5.
    • Permite procesar mes por mes manualmente (ideal para iterar la
      lógica de features sin pagar el costo total cada vez).
    • La función al cargar detecta qué meses ya están listos y los salta.

Asignación de mes por trip
--------------------------
Cada `trip_id` se asigna al mes de su primer ping (`time.min()`). Los
trips que cruzan medianoche del último día del mes (raro: <0.1%) van
al mes donde inician. Esto preserva la coherencia: un trip no se parte.

Cross-shard caveat
------------------
Las features que dependen de contexto inter-bus dentro de una ventana
de tiempo (`headway_ahead`, `headway_behind`, `segment_count_ahead`)
pierden visibilidad en la frontera entre meses. El zone afectado es
~`headway_max_seconds` (default 1800 s = 30 min) al inicio y al final
de cada mes. Sobre ~720 h por mes, esto representa <0.07% de las
filas. Acepto el tradeoff: la alternativa (procesar fronteras con
slices solapados) cancela la ganancia operativa.

Compatibilidad
--------------
`run_enrich_step` devuelve los dos DataFrames combinados y además los
escribe a `PATHS.df_enriched` y `PATHS.df_full`, que es lo que leen el
frame, el XGBoost y el LSTM.

La tabla histórica (hist_p50, …), el modo sample y las features del
camino 2 sin uso se quitaron; están en el tag `features-completas`.

Usage
-----
    from step_07_enrich_trips import run_enrich_step

    # Modo normal: detecta meses cacheados y solo procesa los faltantes
    df_enriched, df_full = run_enrich_step(
        df_trips, df_segments, verbose=True,
    )

    # Forzar re-procesamiento de TODOS los meses
    df_enriched, df_full = run_enrich_step(
        df_trips, df_segments, force=True, verbose=True,
    )

    # Forzar re-procesamiento de un mes específico
    df_enriched, df_full = run_enrich_step(
        df_trips, df_segments,
        force_months=["2024-03"], verbose=True,
    )

    # Sólo ver qué hay cacheado, sin procesar
    from step_07_enrich_trips import list_processed_months
    print(list_processed_months())
"""

from __future__ import annotations

# Ejecutable directo: registra func/ en el path para hallar config (que a su vez registra lib/ y steps/).
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import polars as pl

from config import (
    PATHS, INT_ROUTE_STOP, HEADWAY_IDEAL_MIN,
    SPEED_WINDOW_SECONDS, HEADWAY_MAX_SECONDS,
    SPEED_ZERO_THRESHOLD,
)


# Columnas mínimas que enrich_trip_segments espera.
COLS_TO_KEEP = [
    "bus_code", "speed", "angle", "time",
    "snap_lat", "snap_lon",
    "trip_id", "direction", "trip_valid",
]

# Columnas de df_full: las de df_enriched (7a) + las features que leen los
# modelos — el frame (cum_dist_m), el XGBoost contextual (exo_features._SRC_COLS)
# y el tokenizador por parada del LSTM (link_travel_time). compute_all_features
# produce más (diagnóstico de la interpolación, camino 2 sin uso); se descartan
# aquí y al combinar shards, así que un caché viejo de 37 columnas también sale
# recortado. Las columnas quitadas están en el tag `features-completas`.
DF_FULL_COLUMNS = [
    "bus_code", "speed", "angle", "time", "latitude", "longitude",
    "trip_id", "direction", "trip_valid",
    "next_stop", "m_next_stop", "s_next_stop",
    "cum_dist_m", "link_travel_time", "dwell_time",
    "hour_of_day", "day_of_week",
    "speed_recent_5min", "headway_ahead", "headway_behind", "segment_count_ahead",
]


def _keep_full_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Recorta a DF_FULL_COLUMNS conservando el orden original de df."""
    return df.select([c for c in df.columns if c in DF_FULL_COLUMNS])


# ──────────────────────────────────────────────────────────────────
# HELPERS DE CACHÉ POR MES
# ──────────────────────────────────────────────────────────────────

def _monthly_dir(subname: str) -> Path:
    """
    Directorio que contiene los shards mensuales de un sub-paso.

    Estructura:
        PATHS.df_dir / "monthly" / <subname> / YYYY-MM.parquet
    """
    d = PATHS.df_dir / "monthly" / subname
    d.mkdir(parents=True, exist_ok=True)
    return d


def _weekly_ckpt_dir() -> Path:
    """Directorio de checkpoints semanales para df_rec_full."""
    d = PATHS.df_dir / "weekly" / "df_rec_full"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _process_month_weekly(
    df_month: pl.DataFrame,
    month_key: str,
    df_segments: "pl.DataFrame",
    compute_headways: bool,
    verbose: bool,
) -> pl.DataFrame:
    """
    Procesa un mes en chunks semanales (7 días) con checkpoints en disco.

    Cada semana se guarda en:
        PATHS.df_dir / "weekly" / "df_rec_full" / "{month_key}-W{n}.parquet"

    Si un checkpoint ya existe, se carga y se salta el cómputo.
    Al final se concatenan todas las semanas, se ordena por tiempo y se
    eliminan los checkpoints (el shard mensual actúa como caché definitivo).

    Overlap
    -------
    Para que headway_ahead/behind (ventana ±30 min) y speed_recent (ventana
    5 min) sean correctos en los bordes de semana, cada chunk incluye un
    solapamiento de `overlap_s` segundos a cada lado. Las filas del overlap
    se recortan antes de guardar el checkpoint.
    """
    from _lib_features import compute_all_features

    ckpt_dir = _weekly_ckpt_dir()
    # Overlap = max_headway cuando se computa headway; de lo contrario
    # basta con la ventana de speed_recent (300 s).
    overlap_s = int(HEADWAY_MAX_SECONDS) if compute_headways else SPEED_WINDOW_SECONDS

    # Determinar semanas presentes en el mes
    t_min = df_month["time"].min()
    t_max = df_month["time"].max()
    week_start = t_min.replace(hour=0, minute=0, second=0, microsecond=0)

    weeks: list[tuple[datetime, datetime]] = []
    while week_start <= t_max:
        weeks.append((week_start, week_start + timedelta(days=7)))
        week_start += timedelta(days=7)

    week_dfs: list[pl.DataFrame] = []
    for w_i, (w_start, w_end) in enumerate(weeks):
        ckpt_path = ckpt_dir / f"{month_key}-W{w_i}.parquet"

        if ckpt_path.is_file():
            if verbose:
                print(f"    [W{w_i}] ✔ checkpoint existe — cargando {ckpt_path.name}")
            week_dfs.append(_keep_full_columns(pl.read_parquet(ckpt_path)))
            continue

        overlap = timedelta(seconds=overlap_s)
        df_chunk = df_month.filter(
            (pl.col("time") >= pl.lit(w_start - overlap)) &
            (pl.col("time") <  pl.lit(w_end   + overlap))
        )
        if df_chunk.is_empty():
            continue

        if verbose:
            print(f"    [W{w_i}] ⏳ {w_start.date()} → {w_end.date()} "
                  f"({df_chunk.height:,} filas con overlap) …")

        df_chunk_full = compute_all_features(
            df_chunk,
            df_segments_route    = df_segments,
            speed_window_seconds = SPEED_WINDOW_SECONDS,
            headway_max_seconds  = HEADWAY_MAX_SECONDS,
            speed_zero_threshold = SPEED_ZERO_THRESHOLD,
            compute_headways     = compute_headways,
            verbose              = verbose,
        )

        # Recortar overlap: conservar solo las filas del núcleo de la semana
        df_week = _keep_full_columns(df_chunk_full.filter(
            (pl.col("time") >= pl.lit(w_start)) &
            (pl.col("time") <  pl.lit(w_end))
        ))

        from _lib_data import save_df
        save_df(df_week, ckpt_path, allow_empty=True)  # checkpoint transitorio (partición)
        if verbose:
            print(f"    [W{w_i}] ✔ {df_week.height:,} filas → {ckpt_path.name}")
        week_dfs.append(df_week)

    if not week_dfs:
        return pl.DataFrame()

    df_month_full = pl.concat(week_dfs, how="diagonal_relaxed").sort("time")

    # Eliminar checkpoints: el shard mensual es el caché definitivo
    for w_i in range(len(weeks)):
        p = ckpt_dir / f"{month_key}-W{w_i}.parquet"
        if p.is_file():
            p.unlink()

    return df_month_full


def _month_key_from_time(ts) -> str:
    """Convierte un Datetime polars / pyobj a 'YYYY-MM'."""
    return f"{ts.year:04d}-{ts.month:02d}"


def _trip_to_month(df_trips: pl.DataFrame) -> pl.DataFrame:
    """
    Devuelve una tabla [trip_id, month_key] asignando cada trip al
    mes de su primer ping (`time.min()`).

    Solo considera trips con `trip_valid=True` para evitar gasto en
    los inválidos (que enrich_trip_segments ignoraría igual).
    """
    df = (
        df_trips
        .filter(pl.col("trip_valid") == True)
        .group_by("trip_id")
        .agg(pl.col("time").min().alias("_t0"))
        .with_columns(
            (pl.col("_t0").dt.year().cast(pl.Utf8) + "-" +
             pl.col("_t0").dt.month().cast(pl.Utf8).str.zfill(2))
            .alias("month_key")
        )
        .select("trip_id", "month_key")
    )
    return df


def _months_present_in_trips(df_trips: pl.DataFrame) -> List[str]:
    """Lista única de meses cubiertos por los trips válidos, ordenada."""
    map_ = _trip_to_month(df_trips)
    return sorted(map_["month_key"].unique().to_list())


def _load_shards(shard_dir: Path, verbose: bool = False) -> Optional[pl.DataFrame]:
    """Lee todos los .parquet de shard_dir y los concatena en orden de nombre."""
    files = sorted(shard_dir.glob("*.parquet"))
    if not files:
        return None
    parts = [pl.read_parquet(f) for f in files]
    df = pl.concat(parts, how="diagonal_relaxed")
    if verbose:
        n = sum(p.height for p in parts)
        print(f"  ✔ Cargados {len(files)} shards de {shard_dir.name}: "
              f"{n:,} filas")
    return df


def list_processed_months() -> dict:
    """
    Inspecciona qué shards mensuales existen ya en disco. Útil para
    decidir si correr `run_enrich_step` o no.

    Returns
    -------
    dict con claves:
        enriched : list[str]   meses en df_rec_enrich/
        full     : list[str]   meses en df_rec_full/
    """
    enr_dir  = _monthly_dir("df_rec_enrich")
    full_dir = _monthly_dir("df_rec_full")
    return {
        "enriched": sorted(p.stem for p in enr_dir.glob("*.parquet")),
        "full":     sorted(p.stem for p in full_dir.glob("*.parquet")),
    }


# ──────────────────────────────────────────────────────────────────
# 7a · ENRIQUECIMIENTO DE VIAJES (por mes)
# ──────────────────────────────────────────────────────────────────

def _filter_valid_trips(df_trips: pl.DataFrame) -> pl.DataFrame:
    """
    Filtra los registros que pertenecen a viajes válidos y renombra a
    las columnas esperadas por enrich_trip_segments.
    """
    return (
        df_trips
        .filter(pl.col("trip_valid") == True)
        .select(COLS_TO_KEEP)
        .rename({"snap_lat": "latitude", "snap_lon": "longitude"})
    )


def enrich_trips(
    df_trips: pl.DataFrame,
    df_segments: pl.DataFrame,
    force: bool = False,
    force_months: Optional[Sequence[str]] = None,
    verbose: bool = False,
) -> pl.DataFrame:
    """
    Genera y cachea `df_enriched` con estrategia por mes.

    Cada mes se procesa independientemente (enrich_trip_segments opera
    por-ping sin contexto cross-row, así que es seguro). Al terminar
    todos los meses, se concatenan y se escribe el combinado a
    `PATHS.df_enriched` para compatibilidad downstream.

    Parameters
    ----------
    df_trips : pl.DataFrame
        Salida del paso 6 (con trip_valid).
    df_segments : pl.DataFrame
        Geometría de segmentos del paso 1.
    force : bool
        Si True, ignora caché y re-procesa todos los meses presentes en
        df_trips.
    force_months : list[str] | None
        Lista de meses ("YYYY-MM") a re-procesar selectivamente.
        Combinable con force=False.
    verbose : bool

    Returns
    -------
    pl.DataFrame combinado (todos los meses) ordenado por time.
    """
    PATHS.ensure_dirs()
    shard_dir = _monthly_dir("df_rec_enrich")
    force_set = set(force_months or [])

    # Mapa trip → mes; sin esto no podemos rebanar el df por mes.
    map_trip_month = _trip_to_month(df_trips)
    months_target  = sorted(map_trip_month["month_key"].unique().to_list())

    if not months_target:
        if verbose:
            print("  ⚠ No hay trips válidos en df_trips; df_enriched vacío.")
        empty = pl.DataFrame()
        if PATHS.df_enriched.is_file():
            PATHS.df_enriched.unlink()
        return empty

    if verbose:
        print(f"  Meses presentes en df_trips: {months_target}")

    df_in_all = _filter_valid_trips(df_trips).join(
        map_trip_month, on="trip_id", how="left"
    )

    from _lib_features import enrich_trip_segments

    # Procesar cada mes faltante.
    for month_key in months_target:
        shard_path = shard_dir / f"{month_key}.parquet"
        if (shard_path.is_file()
                and not force
                and month_key not in force_set):
            if verbose:
                print(f"  [7a · {month_key}] ✔ shard ya existe — skip")
            continue

        if verbose:
            reason = ("force" if force else
                      "force_months" if month_key in force_set else
                      "nuevo")
            print(f"  [7a · {month_key}] ⏳ procesando ({reason}) …")

        df_month_in = df_in_all.filter(
            pl.col("month_key") == month_key
        ).drop("month_key")

        if df_month_in.is_empty():
            if verbose:
                print(f"  [7a · {month_key}] ⚠ slice vacío; se omite.")
            continue

        df_month_out = enrich_trip_segments(
            df_month_in,
            df_segments,
            route_prefix = str(INT_ROUTE_STOP),
            verbose      = verbose,
        )
        from _lib_data import save_df
        save_df(df_month_out, shard_path, allow_empty=True)  # shard mensual (partición)
        if verbose:
            print(f"  [7a · {month_key}] ✔ {df_month_out.height:,} filas → "
                  f"{shard_path.name}")

    # Cargar y concatenar todos los shards (incluye meses pre-existentes
    # que no estaban en este df_trips si los hay).
    df_combined = _load_shards(shard_dir, verbose=verbose)
    if df_combined is None:
        if verbose:
            print("  ⚠ No se generó ningún shard; df_enriched vacío.")
        return pl.DataFrame()

    df_combined = df_combined.sort("time")

    # Escribir combinado a la ruta legacy para que paso 08+ funcione igual.
    from _lib_data import save_df
    save_df(df_combined, PATHS.df_enriched)
    if verbose:
        print(f"  ✔ df_enriched combinado: {PATHS.df_enriched} "
              f"({df_combined.height:,} filas)")
    return df_combined


# ──────────────────────────────────────────────────────────────────
# 7b · FEATURE ENGINEERING (por mes)
# ──────────────────────────────────────────────────────────────────

def compute_full_features(
    df_enriched: pl.DataFrame,
    df_segments: pl.DataFrame,
    force: bool = False,
    force_months: Optional[Sequence[str]] = None,
    compute_headways: bool = True,
    verbose: bool = False,
) -> pl.DataFrame:
    """
    Aplica `compute_all_features` por mes con caché por shard y deja solo
    DF_FULL_COLUMNS.

    El orden de operaciones por mes es:
        1. Slice de df_enriched al mes.
        2. compute_all_features por semana (con solapamiento).
        3. Recorte a DF_FULL_COLUMNS y escritura del shard a disco.

    Parameters
    ----------
    df_enriched : pl.DataFrame
        Salida combinada del paso 7a.
    df_segments : pl.DataFrame
    force : bool
        Si True, ignora el caché de TODOS los shards mensuales de df_full.
    force_months : list[str] | None
        Re-procesa selectivamente.

    Returns
    -------
    df_full combinado (todos los meses) ordenado por time.
    """
    PATHS.ensure_dirs()
    shard_dir = _monthly_dir("df_rec_full")
    force_set = set(force_months or [])

    if df_enriched is None or df_enriched.is_empty():
        if verbose:
            print("  ⚠ df_enriched vacío; no hay nada que computar.")
        return pl.DataFrame()

    # ── 1. Determinar meses a procesar ──
    df_enriched_w_month = df_enriched.with_columns(
        (pl.col("time").dt.year().cast(pl.Utf8) + "-" +
         pl.col("time").dt.month().cast(pl.Utf8).str.zfill(2))
        .alias("_month_key")
    )
    months_target = sorted(df_enriched_w_month["_month_key"].unique().to_list())

    if verbose:
        print(f"\n  Meses presentes en df_enriched: {months_target}")

    # ── 2. Bucle por mes ──
    for month_key in months_target:
        shard_path = shard_dir / f"{month_key}.parquet"
        if (shard_path.is_file()
                and not force
                and month_key not in force_set):
            if verbose:
                print(f"  [7b · {month_key}] ✔ shard ya existe — skip")
            continue

        if verbose:
            reason = ("force" if force else
                      "force_months" if month_key in force_set else
                      "nuevo")
            hw_note = "" if compute_headways else "  [headways=False]"
            print(f"  [7b · {month_key}] ⏳ procesando ({reason}){hw_note} …")

        df_month = df_enriched_w_month.filter(
            pl.col("_month_key") == month_key
        ).drop("_month_key")

        if df_month.is_empty():
            continue

        # Limpiar checkpoints semanales obsoletos si se fuerza re-proceso
        if force or month_key in force_set:
            ckpt_dir = _weekly_ckpt_dir()
            for p in ckpt_dir.glob(f"{month_key}-W*.parquet"):
                p.unlink()

        # Procesamiento por semana con checkpoints intermedios
        df_month_full = _process_month_weekly(
            df_month,
            month_key        = month_key,
            df_segments      = df_segments,
            compute_headways = compute_headways,
            verbose          = verbose,
        )

        if df_month_full.is_empty():
            continue

        from _lib_data import save_df
        save_df(df_month_full, shard_path, allow_empty=True)  # shard mensual (partición)
        if verbose:
            print(f"  [7b · {month_key}] ✔ {df_month_full.height:,} filas × "
                  f"{len(df_month_full.columns)} cols → {shard_path.name}")

    # ── 3. Combinar shards ──
    df_combined = _load_shards(shard_dir, verbose=verbose)
    if df_combined is None:
        if verbose:
            print("  ⚠ Sin shards df_full; nada que combinar.")
        return pl.DataFrame()

    # El recorte también aquí: un shard de antes del recorte trae 37 columnas.
    df_combined = _keep_full_columns(df_combined).sort("time")
    from _lib_data import save_df
    save_df(df_combined, PATHS.df_full)
    if verbose:
        print(f"  ✔ df_full combinado: {PATHS.df_full} "
              f"({df_combined.height:,} filas, "
              f"{len(df_combined.columns)} cols)")

    return df_combined


# ──────────────────────────────────────────────────────────────────
# Función pública
# ──────────────────────────────────────────────────────────────────

def run_enrich_step(
    df_trips: pl.DataFrame,
    df_segments: pl.DataFrame,
    force: bool = False,
    force_months: Optional[Sequence[str]] = None,
    compute_headways: bool = True,
    verbose: bool = False,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Ejecuta PASO 7 completo con caché por mes: enrich (7a) + features (7b).

    Procesa cada mes presente en df_trips y guarda un shard `.parquet`
    por mes. En corridas posteriores, detecta qué meses ya están en
    disco y solo procesa los faltantes.

    Parameters
    ----------
    df_trips : pl.DataFrame
        Salida del PASO 6 (con trip_valid).
    df_segments : pl.DataFrame
        Salida del PASO 1.
    force : bool
        Re-procesa TODOS los meses.
    force_months : list[str] | None
        Lista de meses "YYYY-MM" a re-procesar.
    verbose : bool

    Returns
    -------
    df_enriched : pl.DataFrame
    df_full     : pl.DataFrame
    """
    if verbose:
        print("=" * 60)
        print("  PASO 7 · Enrich + feature engineering (caché por mes)")
        print("=" * 60)

    # Reporte previo de estado
    if verbose:
        state = list_processed_months()
        print(f"  Estado del caché:")
        print(f"    enriched ({len(state['enriched'])}): {state['enriched']}")
        print(f"    full     ({len(state['full'])}): {state['full']}")

    # 7a · Enriquecimiento por mes
    if verbose:
        print("\n  [7a] Enriquecimiento de viajes por mes …")
    df_enriched = enrich_trips(
        df_trips, df_segments,
        force        = force,
        force_months = force_months,
        verbose      = verbose,
    )

    # 7b · Features por mes
    if verbose:
        print("\n  [7b] Feature engineering por mes (camino1 + camino2 + ext) …")
    df_full = compute_full_features(
        df_enriched,
        df_segments,
        force             = force,
        force_months      = force_months,
        compute_headways  = compute_headways,
        verbose           = verbose,
    )

    if verbose:
        print(f"\n  → df_enriched: {df_enriched.height:,} filas, "
              f"{len(df_enriched.columns)} cols")
        print(f"  → df_full    : {df_full.height:,} filas, "
              f"{len(df_full.columns)} cols")
        print(f"  → Headway ideal usado: {HEADWAY_IDEAL_MIN} min\n")

    return df_enriched, df_full


if __name__ == "__main__":
    # Smoke test
    df_trips = pl.read_parquet(PATHS.df_trips)
    df_segs  = pl.read_parquet(PATHS.df_segments)
    run_enrich_step(df_trips, df_segs, verbose=True)
