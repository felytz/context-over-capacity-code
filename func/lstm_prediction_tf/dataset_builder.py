"""
lstm_prediction.dataset_builder
=================================
Etapa A del pipeline de adaptación del Sequential LSTM de Sun & Kim (2021)
a la Ruta 101.

Transforma df_full (un ping GPS = una fila) en secuencias por viaje
(una parada visitada = un token), produciendo el input que consume el
modelo SequentialLSTM.

Diferencias clave con arrival_prediction.dataset_builder
--------------------------------------------------------
  - La unidad de análisis es la PARADA, no el punto de grid uniforme.
  - El modelo opera sobre tokens discretos (stop_code, link_travel_time),
    no sobre trayectorias interpoladas a grid métrica.
  - El target NO es la trayectoria completa; se construye por separado
    en cross_validation usando las mismas particiones (l_idx, horizonte)
    que el paso 10.

Referencia
----------
Sun, J. & Kim, J. (2021). "Joint prediction of next location and
travel time from urban vehicle trajectories using long short-term
memory neural networks". Transportation Research Part C, 128.
"""
from __future__ import annotations

import numpy as np
import polars as pl
from typing import Dict, Any, Iterable, List, Optional


def _grid_token(cell: int) -> str:
    """ID de token para una celda de rejilla (orden alfabético == numérico)."""
    return f"g{cell:05d}"


def build_sequence_dataset(
    df_full:             pl.DataFrame,
    direction:           int = 0,
    min_sequence_length: int = 5,
    max_sequence_length: int = 200,
    max_context_length:  Optional[int] = None,
    aggregation:         str = "first",
    context_features:    Optional[List[str]] = None,
    use_interp_time:     bool = True,
    tokenize_by:         str = "grid",
    grid_step_m:         int = 100,
    group_col:           str = "direction",
    group_value:         Any = None,
    with_calendar:       bool = False,
    trip_whitelist:      Optional[Iterable[Any]] = None,
    max_cells:           Optional[int] = None,
    verbose:             bool = False,
) -> Dict[str, Any]:
    """
    Construye el dataset de secuencias por viaje para el SequentialLSTM.

    Cada viaje válido se transforma en una secuencia ordenada de visitas
    a paradas. Para cada parada visitada se conserva:
      - el código de parada (stop_code)
      - el link travel time (segundos para llegar desde la parada anterior)
      - opcionalmente, features de contexto (hour_of_day, day_of_week, etc.)

    Parameters
    ----------
    df_full : pl.DataFrame
        Salida del paso 7 (con feature engineering). Columnas mínimas
        requeridas:
          - trip_id        (str|int)
          - direction      (int)
          - trip_valid     (bool)
          - time           (datetime)
          - next_stop      (str)
          - link_travel_time (float, segundos)   ← o s_next_stop_interp
          - cum_dist_m     (float, opcional pero recomendado para ordenar)

    direction : int, default=0
        Dirección a procesar (0 o 1). Cada dirección produce un dataset
        independiente porque las secuencias de paradas difieren.

    min_sequence_length : int, default=5
        Longitud mínima de la secuencia (en paradas) para incluir el viaje.
        Viajes más cortos se descartan. Coincide con el min del paper
        original (mantener secuencias estadísticamente significativas).

    max_sequence_length : int, default=200
        Tope superior de longitud por viaje. Si una secuencia excede,
        se trunca a las últimas `max_sequence_length` paradas. Default
        200 es holgado para Ruta 101 (~99 paradas/dirección).

    max_context_length : int | None, default=None
        Longitud máxima de la ventana de contexto al construir muestras
        (X, y) en cross_validation. None = ventana creciente (toda la
        historia del viaje hasta el step actual). Expuesto para análisis
        de sensibilidad. NO afecta este builder directamente, solo se
        propaga en metadata como hint para el trainer.

    aggregation : str, default="first"
        Cómo colapsar múltiples pings GPS en la misma parada:
          - "first"  : conserva el primer ping (=> link_travel_time de
                       entrada al segmento). Es el approach Sun & Kim.
          - "median" : mediana de tiempos. Más robusto al ruido AVL.

    context_features : list[str] | None, default=None
        Columnas adicionales de df_full a propagar en la secuencia.
        Por defecto None (modelo Sun & Kim puro). Si se pasa, cada
        token de la secuencia será un dict {"stop", "time", "context": {...}}.
        Útil para futuras variantes que incluyan hour_of_day, headway, etc.

    use_interp_time : bool, default=True
        Si True, usa `link_travel_time` calculado en el paso 7c (recomendado).
        Si False, usa `s_next_stop` cuantizado. El paper original usaría el
        equivalente a s_next_stop_interp; aquí preferimos link_travel_time
        por consistencia con el resto del pipeline.
        (Solo aplica con tokenize_by="stop".)

    tokenize_by : str, default="grid"
        Unidad de tokenización de la secuencia:
          - "grid"  : celdas de rejilla de distancia fija (cada grid_step_m
                      metros), interpolando el tiempo acumulado sobre la grid
                      igual que las baselines del paso 10. NO usa la ubicación
                      de paradas → robusto cuando los stops no son confiables,
                      y da paridad metodológica con el paso 10.
          - "stop"  : un token por parada visitada (next_stop). Enfoque
                      original Sun & Kim; sensible al ruido en ubicación.
        Con "grid", los params aggregation/context_features/use_interp_time
        se ignoran (solo aplican a "stop").

    grid_step_m : int, default=100
        Tamaño de celda en metros para tokenize_by="grid". 100 m = misma
        resolución que la grid del paso 10.

    verbose : bool, default=False

    Returns
    -------
    dict con claves:
      - 'sequences'    : list[dict]
            Cada dict contiene:
              "trip_id"  : str|int
              "stops"    : np.ndarray (n,) dtype=object  — códigos de parada
              "times"    : np.ndarray (n,) dtype=float32 — link travel times (s)
              "context"  : np.ndarray (n, k) o None      — context features
      - 'stop_counts'  : dict[str, int]   — frecuencia de cada parada
      - 'context_cols' : list[str] | None — columnas de contexto incluidas
      - 'metadata'     : dict de diagnóstico

    Raises
    ------
    ValueError
        Si faltan columnas requeridas o si no quedan viajes válidos
        después de los filtros.

    Notes
    -----
    Este builder NO genera muestras (X, y) — eso lo hace el dataset PyTorch
    en `trainer.py` para mantener flexibilidad sobre la ventana de contexto.
    El builder produce SECUENCIAS por viaje; las muestras (un input y un
    target por step) se construyen al vuelo durante el entrenamiento.
    """
    gval = group_value if group_value is not None else direction
    # ── Modo rejilla (como las baselines del paso 10) ────────────────────
    if tokenize_by == "grid":
        return _build_grid_sequences(
            df_full             = df_full,
            direction           = direction,
            grid_step_m         = grid_step_m,
            min_sequence_length = min_sequence_length,
            max_sequence_length = max_sequence_length,
            group_col           = group_col,
            group_value         = gval,
            with_calendar       = with_calendar,
            trip_whitelist      = trip_whitelist,
            max_cells           = max_cells,
            verbose             = verbose,
        )
    elif tokenize_by != "stop":
        raise ValueError(
            f"tokenize_by debe ser 'grid' o 'stop'; recibido: {tokenize_by!r}"
        )

    # ── 0) Validación de columnas ────────────────────────────────────────
    time_col = "link_travel_time" if use_interp_time else "s_next_stop"
    required = {"trip_id", group_col, "trip_valid", "time",
                "next_stop", time_col}
    missing = required - set(df_full.columns)
    if missing:
        raise ValueError(
            f"Columnas faltantes en df_full: {sorted(missing)}. "
            f"Asegúrate de correr el paso 07 (con use_interp_time="
            f"{use_interp_time} se requiere '{time_col}')."
        )

    if aggregation not in ("first", "median"):
        raise ValueError(
            f"aggregation debe ser 'first' o 'median', recibido: {aggregation!r}"
        )

    if context_features:
        missing_ctx = set(context_features) - set(df_full.columns)
        if missing_ctx:
            raise ValueError(
                f"context_features no existen en df_full: {sorted(missing_ctx)}"
            )

    if verbose:
        print("=" * 60)
        print(f"  SEQUENCE DATASET BUILDER — {group_col}={gval}")
        print("=" * 60)
        print(f"  Columna de tiempo        : {time_col}")
        print(f"  Agregación               : {aggregation}")
        if context_features:
            print(f"  Context features        : {context_features}")
        else:
            print(f"  Context features        : (ninguna; modelo paper puro)")

    # ── 1) Filtrar viajes válidos en el grupo elegido (leg/direction) ────
    df = df_full.filter(
        (pl.col("trip_valid") == True) &   # noqa: E712
        (pl.col(group_col)    == gval)
    ).sort(["trip_id", "time"])

    n_records  = df.height
    n_trips_in = df["trip_id"].n_unique()

    if verbose:
        print(f"  Registros tras filtro    : {n_records:,}")
        print(f"  Viajes únicos (entrada)  : {n_trips_in:,}")

    if n_trips_in == 0:
        raise ValueError(
            f"No hay viajes válidos para {group_col}={gval!r}. "
            f"Verifica que df_full tenga trip_valid==True para ese grupo."
        )

    # ── 2) Colapsar a un token por (trip_id, next_stop) ──────────────────
    # Mantenemos un orden cronológico interno: agrupamos por la primera
    # aparición de cada next_stop dentro del viaje.

    # Necesitamos un identificador de orden estable dentro del viaje.
    # Usamos cum_dist_m si está disponible (más robusto que time porque
    # respeta el progreso espacial), si no time.
    if "cum_dist_m" in df.columns:
        order_col = "cum_dist_m"
    else:
        order_col = "time"

    # Para "first": tomamos el primer registro de cada (trip, next_stop)
    # ordenado por order_col. Para "median": agregamos.
    agg_exprs = [
        pl.col(time_col).first().alias("_time_first"),
        pl.col(time_col).median().alias("_time_median"),
        pl.col(order_col).min().alias("_order_key"),
    ]
    if context_features:
        for c in context_features:
            agg_exprs.append(pl.col(c).first().alias(f"_ctx_{c}"))

    df_tokens = (
        df.group_by(["trip_id", "next_stop"], maintain_order=True)
          .agg(agg_exprs)
    )

    # Elegir columna de tiempo según aggregation
    if aggregation == "first":
        df_tokens = df_tokens.with_columns(
            pl.col("_time_first").alias("token_time")
        )
    else:
        df_tokens = df_tokens.with_columns(
            pl.col("_time_median").alias("token_time")
        )

    df_tokens = df_tokens.drop(["_time_first", "_time_median"])

    # Ordenar dentro de cada viaje
    df_tokens = df_tokens.sort(["trip_id", "_order_key"])

    if verbose:
        print(f"  Tokens generados         : {df_tokens.height:,}")
        print(f"  Tokens/viaje (mediana)   : "
              f"{df_tokens.group_by('trip_id').len()['len'].median():.0f}")

    # ── 3) Construir secuencias por viaje ────────────────────────────────
    sequences:    List[Dict[str, Any]] = []
    stop_counts:  Dict[str, int]       = {}

    n_dropped_short      = 0
    n_dropped_no_time    = 0
    n_truncated_long     = 0

    # Iteramos por trip_id; partition_by mantiene orden
    for df_trip in df_tokens.partition_by("trip_id", as_dict=False):
        trip_id = df_trip["trip_id"][0]
        n_steps = df_trip.height

        # Filtrar pasos sin tiempo (NaN al final del viaje, etc.)
        df_trip_clean = df_trip.filter(pl.col("token_time").is_not_null())
        if df_trip_clean.height < n_steps:
            # Algunos pasos cayeron; revisamos longitud final
            pass

        n_clean = df_trip_clean.height

        if n_clean < min_sequence_length:
            if n_clean == 0:
                n_dropped_no_time += 1
            else:
                n_dropped_short += 1
            continue

        # Truncar si excede el máximo (conservamos las últimas N paradas
        # porque el final del viaje suele ser más informativo)
        if n_clean > max_sequence_length:
            df_trip_clean = df_trip_clean.tail(max_sequence_length)
            n_truncated_long += 1
            n_clean = df_trip_clean.height

        stops_arr = df_trip_clean["next_stop"].to_numpy()
        times_arr = df_trip_clean["token_time"].cast(pl.Float32).to_numpy()

        # Contexto opcional
        if context_features:
            ctx_cols = [f"_ctx_{c}" for c in context_features]
            context_arr = (
                df_trip_clean.select(ctx_cols)
                             .to_numpy()
                             .astype(np.float32)
            )
        else:
            context_arr = None

        sequences.append({
            "trip_id": trip_id,
            "stops":   stops_arr,
            "times":   times_arr,
            "context": context_arr,
        })

        # Conteo global de paradas (útil para el vocabulario)
        for s in stops_arr:
            stop_counts[s] = stop_counts.get(s, 0) + 1

    if verbose:
        print(f"  Secuencias generadas (M) : {len(sequences):,}")
        print(f"  Descartadas (sin tiempo) : {n_dropped_no_time:,}")
        print(f"  Descartadas (cortas)     : {n_dropped_short:,}")
        print(f"  Truncadas (largas)       : {n_truncated_long:,}")

    if len(sequences) == 0:
        raise ValueError(
            "Ninguna secuencia pasó los filtros. "
            f"Revisa min_sequence_length={min_sequence_length} y verifica "
            f"que '{time_col}' tenga valores no nulos."
        )

    # ── 4) Estadísticas y metadata ───────────────────────────────────────
    seq_lengths = np.array([len(s["stops"]) for s in sequences])
    total_times = np.array([
        float(s["times"][~np.isnan(s["times"])].sum())
        for s in sequences
    ])

    metadata = {
        "direction":            direction,
        "group_col":            group_col,
        "group_value":          gval,
        "n_trips_input":        n_trips_in,
        "n_trips_output":       len(sequences),
        "n_unique_stops":       len(stop_counts),
        "min_sequence_length":  min_sequence_length,
        "max_sequence_length":  max_sequence_length,
        "max_context_length":   max_context_length,
        "aggregation":          aggregation,
        "time_column":          time_col,
        "context_features":     list(context_features) if context_features else None,
        "dropped_short":        n_dropped_short,
        "dropped_no_time":      n_dropped_no_time,
        "truncated_long":       n_truncated_long,
        "seq_length_mean":      float(seq_lengths.mean()),
        "seq_length_median":    float(np.median(seq_lengths)),
        "seq_length_p5":        float(np.quantile(seq_lengths, 0.05)),
        "seq_length_p95":       float(np.quantile(seq_lengths, 0.95)),
        "trip_total_time_median_s": float(np.median(total_times)),
    }

    if verbose:
        print(f"  Paradas únicas           : {metadata['n_unique_stops']}")
        print(f"  Longitud secuencia (med) : "
              f"{metadata['seq_length_median']:.0f} paradas "
              f"(P5={metadata['seq_length_p5']:.0f}, "
              f"P95={metadata['seq_length_p95']:.0f})")
        print(f"  Tiempo total viaje (med) : "
              f"{metadata['trip_total_time_median_s']/60:.1f} min")
        print("=" * 60)

    return {
        "sequences":    sequences,
        "stop_counts":  stop_counts,
        "context_cols": list(context_features) if context_features else None,
        "metadata":     metadata,
    }


# ──────────────────────────────────────────────────────────────────────────
# Tokenización por rejilla de distancia (paridad con baselines del paso 10)
# ──────────────────────────────────────────────────────────────────────────

def _build_grid_sequences(
    df_full:             pl.DataFrame,
    direction:           int,
    grid_step_m:         int,
    min_sequence_length: int,
    max_sequence_length: int,
    group_col:           str = "direction",
    group_value:         Any = None,
    with_calendar:       bool = False,
    trip_whitelist:      Optional[Iterable[Any]] = None,
    max_cells:           Optional[int] = None,
    verbose:             bool = False,
) -> Dict[str, Any]:
    """
    Construye secuencias por viaje tokenizando por CELDAS DE REJILLA de
    distancia fija, en vez de por paradas.

    Cada viaje se interpola sobre una grid uniforme de `grid_step_m` metros
    (idéntico al `build_trajectory_matrix` del paso 10) y se convierte en:
      - stops : ["g00001", "g00002", ...]  ← índice de celda = posición en ruta
      - times : tiempo (s) para atravesar cada celda de grid_step_m metros
                = diferencia del tiempo acumulado interpolado

    No depende de la ubicación de paradas: solo de `cum_dist_m` y `time`.
    A diferencia del paso 10 (matriz de ancho fijo con extrapolación), aquí
    cada viaje genera una secuencia de longitud variable: solo las celdas
    que el viaje realmente recorrió (hasta su max cum_dist_m).

    Devuelve la misma estructura que `build_sequence_dataset` para que el
    resto del pipeline (vocab, SequenceDataset, CV, trainer) no cambie.
    """
    gval = group_value if group_value is not None else direction
    required = {"trip_id", group_col, "trip_valid", "time", "cum_dist_m"}
    if with_calendar:
        # Fase 4: contexto de calendario por celda. hour_of_day / day_of_week ya
        # los produce el paso 07 (extended.py) y están densos en df_full.
        required |= {"hour_of_day", "day_of_week"}
    missing = required - set(df_full.columns)
    if missing:
        raise ValueError(
            f"Columnas faltantes en df_full para tokenize_by='grid': "
            f"{sorted(missing)}. Se requieren trip_id, {group_col}, trip_valid, "
            f"time, cum_dist_m (todas las produce el paso 07 / el método nuevo)."
        )

    if verbose:
        print("=" * 60)
        print(f"  SEQUENCE DATASET BUILDER (GRID) — {group_col}={gval}, "
              f"grid={grid_step_m}m")
        print("=" * 60)

    # ── 1) Filtrar viajes válidos en el grupo elegido (leg/direction) ────
    df = df_full.filter(
        (pl.col("trip_valid") == True) &   # noqa: E712
        (pl.col(group_col)    == gval)
    ).sort(["trip_id", "time"])

    # Whitelist del frame de evaluación: restringe la población a EXACTAMENTE
    # los viajes que usan los baselines y XGBoost (`func/eval_frame.py`). Sin
    # esto el LSTM entrena con todos los viajes que superan `n_cells >= 5` —con
    # q=0.95 eran ~7,700 por leg contra ~491 de los baselines (15×)— y la
    # comparación deja de ser pareada.
    n_trips_all = df["trip_id"].n_unique()
    n_missing_wl = 0
    if trip_whitelist is not None:
        wl = set(trip_whitelist)
        present = set(df["trip_id"].unique().to_list())
        n_missing_wl = len(wl - present)
        df = df.filter(pl.col("trip_id").is_in(list(wl)))
        if verbose:
            print(f"  Whitelist del frame      : {len(wl):,} viajes pedidos, "
                  f"{df['trip_id'].n_unique():,} presentes "
                  f"(de {n_trips_all:,} en la leg)")
        if n_missing_wl:
            print(f"  ⚠ {n_missing_wl} viaje(s) del frame NO están en df_full "
                  f"con trip_valid+{group_col}={gval!r}")

    n_trips_in = df["trip_id"].n_unique()
    if verbose:
        print(f"  Registros tras filtro    : {df.height:,}")
        print(f"  Viajes únicos (entrada)  : {n_trips_in:,}")

    if n_trips_in == 0:
        raise ValueError(
            f"No hay viajes válidos para {group_col}={gval!r}"
            + (" tras aplicar el whitelist del frame." if trip_whitelist
               is not None else ".")
        )

    # ── 2) Tiempo desde el inicio del viaje (segundos) ───────────────────
    time_dtype = str(df["time"].dtype)
    divisor = 1_000_000 if "us" in time_dtype else (1_000 if "ms" in time_dtype else 1)
    df = df.with_columns(
        (
            (pl.col("time").cast(pl.Int64)
             - pl.col("time").cast(pl.Int64).min().over("trip_id"))
            / divisor
        ).cast(pl.Float64).alias("t_seconds")
    )

    # ── 3) Interpolar cada viaje en su rejilla de celdas recorridas ──────
    sequences:   List[Dict[str, Any]] = []
    stop_counts: Dict[str, int]       = {}
    n_dropped_short  = 0
    n_dropped_nonmono = 0
    n_truncated_long = 0
    n_capped_grid    = 0

    for df_trip in df.partition_by("trip_id", as_dict=False):
        trip_id = df_trip["trip_id"][0]
        d = df_trip["cum_dist_m"].to_numpy().astype(np.float64)
        t = df_trip["t_seconds"].to_numpy().astype(np.float64)

        # Contexto de calendario del viaje (hora de salida + día de la semana).
        # day_of_week es constante por viaje; la hora avanza con el recorrido y
        # se reconstruye por celda desde t_grid, así el modelo ve la hora EN CADA
        # corte, no solo al salir. Misma semántica que Tier B de XGBoost.
        if with_calendar:
            h0  = float(df_trip["hour_of_day"][0])
            dow = float(df_trip["day_of_week"][0])
        else:
            h0 = dow = 0.0

        if len(d) < 5:
            n_dropped_short += 1
            continue

        # Monotonicidad estricta de la distancia (primer registro por valor)
        _, first_idx = np.unique(d, return_index=True)
        d_mono = d[first_idx]
        t_mono = t[first_idx]
        if not np.all(np.diff(d_mono) >= 0):
            n_dropped_nonmono += 1
            continue

        # Celdas realmente recorridas: 1 .. floor(d_max / step)
        d_max = float(d_mono.max())
        n_cells = int(d_max // grid_step_m)

        # Tope de geometría del frame: la celda `c` de este builder equivale a
        # `l_idx = c` de la matriz T de los baselines (ambas miden distancia
        # absoluta desde el inicio del viaje en pasos de `grid_step_m`), pero solo
        # si las dos se cortan en el MISMO `n_route_m`. `max_cells =
        # n_route_m // grid_step_m` impone ese corte: los viajes más largos se
        # TRUNCAN, que es exactamente lo que `build_trajectory_matrix` hace al
        # interpolar sobre una grid más corta. Con esto los índices coinciden y
        # no hace falta el `idx_off` del protocolo matched.
        if max_cells is not None and n_cells > max_cells:
            n_cells = int(max_cells)
            n_capped_grid += 1

        if n_cells < min_sequence_length:
            n_dropped_short += 1
            continue

        # Tiempo acumulado interpolado en 0, step, 2·step, ..., n_cells·step
        grid_d = np.arange(0, n_cells + 1, dtype=np.float64) * grid_step_m
        t_grid = np.interp(grid_d, d_mono, t_mono)

        # Tiempo POR celda = diferencia del acumulado (n_cells valores)
        cell_times = np.diff(t_grid).astype(np.float32)
        cell_ids   = np.arange(1, n_cells + 1)   # posición absoluta en ruta

        # Truncar a las últimas max_sequence_length celdas (como el modo stop),
        # conservando el índice ABSOLUTO de celda (embedding = posición).
        if n_cells > max_sequence_length:
            cell_ids   = cell_ids[-max_sequence_length:]
            cell_times = cell_times[-max_sequence_length:]
            n_truncated_long += 1

        stops_arr = np.array([_grid_token(int(c)) for c in cell_ids], dtype=object)

        # Contexto por celda: [hour_sin, hour_cos, dow/6, is_weekend] en la
        # hora en que el bus alcanzó cada celda (h0 + t_grid/3600, mod 24). El
        # sin/cos maneja el wrap de medianoche sin discontinuidad.
        if with_calendar:
            t_cell = t_grid[1:]                          # tiempo acumulado por celda
            if n_cells > max_sequence_length:
                t_cell = t_cell[-max_sequence_length:]
            hour_cont = (h0 + t_cell / 3600.0) % 24.0
            ctx = np.stack([
                np.sin(2 * np.pi * hour_cont / 24.0),
                np.cos(2 * np.pi * hour_cont / 24.0),
                np.full(len(t_cell), dow / 6.0),
                np.full(len(t_cell), 1.0 if dow >= 5 else 0.0),
            ], axis=1).astype(np.float32)               # (n_cells, 4)
        else:
            ctx = None

        sequences.append({
            "trip_id": trip_id,
            "stops":   stops_arr,
            "times":   cell_times,
            "context": ctx,
        })
        for s in stops_arr:
            stop_counts[s] = stop_counts.get(s, 0) + 1

    if not sequences:
        raise ValueError(
            "Ningún viaje pasó los filtros del builder por rejilla. "
            f"Revisa min_sequence_length={min_sequence_length} y grid_step_m."
        )

    seq_lengths = np.array([len(s["stops"]) for s in sequences])
    total_times = np.array([float(s["times"].sum()) for s in sequences])

    metadata = {
        "direction":            direction,
        "group_col":            group_col,
        "group_value":          gval,
        "tokenize_by":          "grid",
        "grid_step_m":          grid_step_m,
        "n_trips_input":        n_trips_in,
        "n_trips_output":       len(sequences),
        "n_unique_stops":       len(stop_counts),     # nº de celdas distintas
        "n_route_m":            int(seq_lengths.max() * grid_step_m),
        "min_sequence_length":  min_sequence_length,
        "max_sequence_length":  max_sequence_length,
        "max_cells":            (int(max_cells) if max_cells is not None else None),
        "whitelisted":          trip_whitelist is not None,
        "n_trips_leg_total":    int(n_trips_all),
        "whitelist_missing":    int(n_missing_wl),
        "dropped_short":        n_dropped_short,
        "dropped_nonmono":      n_dropped_nonmono,
        "truncated_long":       n_truncated_long,
        "capped_to_grid":       n_capped_grid,
        "seq_length_mean":      float(seq_lengths.mean()),
        "seq_length_median":    float(np.median(seq_lengths)),
        "seq_length_p5":        float(np.quantile(seq_lengths, 0.05)),
        "seq_length_p95":       float(np.quantile(seq_lengths, 0.95)),
        "trip_total_time_median_s": float(np.median(total_times)),
    }

    if verbose:
        print(f"  Secuencias generadas (M) : {len(sequences):,}")
        print(f"  Descartadas (cortas)     : {n_dropped_short:,}")
        print(f"  Descartadas (no mono)    : {n_dropped_nonmono:,}")
        print(f"  Truncadas (largas)       : {n_truncated_long:,}")
        if max_cells is not None:
            print(f"  Recortadas a la grid     : {n_capped_grid:,} "
                  f"(max_cells={max_cells})")
        print(f"  Celdas únicas            : {metadata['n_unique_stops']}")
        print(f"  Longitud secuencia (med) : "
              f"{metadata['seq_length_median']:.0f} celdas "
              f"(P5={metadata['seq_length_p5']:.0f}, "
              f"P95={metadata['seq_length_p95']:.0f})")
        print(f"  Tiempo total viaje (med) : "
              f"{metadata['trip_total_time_median_s']/60:.1f} min")
        print("=" * 60)

    return {
        "sequences":    sequences,
        "stop_counts":  stop_counts,
        "context_cols": (["hour_sin", "hour_cos", "dow", "is_weekend"]
                         if with_calendar else None),
        "context_dim":  4 if with_calendar else 0,
        "metadata":     metadata,
    }
