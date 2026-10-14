"""
_lib_data.py
============
Funciones auxiliares de carga, limpieza y geometría — extraídas de aux_f.py.

Este módulo es parte interna del pipeline; los archivos de paso `step_*.py`
lo importan, pero NO debe usarse directamente desde notebooks o scripts
finales.

Funciones expuestas
-------------------
    Carga de datos
        get_db_rq                — query SQL paralelizada (PostgreSQL)
        get_db_rq_from_to        — query con rango de fechas
        get_stops                — diccionario de paradas por ruta
        get_dic_poly             — diccionario de polilíneas por ruta

    Limpieza
        dc_records               — limpia df_rec (drop NaN, fix tipos, …)

    Geometría
        dp_build_segments        — construye segmentos polilínea ↔ paradas

    Utilitarios privados
        read_json
        get_aux_optimal_partitions
        _crear_dataframe_desde_coordenadas
        _procesar_featurecollection
        _procesar_json_simple
        _strip_prefix

Origen
------
Este archivo es un EXTRACTO de aux_f.py (proyecto BUSONORA original);
solo se incluyeron las 12 funciones requeridas por el pipeline para
hacerlo autocontenido.
"""

from __future__ import annotations

import os
import re
import math
import json
import glob
import calendar
import itertools
import datetime as dt
from datetime import datetime, timedelta, time
from pathlib import Path
from typing import Union, Tuple, List, Optional, Dict, Literal
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor

import numpy as np
import pandas as pd
import polars as pl

# Dependencias opcionales — solo se importan cuando se llamen
# (algunas funciones las usan, otras no)
try:
    import psutil
except ImportError:
    psutil = None

try:
    from sqlalchemy import create_engine
except ImportError:
    create_engine = None


def read_json(file_path) -> dict:
    """
    Read and parse a JSON file into a Python dictionary.
    
    Parameters
    ----------
        path (str):   Path to the JSON file to be read

    Returns
    -------
        data (dict):   Parsed JSON data as a Python dictionary
    """
    with open(file=file_path,mode='r',encoding='utf-8') as f:
        return json.load(f)


###############################################################################
### DataFrame non-empty I/O guards                                          ###
###############################################################################
# Política del proyecto: NUNCA guardar ni leer un DataFrame vacío sin querer.
# Un archivo de 0 bytes, o un parquet con 0 filas, casi siempre es un guardado
# fallido/abortado que corrompe SILENCIOSAMENTE los pasos posteriores del
# pipeline. Estas utilidades centralizan la validación para que el chequeo viva
# en el código y se aplique en CUALQUIER máquina donde se ejecute el pipeline.
#
#   save_df(df, path)            → guarda validando que no quede vacío.
#   load_df(path)                → lee validando que no esté vacío.
#   assert_df_nonempty(df, name) → guard para un df ya cargado en memoria.
#   purge_empty_dataframes(root) → barre y (opcional) elimina dataframes vacíos.

_DF_PARQUET_EXTS = (".parquet", ".pq")
_DF_CSV_EXTS     = (".csv",)


def _df_height(df) -> int:
    """Número de filas de un DataFrame polars o pandas (0 si es None/desconocido)."""
    if df is None:
        return 0
    h = getattr(df, "height", None)        # polars
    if h is not None:
        return int(h)
    if isinstance(df, pd.DataFrame):
        return int(len(df))
    try:
        return int(len(df))
    except TypeError:
        return 0


def assert_df_nonempty(df, name: str = "dataframe"):
    """
    Lanza ``ValueError`` si ``df`` está vacío (0 filas). Devuelve ``df`` para
    poder encadenar. Úsalo al ANALIZAR un DataFrame ya en memoria, antes de
    operar sobre él.
    """
    if df is None:
        raise ValueError(
            f"assert_df_nonempty: '{name}' es None (esperaba un DataFrame con filas)."
        )
    if _df_height(df) == 0:
        raise ValueError(f"assert_df_nonempty: '{name}' está VACÍO (0 filas).")
    return df


def save_df(df, path, *, allow_empty: bool = False, mkdir: bool = True,
            verbose: bool = False, **write_kwargs):
    """
    Guarda un DataFrame (polars o pandas) a parquet/csv validando que NO quede
    vacío. El formato se infiere por la extensión de ``path``.

    Reglas (política del proyecto)
    ------------------------------
    1. Rehúsa guardar un DataFrame de 0 filas, salvo ``allow_empty=True``
       (p.ej. placeholders intencionales de auditoría).
    2. Tras escribir, verifica que el archivo en disco NO sea de 0 bytes; si lo
       es, lo ELIMINA y lanza error (no deja basura corrupta en disco).

    Returns
    -------
    pathlib.Path — la ruta escrita.
    """
    path = Path(path)
    n = _df_height(df)
    if n == 0 and not allow_empty:
        raise ValueError(
            f"save_df: rehúso guardar DataFrame VACÍO (0 filas) en {path}. "
            f"Si es intencional, pasa allow_empty=True."
        )

    if mkdir:
        path.parent.mkdir(parents=True, exist_ok=True)

    ext = path.suffix.lower()
    is_polars = isinstance(df, pl.DataFrame)

    if ext in _DF_PARQUET_EXTS:
        if is_polars:
            df.write_parquet(path, **write_kwargs)
        else:
            df.to_parquet(path, **write_kwargs)
    elif ext in _DF_CSV_EXTS:
        if is_polars:
            df.write_csv(path, **write_kwargs)
        else:
            write_kwargs.setdefault("index", False)
            df.to_csv(path, **write_kwargs)
    else:
        raise ValueError(
            f"save_df: extensión no soportada '{ext}' en {path} (usa .parquet o .csv)."
        )

    # Verificación en disco: 0 bytes ⇒ guardado fallido.
    size = path.stat().st_size if path.exists() else 0
    if size == 0:
        try:
            path.unlink()
        except OSError:
            pass
        raise IOError(
            f"save_df: el archivo escrito quedó VACÍO (0 bytes) y fue eliminado: {path}"
        )

    if verbose:
        print(f"  ✔ save_df: {n:,} filas → {path} ({size:,} B)")
    return path


def load_df(path, *, engine: str = "polars", allow_empty: bool = False,
            verbose: bool = False, **read_kwargs):
    """
    Lee un DataFrame de parquet/csv validando que NO esté vacío.

    Reglas
    ------
    1. Archivo de 0 bytes  → error (salvo ``allow_empty=True``).
    2. DataFrame de 0 filas → error (salvo ``allow_empty=True``).

    Parameters
    ----------
    engine : {"polars", "pandas"}
        Librería de salida (default "polars", para drop-in de ``pl.read_parquet``).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"load_df: no existe el archivo: {path}")

    size = path.stat().st_size
    if size == 0 and not allow_empty:
        raise IOError(f"load_df: archivo VACÍO (0 bytes): {path}")

    ext = path.suffix.lower()
    if ext in _DF_PARQUET_EXTS:
        df = (pl.read_parquet(path, **read_kwargs) if engine == "polars"
              else pd.read_parquet(path, **read_kwargs))
    elif ext in _DF_CSV_EXTS:
        df = (pl.read_csv(path, **read_kwargs) if engine == "polars"
              else pd.read_csv(path, **read_kwargs))
    else:
        raise ValueError(
            f"load_df: extensión no soportada '{ext}' en {path} (usa .parquet o .csv)."
        )

    if _df_height(df) == 0 and not allow_empty:
        raise ValueError(f"load_df: el DataFrame leído está VACÍO (0 filas): {path}")

    if verbose:
        print(f"  ✔ load_df: {_df_height(df):,} filas ← {path} ({size:,} B)")
    return df


def purge_empty_dataframes(
    root: Union[str, Path] = "data",
    *,
    patterns: Tuple[str, ...] = ("*.parquet", "*.csv"),
    check_rows: bool = True,
    csv_header_only: bool = False,
    delete: bool = False,
    verbose: bool = True,
) -> List[Path]:
    """
    Barre ``root`` (recursivo) buscando dataframes guardados VACÍOS y,
    opcionalmente, los elimina. Implementa la política: "en caso de haber
    dataframes vacíos guardados, elimínalos a través del código".

    Criterio de "vacío"
    -------------------
    - Siempre: archivo de 0 bytes.
    - Si ``check_rows`` (default True): parquet con ``num_rows == 0``.
    - CSV solo-cabecera: SOLO se considera vacío si ``csv_header_only=True``
      (default False), porque muchos CSV de auditoría quedan legítimamente con
      solo la cabecera (un resultado "cero" válido, no un guardado corrupto).

    Parameters
    ----------
    delete : bool
        Si True, BORRA los vacíos. Si False (default), solo reporta (dry-run).

    Returns
    -------
    list[pathlib.Path] — archivos vacíos encontrados (borrados si ``delete=True``).
    """
    root = Path(root)
    empties: List[Path] = []
    scanned = 0
    for pat in patterns:
        for f in sorted(root.rglob(pat)):
            scanned += 1
            try:
                size = f.stat().st_size
            except OSError:
                continue
            is_empty = (size == 0)
            if not is_empty and check_rows:
                ext = f.suffix.lower()
                try:
                    if ext in _DF_PARQUET_EXTS:
                        import pyarrow.parquet as _pq
                        is_empty = (_pq.read_metadata(f).num_rows == 0)
                    elif ext in _DF_CSV_EXTS and csv_header_only:
                        with open(f, "r", encoding="utf-8", errors="replace") as fh:
                            is_empty = sum(1 for _ in fh) <= 1
                except Exception as e:
                    if verbose:
                        print(f"  [skip] no se pudo inspeccionar {f}: {e}")
                    continue
            if is_empty:
                empties.append(f)

    if verbose:
        print(f"purge_empty_dataframes: {scanned} archivos escaneados en {root} · "
              f"{len(empties)} vacíos")
        for f in empties:
            print(f"   {'BORRADO' if delete else 'VACÍO  '}: {f}")

    if delete:
        for f in empties:
            try:
                f.unlink()
            except OSError as e:
                if verbose:
                    print(f"  [error] no se pudo borrar {f}: {e}")
    return empties


##############################
### Data loading Functions ###
##############################
def get_db_rq(
    query: Optional[str] = None, 
    config_path: str = '../data/01_raw/gp_connectivity.json',
    parallel: bool = True,
    n_partitions: int = 8,
    batch_size: int = 100_000,
    partition_column: str = 'id_registro',
    verbose: bool = False
):
    """
    Executes a SQL query against a PostgreSQL database with parallel processing for large datasets.
    
    Parameters
    ----------
        query (str):            SQL query string to execute
        config_path (str):      Path to JSON file containing database connection credentials
        parallel (bool):        Enable parallel fetching (default: True)
        n_partitions (int):     Number of parallel connections (default: 8, adjust based on HPC cores)
        batch_size (int):       Rows per batch for streaming (default: 100k)
        partition_column (str): Column to use for partitioning (default: 'id_registro')
                               Must be unique and indexed for best performance
        verbose (bool):         Print detailed debug information (default: False)
    
    Returns
    -------
        data (pl.DataFrame):   Query results as Polars DataFrame
    
    Notes
    -----
        - Uses indexed column (id_registro) for efficient range-based partitioning
        - Much faster than ROW_NUMBER() approach
        - Requires unique, preferably indexed column
        - Falls back to ROW_NUMBER() if partition_column doesn't exist
    """
    try:
        # Read database credentials
        config = read_json(config_path)
        host = config.get('host')
        port = config.get('port')
        db = config.get('db')
        user = config.get('user')
        password = config.get('password')
        table = config.get('table')
        
        if not all([user, password, host, port, db]):
            raise ValueError("Missing database credentials in config file")
        
        connection_string = f"postgresql+pg8000://{user}:{password}@{host}:{port}/{db}"
        
        # -------------------------
        # IF NO QUERY: return metadata
        # -------------------------
        if query is None:
            engine = create_engine(connection_string, pool_pre_ping=True)
            with engine.connect() as connection:
                cols_query = f"""
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_name = '{table}'
                    ORDER BY ordinal_position;
                """
                cols_df = pl.read_database(cols_query, connection)
                columns = cols_df['column_name'].to_list()
                
                date_columns = [c for c in columns if 'utc_datetime' in c.lower() or 'fecha_registro' in c.lower()]
                date_col = date_columns[0] if date_columns else None
                
                if date_col:
                    summary_query = f"""
                        SELECT
                            MIN({date_col}) AS first_record,
                            MAX({date_col}) AS last_record,
                            COUNT(*) AS total_records
                        FROM {table};
                    """
                else:
                    summary_query = f"""
                        SELECT
                            NULL AS first_record,
                            NULL AS last_record,
                            COUNT(*) AS total_records
                        FROM {table};
                    """
                
                summary_df = pl.read_database(summary_query, connection)
                summary_df = summary_df.with_columns(pl.lit(columns).alias("columns"))
                
                print(f"Database name: {db}")
                print(f"Table name: {table}")
                return summary_df
        
        # -------------------------
        # PARALLEL QUERY EXECUTION
        # -------------------------
        if not parallel:
            # Simple single-threaded fetch
            if verbose:
                print("Executing query in single-threaded mode...")
            engine = create_engine(connection_string, pool_pre_ping=True)
            with engine.connect() as connection:
                return pl.read_database(query, connection)
        
        # Get total count and range for partitioning
        engine = create_engine(connection_string, pool_pre_ping=True, pool_size=n_partitions + 2)
        
        with engine.connect() as connection:
            # Check if partition column exists in the query result
            check_query = f"SELECT * FROM ({query}) as subquery LIMIT 1"
            sample_df = pl.read_database(check_query, connection)
            
            use_indexed_partitioning = partition_column in sample_df.columns
            
            if verbose:
                if use_indexed_partitioning:
                    print(f"✓ Using indexed column '{partition_column}' for partitioning")
                else:
                    print(f"⚠ Column '{partition_column}' not found, falling back to ROW_NUMBER()")
            
            # Get count and range
            if use_indexed_partitioning:
                stats_query = f"""
                    SELECT 
                        COUNT(*) as total,
                        MIN({partition_column}) as min_id,
                        MAX({partition_column}) as max_id
                    FROM ({query}) as subquery
                """
                stats = pl.read_database(stats_query, connection)
                total_rows = stats['total'][0]
                min_id = stats['min_id'][0]
                max_id = stats['max_id'][0]
                
                if verbose:
                    print(f"  ID range: {min_id} to {max_id}")
            else:
                count_query = f"SELECT COUNT(*) as total FROM ({query}) as subquery"
                total_rows = pl.read_database(count_query, connection)['total'][0]
                min_id = None
                max_id = None
            
            if total_rows == 0:
                if verbose:
                    print("No rows found in query result")
                return pl.DataFrame()
            
            print(f"Total rows to fetch: {total_rows:,}")
            print(f"Fetching with {n_partitions} parallel connections...")
        
        # Calculate partition ranges
        if use_indexed_partitioning:
            # Use ID-based range partitioning (MUCH FASTER)
            id_range = max_id - min_id + 1
            ids_per_partition = math.ceil(id_range / n_partitions)
            
            if verbose:
                print(f"ID range size: {id_range:,}")
                print(f"IDs per partition: ~{ids_per_partition:,}")
        else:
            # Fallback to row-based partitioning
            rows_per_partition = math.ceil(total_rows / n_partitions)
            
            if verbose:
                print(f"Rows per partition: {rows_per_partition:,}")
        
        def fetch_partition_indexed(partition_id: int, start_id: int, end_id: int) -> pl.DataFrame:
            """Fetch partition using ID range (indexed column)"""
            try:
                local_engine = create_engine(connection_string, pool_pre_ping=True)
                with local_engine.connect() as conn:
                    # Use indexed column for fast range scan
                    partition_query = f"""
                        SELECT *
                        FROM ({query}) as base_query
                        WHERE {partition_column} >= {start_id}
                        AND {partition_column} < {end_id}
                    """
                    
                    df = pl.read_database(partition_query, conn)
                    
                    if verbose:
                        print(f"  ✓ Partition {partition_id + 1}/{n_partitions} [ID {start_id} to {end_id-1}]: {len(df):,} rows")
                    else:
                        print(f"  ✓ Partition {partition_id + 1}/{n_partitions} fetched: {len(df):,} rows")
                    
                    return df
            except Exception as e:
                print(f"  ✗ Error in partition {partition_id}: {e}")
                return pl.DataFrame()
        
        def fetch_partition_rownumber(partition_id: int, offset: int, limit: int) -> pl.DataFrame:
            """Fetch partition using ROW_NUMBER (fallback)"""
            try:
                local_engine = create_engine(connection_string, pool_pre_ping=True)
                with local_engine.connect() as conn:
                    partition_query = f"""
                        SELECT *
                        FROM (
                            SELECT *, ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) as rn
                            FROM ({query}) as base_query
                        ) as numbered_query
                        WHERE rn > {offset} AND rn <= {offset + limit}
                    """
                    
                    df = pl.read_database(partition_query, conn)
                    
                    # Drop row number column
                    if 'rn' in df.columns:
                        df = df.drop('rn')
                    
                    if verbose:
                        print(f"  ✓ Partition {partition_id + 1}/{n_partitions} [rows {offset+1} to {offset+limit}]: {len(df):,} rows")
                    else:
                        print(f"  ✓ Partition {partition_id + 1}/{n_partitions} fetched: {len(df):,} rows")
                    
                    return df
            except Exception as e:
                print(f"  ✗ Error in partition {partition_id}: {e}")
                return pl.DataFrame()
        
        # Parallel execution
        partitions: List[pl.DataFrame] = []
        with ThreadPoolExecutor(max_workers=n_partitions) as executor:
            futures = []
            
            if use_indexed_partitioning:
                # ID-based partitioning
                for i in range(n_partitions):
                    start_id = min_id + (i * ids_per_partition)
                    end_id = min(start_id + ids_per_partition, max_id + 1)
                    
                    if start_id >= max_id + 1:
                        break
                    
                    future = executor.submit(fetch_partition_indexed, i, start_id, end_id)
                    futures.append(future)
            else:
                # Row-based partitioning (fallback)
                for i in range(n_partitions):
                    offset = i * rows_per_partition
                    limit = min(rows_per_partition, total_rows - offset)
                    
                    if limit <= 0:
                        break
                    
                    future = executor.submit(fetch_partition_rownumber, i, offset, limit)
                    futures.append(future)
            
            # Collect results as they complete
            for future in as_completed(futures):
                result = future.result()
                if len(result) > 0:
                    partitions.append(result)
        
        # Concatenate all partitions
        if not partitions:
            return pl.DataFrame()
        
        final_df = pl.concat(partitions, how="vertical_relaxed")
        
        # Verify no duplicates if verbose
        if verbose and partition_column in final_df.columns:
            unique_ids = final_df[partition_column].n_unique()
            total_rows_fetched = len(final_df)
            
            print(f"\nVerificación de duplicados:")
            print(f"  Total filas: {total_rows_fetched:,}")
            print(f"  IDs únicos: {unique_ids:,}")
            
            if unique_ids == total_rows_fetched:
                print(f"  ✓ Sin duplicados")
            else:
                print(f"  ✗ ADVERTENCIA: {total_rows_fetched - unique_ids:,} duplicados detectados!")
        
        print(f"✓ Total fetched: {len(final_df):,} rows")
        
        return final_df
    
    except FileNotFoundError:
        print(f"Config file not found: {config_path}")
        return pl.DataFrame()
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON config file: {e}")
        return pl.DataFrame()
    except Exception as e:
        print(f"Database error: {e}")
        return pl.DataFrame()


def get_aux_optimal_partitions(n_cores: Optional[int] = None) -> int:
    """
    Determina el número óptimo de particiones basado en cores disponibles.
    
    Parameters
    ----------
    n_cores : int, optional
        Número de cores a usar. Si None, detecta automáticamente.
    
    Returns
    -------
    int
        Número óptimo de particiones
    """
    if n_cores is not None:
        return n_cores
    
    # Detectar cores disponibles
    try:
        # En HPC con SLURM
        slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
        if slurm_cpus:
            return int(slurm_cpus)
        
        # Fallback: detectar cores físicos
        cores = psutil.cpu_count(logical=False) or os.cpu_count()
        return cores if cores else 4
    except:
        return 4  # Default conservador


def get_db_rq_from_to(
    fecha_inicio: str, 
    fecha_final: str, 
    table: str = 'mobility_data',
    config_path: str = '../data/01_raw/gp_connectivity.json',
    parallel: bool = True,
    n_partitions: Optional[int] = None,  # Auto-detect si None
    int_route_rec: Optional[int] = None,
    gmt_offset: int = -7,
    verbose: bool = False
) -> pl.DataFrame:
    """
    Retrieves transportation records from database within a date range.
    Optimized for HPC environments with automatic core detection.
    
    Parameters
    ----------
        fecha_inicio (str):    Start date in LOCAL TIME (format: 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS')
        fecha_final (str):     End date in LOCAL TIME (format: 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS')
        table (str):           Database table name (default: 'mobility_data')
        config_path (str):     Path to DB credentials JSON
        parallel (bool):       Enable parallel fetching (default: True)
        n_partitions (int):    Number of parallel connections (default: auto-detect cores)
        gmt_offset (int):      GMT offset in hours (default: -7 for GMT-7/MST)
        verbose (bool):        Print detailed progress information (default: False)
    
    Returns
    -------
        df (pl.DataFrame):     Polars DataFrame with transportation records
    
    Notes
    -----
        - Auto-detects cores in HPC (SLURM) or local environment
        - Optimizes n_partitions to match available cores
        - Prevents oversubscription that causes slowdown
    
    Example
    -------
        >>> # Auto-detect cores (recommended for HPC)
        >>> df = get_db_rq_from_to(
        ...     fecha_inicio='2024-01-01',
        ...     fecha_final='2024-01-01',
        ...     verbose=True
        ... )
        >>> 
        >>> # Manual override
        >>> df = get_db_rq_from_to(
        ...     fecha_inicio='2024-01-01',
        ...     fecha_final='2024-01-01',
        ...     n_partitions=3,  # Force 3 cores
        ...     verbose=True
        ... )
    """
    
    # Auto-detect optimal partitions if not specified
    if n_partitions is None:
        n_partitions = get_aux_optimal_partitions()
        if verbose:
            print(f"Auto-detected {n_partitions} cores for parallel processing")
    
    if verbose:
        print("=" * 70)
        print("FETCHING DATA FROM DATABASE (HPC-Optimized)")
        print("=" * 70)
        print(f"Cores/Partitions: {n_partitions}")
        gmt_str = f"GMT{gmt_offset:+d}" if gmt_offset != 0 else "GMT"
        print(f"Local timezone: {gmt_str}")
        print(f"Date range (local time): {fecha_inicio} to {fecha_final}")
    
    # Parse local time dates
    try:
        local_start_dt = datetime.fromisoformat(fecha_inicio.replace(' ', 'T'))
        local_end_dt = datetime.fromisoformat(fecha_final.replace(' ', 'T'))
    except ValueError:
        local_start_dt = datetime.strptime(fecha_inicio.split()[0], '%Y-%m-%d')
        local_end_dt = datetime.strptime(fecha_final.split()[0], '%Y-%m-%d')
        if ' ' not in fecha_final:
            local_end_dt = local_end_dt.replace(hour=23, minute=59, second=59)
    
    # Convert local time to UTC for database query
    utc_start_dt = local_start_dt - timedelta(hours=gmt_offset)
    utc_end_dt = local_end_dt - timedelta(hours=gmt_offset)
    
    if verbose:
        print(f"Converted to UTC: {utc_start_dt} to {utc_end_dt}\n")
    
    # Build the query using UTC times
    utc_inicio = utc_start_dt.strftime('%Y-%m-%d %H:%M:%S')
    utc_final = utc_end_dt.strftime('%Y-%m-%d %H:%M:%S')
    
    if int_route_rec:
        query = f"""
            SELECT *
            FROM {table}
            WHERE utc_datetime >= '{utc_inicio}'
            AND utc_datetime <= '{utc_final}'
            AND group_id = '{int_route_rec}'
            ORDER BY utc_datetime ASC
        """
    else:
        query = f"""
            SELECT *
            FROM {table}
            WHERE utc_datetime >= '{utc_inicio}'
            AND utc_datetime <= '{utc_final}'
            ORDER BY utc_datetime ASC
        """
    
    # Use get_db_rq for optimized parallel fetching
    df = get_db_rq(
        query=query,
        config_path=config_path,
        parallel=parallel,
        n_partitions=n_partitions,
        partition_column='id_registro',  # Use indexed column
        verbose=verbose
    )
    
    if len(df) == 0:
        if verbose:
            print("⚠️ No data found in specified date range")
            print(f"{'=' * 70}\n")
        return df
    
    # Add local time column
    df = df.with_columns([
        (pl.col('utc_datetime') + pl.duration(hours=gmt_offset)).alias('time')
    ])
    
    if verbose:
        gmt_str = f"GMT{gmt_offset:+d}" if gmt_offset != 0 else "GMT"
        print(f"\n{'─' * 70}")
        print("RESULTS SUMMARY")
        print(f"{'─' * 70}")
        print(f"Total rows fetched: {len(df):,}")
        print(f"Parallelization efficiency: {n_partitions} cores used")
        print(f"{'=' * 70}\n")
    else:
        print(f"✓ Total fetched: {len(df):,} rows using {n_partitions} cores")
   
    return df


# Example usage:
# df = get_db_rq(
#     query="SELECT * FROM transport_data WHERE date >= '2024-01-01'",
#     parallel=True,
#     n_partitions=16  # Adjust based on your HPC cores
# )
def get_stops(
    routes_path: str = '../data/01_raw/stops/', 
    json_path: str = '../data/01_raw/stops.json',
    version: str = 'new',
    minimal: bool = False,
    parallel: bool = True,
    n_workers: int = 8,
    verbose: bool = False,
    # ── Modo "joined": stops.json + stops_routes.json ──
    stops_path: Optional[str] = None,
    stops_routes_path: Optional[str] = None,
    focus_route: Optional[int] = None,
) -> Dict[int, pl.DataFrame]:
    """
    Loads and processes route/stop data into a dictionary of Polars DataFrames.
    Supports two versions: 'old' (multiple JSON files) and 'new' (single JSON file).

    Modo "joined" (stops_path + stops_routes_path):
        Si se pasan ambos argumentos ``stops_path`` y ``stops_routes_path``,
        se ignoran ``version``/``json_path``/``routes_path`` y se usa la lógica
        de join entre los dos archivos del formato nuevo del pipeline:

        - stops.json          → metadatos de cada parada (code, lat, lon, …).
        - stops_routes.json   → mapeo {route_id: [{code, stop_order}, …]}.

        El resultado es un dict {route_id (int): pl.DataFrame} con columnas:
            code, latitude, longitude, codigo_original, stop_order.

    Args:
        routes_path (str): Path to directory with route JSON files (used for version='old')
        json_path (str): Path to single JSON file (used for version='new')
        version (str): 'old' or 'new' - determines loading strategy
        minimal (bool): If True, returns only essential columns (only for version='old')
        parallel (bool): If True, loads routes in parallel (only for version='old')
        n_workers (int): Number of parallel workers (only for version='old')
        verbose (bool): If True, prints progress information
        stops_path (str, optional): Path to stops.json (modo joined).
        stops_routes_path (str, optional): Path to stops_routes.json (modo joined).
        focus_route (int, optional): Solo modo joined + verbose. El catálogo trae
            TODAS las rutas del sistema; con focus_route el detalle por ruta se
            imprime solo para esa (el resto queda en el resumen "Rutas cargadas").

    Returns:
        Dict[int, pl.DataFrame]: Dictionary mapping route_id (int) to DataFrame with stops

        Version 'old' minimal columns:
            * 'codigo', 'latitude', 'longitude', 'distance_to_next_stop_km'

        Version 'new' columns:
            * 'codigo', 'linea', 'name', 'latitude', 'longitude', 'lineas', 'codigo_original'

        Joined columns:
            * 'code', 'latitude', 'longitude', 'codigo_original', 'stop_order'

    Example:
        >>> # Old version with multiple files
        >>> routes = get_stops(version='old', minimal=True, verbose=True)
        >>>
        >>> # New version with single file
        >>> routes = get_stops(version='new', verbose=True)
        >>>
        >>> # Joined mode (pipeline format)
        >>> routes = get_stops(stops_path='stops.json', stops_routes_path='stops_routes.json')
    """

    # ========== MODO JOINED (stops.json + stops_routes.json) ==========
    if stops_path is not None and stops_routes_path is not None:
        if verbose:
            print("=" * 60)
            print("LOADING STOPS (JOINED MODE)")
            print(f"  stops_path       : {stops_path}")
            print(f"  stops_routes_path: {stops_routes_path}")
            print("=" * 60)

        with open(stops_path, "r", encoding="utf-8") as f:
            stops_data = json.load(f)

        with open(stops_routes_path, "r", encoding="utf-8") as f:
            stops_routes_data = json.load(f)

        # ── Normalizar stops.json a lista de dicts ────────────────────────────
        # Formato esperado: lista plana [{code, codigo_original, name, latitude, longitude}, ...]
        # Soporta también dict-de-listas {linea_id: [...]} aplanándolo.
        if isinstance(stops_data, list):
            stops_list = stops_data
        elif isinstance(stops_data, dict):
            # Si el dict tiene una clave "stops" estándar:
            if "stops" in stops_data and isinstance(stops_data["stops"], list):
                stops_list = stops_data["stops"]
            else:
                # Aplanar: cada valor puede ser una lista de paradas
                stops_list = []
                for v in stops_data.values():
                    if isinstance(v, list):
                        stops_list.extend(v)
                    elif isinstance(v, dict):
                        stops_list.append(v)
        else:
            stops_list = []

        # ── Índice code → metadatos ───────────────────────────────────────────
        # Las claves posibles en cada parada de stops.json: code | codigo | stop_code | id
        stop_meta: Dict[str, dict] = {}
        for s in stops_list:
            if not isinstance(s, dict):
                continue
            code = (s.get("code") or s.get("codigo")
                    or s.get("stop_code") or s.get("id"))
            if code is not None:
                stop_meta[str(code)] = s

        if verbose:
            print(f"  ✔ stops.json: {len(stops_list)} paradas totales · "
                  f"{len(stop_meta)} indexadas por code")

        # ── stops_routes.json: {route_id: [{stop_code, stop_order, ...}, ...]} ──
        # Las claves posibles en cada entrada: stop_code | code | codigo
        # El orden viene de stop_order | order | orden.
        result: Dict[int, pl.DataFrame] = {}
        for route_id_str, route_stops in stops_routes_data.items():
            try:
                route_id = int(route_id_str)
            except (ValueError, TypeError):
                continue

            if not route_stops:
                continue

            rows = []
            missing = 0
            for entry in route_stops:
                if not isinstance(entry, dict):
                    continue
                code = entry.get("stop_code") or entry.get("code") or entry.get("codigo")
                code = str(code) if code is not None else ""
                if not code:
                    continue

                # stop_order: usar 0 si falta, pero 0 es válido — usar None centinela.
                so_raw = entry.get("stop_order")
                if so_raw is None: so_raw = entry.get("order")
                if so_raw is None: so_raw = entry.get("orden")
                try:
                    stop_order = int(so_raw) if so_raw is not None else len(rows)
                except (ValueError, TypeError):
                    stop_order = len(rows)

                meta = stop_meta.get(code)
                if meta is None:
                    missing += 1
                    continue   # sin coords no sirve para segmentar

                lat = meta.get("latitude")  if meta.get("latitude")  is not None else meta.get("lat")
                lon = meta.get("longitude") if meta.get("longitude") is not None else meta.get("lon")

                # Las coordenadas en stops.json vienen como string en tu archivo
                try:
                    lat_f = float(lat) if lat is not None else float("nan")
                except (ValueError, TypeError):
                    lat_f = float("nan")
                try:
                    lon_f = float(lon) if lon is not None else float("nan")
                except (ValueError, TypeError):
                    lon_f = float("nan")

                cod_orig = (meta.get("codigo_original") or meta.get("codigo") or code)

                rows.append({
                    "code":            code,
                    "latitude":        lat_f,
                    "longitude":       lon_f,
                    "codigo_original": str(cod_orig),
                    "stop_order":      stop_order,
                })

            _show = verbose and (focus_route is None or route_id == focus_route)
            if rows:
                result[route_id] = pl.DataFrame(rows)
                if _show:
                    extra = f" ({missing} sin metadatos)" if missing else ""
                    print(f"  Ruta {route_id}: {len(rows)} paradas{extra}")
            elif _show:
                print(f"  Ruta {route_id}: 0 paradas — {missing} sin metadatos")

        if verbose:
            print(f"  ✔ Rutas cargadas: {len(result)}"
                  + (f" (detalle solo de la {focus_route})" if focus_route is not None else ""))

        return result

    if version not in ['old', 'new']:
        raise ValueError(f"version must be 'old' or 'new', got '{version}'")
    
    # ========== VERSION OLD ==========
    if version == 'old':
        if verbose:
            print("=" * 60)
            print("LOADING STOPS (OLD VERSION - Multiple Files)")
            print("=" * 60)
            print(f"Directory: {routes_path}")
            print(f"Minimal mode: {minimal}")
            print(f"Parallel: {parallel} (workers: {n_workers})")
        
        # Get list of JSON files
        all_routes_files = os.listdir(routes_path)
        
        # Extract route IDs from filenames
        all_routes = [
            int(route.split('_')[1].split('.')[0]) 
            for route in all_routes_files 
            if route.endswith('.json')
        ]
        
        if verbose:
            print(f"Files found: {len(all_routes)}\n")
        
        # Define minimal columns to keep
        minimal_columns = ['codigo', 'latitude', 'longitude', 'distance_to_next_stop_km']
        
        def load_single_route(route_id: int) -> tuple[int, pl.DataFrame]:
            """Load a single route JSON file and convert to Polars DataFrame"""
            try:
                json_file = f"{routes_path}stop_{route_id}.json"
                route_stops = read_json(json_file)
                
                df = pl.DataFrame(route_stops['stops'])
                
                if minimal:
                    existing_columns = [col for col in minimal_columns if col in df.columns]
                    df = df.select(existing_columns)
                
                return (route_id, df)
                
            except Exception as e:
                if verbose:
                    print(f"✗ Error loading route {route_id}: {e}")
                return (route_id, pl.DataFrame())
        
        # Load routes
        all_dict = {}
        
        if not parallel or len(all_routes) < 5:
            if verbose:
                print("Loading sequentially...")
            for route in all_routes:
                route_id, df = load_single_route(route)
                all_dict[route_id] = df
        else:
            if verbose:
                print(f"Loading with {n_workers} parallel workers...")
            
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = {executor.submit(load_single_route, route): route for route in all_routes}
                
                completed = 0
                for future in as_completed(futures):
                    route_id, df = future.result()
                    all_dict[route_id] = df
                    completed += 1
                    
                    if verbose and completed % 10 == 0:
                        print(f"  Progress: {completed}/{len(all_routes)} routes...")
        
        if verbose:
            print(f"\n{'─' * 60}")
            print("SUMMARY")
            print(f"{'─' * 60}")
            print(f"✓ Successfully loaded {len(all_dict)} routes")
            total_stops = sum(len(df) for df in all_dict.values())
            print(f"Total stops: {total_stops:,}")
            print(f"{'=' * 60}\n")
        
        return all_dict
    
    # ========== VERSION NEW ==========
    else:  # version == 'new'
        if verbose:
            print("=" * 60)
            print("LOADING STOPS (NEW VERSION - Single File)")
            print("=" * 60)
            print(f"File: {json_path}")
        
        # Read JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Access real level (skip top "1" key)
        lineas_data = list(data.values())[0]
        
        if verbose:
            print(f"Lines found: {len(lineas_data)}\n")
        
        # Convert each line to DataFrame with integer keys
        stops = {}
        total_stops = 0
        
        for linea_id, paradas_list in lineas_data.items():
            if paradas_list:
                # Convert string key to integer
                stops[int(linea_id)] = pl.DataFrame(paradas_list)
                total_stops += len(paradas_list)
                
                if verbose:
                    print(f"  Line {linea_id}: {len(paradas_list)} stops")
        
        if verbose:
            print(f"\n{'─' * 60}")
            print("SUMMARY")
            print(f"{'─' * 60}")
            print(f"Total lines: {len(stops)}")
            print(f"Total stops: {total_stops:,}")
            
            # Show example
            if stops:
                example_key = list(stops.keys())[0]
                print(f"\nExample - Line {example_key}:")
                print(stops[example_key].head(3))
                print(f"\nSchema:")
                print(stops[example_key].schema)
            
            print(f"{'=' * 60}\n")
        
        return stops


###############################
### Data cleaning Functions ###
###############################
def dc_records(
    df: pl.DataFrame, 
    verbose: bool = False,
    drop_cols: bool = True,
    cols_to_drop: List[str] = ['id', 'id_registro', 'brand', 'plate_number', 'engine_on', 'route_direction'],
    h3_resolution: int = 9,
    compute_h3: bool = False,
    work_hours_filter: bool = True,
) -> pl.DataFrame:
    """
    Prepara y limpia un DataFrame de datos de transporte usando Polars para máximo rendimiento.
    Realiza conversiones de tipos, limpieza de valores nulos y creación de características temporales.
    
    Parameters
    ----------
    df : pl.DataFrame
        DataFrame original con datos de transporte que contiene columnas como 
        'utc_datetime', 'speed', 'code', 'route', 'disabled', etc.
    verbose : bool, default False
        Si True, imprime información de progreso y estadísticas del proceso.
    drop_cols : bool, default True
        Si True, elimina las columnas especificadas en cols_to_drop.
    cols_to_drop : list, default None
        Lista de columnas adicionales a eliminar. Si es None, usa ['id', 'id_registro'] por defecto.
    h3_resolution : int, default 9
        H3 resolution level used to compute the H3 cell index from latitude/longitude.
        Only used when compute_h3=True.
    compute_h3 : bool, default False
        If True, computes the H3 cell index from latitude/longitude at the given
        h3_resolution and adds it as a column named 'h3'.
    
    Returns
    -------
    pl.DataFrame
        DataFrame procesado con conversión a GMT-7, filtrado de ventana operativa (05:00-22:00),
        valores nulos eliminados en columnas críticas, y duplicados removidos.
    
    Notes
    -----
    - Filtra registros con disabled == 0 (solo rutas activas)
    - Convierte UTC a GMT-7 (UTC-7) si la columna 'time' no existe
    - Filtra ventana operativa: 05:00:00 a 22:00:00 (si work_hours_filter=True)
    - Si la columna 'time' ya existe, la usa directamente
    - Elimina duplicados basados en ('code', 'utc_datetime'), conservando el registro con menor 'id_registro'
    """
    
    if verbose:
        print("=" * 60)
        print("INICIO DE LIMPIEZA DE DATOS")
        print("=" * 60)
        print(f"Filas iniciales: {df.height:,}")
        print(f"Columnas: {df.width}")
        print("\nTipos de datos originales:")
        print(df.schema)
    
    # Verificar si la columna 'time' ya existe
    has_time_column = 'time' in df.columns
    
    if verbose:
        if has_time_column:
            print("\n✓ Columna 'time' ya existe, se usará directamente")
        else:
            print("\n→ Columna 'time' no existe, se creará desde 'utc_datetime' (GMT-7)")
        
        if drop_cols:
            existing_cols = [col for col in cols_to_drop if col in df.columns]
            if existing_cols:
                print(f"\n→ Se eliminarán columnas adicionales: {existing_cols}")
    
    # Verificar si la columna "group_id" existe, para saber si se trabaja con una o varias rutas
    has_route_column = 'group_id' in df.columns
    
    if has_route_column:
        if verbose:
            print("\n✓ Columna 'group_id' existe, se procesaran rutas")
        columns_renamed = {'code':'bus_code', 'cape':'angle', 'group_id':'route', 'stop':'last_stop'}
    else:
        if verbose:
            print("\n→ Columna 'group_id' no existe, se procesaran datos de la ruta proporcionada")
        columns_renamed = {'code':'bus_code', 'cape':'angle', 'stop':'last_stop'}
    # step_01 lee solo las columnas que usa (RAW_COLUMNS): sin 'stop'.
    columns_renamed = {k: v for k, v in columns_renamed.items() if k in df.columns}

    # ============================================================================
    # PASO 1: IDENTIFICAR Y ELIMINAR DUPLICADOS
    # ============================================================================
    
    if verbose:
        print("\n" + "─" * 60)
        print("ANÁLISIS DE DUPLICADOS (code, utc_datetime)")
        print("─" * 60)
    
    # Contar duplicados antes de eliminarlos
    duplicados_antes = (
        df.lazy()
        .with_columns(
            pl.len().over(['code', 'utc_datetime']).alias('_count_duplicados')
        )
        .filter(pl.col('_count_duplicados') > 1)
        .collect()
        .height
    )
    
    if verbose:
        grupos_duplicados = (
            df.lazy()
            .group_by(['code', 'utc_datetime'])
            .agg(pl.len().alias('count'))
            .filter(pl.col('count') > 1)
            .collect()
            .height
        )
        
        print(f"Registros duplicados encontrados: {duplicados_antes:,}")
        print(f"Grupos de duplicados: {grupos_duplicados:,}")
        
        if duplicados_antes > 0:
            promedio = duplicados_antes / grupos_duplicados if grupos_duplicados > 0 else 0
            print(f"Promedio de duplicados por grupo: {promedio:.2f}")
            
            # Mostrar ejemplo de duplicados
            ejemplo_dup = (
                df.lazy()
                .with_columns(
                    pl.len().over(['code', 'utc_datetime']).alias('_count_duplicados')
                )
                .filter(pl.col('_count_duplicados') > 1)
                .sort(['code', 'utc_datetime', 'id_registro'])
                .select(['id_registro', 'code', 'utc_datetime', '_count_duplicados'])
                .head(5)
                .collect()
            )
            print("\nEjemplo de registros duplicados (ordenados por id_registro):")
            print(ejemplo_dup)
    
    # Eliminar duplicados conservando el que tenga menor id_registro
    df_sin_duplicados = (
        df.lazy()
        .sort(['code', 'utc_datetime', 'id_registro'])  # Ordenar primero
        .unique(subset=['code', 'utc_datetime'], keep='first')  # Mantener el primero (menor id_registro)
        .collect()
    )
    
    duplicados_eliminados = df.height - df_sin_duplicados.height
    
    if verbose:
        print(f"\n✓ Duplicados eliminados: {duplicados_eliminados:,}")
        print(f"  Registros conservados: {df_sin_duplicados.height:,}")
        print("  Criterio: Se conservó el registro con menor 'id_registro'")
    
    # ============================================================================
    # PASO 2: PIPELINE DE LIMPIEZA
    # ============================================================================
    
    if verbose:
        print("\n" + "─" * 60)
        print("LIMPIEZA Y TRANSFORMACIÓN DE DATOS")
        print("─" * 60)
    
    # Pipeline de limpieza usando lazy evaluation para optimización
    df_lazy = (
        df_sin_duplicados.lazy()
        
        # 0. Rename columns
        .rename(columns_renamed)
        
        # 1. Eliminar nulos en columnas críticas (ya se checaron y no hay nulos)
        # .filter(
        #     pl.col('speed').is_not_null() &
        #     pl.col('bus_code').is_not_null() &
        #     pl.col('route').is_not_null() &
        #     (pl.col('no_passengers') < 150)   # no tomar en cuenta columna de pasajeros
        # )
        
        # 2. Filtrar solo rutas activas (disabled == 0)
        .filter(pl.col('disabled') == 0)
    )
    

    # 3. Crear columna de tiempo local (GMT-7) solo si no existe
    if not has_time_column:
        df_lazy = df_lazy.with_columns([
            (pl.col('utc_datetime') - pl.duration(hours=7)).alias('time')
        ])

    # ── NEW: compute H3 cell index ──────────────────────────────────────────────
    if compute_h3:
        h3_col_name = f'h3'
        df_lazy = df_lazy.with_columns(
            pl.struct(['latitude', 'longitude'])
            .map_elements(
                lambda row: h3.latlng_to_cell(row['latitude'], row['longitude'], h3_resolution),
                return_dtype=pl.String
            )
            .alias(h3_col_name)
        )
    # ────────────────────────────────────────────────────────────────────────────

    # 4. Preparar lista de columnas a eliminar
    cols_to_remove = [c for c in ['utc_datetime', 'fecha_registro', 'disabled', 'no_passengers']
                      if c in df_sin_duplicados.columns]

    if drop_cols:
        cols_to_remove.extend([col for col in cols_to_drop if col in df_sin_duplicados.columns])
        # ── NEW: also drop lat/lon when drop_cols=True ──────────────────────────
        #cols_to_remove.extend([col for col in ['latitude', 'longitude'] if col in df_sin_duplicados.columns]) #################################################################
    
    # 5. Filtrar horario laboral (ventana operativa 05:00-22:00) y eliminar columnas.
    #    work_hours_filter=False conserva TODAS las horas (p.ej. para ver salidas/retornos
    #    a depósito de madrugada/noche que este filtro recorta).
    if work_hours_filter:
        df_lazy = df_lazy.filter(
            (pl.col('time').dt.time() > time(5, 00, 0)) &
            (pl.col('time').dt.time() < time(22, 00, 0))
        )
    df_cleaned = (
        df_lazy
        .drop(cols_to_remove)
        .sort('time')  # Ordenar por tiempo
        .collect()  # Ejecuta todo el pipeline de una vez
    )
    
    if verbose:
        print(f"\n{'─' * 60}")
        print("RESULTADOS FINALES")
        print(f"{'─' * 60}")
        print(f"Filas iniciales:                {df.height:,}")
        print(f"Duplicados eliminados:          {duplicados_eliminados:,}")
        print(f"Después de eliminar duplicados: {df_sin_duplicados.height:,}")
        print(f"Filas después de limpieza:      {df_cleaned.height:,}")
        print(f"Total de filas eliminadas:      {df.height - df_cleaned.height:,}")
        print(f"Porcentaje retenido:            {(df_cleaned.height / df.height * 100):.2f}%")
        
        print("\nDesglose de eliminaciones:")
        print(f"  - Por duplicados:             {duplicados_eliminados:,}")
        print(f"  - Por filtros de limpieza:    {df_sin_duplicados.height - df_cleaned.height:,}")
        
        print("\nColumnas finales:")
        print(df_cleaned.columns)
        
        print("\nEjemplo de datos (primeras 3 filas):")
        if df_cleaned.height > 0:
            sample = df_cleaned.select(['time', 'route', 'bus_code', 'speed']).head(3)
            print(sample)
        
        print(f"\n{'=' * 60}\n")
    
    return df_cleaned


def _strip_prefix(codigo: str) -> str:
    """
    Extrae el sufijo de un código de parada eliminando todo hasta el último '-'.

    Parameters
    ----------
    codigo : str
        Código completo de parada, e.g. ``"1601-FQAE"`` o ``"FQAE"``.

    Returns
    -------
    str
        Sufijo del código, e.g. ``"FQAE"``.

    Examples
    --------
    >>> _strip_prefix("1601-FQAE")
    'FQAE'
    >>> _strip_prefix("FQAE")
    'FQAE'
    >>> _strip_prefix("101-82")
    '82'
    """
    return codigo.rsplit("-", 1)[-1] if "-" in codigo else codigo


def dp_build_segments(
    df_poly: pl.DataFrame,
    df_stops: pl.DataFrame,
    codigo_col: str = "auto",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    orden_col: str = "punto_orden",
    strip_prefix: bool = True,
    verbose: bool = False,
) -> pl.DataFrame:
    """
    Proyecta las paradas de df_stops sobre la polilínea de df_poly y divide
    la polilínea en segmentos entre paradas consecutivas (orden cíclico).

    La columna ``stop`` del resultado contiene por defecto solo el **sufijo**
    del código (parte después del último ``-``), compatible con el parámetro
    ``route_prefix`` de ``resolve_stop_codes``. Para conservar el código
    completo usar ``strip_prefix=False``.

    Parameters
    ----------
    df_poly : pl.DataFrame
        Polilínea de la ruta. Debe contener columnas de latitud, longitud y
        orden de punto. Los tipos se castean internamente a Float64.
    df_stops : pl.DataFrame
        Paradas de la ruta. Debe contener ``codigo_col``, ``lat_col`` y
        ``lon_col``. El orden de filas no importa; se reordenan por posición
        proyectada en la polilínea.
    codigo_col : str, optional
        Columna de código de parada en ``df_stops``. Default: ``"codigo"``.
    lat_col : str, optional
        Columna de latitud en ambos DataFrames. Default: ``"latitude"``.
    lon_col : str, optional
        Columna de longitud en ambos DataFrames. Default: ``"longitude"``.
    orden_col : str, optional
        Columna de orden de puntos en ``df_poly``. Default: ``"punto_orden"``.
    strip_prefix : bool, optional
        Si ``True`` (default), la columna ``stop`` contendrá solo el sufijo
        del código (e.g. ``"FQAE"`` en vez de ``"1601-FQAE"``).
        Usar ``False`` para conservar el código completo.
    verbose : bool, optional
        Si ``True``, imprime estadísticas del proceso. Default: ``False``.

    Returns
    -------
    pl.DataFrame
        DataFrame con columnas:

        - ``stop``     : str — código de inicio del segmento (sufijo si
          ``strip_prefix=True``, código completo si ``False``).
        - ``polyline`` : list[struct[latitude: f64, longitude: f64]] —
          puntos desde la parada ``stop`` hasta la siguiente (exclusive).

    Examples
    --------
    >>> df_segments = dp_build_segments(
    ...     df_poly=df_poly_1601,
    ...     df_stops=df_stops_1601,
    ...     verbose=True,
    ... )
    [dp_build_segments] Polilínea: 312 puntos, 311 segmentos
    [dp_build_segments] Paradas  : 85
    [dp_build_segments] Segmentos generados: 85
    >>> df_segments["stop"].to_list()[:3]
    ['FQAE', 'PEPM', 'E5TM']
    """

    #──────────────────────────────────────────────────────────────────────────#
    # ── 2.1 Preparar arrays de la polilínea ── #
    #──────────────────────────────────────────────────────────────────────────#
    poly_np    = df_poly.select([
        pl.col(lat_col).cast(pl.Float64),
        pl.col(lon_col).cast(pl.Float64),
    ]).to_numpy()
    poly_orden = df_poly[orden_col].cast(pl.Float64).to_numpy()
    n_poly     = len(poly_np)

    sort_poly  = np.argsort(poly_orden)
    poly_np    = poly_np[sort_poly]

    if verbose:
        print(f"[dp_build_segments] Polilínea: {n_poly} puntos, {n_poly - 1} segmentos")

    #──────────────────────────────────────────────────────────────────────────#
    # ── 2.2 Auto-detectar columnas y preparar paradas ── #
    #──────────────────────────────────────────────────────────────────────────#
    if codigo_col == "auto":
        for _candidate in ("codigo", "code", "stop_code", "id"):
            if _candidate in df_stops.columns:
                codigo_col = _candidate
                break
        else:
            raise ValueError(
                f"dp_build_segments: no se encontró columna de código en df_stops. "
                f"Columnas disponibles: {df_stops.columns}."
            )

    # Detectar columna de orden de paradas. Si existe, es la fuente de verdad
    # del orden de visita (respeta duplicados: misma parada con distintos orders).
    order_col = None
    for _cand in ("stop_order", "order", "orden"):
        if _cand in df_stops.columns:
            order_col = _cand
            break

    if order_col is not None:
        df_stops_sorted = df_stops.sort(order_col)
    else:
        df_stops_sorted = df_stops

    stops_np = df_stops_sorted.select([
        pl.col(lat_col).cast(pl.Float64),
        pl.col(lon_col).cast(pl.Float64),
    ]).to_numpy()
    codigos  = df_stops_sorted[codigo_col].to_numpy()
    n_stops  = len(stops_np)
    n_segs   = n_poly - 1

    if verbose:
        print(f"[dp_build_segments] Paradas  : {n_stops}"
              + (f" (ordenadas por {order_col!r})" if order_col else ""))

    #──────────────────────────────────────────────────────────────────────────#
    # ── 2.3 Proyección monotónica de paradas sobre la polilínea ── #
    #──────────────────────────────────────────────────────────────────────────#
    # Para cada parada (en orden de visita), encontrar el segmento de polilínea
    # más cercano CON proj_orden ≥ proj_orden de la parada anterior. Esto:
    #   - respeta el orden de visita (stop_order).
    #   - permite que un mismo `code` caiga dos veces en la polilínea (ida/vuelta).
    #   - no obliga a una elección global perpendicular que rompería el orden.

    # Pre-calcular vectores y normas de cada segmento de polilínea
    A_all  = poly_np[:n_segs]              # inicio de cada segmento
    B_all  = poly_np[1:]                   # fin de cada segmento
    AB     = B_all - A_all                 # vector segmento
    AB_sq  = np.sum(AB ** 2, axis=1)
    AB_sq  = np.where(AB_sq < 1e-20, 1e-20, AB_sq)

    proj_orden  = np.zeros(n_stops, dtype=np.float64)
    best_seg    = np.zeros(n_stops, dtype=np.int64)
    best_t      = np.zeros(n_stops, dtype=np.float64)
    snap_coords = np.zeros((n_stops, 2), dtype=np.float64)

    # Ventana de búsqueda: empieza desde el último best_seg encontrado para
    # mantener monotonicidad. La ventana se extiende hacia adelante por
    # toda la polilínea; si la parada está más allá del fin, se permite
    # buscar desde el inicio (rutas circulares).
    cursor = 0   # índice mínimo de segmento permitido para la siguiente parada

    for i in range(n_stops):
        p_lat, p_lon = stops_np[i, 0], stops_np[i, 1]
        if not (np.isfinite(p_lat) and np.isfinite(p_lon)):
            # Parada con coordenadas inválidas: hereda la posición anterior.
            proj_orden[i]  = proj_orden[i - 1] if i > 0 else 0.0
            best_seg[i]    = cursor
            best_t[i]      = 0.0
            snap_coords[i] = poly_np[cursor]
            continue

        # Calcular proyección perpendicular sobre cada segmento desde cursor
        AP    = np.array([p_lat - A_all[cursor:, 0], p_lon - A_all[cursor:, 1]]).T
        t     = np.clip(np.sum(AP * AB[cursor:], axis=1) / AB_sq[cursor:], 0.0, 1.0)
        snaps = A_all[cursor:] + t[:, np.newaxis] * AB[cursor:]
        diff  = np.array([p_lat - snaps[:, 0], p_lon - snaps[:, 1]]).T
        dists = np.sqrt(np.sum(diff ** 2, axis=1))

        rel_idx = int(np.argmin(dists))
        seg_idx = cursor + rel_idx

        best_seg[i]    = seg_idx
        best_t[i]      = float(t[rel_idx])
        snap_coords[i] = snaps[rel_idx]
        proj_orden[i]  = float(seg_idx) + float(t[rel_idx])

        # Avanzar cursor SOLO si la parada no quedó pegada al inicio del
        # segmento actual (esto permitiría que la siguiente parada caiga
        # en el mismo segmento si está muy cerca).
        cursor = seg_idx

    #──────────────────────────────────────────────────────────────────────────#
    # ── 2.4 Construir segmentos: uno por parada (en el orden de visita) ── #
    #──────────────────────────────────────────────────────────────────────────#
    # Para cada parada i, el segmento va desde snap_coords[i] hasta snap_coords[i+1],
    # pasando por todos los vértices de polilínea estrictamente entre ambos.
    # Para la última parada (cíclica), se cierra hacia snap_coords[0] dando la vuelta
    # por el final de la polilínea + el inicio.
    segments: list = []
    for i in range(n_stops):
        j = (i + 1) % n_stops  # parada destino

        start_lat, start_lon = float(snap_coords[i, 0]), float(snap_coords[i, 1])
        end_lat,   end_lon   = float(snap_coords[j, 0]), float(snap_coords[j, 1])

        seg_a = int(best_seg[i])  # segmento donde cae parada origen
        seg_b = int(best_seg[j])  # segmento donde cae parada destino

        pts = [{"latitude": start_lat, "longitude": start_lon}]

        if j > i:
            # Caso normal: avanzar de seg_a+1 a seg_b (vértices interiores)
            if seg_b > seg_a:
                for k in range(seg_a + 1, seg_b + 1):
                    pts.append({
                        "latitude":  float(poly_np[k, 0]),
                        "longitude": float(poly_np[k, 1]),
                    })
            elif seg_b == seg_a and best_t[j] < best_t[i]:
                # Misma segmento de polilínea pero la parada destino quedó
                # antes en t: tomar el camino largo (raro pero posible).
                for k in range(seg_a + 1, n_poly):
                    pts.append({
                        "latitude":  float(poly_np[k, 0]),
                        "longitude": float(poly_np[k, 1]),
                    })
                for k in range(0, seg_b + 1):
                    pts.append({
                        "latitude":  float(poly_np[k, 0]),
                        "longitude": float(poly_np[k, 1]),
                    })
        else:
            # Cierre del loop: i = última parada, j = primera. Recorrer del
            # segmento seg_a+1 hasta el final de la polilínea y luego del
            # inicio hasta seg_b.
            for k in range(seg_a + 1, n_poly):
                pts.append({
                    "latitude":  float(poly_np[k, 0]),
                    "longitude": float(poly_np[k, 1]),
                })
            for k in range(0, seg_b + 1):
                pts.append({
                    "latitude":  float(poly_np[k, 0]),
                    "longitude": float(poly_np[k, 1]),
                })

        # Cerrar con el snap de la parada destino
        pts.append({"latitude": end_lat, "longitude": end_lon})

        # Etiqueta: código de la parada DESTINO (a la que llega el segmento)
        cod_dst = codigos[j]
        cod_dst = str(cod_dst) if cod_dst is not None else ""
        stop_key = _strip_prefix(cod_dst) if strip_prefix and cod_dst else cod_dst

        segments.append((stop_key, pts))

    #──────────────────────────────────────────────────────────────────────────#
    # ── 2.5 Construir DataFrame resultado ── #
    #──────────────────────────────────────────────────────────────────────────#
    df_result = pl.DataFrame({
        "stop":     [s[0] for s in segments],
        "polyline": [s[1] for s in segments],
    })

    if verbose:
        print(f"[dp_build_segments] Segmentos generados: {df_result.height}")
        ejemplo = df_result["stop"].to_list()[:5]
        label   = "sufijo" if strip_prefix else "código completo"
        print(f"[dp_build_segments] Ejemplo stop ({label}): {ejemplo}")

    return df_result


def get_dic_poly(
    file_path: str = '../data/02_interim/polylines.geojson',
    libreria: Literal['pandas', 'polars'] = 'pandas',
    incluir_metadata: bool = False
) -> Dict[str, Union[pd.DataFrame, pl.DataFrame]]:
    """
    Lee un archivo JSON/GeoJSON combinado y lo convierte en un diccionario de DataFrames.
    
    Args:
        file_path: Ruta al archivo combinado (.json o .geojson)
        libreria: 'pandas' o 'polars' para elegir el tipo de DataFrame
        incluir_metadata: Si True, incluye columnas adicionales con información de la geometría
    
    Returns:
        Dict[str, DataFrame]: Diccionario con ID como llave y DataFrame con coordenadas como valor
    """
    
    # Verificar que el archivo existe
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No se encuentra el archivo: {file_path}")
    
    # Leer el archivo JSON
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Detectar el formato del archivo
    if 'type' in data and data['type'] == 'FeatureCollection':
        # Es GeoJSON FeatureCollection
        return _procesar_featurecollection(data, libreria, incluir_metadata)
    else:
        # Es JSON simple (diccionario con IDs)
        return _procesar_json_simple(data, libreria, incluir_metadata)


def _procesar_json_simple(
    data: dict,
    libreria: str,
    incluir_metadata: bool
) -> Dict[str, Union[pd.DataFrame, pl.DataFrame]]:
    """
    Procesa formato JSON simple.

    Soporta tres estructuras:
      1. Plana:   {route_id: {type:'LineString', coordinates:[...]}}
      2. Anidada: {operator_id: {route_id: {type:'LineString', coordinates:[...]}}}
      3. Con clave polyline:
                  {operator_id: {route_id: {polyline: {type:'LineString', coordinates:[...]}}}}
    En los casos 2 y 3 el route_id interior se usa como clave del resultado.
    """
    dataframes_dict = {}

    for id_key, val in data.items():
        if not isinstance(val, dict):
            continue

        # Caso 1: LineString directo en nivel raiz
        if val.get('type') == 'LineString':
            df = _crear_dataframe_desde_coordenadas(
                val['coordinates'], id_key, libreria, incluir_metadata
            )
            dataframes_dict[str(id_key)] = df

        # Casos 2 y 3: nivel raiz es operador, valores son rutas
        else:
            for route_id, geom in val.items():
                if not isinstance(geom, dict):
                    continue
                if geom.get('type') == 'LineString':
                    coordinates = geom['coordinates']
                elif isinstance(geom.get('polyline'), dict) and geom['polyline'].get('type') == 'LineString':
                    coordinates = geom['polyline']['coordinates']
                else:
                    continue
                df = _crear_dataframe_desde_coordenadas(
                    coordinates, route_id, libreria, incluir_metadata
                )
                dataframes_dict[str(route_id)] = df

    return dataframes_dict


def _procesar_featurecollection(
    data: dict,
    libreria: str,
    incluir_metadata: bool
) -> Dict[str, Union[pd.DataFrame, pl.DataFrame]]:
    """
    Procesa formato GeoJSON FeatureCollection.

    Soporta dos estructuras:
      1. FeatureCollection estándar: lista de features con geometry.type == 'LineString'.
      2. Formato anidado del pipeline: {operator: {route_id: {type:'LineString', ...}, ...}, ...}
    """
    dataframes_dict = {}

    features = data.get('features', [])

    # ── Formato anidado: {operator: {route_id: LineString, ...}, ...} ──────────
    if not features:
        for operator_key, operator_val in data.items():
            if operator_key in ('type', 'crs', 'bbox', 'name'):
                continue
            if not isinstance(operator_val, dict):
                continue
            for route_id, geom in operator_val.items():
                if not isinstance(geom, dict):
                    continue
                # Puede ser el LineString directamente o bajo clave 'polyline'
                if geom.get('type') == 'LineString':
                    coordinates = geom['coordinates']
                elif isinstance(geom.get('polyline'), dict) and geom['polyline'].get('type') == 'LineString':
                    coordinates = geom['polyline']['coordinates']
                else:
                    continue
                df = _crear_dataframe_desde_coordenadas(
                    coordinates, route_id, libreria, incluir_metadata
                )
                dataframes_dict[str(route_id)] = df
        return dataframes_dict

    # ── Formato FeatureCollection estándar ───────────────────────────────────
    for feature in features:
        geometry = feature.get('geometry', {})

        if geometry.get('type') == 'LineString':
            coordinates = geometry['coordinates']

            # Obtener ID (puede estar en 'id' o en properties)
            feature_id = feature.get('id')
            if feature_id is None:
                feature_id = feature.get('properties', {}).get('id', str(len(dataframes_dict)))
            
            # Crear DataFrame con las coordenadas
            df = _crear_dataframe_desde_coordenadas(
                coordinates, 
                feature_id, 
                libreria, 
                incluir_metadata,
                properties=feature.get('properties', {})
            )
            dataframes_dict[str(feature_id)] = df
    
    return dataframes_dict


def _crear_dataframe_desde_coordenadas(
    coordinates: list,
    id_key: str,
    libreria: str,
    incluir_metadata: bool,
    properties: dict = None
) -> Union[pd.DataFrame, pl.DataFrame]:
    """Crea un DataFrame a partir de una lista de coordenadas"""
    
    # Separar coordenadas en listas
    longitudes = [coord[0] for coord in coordinates]
    latitudes = [coord[1] for coord in coordinates]
    
    # Crear datos base
    data = {
        'longitude': longitudes,
        'latitude': latitudes,
        'punto_orden': list(range(len(coordinates)))  # Para mantener el orden
    }
    
    # Agregar metadata si se solicita
    if incluir_metadata:
        data['id'] = id_key
        data['num_puntos'] = len(coordinates)
        
        # Agregar propiedades adicionales (para GeoJSON)
        if properties:
            for key, value in properties.items():
                if key not in data:  # Evitar sobrescribir columnas existentes
                    data[f'prop_{key}'] = [value] * len(coordinates)
    
    # Crear DataFrame según la librería elegida
    if libreria == 'pandas':
        return pd.DataFrame(data)
    else:  # polars
        return pl.DataFrame(data)