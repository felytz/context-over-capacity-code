"""
step_01_load_data.py
====================
PASO 1 · Carga de datos crudos y construcción de geometrías base.

Sub-pasos:
    1.1  df_rec_route_raw   — registros GPS crudos de la BD (cache parquet).
    1.2  df_rec_route_clean — registros limpios (drop NaN, fix tipos).
    1.3  df_stop_route      — paradas de la ruta (lat, lon, codigo, order).
    1.4  df_segments_route  — segmentos precalculados desde ``segments.geojson``
                              (raw data nuevo). Fallback a dp_build_segments
                              si el archivo no existe.

v2 (mayo 2026): el sub-paso 1.4 lee los segmentos directamente del GeoJSON
raw en lugar de calcularlos con ``dp_build_segments``. Esto elimina el bug
de longitud 2× y la dependencia de proyección de paradas sobre polilínea.

v3 (jun 2026): ``segments.geojson`` puede venir como FeatureCollection plana
(features con ``properties.{line, code, start_stop, end_stop}``) acompañada de
``segments_route.json`` (``{route_id: [{segment_code, segment_order}]}``) para
el orden. El loader autodetecta este formato y el anidado legacy.

Si los parquets cacheados existen, se cargan; si no, se generan y se escriben
a disco para corridas posteriores.

Usage
-----
    from step_01_load_data import load_data
    df_rec_clean, df_stops, df_segments = load_data(verbose=True)
"""

from __future__ import annotations

# Ejecutable directo: registra func/ en el path para hallar config (que a su vez registra lib/, steps/ y audits/).
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from typing import Tuple

import polars as pl

import config
from config import PATHS, INT_ROUTE, INT_ROUTE_REC, INT_ROUTE_STOP, YEAR, MONTH1


# ──────────────────────────────────────────────────────────────────
# 1.0 · GUARD: ruta de los DATOS vs ruta de las PARADAS/SEGMENTOS
# ──────────────────────────────────────────────────────────────────

def assert_data_route_consistency(
    df_raw: pl.DataFrame,
    verbose: bool = False,
) -> None:
    """
    Verifica que el código de ruta de los DATOS (``group_id`` de los pings GPS)
    sea consistente con la ruta configurada para la geometría (paradas,
    segmentos y polilínea, que se cargan por ``INT_ROUTE_STOP``).

    Bug que previene
    ----------------
    Procesar los pings de una ruta (``group_id=X``) contra la geometría de OTRA
    ruta (``INT_ROUTE_STOP=Y``). Pasó en jun 2026: el extracto de la **1701**
    estaba etiquetado como **101**, así que los pings de la 1701 se
    map-matcheaban contra las paradas/segmentos de la 101. El ``group_id`` del
    parquet raw es la única señal de identidad real de los datos, por eso la
    validación vive aquí (paso 01, donde el raw aún conserva esa columna; el
    paso 02 la elimina en ``dc_records``).

    Reglas (alineadas con lo pedido: "el código de los datos y el de las
    paradas/polilíneas son el mismo, o al menos uno de los 3 ints de config")
    --------------------------------------------------------------------------
    - **ERROR duro** si algún ``group_id`` NO está en
      ``{INT_ROUTE, INT_ROUTE_STOP, INT_ROUTE_REC}``: los datos son de una ruta
      que la config no referencia en absoluto (casi seguro el archivo raw
      equivocado).
    - **WARNING fuerte** (siempre, aunque ``verbose=False``) si el ``group_id``
      está dentro de los 3 ints pero NO coincide con ``INT_ROUTE_STOP`` (la ruta
      cuya geometría se usa para el map-match). Es la forma exacta del footgun;
      se deja como warning y no error porque un ``INT_ROUTE_REC`` distinto puede
      ser un recorrido legítimo de la BD.
    """
    if "group_id" not in df_raw.columns:
        if verbose:
            print("  ⚠ raw sin columna 'group_id'; no se valida la ruta de los datos.")
        return

    groups = df_raw.get_column("group_id").drop_nulls().unique().to_list()
    if not groups:
        return

    try:
        groups_int = sorted({int(g) for g in groups})
    except (TypeError, ValueError):
        if verbose:
            print(f"  ⚠ group_id no numérico ({groups}); validación de ruta omitida.")
        return

    valid = {INT_ROUTE, INT_ROUTE_STOP, INT_ROUTE_REC}
    outside = [g for g in groups_int if g not in valid]
    if outside:
        raise ValueError(
            f"Inconsistencia ruta ↔ datos: el raw tiene group_id={groups_int}, "
            f"pero {outside} no está en la config "
            f"(INT_ROUTE={INT_ROUTE}, INT_ROUTE_STOP={INT_ROUTE_STOP}, "
            f"INT_ROUTE_REC={INT_ROUTE_REC}). Estás procesando los pings de una "
            f"ruta contra las paradas/segmentos de otra. Revisa config.py y el "
            f"archivo raw ({PATHS.df_rec_raw.name})."
        )

    if any(g != INT_ROUTE_STOP for g in groups_int):
        # Dentro de los 3 ints, pero la geometría es de otra ruta: footgun.
        print(
            f"  ⚠⚠ ADVERTENCIA ruta ↔ datos: los pings tienen group_id={groups_int} "
            f"pero las paradas/segmentos/polilínea se cargan para "
            f"INT_ROUTE_STOP={INT_ROUTE_STOP}. Coincide con la config solo vía "
            f"INT_ROUTE={INT_ROUTE} / INT_ROUTE_REC={INT_ROUTE_REC}. Si NO es un "
            f"recorrido (REC) intencional, los datos y la geometría son de rutas "
            f"distintas — revisa config.py."
        )
    elif verbose:
        print(f"  ✔ ruta consistente: datos group_id={groups_int} == "
              f"INT_ROUTE_STOP={INT_ROUTE_STOP} (paradas/segmentos)")


# ──────────────────────────────────────────────────────────────────
# 1.1 · REGISTROS GPS CRUDOS
# ──────────────────────────────────────────────────────────────────

# Columnas del crudo que usa el pipeline: las de dc_records (duplicados por
# (code, utc_datetime) con desempate por id_registro, filtro disabled==0, hora
# local, posición, velocidad y rumbo) y group_id para la verificación de ruta.
# Se lee solo esto: plate_number, brand, id, stop, no_passengers, etc. no entran.
RAW_COLUMNS = ("id_registro", "code", "latitude", "longitude", "speed", "cape",
               "utc_datetime", "disabled", "group_id", "time")


def load_records_raw(verbose: bool = False) -> pl.DataFrame:
    """
    Carga `df_rec_route_raw` (solo RAW_COLUMNS). Si el parquet existe y tiene
    filas, lo lee; si no existe, o existe pero está vacío (0 filas), descarga
    desde la BD usando `aux_f.get_db_rq_from_to` y reescribe el cache sin la
    columna de matrícula.

    Returns
    -------
    pl.DataFrame
        Registros GPS crudos (1 fila por ping).
    """
    PATHS.ensure_dirs()
    cache = PATHS.df_rec_raw

    if cache.is_file():
        # Guard de no-vacío: un cache de 0 bytes / corrupto / 0 filas no sirve y
        # haría crashear a polars. Lo eliminamos vía código (política del
        # proyecto: no dejar dataframes vacíos guardados) y re-descargamos.
        from _lib_data import load_df
        try:
            cols = [c for c in pl.read_parquet_schema(cache) if c in RAW_COLUMNS]
            df = load_df(cache, columns=cols)   # lanza si 0 bytes / 0 filas / corrupto
        except Exception as e:
            if verbose:
                print(f"  ⚠ Cache {cache.name} vacío/corrupto ({type(e).__name__}); "
                      f"se elimina y se re-descarga desde BD …")
            try:
                cache.unlink()
            except OSError:
                pass
        else:
            if verbose:
                print(f"  ✔ df_rec_raw cargado de cache: {cache.name} "
                      f"({df.height:,} filas)")
            assert_data_route_consistency(df, verbose=verbose)
            return df

    if verbose:
        print(f"  ⤓ Descargando df_rec_raw desde BD …")
    from _lib_data import get_db_rq_from_to
    df = get_db_rq_from_to(
        fecha_inicio=f"{YEAR}-{MONTH1}-01",
        fecha_final=f"{int(YEAR) + 1}-01-01",
        int_route_rec=INT_ROUTE_REC,
    ).drop("plate_number", strict=False)   # como el crudo publicado
    from _lib_data import save_df
    save_df(df, cache)
    if verbose:
        print(f"  ✔ df_rec_raw guardado: {cache} ({df.height:,} filas)")
    df = df.select([c for c in df.columns if c in RAW_COLUMNS])
    assert_data_route_consistency(df, verbose=verbose)
    return df


# ──────────────────────────────────────────────────────────────────
# 1.2 · REGISTROS LIMPIOS
# ──────────────────────────────────────────────────────────────────

def load_records_clean(
    df_raw: pl.DataFrame | None = None,
    verbose: bool = False,
    force: bool = False,
    work_hours_filter: bool = True,
) -> pl.DataFrame:
    """
    Aplica `aux_f.dc_records` (limpia tipos, fechas, drop columnas auxiliares)
    Y restringe los registros al periodo válido de análisis (ene–ago 2024,
    Opción A — ver `config.ANALYSIS_START` / `ANALYSIS_END`).

    El filtro temporal se aplica AQUÍ, en el punto más temprano del pipeline,
    de modo que todos los pasos posteriores operan sobre la ventana válida.

    Parameters
    ----------
    df_raw : pl.DataFrame, opcional
        Si se provee, evita el round-trip por disco.
    force : bool
        Si True, ignora el cache y recomputa df_clean desde df_raw.
    work_hours_filter : bool
        Si True (default) aplica el filtro de ventana operativa (hora local
        06:00–22:00) en `dc_records`. Si False, conserva TODAS las horas y
        cachea la variante en un parquet aparte (`*_allhours.parquet`) para no
        pisar el df_clean canónico que consume el resto del pipeline.
    """
    from config import (ANALYSIS_START, ANALYSIS_END,
                        filter_to_analysis_window)

    cache = PATHS.df_rec_clean
    if not work_hours_filter:
        cache = cache.with_name(f"{cache.stem}_allhours{cache.suffix}")

    if cache.is_file() and not force:
        # Guard de no-vacío: cache de 0 bytes / corrupto / 0 filas → eliminar
        # vía código y recomputar desde df_raw.
        from _lib_data import load_df
        try:
            df = load_df(cache)
        except Exception as e:
            if verbose:
                print(f"  ⚠ Cache {cache.name} vacío/corrupto ({type(e).__name__}); "
                      f"se elimina y se recomputa.")
            try:
                cache.unlink()
            except OSError:
                pass
        else:
            if verbose:
                print(f"  ✔ df_rec_clean cargado de cache: {cache.name} "
                      f"({df.height:,} filas)")
            return df

    if df_raw is None:
        df_raw = load_records_raw(verbose=verbose)

    n_raw = df_raw.height

    from _lib_data import dc_records
    df = df_raw.drop("group_id", strict=False)
    df = dc_records(df, work_hours_filter=work_hours_filter)
    df = df.drop("last_stop", strict=False)
    if verbose and not work_hours_filter:
        print("  ⚠ ventana operativa DESACTIVADA (work_hours_filter=False): "
              "se conservan todas las horas (cache aparte *_allhours).")

    # ── Restricción al periodo válido (Opción A) ──
    n_pre_filter = df.height
    df = filter_to_analysis_window(df, time_col="time")
    n_post_filter = df.height

    if verbose:
        n_dropped = n_pre_filter - n_post_filter
        pct       = 100 * n_dropped / n_pre_filter if n_pre_filter else 0
        print(f"  ⊘ Filtro temporal {ANALYSIS_START} ≤ time < {ANALYSIS_END}")
        print(f"      Antes : {n_pre_filter:,}")
        print(f"      Después: {n_post_filter:,}  "
              f"(descartados: {n_dropped:,} · {pct:.1f}%)")

    from _lib_data import save_df
    save_df(df, cache)
    if verbose:
        print(f"  ✔ df_rec_clean guardado: {cache} ({df.height:,} filas)")
    return df


# ──────────────────────────────────────────────────────────────────
# 1.3 · PARADAS DE LA RUTA
# ──────────────────────────────────────────────────────────────────

def load_stops(verbose: bool = False) -> pl.DataFrame:
    """
    Carga las paradas de la ruta desde el nuevo formato (stops.json +
    stops_routes.json). Selecciona solo la ruta `INT_ROUTE_STOP` y deja
    las columnas mínimas necesarias.

    Columnas resultantes:
        code, latitude, longitude, codigo_original, stop_order
    """
    from _lib_data import get_stops
    dict_stops = get_stops(
        stops_path        = str(PATHS.stops_json),
        stops_routes_path = str(PATHS.stops_routes_json),
        focus_route       = INT_ROUTE_STOP,
        verbose           = verbose,
    )
    if INT_ROUTE_STOP not in dict_stops:
        raise KeyError(
            f"Ruta {INT_ROUTE_STOP} no encontrada en stops_routes.json. "
            f"Rutas disponibles: {sorted(dict_stops.keys())[:10]}…"
        )
    df = dict_stops[INT_ROUTE_STOP]
    df = df.select(
        "code", "latitude", "longitude", "codigo_original", "stop_order"
    ).clone()
    if verbose:
        print(f"  ✔ df_stop_route: {df.height} paradas")
    return df


# ──────────────────────────────────────────────────────────────────
# 1.4 · SEGMENTOS (precalculados en segments.geojson)
# ──────────────────────────────────────────────────────────────────

def _coords_to_points(coords: list) -> list[dict]:
    """Convierte coordenadas GeoJSON ``[[lng, lat], ...]`` a una lista de
    dicts ``{"latitude": float, "longitude": float}`` (schema interno)."""
    return [
        {"latitude": float(c[1]), "longitude": float(c[0])}
        for c in coords
        if isinstance(c, (list, tuple)) and len(c) >= 2
    ]


def _segments_df_from_rows(
    rows_stop: list[str],
    rows_polyline: list[list],
    verbose: bool = False,
) -> pl.DataFrame:
    """Construye el DataFrame final ``{stop, polyline}`` y, si ``verbose``,
    reporta la longitud total de la polilínea para verificación."""
    df = pl.DataFrame({
        "stop":     rows_stop,
        "polyline": rows_polyline,
    })

    if verbose:
        from _lib_turn_hex import _haversine_m
        total = 0.0
        for pts_raw in rows_polyline:
            pts = [(p["latitude"], p["longitude"]) for p in pts_raw]
            total += sum(
                _haversine_m(pts[i - 1][0], pts[i - 1][1], pts[i][0], pts[i][1])
                for i in range(1, len(pts))
            )
        print(f"  [segments.geojson] ✔ {len(rows_stop)} segmentos "
              f"| total polilínea: {total:,.0f} m ({total / 1000:.2f} km)")

    return df


def _load_segments_featurecollection(
    data: dict,
    verbose: bool = False,
) -> pl.DataFrame:
    """
    Handler del formato FeatureCollection plano (v3)::

        {"type": "FeatureCollection", "features": [
            {"type": "Feature",
             "properties": {"line": "101", "code": "QMSS__EJ5E",
                            "start_stop": "QMSS", "end_stop": "EJ5E",
                            "name": "QMSS → EJ5E"},
             "geometry": {"type": "LineString", "coordinates": [[lng, lat], ...]}}
        ]}

    El orden de los segmentos por ruta lo da ``segments_route.json``
    (``{route_id: [{segment_code, segment_order}, ...]}``):

        - Si existe, se recorre la lista ordenada de la ruta ``INT_ROUTE_STOP``
          y se resuelve cada ``segment_code`` contra el índice de features.
        - Si no existe, se hace fallback: se filtran las features por
          ``properties.line == INT_ROUTE_STOP`` en el orden del archivo.

    La etiqueta ``stop`` de cada segmento es el ``end_stop`` (parada destino),
    consistente con el resto del pipeline.
    """
    import json

    key_str = str(INT_ROUTE_STOP)

    # ── Índice code → feature ──────────────────────────────────────────────
    features = data.get("features", []) or []
    by_code: Dict[str, dict] = {}
    for feat in features:
        if not isinstance(feat, dict):
            continue
        props = feat.get("properties", {}) or {}
        code = props.get("code") or props.get("segment_code")
        if code is not None:
            by_code[str(code)] = feat

    if verbose:
        print(f"  [segments.geojson] FeatureCollection: "
              f"{len(features)} features · {len(by_code)} indexadas por code")

    # ── Determinar la secuencia ordenada de segment_codes de la ruta ───────
    ordered_codes: list[str] = []
    seg_routes_path = PATHS.segments_route_json

    if seg_routes_path.is_file():
        with open(seg_routes_path, "r", encoding="utf-8") as f:
            seg_routes = json.load(f)

        # Localizar la entrada de la ruta (la clave puede ser "101" o contenerla)
        route_segs = seg_routes.get(key_str)
        if route_segs is None:
            for rk, rv in seg_routes.items():
                if key_str in str(rk):
                    route_segs = rv
                    break

        if route_segs:
            def _order_key(entry: dict):
                so = entry.get("segment_order")
                if so is None:
                    so = entry.get("order")
                try:
                    return int(so)
                except (ValueError, TypeError):
                    return 10**9  # sin orden válido → al final, en orden de aparición

            for entry in sorted(route_segs, key=_order_key):
                if not isinstance(entry, dict):
                    continue
                code = entry.get("segment_code") or entry.get("code")
                if code is not None:
                    ordered_codes.append(str(code))

            if verbose:
                print(f"  [segments.geojson] segments_route.json → ruta {key_str}: "
                      f"{len(ordered_codes)} segmentos ordenados")
        elif verbose:
            print(f"  ⚠ segments_route.json no contiene la ruta {key_str}; "
                  f"usando fallback por properties.line")

    # ── Fallback: filtrar features por properties.line (orden del archivo) ──
    if not ordered_codes:
        for feat in features:
            props = feat.get("properties", {}) or {}
            line = props.get("line")
            if line is not None and key_str in str(line):
                code = props.get("code") or props.get("segment_code")
                if code is not None:
                    ordered_codes.append(str(code))
        if verbose:
            print(f"  [segments.geojson] fallback por line=={key_str}: "
                  f"{len(ordered_codes)} segmentos")

    if not ordered_codes:
        raise KeyError(
            f"No se encontraron segmentos para la ruta {key_str} en "
            f"segments.geojson (ni vía segments_route.json ni por properties.line). "
            f"Líneas presentes en features: "
            f"{sorted({str((f.get('properties') or {}).get('line')) for f in features})[:10]}"
        )

    # ── Construir filas en orden, resolviendo cada code contra el índice ───
    rows_stop:     list[str]  = []
    rows_polyline: list[list] = []
    missing = 0

    for code in ordered_codes:
        feat = by_code.get(code)
        if feat is None:
            missing += 1
            continue
        props = feat.get("properties", {}) or {}
        # Parada destino; si falta, derivarla del code "AAAA__BBBB"
        stop_code = props.get("end_stop")
        if not stop_code and "__" in code:
            stop_code = code.split("__")[-1]
        rows_stop.append(str(stop_code if stop_code is not None else code))

        geom = feat.get("geometry", {}) or {}
        coords = geom.get("coordinates", []) if isinstance(geom, dict) else []
        rows_polyline.append(_coords_to_points(coords))

    if verbose and missing:
        print(f"  ⚠ {missing} segment_code de segments_route.json no estaban "
              f"en segments.geojson (omitidos)")

    if not rows_stop:
        raise KeyError(
            f"Ningún segment_code de la ruta {key_str} se resolvió contra las "
            f"features de segments.geojson ({missing} sin geometría)."
        )

    return _segments_df_from_rows(rows_stop, rows_polyline, verbose=verbose)


def _load_segments_nested(
    data: dict,
    verbose: bool = False,
) -> pl.DataFrame:
    """
    Handler del formato anidado legacy (v2)::

        {operator: {lineKey: {segments: [{code, start_stop, end_stop,
                                          polyline: LineString}]}}}

    Selecciona la lineKey cuyo nombre contiene ``INT_ROUTE_STOP``.
    """
    key_str = str(INT_ROUTE_STOP)
    segments_raw: list | None = None

    for operator_id, routes in data.items():
        if not isinstance(routes, dict):
            continue
        for line_key, line_data in routes.items():
            if key_str in str(line_key) and isinstance(line_data, dict):
                segs = line_data.get("segments")
                if isinstance(segs, list) and len(segs) > 0:
                    segments_raw = segs
                    if verbose:
                        print(f"  [segments.geojson] operador={operator_id!r} "
                              f"lineKey={line_key!r} → {len(segs)} segmentos")
                    break
        if segments_raw is not None:
            break

    if segments_raw is None:
        raise KeyError(
            f"No se encontró la ruta {key_str} en segments.geojson (formato anidado). "
            f"Operadores/lineKeys disponibles: "
            f"{[(op, list(routes.keys())[:3]) for op, routes in data.items() if isinstance(routes, dict)]}"
        )

    rows_stop:     list[str]  = []
    rows_polyline: list[list] = []

    for seg in segments_raw:
        stop_code = str(seg.get("end_stop", seg.get("code", "")))
        rows_stop.append(stop_code)

        geom = seg.get("polyline", {})
        coords = geom.get("coordinates", []) if isinstance(geom, dict) else []
        rows_polyline.append(_coords_to_points(coords))

    return _segments_df_from_rows(rows_stop, rows_polyline, verbose=verbose)


def _load_segments_from_geojson(verbose: bool = False) -> pl.DataFrame:
    """
    Lee los segmentos precalculados de ``PATHS.segments_geojson`` y devuelve
    un DataFrame ``{stop, polyline}``.

    Autodetecta el formato del archivo:

      * **FeatureCollection plana (v3, preferido)** — features con
        ``properties.{line, code, start_stop, end_stop}`` y geometría
        ``LineString``. El orden por ruta lo da ``segments_route.json``.
      * **Anidado (v2, legacy)** — ``{operator: {lineKey: {segments: [...]}}}``
        con la polilínea bajo la clave ``polyline`` de cada segmento.

    Las coordenadas GeoJSON ``[longitude, latitude]`` se convierten a dicts
    ``{"latitude": float, "longitude": float}`` para el schema interno.

    Returns
    -------
    pl.DataFrame con columnas:
        stop     (str)           : código de la parada DESTINO del segmento
        polyline (List[Struct])  : lista de puntos {"latitude", "longitude"}
    """
    import json
    path = PATHS.segments_geojson

    if not path.is_file():
        raise FileNotFoundError(
            f"segments.geojson no encontrado en {path}. "
            f"Asegúrate de que el archivo raw esté en {PATHS.data_raw}."
        )

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Detección de formato: FeatureCollection vs anidado legacy.
    is_feature_collection = (
        isinstance(data, dict)
        and (data.get("type") == "FeatureCollection" or "features" in data)
    )

    if is_feature_collection:
        return _load_segments_featurecollection(data, verbose=verbose)
    return _load_segments_nested(data, verbose=verbose)


def load_segments(
    df_stops: pl.DataFrame | None = None,
    verbose: bool = False,
) -> pl.DataFrame:
    """
    Carga la geometría de segmentos ``df_segments_route``.

    v2 (mayo 2026): lee directamente de ``segments.geojson`` (raw data nuevo)
    en lugar de calcularlos con ``dp_build_segments``. Esto elimina el bug de
    longitud 2× y la dependencia de proyección de paradas sobre polilínea.

    Si ``segments.geojson`` no existe, cae back al método legacy (dp_build_segments).

    Returns
    -------
    pl.DataFrame con columnas ``stop`` (str) y ``polyline`` (List[Struct]).
    """
    PATHS.ensure_dirs()
    cache = PATHS.df_segments

    if cache.is_file():
        df = pl.read_parquet(cache)
        if verbose:
            print(f"  ✔ df_segments cargado de cache: {cache.name} "
                  f"({df.height} segmentos)")
        return df

    # Intentar cargar desde segments.geojson (fuente preferida)
    if PATHS.segments_geojson.is_file():
        df = _load_segments_from_geojson(verbose=verbose)
        from _lib_data import save_df
        save_df(df, cache)
        if verbose:
            print(f"  ✔ df_segments guardado desde segments.geojson: {cache}")
        return df

    # Fallback: método legacy (dp_build_segments)
    if verbose:
        print(f"  ⚠ segments.geojson no encontrado en {PATHS.segments_geojson}")
        print(f"    Calculando segmentos con dp_build_segments (método legacy)…")

    if df_stops is None:
        df_stops = load_stops(verbose=verbose)

    from _lib_data import get_dic_poly, dp_build_segments
    dic_poly = get_dic_poly(
        file_path        = str(PATHS.polylines_json),
        libreria         = "polars",
        incluir_metadata = False,
    )
    key = str(INT_ROUTE_STOP)
    if key not in dic_poly:
        raise KeyError(
            f"Ruta {key} no encontrada en polylines.json. "
            f"Claves disponibles: {sorted(dic_poly.keys())[:10]}…"
        )
    df_poly = dic_poly[key]
    df = dp_build_segments(df_poly, df_stops)
    from _lib_data import save_df
    save_df(df, cache)
    if verbose:
        print(f"  ✔ df_segments guardado (legacy): {cache} ({df.height} segmentos)")
    return df


# ──────────────────────────────────────────────────────────────────
# Función pública
# ──────────────────────────────────────────────────────────────────

def load_data(
    verbose: bool = False,
    force: bool = False,
    work_hours_filter: bool = True,
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Pipeline completo del PASO 1.

    Parameters
    ----------
    force : bool
        Si True, regenera df_clean ignorando el cache (paso 1.2).
    work_hours_filter : bool
        Si False, df_clean conserva todas las horas (sin filtro de ventana
        operativa 06:00–22:00); se cachea aparte (`*_allhours`). Ver
        `load_records_clean`.

    Returns
    -------
    df_rec_clean : pl.DataFrame
    df_stops     : pl.DataFrame
    df_segments  : pl.DataFrame
    """
    if verbose:
        print("=" * 60)
        print(f"  PASO 1 · Carga de datos · Ruta {INT_ROUTE}")
        print("=" * 60)

    df_rec_clean = load_records_clean(
        verbose=verbose, force=force, work_hours_filter=work_hours_filter)
    df_stops     = load_stops(verbose=verbose)
    df_segments  = load_segments(df_stops=df_stops, verbose=verbose)

    if verbose:
        print("  → Datos listos para PASO 2 (overview)\n")

    return df_rec_clean, df_stops, df_segments


if __name__ == "__main__":
    load_data(verbose=True)