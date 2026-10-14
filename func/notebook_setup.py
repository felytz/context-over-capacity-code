"""
notebook_setup.py
=================
Arranque compartido de los notebooks de ``notebooks/``.

Los 4 notebooks (``pipeline_first``, ``pipeline_after``,
``run_all_notebook_version``, ``integrity_check``) repetían byte-a-byte las
mismas celdas de parámetros (0a · extracto, 0b · ventana, 0c · consistencia
geométrica, ⚡ carga desde caché). Aquí viven una sola vez.

Diseño
------
- Imports a nivel de módulo **mínimos** (``os``, ``sys``, ``re``, ``pathlib``,
  ``datetime``). ``polars``, ``config`` y los ``_lib_*``/``step_*`` se importan
  **dentro** de cada función.
- En particular ``config`` NO se importa al cargar este módulo: hay que fijar los
  env vars (``BUS_*``) **antes** de importarlo, y eso solo pasa en
  ``configure_window``. Importarlo antes dejaría ``config`` cacheado con los
  defaults y haría fallar el guard de re-parametrización.

Uso típico en un notebook (celda de bootstrap, ~4 líneas)::

    import sys
    if "../func" not in sys.path:
        sys.path.insert(0, "../func")
    import notebook_setup as nb

    data_min, data_max = nb.show_raw_extracts(RUTA, DATE_RANGE)        # 0a
    cfg = nb.configure_window(ruta=RUTA, ..., modo=MODO, ...)          # 0b
    nb.check_geometry_consistency()                                    # 0c
    data = nb.load_from_cache(need_raw=False)                          # ⚡
"""

from __future__ import annotations

import os
import sys
from datetime import timedelta
from pathlib import Path

# Anclado a la ubicación del archivo (no al cwd), igual que config.py, para que
# funcione sin importar desde qué carpeta se lance el kernel. Respeta el override
# BUSONORA_ROOT por consistencia con config.
_FUNC_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = Path(os.environ.get("BUSONORA_ROOT", _FUNC_DIR.parent)).resolve()
_RECORDS_DIR = _PROJECT_ROOT / "data" / "01_raw" / "records"


# Encoding de stdout/stderr (consola Windows cp1252 no soporta →/✔/█…). Este
# módulo se importa ANTES que config.py en los notebooks (ver docstring), así
# que el fix también vive aquí — config.py lo repite por si se importa solo.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass


# ────────────────────────────────────────────────────────────────────
# 0a · SELECCIÓN DE RUTAS
# ────────────────────────────────────────────────────────────────────
# Etiqueta canónica del extracto de AÑO COMPLETO (config.DATE_RANGE por defecto:
# f"{YEAR}_{MONTH1}-{MONTH2-1}"). El paso 01 descarga el año entero de la BD y lo
# guarda bajo este nombre; los extractos de una semana (p. ej. "01-07_2024-01")
# son muestras locales para iterar rápido.
FULL_RANGE_LABEL = "2024_01-12"


def resolve_routes(rutas="all", excluir=None, date_range=None, *,
                   prefer_full_range=False, full_range=FULL_RANGE_LABEL,
                   verbose=True):
    """Resuelve la selección de rutas contra los extractos raw en disco.

    Parameters
    ----------
    rutas : int | str | list | "all"
        * ``1602`` o ``"1602"`` — una sola ruta.
        * ``[1602, 1701]`` — lista explícita.
        * ``"all"`` — todas las rutas con al menos un extracto raw en
          ``data/01_raw/records/<ruta>/``.
    excluir : list | None
        Rutas a quitar de la selección (útil con ``rutas="all"``).
    date_range : None | str | dict
        * ``None`` — auto: el extracto MÁS GRANDE en disco de cada ruta.
        * ``str`` — el mismo date_range para todas.
        * ``dict {ruta: date_range}`` — por ruta (las que falten → auto).
    prefer_full_range : bool, default False
        Solo aplica al modo auto (``date_range=None``). Si en disco NO está el
        extracto de año completo (``full_range``) sino solo muestras parciales
        (p. ej. la semana ``01-07_2024-01``), se apunta a ``full_range`` de
        todas formas: como ese parquet no existe, el paso 01 lo DESCARGA de la
        BD (año completo) y lo cachea. Útil en la nube, donde sí hay BD;
        déjalo en False para iterar local con la muestra que ya tengas.
        (Requiere BD accesible; en local sin BD el paso 01 fallaría.)
    full_range : str, default ``FULL_RANGE_LABEL`` ("2024_01-12")
        Etiqueta del extracto completo a descargar cuando ``prefer_full_range``.

    Returns
    -------
    list[tuple[int, str]]
        ``[(ruta, date_range), ...]`` ordenado por ruta. Las rutas sin
        extracto en disco y sin date_range explícito se SALTAN con aviso
        (salvo con ``prefer_full_range``, que apunta a ``full_range`` para
        descargarlo).
    """
    import re

    excl = {int(x) for x in (excluir or [])}

    # Inventario de extractos: {ruta: [(label, bytes), ...]}
    extracts: dict = {}
    for p in sorted(_RECORDS_DIR.glob("*/df_rec_raw_*.parquet")):
        m = re.match(r"df_rec_raw_(\d+)_(.+)\.parquet", p.name)
        if m:
            extracts.setdefault(int(m.group(1)), []).append(
                (m.group(2), p.stat().st_size))

    if isinstance(rutas, str) and rutas.strip().lower() == "all":
        selected = sorted(extracts)
    elif isinstance(rutas, (int, str)):
        selected = [int(rutas)]
    else:
        selected = sorted({int(r) for r in rutas})
    selected = [r for r in selected if r not in excl]

    plan = []
    for r in selected:
        if isinstance(date_range, dict):
            dr = date_range.get(r)
        else:
            dr = date_range
        if dr is None:
            opts = extracts.get(r)
            labels = [lbl for lbl, _ in (opts or [])]
            if opts and full_range in labels:
                dr = full_range                      # el completo ya está en disco
            elif prefer_full_range:
                # Solo muestras parciales (o nada) → apuntar al año completo para
                # que el paso 01 lo descargue de la BD. Aviso claro de la fuente.
                dr = full_range
                if verbose:
                    _tiene = f"solo {labels}" if labels else "nada"
                    print(f"  ⤓ ruta {r}: en disco {_tiene}; "
                          f"prefer_full_range → DATE_RANGE='{full_range}' "
                          f"(el paso 01 lo descargará de la BD)")
            elif opts:
                dr = max(opts, key=lambda t: t[1])[0]   # el extracto más grande
            else:
                if verbose:
                    print(f"  ⚠ ruta {r}: sin extracto raw en disco y sin "
                          f"date_range explícito → SALTADA "
                          f"(usa prefer_full_range=True para descargar de la BD)")
                continue
        plan.append((r, dr))

    if verbose:
        print(f"Rutas seleccionadas ({len(plan)}):")
        for r, dr in plan:
            p = _RECORDS_DIR / str(r) / f"df_rec_raw_{r}_{dr}.parquet"
            estado = (f"{p.stat().st_size / 1e6:6.1f} MB" if p.is_file()
                      else "NO existe → BD")
            print(f"  · RUTA={r:>5}  DATE_RANGE='{dr}'  {estado}")
    if not plan:
        raise ValueError(f"La selección {rutas!r} (excluir={sorted(excl)}) "
                         f"no dejó ninguna ruta con extracto.")
    return plan


# ────────────────────────────────────────────────────────────────────
# 0a-bis · Nº DE TERMINALES → (dominance_frac, tag)
# ────────────────────────────────────────────────────────────────────
def resolve_terminals(terminals):
    """Traduce una especificación amigable de **nº de terminales** a
    ``(dominance_frac, tag)``:

    - ``dominance_frac`` alimenta ``_lib_new_trips.build_trips`` (descubrimiento
      de terminales por dwell; ver ``_resolve_dominance_frac``).
    - ``tag`` etiqueta los outputs (se AÑADE al ``analysis_range``), de modo que
      cada configuración de terminales produce un **análisis separado** en disco
      en vez de sobrescribir al anterior.

    Formas aceptadas de ``terminals``:
      * ``1``, ``2``, … (int)          → top-N terminales; tag ``t1``, ``t2``, …
      * ``"auto"`` / ``"full_AUTO"``   → auto por ruta (1-2 dominantes); tag ``tauto``
      * ``"auto_min"`` / ``"AUTO_min"``→ variante min_auto; tag ``tmin``
      * ``0.2`` (float)                → fracción de dominancia directa; tag ``tf0_2``
    """
    if isinstance(terminals, bool):
        raise ValueError("terminals no puede ser bool")
    if isinstance(terminals, str):
        t = terminals.strip().lower()
        if t in ("auto", "full_auto"):
            return "full_AUTO", "tauto"
        if t in ("auto_min", "min", "min_auto"):
            return "AUTO_min", "tmin"
        try:                       # cadena numérica: "1", "2", "0.2"
            terminals = int(t)
        except ValueError:
            try:
                terminals = float(t)
            except ValueError:
                raise ValueError(f"terminals no reconocido: {terminals!r}")
    if isinstance(terminals, int):
        n = int(terminals)
        if n < 1:
            raise ValueError("terminals (int) debe ser >= 1")
        return n, f"t{n}"
    f = float(terminals)           # float → fracción de dominancia directa
    return f, "tf" + f"{f:g}".replace(".", "_")


# ────────────────────────────────────────────────────────────────────
# 0a · RUTA Y EXTRACTO
# ────────────────────────────────────────────────────────────────────
def show_raw_extracts(ruta, date_range):
    """Lista los extractos raw en disco (marca el elegido con →) y devuelve el
    rango temporal ``(data_min, data_max)`` del extracto seleccionado.

    Devuelve ``(None, None)`` si el parquet del extracto no existe localmente.
    """
    import re
    import polars as pl

    print("Extractos raw disponibles (data/01_raw/records/):")
    for p in sorted(_RECORDS_DIR.glob("*/df_rec_raw_*.parquet")):
        m = re.match(r"df_rec_raw_(\w+)_(.+)\.parquet", p.name)
        if m:
            sel = "→" if (str(ruta) == m.group(1) and date_range == m.group(2)) else " "
            print(f" {sel} RUTA={m.group(1):>5}  DATE_RANGE='{m.group(2)}'  "
                  f"{p.stat().st_size / 1e6:6.1f} MB")

    raw_path = _RECORDS_DIR / str(ruta) / f"df_rec_raw_{ruta}_{date_range}.parquet"
    data_min = data_max = None
    if raw_path.is_file():
        # Guard de no-vacío: un extracto de 0 bytes / 0 filas hace que polars
        # falle aguas abajo con un error críptico ("RESOLVING 'sink'",
        # ESTIMATED ROWS: 0). Lo detectamos, lo ELIMINAMOS vía código (política
        # del proyecto: no dejar dataframes vacíos guardados) y avisamos claro.
        import pyarrow.parquet as pq
        _size = raw_path.stat().st_size
        _nrows = None
        if _size > 0:
            try:
                _nrows = pq.read_metadata(raw_path).num_rows
            except Exception as e:
                print(f"\n⚠ Extracto ILEGIBLE/corrupto (NO se elimina): "
                      f"{raw_path.name}\n   {e}")
                return None, None
        if _size == 0 or _nrows == 0:
            raw_path.unlink()
            print(f"\n⚠ Extracto VACÍO (0 filas / 0 bytes) ELIMINADO vía código: "
                  f"{raw_path.name}")
            print(f"   Re-genera el sub-extracto en 0a, o el paso 01 descargará de la BD.")
            return None, None
        _t = (pl.scan_parquet(raw_path)
                .select(pl.col("time").min().alias("lo"),
                        pl.col("time").max().alias("hi"))
                .collect())
        data_min, data_max = _t["lo"][0], _t["hi"][0]
        print(f"\nRango disponible en el extracto: {data_min}  →  {data_max}  "
              f"(~{(data_max - data_min).days + 1} días)")
    else:
        print(f"\n⚠ No existe {raw_path.name} — el paso 01 intentaría descargar de la BD.")
    return data_min, data_max


# ────────────────────────────────────────────────────────────────────
# 0b · VENTANA DE ANÁLISIS + env vars + import config + guard
# ────────────────────────────────────────────────────────────────────
def configure_window(*, ruta, ruta_rec=None, date_range, modo,
                     terminals=1,
                     data_min=None, data_max=None,
                     custom_start=None, custom_end=None):
    """Deriva la ventana de análisis según ``modo`` (full/rapido/custom), fija los
    env vars ``BUS_*``, importa ``config`` y verifica con un ``assert`` que no
    quedara importado con otros parámetros (→ Kernel → Restart). Devuelve el
    módulo ``config`` ya configurado (``config.PATHS`` para las rutas).

    ``ruta_rec`` (group_id de BD, solo para DESCARGAR de la BD) default =
    ``ruta``: los extractos en disco ya están nombrados por el id público.
    Pásalo explícito solo si el group_id interno difiere del id público.

    ``terminals`` (default ``1``) fija el nº de terminales del método de viajes
    (ver ``resolve_terminals``) y AÑADE su etiqueta al ``analysis_range``
    (``…_t1``, ``…_t2``, ``…_tauto``, …), de modo que correr con otra
    configuración de terminales crea un **análisis aparte** sin pisar el previo.
    Pasa ``terminals=None`` para el comportamiento antiguo (sin etiqueta).
    El ``dominance_frac`` resuelto queda en ``config.DOMINANCE_FRAC`` (fuente
    única para el corte de viajes).
    """
    import warnings

    ruta_rec = ruta if ruta_rec is None else ruta_rec
    if terminals is None:
        dominance_frac, tag = 1, ""
    else:
        dominance_frac, tag = resolve_terminals(terminals)

    if modo == "full":
        analysis_start, analysis_end = "2024-01-01", "2024-09-01"
        analysis_range = "2024_01-08"     # sufijo estándar → reusa caches existentes
    elif modo == "rapido":
        assert data_min is not None, "Sin extracto raw no se puede derivar la ventana."
        _ini = data_min.date()
        _fin = min(_ini + timedelta(days=28), data_max.date() + timedelta(days=1))
        analysis_start, analysis_end = str(_ini), str(_fin)
        analysis_range = f"{_ini:%Y%m%d}-{_fin:%Y%m%d}"
    elif modo == "custom":
        assert custom_start and custom_end, "MODO='custom' requiere custom_start y custom_end."
        analysis_start, analysis_end = custom_start, custom_end
        analysis_range = f"{custom_start.replace('-', '')}-{custom_end.replace('-', '')}"
    else:
        raise ValueError(f"MODO desconocido: {modo!r}")

    # Etiqueta de terminales → sufijo del analysis_range (análisis separados).
    if tag:
        analysis_range = f"{analysis_range}_{tag}"

    os.environ.update({
        "BUS_ROUTE":             str(ruta),
        "BUS_ROUTE_STOP":        str(ruta),
        "BUS_ROUTE_REC":         str(ruta_rec),
        "BUS_DATE_RANGE":        date_range,
        "BUS_ANALYSIS_START":    analysis_start,
        "BUS_ANALYSIS_END":      analysis_end,
        "BUS_ANALYSIS_RANGE":    analysis_range,
    })

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    if str(_FUNC_DIR) not in sys.path:
        sys.path.insert(0, str(_FUNC_DIR))

    import config
    from config import PATHS

    # Guard: si config quedó importado con OTROS parámetros → reiniciar kernel.
    assert (config.INT_ROUTE == ruta and config.DATE_RANGE == date_range
            and config.ANALYSIS_RANGE == analysis_range
            and config.ANALYSIS_START == analysis_start
            and config.ANALYSIS_END == analysis_end), \
        "config ya estaba importado con otros parámetros -> Kernel -> Restart"

    # Fuente única del corte de viajes para el resto del pipeline/notebook.
    config.DOMINANCE_FRAC = dominance_frac
    config.TERMINALS = terminals
    config.ANALYSIS_TAG = tag

    print(f"  Ruta       : {config.INT_ROUTE}  (REC BD: {config.INT_ROUTE_REC})")
    print(f"  Extracto   : {PATHS.df_rec_raw.name}  "
          f"({'existe' if PATHS.df_rec_raw.is_file() else 'NO existe → BD'})")
    print(f"  MODO       : {modo}")
    print(f"  Terminales : {terminals}  (dominance_frac={dominance_frac}, tag='{tag or '—'}')")
    print(f"  Ventana    : {analysis_start} → {analysis_end}   (caches *{analysis_range}*)")
    print(f"  df_dir     : {PATHS.df_dir}")
    print(f"  audit_dir  : {PATHS.audit_dir}")
    return config


# ────────────────────────────────────────────────────────────────────
# 0c · CONSISTENCIA GEOMÉTRICA
# ────────────────────────────────────────────────────────────────────
def _polyline_length_km(poly) -> float:
    """Longitud haversine (km) de una polilínea ``[{latitude, longitude}, ...]``.

    Vectorizado con numpy (mismo R que ``_lib_*._haversine_m`` → 6 371 000 m), en
    vez de un loop Python por vértice.
    """
    import numpy as np

    if poly is None or len(poly) < 2:
        return 0.0
    lat = np.radians(np.fromiter((q["latitude"] for q in poly), dtype=float, count=len(poly)))
    lon = np.radians(np.fromiter((q["longitude"] for q in poly), dtype=float, count=len(poly)))
    dlat = np.diff(lat)
    dlon = np.diff(lon)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat[:-1]) * np.cos(lat[1:]) * np.sin(dlon / 2) ** 2
    d = 2 * 6_371_000.0 * np.arcsin(np.sqrt(np.maximum(0.0, a)))
    return float(d.sum()) / 1000


def check_geometry_consistency():
    """Verifica que ``polylines.geojson``, ``df_segments.parquet`` y
    ``turn_hex.json`` midan la misma polilínea (drift < 5 %). Imprime el veredicto
    y devuelve la lista de caches STALE a borrar (vacía si todo es consistente).

    Asume que ``configure_window`` ya corrió (config importado → ``sys.path`` con
    ``lib/``).
    """
    import json
    import polars as pl
    import config
    from config import PATHS
    from _lib_map_match import _load_geojson_branches, _cumulative_distances

    # 1 · Polilínea oficial de la ruta (la que usan los pasos 4/6, ya filtrada)
    _brA, _brB = _load_geojson_branches(str(PATHS.geojson_path), df_end_stop=None)
    _lenA = _cumulative_distances(_brA)[-1] / 1000 if len(_brA) > 1 else 0.0
    _lenB = _cumulative_distances(_brB)[-1] / 1000 if len(_brB) > 1 else 0.0
    # Con 1 LineString la rama B es el inverso (misma longitud) → el circuito es A.
    circle_km = _lenA if abs(_lenA - _lenB) < 1e-6 else _lenA + _lenB
    print(f"  polylines.geojson (ruta {config.INT_ROUTE}) : {circle_km:8.2f} km")

    stale = []

    # 2 · df_segments.parquet (fuente de mapas del paso 2 y features del 7)
    if PATHS.df_segments.is_file():
        _seg = pl.read_parquet(PATHS.df_segments)
        seg_km = sum(_polyline_length_km(_poly) for _poly in _seg["polyline"].to_list())
        _d = abs(seg_km - circle_km) / circle_km * 100
        print(f"  df_segments.parquet ({_seg.height} segs)    : {seg_km:8.2f} km  "
              f"(drift {_d:4.1f}% → {'OK' if _d < 5 else 'STALE'})")
        if _d >= 5:
            stale.append(PATHS.df_segments)
    else:
        print("  df_segments.parquet                  : (no existe; el paso 1 lo genera)")

    # 3 · turn_hex.json (terminales → dirección y corte de viajes)
    if PATHS.turn_hex_json.is_file():
        with open(PATHS.turn_hex_json, encoding="utf-8") as f:
            _th = json.load(f)
        th_km = _th.get("polyline_length_m", 0) / 1000
        _d = abs(th_km - circle_km) / circle_km * 100
        print(f"  turn_hex.json (polyline_length)      : {th_km:8.2f} km  "
              f"(drift {_d:4.1f}% → {'OK' if _d < 5 else 'STALE'})")
        if _d >= 5:
            stale.append(PATHS.turn_hex_json)
    else:
        print("  turn_hex.json                        : (no existe; el paso 4 lo genera)")

    if stale:
        print("\n  ⚠ CACHES INCONSISTENTES con la polilínea actual de la ruta.")
        print("  Borra estos archivos y re-corre los pasos 1/4/6:")
        for s in stale:
            print(f"     {s}")
        print("  ...y los derivados (se regeneran solos):")
        for prop in ("df_route_direction", "df_trips", "df_enriched", "df_full"):
            print(f"     {getattr(PATHS, prop)}")
    else:
        print("\n  ✔ Geometría consistente: mapas, dirección y viajes usan la misma polilínea.")

    return stale


# ────────────────────────────────────────────────────────────────────
# ⚡ CARGA DESDE CACHÉ
# ────────────────────────────────────────────────────────────────────
def list_sample_caches() -> list:
    """Lista los archivos sample disponibles en PATHS.df_dir.

    Devuelve una lista de dicts con keys: ``full_path``, ``enriched_path``,
    ``label`` (p.ej. ``"3d_from_2024-01-01"``), ordenada por fecha de
    modificación (más reciente primero).
    """
    import config
    from config import PATHS
    samples = []
    for p in sorted(PATHS.df_dir.glob("df_rec_sample_*.parquet")):
        if p.name.endswith("_enriched.parquet"):
            continue
        label = p.stem.replace("df_rec_sample_", "")
        enriched = p.with_name(p.stem + "_enriched.parquet")
        samples.append({
            "full_path":     p,
            "enriched_path": enriched if enriched.is_file() else None,
            "label":         label,
            "mtime":         p.stat().st_mtime,
        })
    return sorted(samples, key=lambda x: x["mtime"], reverse=True)


def load_from_cache(need_raw: bool = True, sample=None) -> dict:
    """Carga todo lo disponible en disco (generado por ``pipeline_first`` o
    ``run_all.py``) para iterar sin re-ejecutar el pipeline de datos.

    Devuelve un dict con los DataFrames (``None`` los que no existan).

    Parameters
    ----------
    need_raw : bool
        ``df_rec_raw`` es el parquet crudo grande; solo lo necesita el
        funnel (paso 09). Pasa ``need_raw=False`` (p.ej. en
        ``pipeline_after``, pasos 8/10/11) para ahorrarte cargarlo en
        memoria.
    sample : None | bool | str
        * ``None`` (default) — modo normal: busca ``df_full`` y shards
          mensuales.
        * ``True`` — modo sample automático: carga el sample más reciente
          disponible en ``PATHS.df_dir``. Si no hay ninguno, advierte y
          cae de vuelta al modo normal.
        * ``"Nd_from_YYYY-MM-DD"`` — carga el sample específico con ese
          label (p.ej. ``"3d_from_2024-01-01"``).
    """
    import polars as pl
    import config  # noqa: F401  (asegura que configure_window ya corrió)
    from config import PATHS

    def _try_parquet(path, label):
        p = Path(path)
        if p.is_file():
            df = pl.read_parquet(p)
            flag = "  ⚠ VACÍO (0 filas)" if df.height == 0 else ""
            print(f"  ✔ {label:<30} {df.height:>10,} filas  {p.name}{flag}")
            return df
        print(f"  · {label:<30} (no encontrado: {p.name})")
        return None

    print("=" * 64)
    print("  CARGANDO DESDE CACHÉ")
    print("=" * 64)

    # ── Resolver sample ──────────────────────────────────────────────
    _sample_full = _sample_enriched = None

    if sample is not None:
        available = list_sample_caches()
        if not available:
            print("  ⚠ sample=True pero no hay archivos sample en disco; "
                  "cayendo a modo normal.")
            sample = None
        elif sample is True:
            hit = available[0]  # más reciente
            _sample_full     = hit["full_path"]
            _sample_enriched = hit["enriched_path"]
            print(f"  [sample] {hit['label']}  "
                  f"(más reciente de {len(available)} disponible(s))")
        else:
            label_want = str(sample)
            hits = [s for s in available if s["label"] == label_want]
            if not hits:
                labels = [s["label"] for s in available]
                raise FileNotFoundError(
                    f"Sample '{label_want}' no encontrado. "
                    f"Disponibles: {labels}"
                )
            _sample_full     = hits[0]["full_path"]
            _sample_enriched = hits[0]["enriched_path"]
            print(f"  [sample] {label_want}")

    # ── Paso 1 ──
    if need_raw:
        df_rec_raw = _try_parquet(PATHS.df_rec_raw, "df_rec_raw")
    else:
        df_rec_raw = None
        print(f"  · {'df_rec_raw':<30} (omitido: need_raw=False)")
    df_rec_clean = _try_parquet(PATHS.df_rec_clean, "df_rec_clean")

    try:
        from step_01_load_data import load_stops, load_segments
        df_stops = load_stops(verbose=False)
        df_segments = load_segments(verbose=False)
        print(f"  ✔ {'df_stops':<30} {df_stops.height:>10,} filas  (stops.json)")
        print(f"  ✔ {'df_segments':<30} {df_segments.height:>10,} segmentos")
    except Exception as e:
        df_stops = df_segments = None
        print(f"  · Paradas/segmentos no disponibles: {e}")

    # ── Pasos 4 y 6 ──
    df_route_direction = _try_parquet(PATHS.df_route_direction, "df_route_direction")
    df_filter_temporal = _try_parquet(PATHS.df_filter_temporal, "df_filter_temporal")
    df_trips           = _try_parquet(PATHS.df_trips,           "df_trips")

    # ── Paso 7 — df_enriched y df_full ──────────────────────────────
    if _sample_full is not None:
        # Modo sample: cargar directamente los archivos sample
        df_full     = _try_parquet(_sample_full,     "df_full [sample]")
        df_enriched = (_try_parquet(_sample_enriched, "df_enriched [sample]")
                       if _sample_enriched else None)
    else:
        # Modo normal: parquet combinado → shards mensuales
        df_enriched = _try_parquet(PATHS.df_enriched, "df_enriched")
        df_full     = _try_parquet(PATHS.df_full,     "df_full")

        if df_full is None:
            from step_07_enrich_trips import _monthly_dir
            shard_files = sorted(_monthly_dir("df_rec_full").glob("*.parquet"))
            if shard_files:
                parts = [pl.read_parquet(sf) for sf in shard_files]
                df_full = pl.concat(parts, how="diagonal_relaxed").sort("time")
                print(f"  → df_full (shards concat)      {df_full.height:>10,} filas  "
                      f"({len(shard_files)} meses)")
            else:
                print("  · df_full                        (sin shards disponibles)")

    # ── Alinear df_enriched a los meses de df_full (consistencia) ──
    if df_full is not None and df_enriched is not None and "time" in df_full.columns:
        months_full = df_full["time"].dt.strftime("%Y-%m").unique().sort().to_list()
        df_enriched = df_enriched.filter(
            pl.col("time").dt.strftime("%Y-%m").is_in(months_full))

    # ── Resumen de cobertura temporal ──
    print()
    if df_full is not None and "time" in df_full.columns:
        resumen = (
            df_full
            .with_columns(pl.col("time").dt.strftime("%Y-%m").alias("_m"))
            .group_by("_m")
            .agg(pl.len().alias("n_rows"),
                 pl.col("trip_id").n_unique().alias("n_trips"),
                 pl.col("bus_code").n_unique().alias("n_buses"))
            .sort("_m")
        )
        print(f"  df_full cubre {resumen.height} mes(es):")
        for r in resumen.iter_rows(named=True):
            print(f"    {r['_m']}  {r['n_rows']:>8,} filas  "
                  f"{r['n_trips']:>5,} trips  {r['n_buses']:>3} buses")
    else:
        print("  ⚠ df_full no disponible. Corre antes pipeline_first.ipynb o run_all.py.")

    # ── Samples disponibles (informativo) ───────────────────────────
    if sample is None:
        available = list_sample_caches()
        if available:
            print(f"\n  💡 Hay {len(available)} sample(s) en disco. "
                  f"Usa load_from_cache(sample=True) para cargarlos:")
            for s in available:
                sz = f"{s['full_path'].stat().st_size/1e6:.1f} MB"
                print(f"     '{s['label']}'  ({sz})")

    return {
        "df_rec_raw":         df_rec_raw,
        "df_rec_clean":       df_rec_clean,
        "df_stops":           df_stops,
        "df_segments":        df_segments,
        "df_route_direction": df_route_direction,
        "df_filter_temporal": df_filter_temporal,
        "df_trips":           df_trips,
        "df_enriched":        df_enriched,
        "df_full":            df_full,
    }
