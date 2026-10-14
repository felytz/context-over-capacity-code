"""
_lib_package.py
===============
Empaquetado de los OUTPUTS de una ruta en un .zip dentro de
``data/03_processed/``.

Pensado para el final de una corrida del método nuevo (``run_new_pipeline.py``
con ``--zip``, o el notebook ``package_outputs``).

MODO DEFAULT (``include=None``) — TODO el árbol de la ruta menos dataframes:
entra cada archivo bajo ``data/02_interim/routes/<ruta>/`` EXCEPTO
  - el directorio ``dataframes/`` (df_full, df_final, trips_cache, shards…),
  - ``df_pipeline.parquet`` (dataframe grande del paso 8.2),
  - checkpoints de entrenamiento (``cv_ckpt_fold*``, estado reanudable, no
    resultado).
Los df_errors/summaries de predicción SÍ van (son los resultados que los
análisis offline del paper necesitan).

MODO CATEGORÍAS (``include=[...]``) — subconjunto por globs, para paquetes
parciales (p. ej. solo ``maps``+``dashboards``):

- ``dashboards`` — dashboards del paso 8.
- ``pipeline``   — dashboard + estáticos del paso 8 (df_pipeline NO va).
- ``arrival``    — reportes de baselines (paso 10) + dashboard/estáticos por leg
                   + summary/metrics parquet + skips.
- ``lstm``       — reportes del LSTM (paso 11) + dashboards por leg + summaries.
- ``maps``       — mapas HTML (viajes nuevos, snap, overview, stopped).
- ``reports``    — reportes de gaps/snap + trayectorias.
- ``perf``       — reporte técnico de rendimiento + funnel.
- ``audit``      — integrity_report.html + plots de las auditorías.

Si se pasa ``analysis_range``, los archivos cuyo nombre (o directorio) lleva
OTRA ventana de análisis (p. ej. ``arrival_report_20240101-20240201.html``
cuando la corrida es ``2024_01-08``) se excluyen; los archivos sin sufijo de
ventana (``dashboard.html``, estáticos, mapas) entran siempre.
"""
from __future__ import annotations

import hashlib
import re
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

# Globs por categoría. Solo artefactos de REPORTE (html/png/pdf/csv y los
# summaries chicos); nunca los parquets de datos.
_CATEGORIES: Dict[str, List[str]] = {
    "dashboards": ["dashboards/*.html"],
    "pipeline":   ["pipeline/dashboard.html", "pipeline/static/*"],
    "arrival":    ["arrival/*.html", "arrival/arrival_summary_*.parquet",
                   "arrival/arrival_metrics_*.parquet", "arrival/arrival_skips_*.json",
                   # dirs por leg anidados bajo la ventana: arrival/<rango>/legX/
                   "arrival/*/leg*/dashboard.html", "arrival/*/leg*/static/*"],
    "lstm":       ["lstm_tf/*.html", "lstm_tf/lstm_tf_summary_*.parquet",
                   "lstm_tf/lstm_tf_vs_baselines_*.parquet",
                   "lstm_tf/lstm_tf_skips_*.json",
                   "lstm_tf/*/leg*/*/dashboard.html", "lstm_tf/*/leg*/*/static/*"],
    "maps":       ["maps/*.html"],
    "reports":    ["reports/*.html", "reports/*.png"],
    "perf":       ["perf/*.json"],
    "audit":      ["audit/*/integrity_report.html", "audit/*/plot_*.png",
                   "audit/*/plot_*.html"],
}

# Sufijos de ventana de análisis en nombres de archivo: "2024_01-08" o
# "20240101-20240201" (modo custom), con la etiqueta de terminales opcional
# ("_t1", "_t2", "_tauto", "_tf0_2") que añade configure_window para separar
# análisis. La etiqueta se captura junto con la ventana para que un zip de _t1
# no arrastre archivos de _t2.
_RANGE_RX = re.compile(
    r"((?:\d{4}_\d{2}-\d{2}|\d{8}-\d{8})(?:_t\w+)?)")


def _matches_range(rel_path: str, analysis_range: Optional[str]) -> bool:
    """True si el archivo pertenece a esta ventana+etiqueta (o no lleva sufijo).

    Recibe el PATH RELATIVO (no solo el nombre) porque la etiqueta de la ventana
    puede vivir en un directorio (``arrival/<rango>/legX/dashboard.html``), no
    solo en el nombre del archivo (``arrival_report_<rango>.html``)."""
    if not analysis_range:
        return True
    found = _RANGE_RX.findall(rel_path.replace("\\", "/"))
    return (not found) or (analysis_range in found)


# Exclusiones del modo full (todo menos dataframes):
_EXCLUDE_DIRS = {"dataframes"}          # df_full/df_final/trips_cache/shards
_EXCLUDE_NAMES = {"df_pipeline.parquet"}  # dataframe grande del paso 8.2
_EXCLUDE_PREFIXES = ("cv_ckpt_fold",)   # checkpoints TF (estado, no resultado)

# df_errors por-leg del LSTM: ``lstm_tf/<...>/leg<x>/df_errors[...].parquet``.
# Es el insumo que el paso 11 concatena en ``lstm_tf_vs_baselines_<rango>.parquet``
# (que es además el archivo que lee el análisis del paper). OJO: el esquema del
# combinado VARÍA entre corridas — a veces trae las columnas de segmento
# (``*_seg``) y a veces no. Por eso solo se omite el per-leg si el combinado
# CUBRE todas sus columnas (guardia en :func:`_drop_redundant`).
_LSTM_LEG_DFERR_RX = re.compile(r"(?:^|/)lstm_tf/(?:.*/)?leg[^/]*/.*df_errors[^/]*\.parquet$")


def _file_digest(p: Path, chunk: int = 1 << 20) -> bytes:
    """SHA1 del contenido (para detectar copias byte-idénticas)."""
    h = hashlib.sha1()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.digest()


def _parquet_columns(p: Path) -> Optional[set]:
    """Columnas de un parquet leyendo SOLO el footer (sin cargar datos).
    Devuelve None si no hay un lector disponible (polars/pyarrow)."""
    try:
        import polars as pl
        return set(pl.scan_parquet(p).collect_schema().names())
    except Exception:
        pass
    try:
        import pyarrow.parquet as pq
        return set(pq.ParquetFile(p).schema.names)
    except Exception:
        return None


# Clave que identifica una predicción (fila) del df_errors del LSTM. Se usa para
# verificar CONTENCIÓN DE FILAS del per-leg en el combinado: que las columnas
# sean subset NO basta (un leg distinto puede tener las mismas columnas pero
# filas que el combinado no contiene → dropearlo perdería datos).
_LSTM_ROW_KEY = ("algorithm", "fold", "trip_id", "l_idx", "h_idx")


def _lstm_rows_contained(per_leg: Path, combined: Path, pe_cols: set) -> bool:
    """True si TODAS las filas del df_errors por-leg están representadas en el
    combinado (por clave). Anti-join perezoso con corte temprano; sin polars
    devuelve False (conservador → se conserva el per-leg)."""
    try:
        import polars as pl
    except Exception:
        return False
    key = [c for c in _LSTM_ROW_KEY if c in pe_cols]
    if not key:
        return False   # sin clave fiable → no arriesgar
    try:
        pe = pl.scan_parquet(per_leg).select(key).unique()
        vs = pl.scan_parquet(combined).select(key).unique()
        missing = pe.join(vs, on=key, how="anti").head(1).collect(engine="streaming")
        return missing.height == 0
    except Exception:
        return False


def _drop_redundant(files: List[Path], base_dir: Path,
                    verbose: bool = False) -> List[Path]:
    """Quita, SIN pérdida de información, copias redundantes de ``df_errors``
    (el grueso del peso del zip). Dos reglas:

    1. **LSTM per-leg cubierto** — descarta ``lstm_tf/.../leg*/df_errors*.parquet``
       SOLO si el combinado ``lstm_tf_vs_baselines_<rango>.parquet`` está en el
       paquete, sus columnas son **superset** de las del per-leg, no es más viejo
       (mtime), y —lo decisivo— **contiene todas sus filas** (anti-join por
       clave). Si el combinado no cubre alguna columna (p. ej. las ``*_seg`` en
       algunas corridas) o no tiene las filas de ese leg, el per-leg se CONSERVA
       — no se pierde nada.
    2. **Byte-idénticas** — entre parquets con el mismo tamaño+SHA1 (p. ej. el
       ``df_errors.parquet`` plano de arrival y su gemelo range-aware
       ``df_errors_<leg>_<rango>.parquet``) conserva UNO — el de nombre más
       largo, i.e. la copia range-aware canónica — y descarta el resto.

    No borra nada de disco: solo decide qué NO entra al zip.
    """
    keep = list(files)
    dropped: List[Path] = []

    # (1) per-leg del LSTM cubierto por el vs_baselines combinado
    vs_files = [p for p in keep if p.name.startswith("lstm_tf_vs_baselines_")]
    candidates = [p for p in keep
                  if _LSTM_LEG_DFERR_RX.search(p.relative_to(base_dir).as_posix())]
    if vs_files and candidates:
        # collect_route_outputs ya filtró por ventana → hay un solo combinado
        vs = vs_files[0]
        vs_cols = _parquet_columns(vs)
        vs_mtime = vs.stat().st_mtime
        drop_set = set()
        for p in candidates:
            pe_cols = _parquet_columns(p)
            readable = vs_cols is not None and pe_cols is not None
            covered = readable and pe_cols <= vs_cols
            fresh = vs_mtime >= p.stat().st_mtime
            # columnas subset + no-más-viejo son prefiltros baratos; la condición
            # decisiva es la CONTENCIÓN DE FILAS (evita tirar un leg cuyas filas
            # no están en el combinado aunque compartan columnas).
            contained = covered and fresh and _lstm_rows_contained(p, vs, pe_cols)
            if contained:
                drop_set.add(p)
                dropped.append(p)
            elif verbose:
                if not readable:
                    why = "no pude leer el esquema (¿sin polars/pyarrow?)"
                elif not covered:
                    why = "columnas no cubiertas por vs_baselines"
                elif not fresh:
                    why = "vs_baselines más viejo"
                else:
                    why = "sus filas no están en vs_baselines"
                print(f"  • conservo {p.relative_to(base_dir).as_posix()} "
                      f"({why})")
        keep = [p for p in keep if p not in drop_set]

    # (2) copias byte-idénticas: agrupa por tamaño, hashea solo las colisiones
    by_size: Dict[int, List[Path]] = defaultdict(list)
    for p in keep:
        if p.suffix == ".parquet":
            by_size[p.stat().st_size].append(p)
    to_drop = set()
    for group in by_size.values():
        if len(group) < 2:
            continue
        by_hash: Dict[bytes, List[Path]] = defaultdict(list)
        for p in group:
            by_hash[_file_digest(p)].append(p)
        for dups in by_hash.values():
            if len(dups) < 2:
                continue
            # conserva el nombre más largo (la copia range-aware)
            dups.sort(key=lambda q: (len(q.name), q.name), reverse=True)
            for q in dups[1:]:
                to_drop.add(q)
                dropped.append(q)
    keep = [p for p in keep if p not in to_drop]

    if verbose and dropped:
        mb = sum(p.stat().st_size for p in dropped) / 1e6
        print(f"  ♻ dedup: -{len(dropped)} archivo(s) redundante(s) "
              f"({mb:.1f} MB) omitidos del zip (cubiertos por otra copia)")
    return keep


def collect_route_outputs_full(base_dir: Path,
                               analysis_range: Optional[str] = None,
                               dedup: bool = True,
                               verbose: bool = False) -> List[Path]:
    """Modo full: TODOS los archivos bajo la ruta menos dataframes/checkpoints
    (ver docstring del módulo), filtrados por ventana de análisis.

    ``dedup=True`` (default) quita copias redundantes de df_errors sin pérdida
    de información (ver :func:`_drop_redundant`)."""
    base_dir = Path(base_dir)
    files: List[Path] = []
    for p in sorted(base_dir.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(base_dir)
        if rel.parts and rel.parts[0] in _EXCLUDE_DIRS:
            continue
        if p.name in _EXCLUDE_NAMES:
            continue
        if any(p.name.startswith(pre) for pre in _EXCLUDE_PREFIXES):
            continue
        if not _matches_range(str(rel), analysis_range):
            continue
        files.append(p)
    if dedup:
        files = _drop_redundant(files, base_dir, verbose=verbose)
    return files


def collect_route_outputs(base_dir: Path,
                          include: Optional[Iterable[str]] = None,
                          analysis_range: Optional[str] = None,
                          dedup: bool = True,
                          verbose: bool = False) -> List[Path]:
    """Lista (sin zipear) los archivos que entrarían al paquete.

    ``include=None`` (default) → modo FULL (todo menos dataframes);
    ``include=[categorías]`` → modo por globs (paquete parcial).
    ``dedup`` solo aplica al modo full (las categorías no arrastran los df_errors
    pesados)."""
    base_dir = Path(base_dir)
    if include is None:
        return collect_route_outputs_full(base_dir, analysis_range,
                                          dedup=dedup, verbose=verbose)
    files: List[Path] = []
    for cat in include:
        for pattern in _CATEGORIES[cat]:
            for p in sorted(base_dir.glob(pattern)):
                if p.is_file() and _matches_range(
                        str(p.relative_to(base_dir)), analysis_range):
                    files.append(p)
    # sin duplicados, orden estable
    seen, uniq = set(), []
    for p in files:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def package_route_outputs(paths=None, *,
                          route: Optional[int] = None,
                          analysis_range: Optional[str] = None,
                          include: Optional[Iterable[str]] = None,
                          out_dir: Optional[Path] = None,
                          dedup: bool = True,
                          verbose: bool = True) -> Optional[Path]:
    """Empaqueta los outputs de UNA ruta en un .zip de 03_processed.

    Parameters
    ----------
    paths : config.RoutePaths | None
        Si es None usa ``config.PATHS`` (requiere ``configure_window`` corrido).
    route, analysis_range :
        Overrides; por default salen de ``paths``.
    include :
        None (default) → TODO el árbol de la ruta menos dataframes/checkpoints.
        Lista → solo esas categorías: {'dashboards','pipeline','arrival',
        'lstm','maps','reports','perf','audit'}.
    out_dir :
        Destino; default ``data/03_processed/``.
    dedup :
        True (default) → omite del zip copias redundantes de df_errors sin
        pérdida de información (per-leg del LSTM cubierto por vs_baselines, y
        copias byte-idénticas). Solo aplica al modo full.

    Returns
    -------
    Path | None — ruta del .zip creado, o None si no había nada que empaquetar.
    """
    if paths is None:
        import config
        paths = config.PATHS
    route = paths.int_route if route is None else route
    analysis_range = paths.analysis_range if analysis_range is None else analysis_range
    base_dir = Path(paths.interim_route_dir)
    if out_dir is None:
        out_dir = base_dir.parents[2] / "03_processed"   # data/02_interim/routes/<r> → data/
    out_dir = Path(out_dir)

    if include is not None:
        bad = set(include) - set(_CATEGORIES)
        if bad:
            raise ValueError(f"Categorías desconocidas: {sorted(bad)}; "
                             f"válidas: {sorted(_CATEGORIES)}")

    files = collect_route_outputs(base_dir, include=include,
                                  analysis_range=analysis_range,
                                  dedup=dedup, verbose=verbose)
    if not files:
        if verbose:
            print(f"  ⚠ ruta {route}: nada que empaquetar en {base_dir} "
                  f"(¿corrió el pipeline?)")
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{analysis_range}" if analysis_range else ""
    zip_path = out_dir / f"new_pipeline_{route}{suffix}.zip"

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in files:
            zf.write(p, arcname=f"{route}/{p.relative_to(base_dir).as_posix()}")

    if verbose:
        print(f"  📦 {zip_path.name}: {len(files)} archivos "
              f"({zip_path.stat().st_size / 1e6:.1f} MB) → {zip_path.parent}")
    return zip_path
