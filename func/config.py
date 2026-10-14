"""
config.py
=========
Configuración centralizada del pipeline de la Ruta 1701 (BUSONORA · Hermosillo).

Aquí se concentran TODAS las constantes, rutas y parámetros del pipeline para
que no estén dispersos por el código. Si quieres procesar otra ruta o cambiar
un umbral, este es el único archivo que tocar.

Notas metodológicas
-------------------
Los umbrales (max_gap_min=7, min_completeness=0.75, max_implied_speed_kmh=180)
se justifican en el documento Route_101_Trip_Definition_Validation.pdf y
están alineados con la literatura canónica:

    - Lin & Zeng (1999)            → corte temporal, P90
    - Du & Aultman-Hall (2007)     → completitud mínima
    - Cathey & Dailey (2003)       → end-stops + cooldown
    - Newson & Krumm (2009)        → map-matching HMM
    - Schuessler & Axhausen (2009) → filtro de outlier por velocidad implícita
                                     (V_max=50 m/s = 180 km/h)

Cambios en v2 (mayo 2026)
-------------------------

1. **Filtro espacial → filtro de velocidad** (Schuessler & Axhausen 2009).
   El filtro espacial absoluto `max_gap_dist_m` (P95 automático, Lin & Zeng 1999)
   fue reemplazado por `max_implied_speed_kmh`. En sistemas AVL con muestreo
   regular (~1 min), un gap espacial de 500–800m es movimiento legítimo del bus,
   no error GPS. El umbral espacial absoluto fragmentaba ~60% de los trips reales.
   Recuperación empírica: +455% trips válidos (de 3,051 → 16,923 trips en
   Ruta 101, ene–ago 2024).

2. **end_stops → turn_hex** (hexágonos H3 detectados por densidad).
   Las terminales ya no son paradas GTFS codificadas, sino entidades
   geométricas detectadas automáticamente por densidad de registros parados.
   Detección: cluster contiguo más denso globalmente + segundo más denso a
   distancia ≥ 1/4 del branch_length.

3. **Cooldown 3 → 10 min**. Con zonas de turn_hex (clusters de 3-9 hex H3),
   el bus tarda más en salir y reaparecer que en una parada GTFS individual.
   Empíricamente, cooldown=3 generaba 4,956 cortes con gap_time mediano de
   1.02 min (99.3% bajo el umbral original); cooldown=10 reduce a 2,702
   cortes sin perder ningún trip válido real.

No tocar estos valores sin re-ejecutar el análisis de sensibilidad.
"""

from __future__ import annotations

import os
import sys as _sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional


# ────────────────────────────────────────────────────────────────────
# 0. BOOTSTRAP DE sys.path
# ────────────────────────────────────────────────────────────────────
# Los módulos del pipeline viven en subcarpetas de func/ (lib/, steps/)
# pero se importan por nombre simple ("from _lib_data import …",
# "from step_04_direction import …"). Este bloque registra esas carpetas
# en sys.path la primera vez que algo importa config, de modo que todos
# los imports históricos sigan funcionando sin reescribirlos.
_FUNC_DIR = Path(__file__).resolve().parent
for _sub in ("", "lib", "steps"):
    _p = str(_FUNC_DIR / _sub) if _sub else str(_FUNC_DIR)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)


# ────────────────────────────────────────────────────────────────────
# 0.bis2 · ENCODING DE STDOUT/STDERR (consola Windows)
# ────────────────────────────────────────────────────────────────────
# La consola de Windows usa cp1252 por defecto, que no soporta los
# caracteres unicode (✔, →, █…) que usan los prints del pipeline. Eso
# revienta con UnicodeEncodeError tanto en el método "viejo" (run_all.py,
# direction-based) como en el "nuevo" (notebooks, leg-based) cada vez que
# se corre fuera de una terminal UTF-8 sin PYTHONIOENCODING seteado.
# Reconfiguramos a UTF-8 una sola vez, sin requerir variables de entorno.
for _stream in (_sys.stdout, _sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass


# ────────────────────────────────────────────────────────────────────
# 0.bis · BACKEND DE MATPLOTLIB (no interactivo por defecto)
# ────────────────────────────────────────────────────────────────────
# El pipeline GUARDA figuras a disco (gap_distribution.png, dashboards,
# reportes de baselines); nunca necesita una ventana GUI. Si matplotlib
# elige un backend interactivo, una llamada como `plt.show()` (p.ej. en
# `_lib_temporal.run_temporal_consistency`, paso 06) BLOQUEA el proceso en
# ejecuciones batch/headless (run_all.py, clúster, subprocesos). Fijamos
# Agg por defecto. `setdefault`: respeta un MPLBACKEND ya exportado y un
# notebook puede pedir inline con `%matplotlib inline` (lo sobreescribe).
# config se importa antes que matplotlib en todos los entrypoints, así que
# matplotlib lee este valor al importarse.
os.environ.setdefault("MPLBACKEND", "Agg")


# ────────────────────────────────────────────────────────────────────
# 1. IDENTIDAD DE LA RUTA
# ────────────────────────────────────────────────────────────────────

# Corrección (jun 2026): el extracto local es realmente la Ruta 1701, no la 101.
# El parquet raw tiene group_id=1701 (confirmado) y la 1701 existe como ruta propia
# en stops_routes.json / segments.geojson / segments_route.json. Antes estaba mal
# etiquetado como 101, lo que map-matcheaba los pings de la 1701 contra la geometría
# de la 101 (paradas y segmentos equivocados). INT_ROUTE pasa a 1701 para que paths,
# paradas (INT_ROUTE_STOP) y segmentos usen la ruta correcta; INT_ROUTE_REC (group_id
# de descarga desde la BD) ya era 1701, así que ahora los tres identificadores coinciden.
INT_ROUTE      = int(os.environ.get("BUS_ROUTE",      1701))
INT_ROUTE_STOP = int(os.environ.get("BUS_ROUTE_STOP", INT_ROUTE))
INT_ROUTE_REC  = int(os.environ.get("BUS_ROUTE_REC",  1701))

# ────────────────────────────────────────────────────────────────────
# 2. VENTANA TEMPORAL DEL ANÁLISIS
# ────────────────────────────────────────────────────────────────────

# Periodo de DESCARGA (lo que se baja de la BD y se cachea en parquet).
YEAR   = "2024"
MONTH1 = "01"
MONTH2 = "13"   # exclusivo: 13 ⇒ enero–diciembre 2024

DATE_RANGE = f"{YEAR}_{MONTH1}-{int(MONTH2) - 1:02d}"   # → "2024_01-12"

# ⚠️ Override (corrida con extracto de UNA SEMANA): el único parquet raw presente
# en disco es data/01_raw/records/1701/df_rec_raw_1701_01-07_2024-01.parquet
# (1–7 ene 2024). Forzamos DATE_RANGE a coincidir con ese nombre para que el
# paso 01 lo lea del cache en vez de intentar descargar de la BD. Para volver a
# la ventana completa, cambia el default o exporta BUS_DATE_RANGE.
# La env var BUS_DATE_RANGE permite parametrizarlo desde notebooks/subprocesos
# (mismo patrón que BUS_ROUTE).
DATE_RANGE = os.environ.get("BUS_DATE_RANGE", "01-07_2024-01")


# ────────────────────────────────────────────────────────────────────
# 2.bis · VENTANA OPERACIONAL VÁLIDA (Opción A — restricción ene–ago)
# ────────────────────────────────────────────────────────────────────
#
# A partir de septiembre 2024 el proveedor de telemetría degradó la
# frecuencia de muestreo GPS (~60s → 3-6 min). El análisis se restringe
# a ene-ago 2024 para mantener la consistencia metodológica.
#
# ⚠️ ITERACIÓN RÁPIDA (mayo 2026): mientras se validan los cambios del
# rebrand v2 (turn_hex + max_implied_speed_kmh), se reduce la ventana a
# UN SOLO MES (enero 2024) para que el pipeline corra en minutos en vez
# de horas. Los archivos cacheados usan `ANALYSIS_RANGE` en el nombre,
# así que NO sobrescriben los caches de la ventana completa de 8 meses.
# Para volver a la ventana completa, restaura los valores comentados.

# # ─── Ventana actual: enero 2024 únicamente (iteración rápida) ────────
# ANALYSIS_START = "2024-01-01"  # inclusivo
# ANALYSIS_END   = "2024-02-01"  # exclusivo (= hasta fin de enero)
# ANALYSIS_RANGE = "2024_01-01"

# ─── Ventana completa de la tesis (ene–ago 2024) ────────────────────
# Las env vars BUS_ANALYSIS_* permiten parametrizar el periodo desde
# notebooks/subprocesos sin tocar este archivo (mismo patrón que BUS_ROUTE).
ANALYSIS_START = os.environ.get("BUS_ANALYSIS_START", "2024-01-01")  # inclusivo
ANALYSIS_END   = os.environ.get("BUS_ANALYSIS_END",   "2024-09-01")  # exclusivo
ANALYSIS_RANGE = os.environ.get("BUS_ANALYSIS_RANGE", "2024_01-08")


def filter_to_analysis_window(df, time_col: str = "time"):
    """Filtra un DataFrame al periodo válido configurado en ANALYSIS_START/END."""
    import polars as pl
    return df.filter(
        (pl.col(time_col) >= pl.lit(ANALYSIS_START).str.to_datetime()) &
        (pl.col(time_col) <  pl.lit(ANALYSIS_END).str.to_datetime())
    )


# ────────────────────────────────────────────────────────────────────
# 3. TURN_HEX Y CRITERIOS DE VIAJE
# ────────────────────────────────────────────────────────────────────
#
# Glosario (v2):
#   turn_hex     → Hexágono H3 (res 12 por defecto) que representa el centro
#                  de una zona de retorno (terminal). Ya no es una parada
#                  GTFS sino una entidad puramente geométrica detectada por
#                  densidad de registros parados.
#   turn_cluster → Componente conexa (k-ring=1) de hex parados outliers que
#                  contiene al turn_hex como su miembro más denso.
#   turn_zone    → turn_hex + sus vecinos k-ring=N que también contienen
#                  polilínea. Define la zona efectiva para corte de viajes.
#                  Reemplaza la antigua `neighbor_window` por paradas.
#
# Los nombres antiguos `END_STOPS_*` se mantienen como aliases deprecated
# al final de este bloque para retrocompatibilidad.

# ── Modo de resolución de turn_hex ──────────────────────────────────
# "auto"   → detección automática: cluster contiguo más denso globalmente
#            (vía k-ring=1 + filtro de outliers por percentil), luego el
#            segundo más denso a distancia geográfica ≥ TURN_HEX_MIN_DIST_FRACTION
#            del primero. Default.
# "manual" → usa la lista `TURN_HEX_MANUAL` (códigos H3 directos).
TURN_HEX_MODE: str = "auto"

# Lista para modo "manual": dos códigos H3 (strings hex).
# ⚠️ Antes era una lista de stop_codes GTFS; ahora son códigos H3 directos.
TURN_HEX_MANUAL: List[str] = []   # ej.: ["8c4ade17b30b7ff", "8c4ade19e9697ff"]

# ── Parámetros de detección automática (modo "auto") ────────────────
TURN_HEX_RESOLUTION:        int   = 12          # resolución H3 (~9 m diámetro)
TURN_HEX_MIN_HEX_COUNT:     int   = 1           # piso absoluto suave; el filtro
                                                # real es por outliers (abajo)
TURN_HEX_MIN_DIST_FRACTION: float = 1.0 / 5.0   # distancia mín. entre turn_hex,
                                                # como fracción del branch_length
                                                # (= polyline_total / 2)
TURN_HEX_MAX_SPEED:         float = 0.0         # umbral "parado" (m/s o km/h
                                                # según unidad de speed)

# ── Filtro de outliers (v2) ─────────────────────────────────────────
# Define qué hex son "candidatos a terminal":
#   "percentile" → stopped_count ≥ P_q (default q=95)
#   "tukey"      → stopped_count ≥ Q3 + 1.5·IQR
#   "absolute"   → stopped_count ≥ TURN_HEX_OUTLIER_ABSOLUTE
TURN_HEX_OUTLIER_METHOD:     str   = "percentile"
TURN_HEX_OUTLIER_PERCENTILE: float = 95.0
TURN_HEX_OUTLIER_ABSOLUTE:   int   = 50

# ── Selección entre clusters candidatos ─────────────────────────────
# "center_count"        → el hex pico individual más denso (default)
# "cluster_total_count" → la suma de conteos del cluster contiguo
TURN_HEX_SELECTION_METRIC: str = "center_count"

# ── k-ring para la turn_zone (corte de viajes) ──────────────────────
# Número de anillos H3 vecinos al turn_hex que forman la zona efectiva
# de corte. 0=solo el centro, 1=+6 hex, 2=+18, etc. Filtrado a hex
# que la polilínea atraviesa.
TURN_ZONE_KRING: int = 1


# ── Density bus-aware (v3) ──────────────────────────────────────────
# Cap por (bus_code, h3_index): cada bus puede aportar como mucho K pings
# al stopped_count de un hex. Mitiga el sesgo de unidades con telemetría
# estacionaria (un bus dormido reportando miles de pings en el mismo hex
# se queda capeado a K, en lugar de dominar la densidad y arrastrar la
# detección hacia el punto donde se estaciona).
#
# Valores:
#   None  → conteo crudo (comportamiento original v2; usar solo si la
#           auditoría confirma que no hay sesgo de telemetría estacionaria).
#   1     → equivalente a n_unique(bus_code) por hex. Suele descartar
#           información operacional legítima (pasadas reales en terminal
#           con varios pings por espera de turno).
#   ~10   → punto intermedio principiado: permite ~10 pings por pasada
#           operacional (espera de turno típica) y acota a buses dormidos.
#           Default recomendado, validado para Ruta 101 contra ubicación
#           real conocida de terminales (escenario G de la auditoría
#           turn_hex_robustness).
#
# Justificación bibliográfica: Cathey & Dailey (2003) y la línea AVL
# subsiguiente recomiendan ponderar por unidad vehicular en lugar de
# pings crudos al estimar densidad de paradas, para evitar exactamente
# este sesgo.
TURN_HEX_MAX_PINGS_PER_BUS_PER_HEX: Optional[int] = 10


# ── Retro-compatibilidad (deprecated, no usar en código nuevo) ──────
# Los nombres antiguos END_STOPS_* se mapean a sus equivalentes TURN_HEX_*.
# Esto permite que `from config import END_STOPS_MODE` siga funcionando
# durante la migración. Emite DeprecationWarning si se importa con `from`
# inspeccionando __getattr__ del módulo (PEP 562).
END_STOPS_MODE              = TURN_HEX_MODE                # alias
END_STOPS_MANUAL: List[str] = []                           # ya no aplica (era stop_codes)
END_STOPS_RESOLUTION        = TURN_HEX_RESOLUTION          # alias
END_STOPS_MIN_HEX_COUNT     = TURN_HEX_MIN_HEX_COUNT       # alias
END_STOPS_MIN_DIST_FRACTION = TURN_HEX_MIN_DIST_FRACTION   # alias (semántica cambió: ahora es del branch_length)
END_STOPS_MAX_SPEED         = TURN_HEX_MAX_SPEED           # alias
END_STOPS: List[str] = END_STOPS_MANUAL                    # alias legacy del legacy


@dataclass(frozen=True)
class TripParams:
    """
    Parámetros de la función `make_dir_trips` (segmentación de viajes).

    Justificación de defaults (v2, mayo 2026):
    - ``turn_zone_kring=1``: vecindad H3 del turn_hex (~50-80m efectivos).
    - ``end_stop_radius_m=80``: radio (m) sobre cada hex de la zona.
    - ``end_stop_cooldown_min=10``: subido de 3 → 10 min. Razón empírica:
      con zonas de turn_hex (clusters de 3-9 hex H3), el bus tarda en salir
      y reaparecer; 99.3% de los cortes ocurrían a <3 min cuando cooldown=3.
      Cooldown=10 reduce los cortes redundantes de 4,956 → 2,702 sin perder
      ningún trip válido real (los mismos 1,650 trips válidos se identifican
      con cooldown ∈ {3, 5, 10}; el cooldown solo filtra ruido de
      fragmentación).
    - ``max_gap_min=7``: P90 empírico (Lin & Zeng 1999).
    - ``max_implied_speed_kmh=180``: Schuessler & Axhausen (2009).
    - ``min_completeness=0.75``: valle bimodal empírico (Du & Aultman-Hall 2007).
    """
    # Corte por zona terminal (turn_zone)
    turn_zone_kring: int          = 1     # k-ring del turn_hex (v2)
    neighbor_window: int          = 1     # ⚠️ DEPRECATED alias de turn_zone_kring
    end_stop_radius_m: float      = 80.0  # radio (m) para detectar paso por turn_zone
    end_stop_cooldown_min: float  = 10.0  # minutos mínimos entre cortes en la misma zona

    # Cortes por gap
    max_gap_min: float            = 7.0   # P90 empírico (Lin & Zeng 1999)

    # Outlier de velocidad implícita (Schuessler & Axhausen 2009).
    # Default 180 km/h = 50 m/s (valor canónico del paper original).
    # Cortes solo si la velocidad implícita es físicamente imposible para un bus.
    max_implied_speed_kmh: float  = 180.0

    # ⚠️ DEPRECATED: max_gap_dist_m fragmentaba trips legítimos en sistemas
    # AVL con muestreo regular. Reemplazado por max_implied_speed_kmh.
    # Se mantiene para retrocompatibilidad; si != None, emite warning y se ignora.
    max_gap_dist_m: Optional[float] = None

    # Ventana operacional
    min_hour: float               = 6.5   # 06:30
    max_hour: float               = 20.0  # 20:00

    # Filtros de validez
    min_points: int               = 5
    min_dist_m: float             = 200.0
    min_completeness: float       = 0.75  # valle bimodal empírico


# Filtro de consistencia temporal (gap_filter = percentil)
# float 0-1  → percentil sobre la distribución empírica de huecos (P90 = 0.9)
# int > 1    → umbral absoluto en minutos
# None       → sin filtro (se devuelven los registros sin modificar)
GAP_FILTER_PCTL: float = 0.9    # P90
MAX_GAP_DISPLAY: float = 10      # corte empirico

# ────────────────────────────────────────────────────────────────────
# 3.bis · FEATURE ENGINEERING (paso 07)
# ────────────────────────────────────────────────────────────────────
#
# Parámetros de las features extendidas y el split histórico.

# Headway teórico de la ruta (minutos). Ajustar a la frecuencia real.
HEADWAY_IDEAL_MIN: float = 15.0

# Ventanas de las features dinámicas
SPEED_WINDOW_SECONDS:  int   = 300     # 5 min para speed_recent
HEADWAY_MAX_SECONDS:   float = 1800.0  # 30 min como máximo razonable
SPEED_ZERO_THRESHOLD:  float = 2.0     # km/h para considerar bus parado


# ────────────────────────────────────────────────────────────────────
# 4. RUTAS DEL FILESYSTEM
# ────────────────────────────────────────────────────────────────────

# Anclado a la ubicación de config.py (no al cwd), para que los pasos puedan
# ejecutarse directo desde cualquier carpeta. BUSONORA_ROOT lo sobreescribe.
PROJECT_ROOT = Path(os.environ.get("BUSONORA_ROOT", _FUNC_DIR.parent)).resolve()
DATA_RAW     = PROJECT_ROOT / "data" / "01_raw"
DATA_INTERIM = PROJECT_ROOT / "data" / "02_interim"


@dataclass(frozen=True)
class RoutePaths:
    """Construye las rutas de los artefactos de UNA ruta concreta."""
    int_route: int
    date_range: str
    analysis_range: str = ""
    data_raw: Path     = field(default_factory=lambda: DATA_RAW)
    data_interim: Path = field(default_factory=lambda: DATA_INTERIM)

    def __post_init__(self):
        if not self.analysis_range:
            object.__setattr__(self, "analysis_range", self.date_range)

    # ── Directorios base ──
    @property
    def raw_records_dir(self) -> Path:
        return self.data_raw / "records" / str(self.int_route)

    @property
    def raw_polylines_dir(self) -> Path:
        # Conservada por compatibilidad histórica. Los polylines ahora se cargan
        # desde `polylines_json`; este dir ya no contiene un .geojson por ruta.
        return self.data_raw / "polylines"

    # ── Archivos JSON globales (formato nuevo, 3 archivos en raw) ──
    @property
    def stops_json(self) -> Path:
        """stops.json — lista global de paradas con metadatos."""
        return self.data_raw / "stops.json"

    @property
    def stops_routes_json(self) -> Path:
        """stops_routes.json — mapeo {route_id: [{stop_code, stop_order}, ...]}."""
        return self.data_raw / "stops_routes.json"

    @property
    def polylines_json(self) -> Path:
        """polylines.geojson — anidado {operator: {route_id: {polyline: LineString}}}."""
        return self.data_raw / "polylines.geojson"

    @property
    def segments_geojson(self) -> Path:
        """segments.geojson — segmentos precalculados por tramo entre paradas.

        Soporta dos formatos (autodetectados por el loader):

        v3 (FeatureCollection plana, formato preferido):
            {"type": "FeatureCollection", "features": [
                {"type": "Feature",
                 "properties": {"line": "101", "code": "QMSS__EJ5E",
                                "start_stop": "QMSS", "end_stop": "EJ5E",
                                "name": "QMSS → EJ5E"},
                 "geometry": {"type": "LineString", "coordinates": [[lng, lat], ...]}}
            ]}
            El orden de los segmentos por ruta lo da `segments_route_json`.

        v2 (anidado, legacy):
            {operator: {lineKey: {segments: [{code, start_stop, end_stop,
                                              polyline: LineString}]}}}.
        """
        return self.data_raw / "segments.geojson"

    @property
    def segments_route_json(self) -> Path:
        """segments_route.json — mapeo {route_id: [{segment_code, segment_order}, ...]}.

        Define qué segmentos de `segments.geojson` pertenecen a cada ruta y en
        qué orden. Análogo a `stops_routes.json` pero para tramos. Solo se usa
        con el formato FeatureCollection (v3) de `segments.geojson`."""
        return self.data_raw / "segments_route.json"

    @property
    def interim_route_dir(self) -> Path:
        return self.data_interim / "routes" / str(self.int_route)

    @property
    def df_dir(self) -> Path:
        return self.interim_route_dir / "dataframes"

    @property
    def maps_dir(self) -> Path:
        return self.interim_route_dir / "maps"

    @property
    def reports_dir(self) -> Path:
        return self.interim_route_dir / "reports"

    @property
    def dashboards_dir(self) -> Path:
        return self.interim_route_dir / "dashboards"

    @property
    def pipeline_dir(self) -> Path:
        return self.interim_route_dir / "pipeline"

    @property
    def audit_dir(self) -> Path:
        return self.interim_route_dir / "audit"

    # ── Archivos individuales ──
    @property
    def df_rec_raw(self) -> Path:
        return self.raw_records_dir / f"df_rec_raw_{self.int_route}_{self.date_range}.parquet"

    @property
    def df_rec_clean(self) -> Path:
        return self.df_dir / f"df_rec_clean_{self.analysis_range}.parquet"

    @property
    def df_segments(self) -> Path:
        return self.df_dir / "df_segments.parquet"

    @property
    def turn_hex_json(self) -> Path:
        """Cache JSON de turn_hex resueltos (auto o manual). Formato v2."""
        return self.interim_route_dir / "turn_hex.json"

    @property
    def end_stops_json(self) -> Path:
        """⚠️ DEPRECATED. Cache JSON legacy de end_stops; use `turn_hex_json`."""
        return self.interim_route_dir / "end_stops.json"

    @property
    def df_route_direction(self) -> Path:
        return self.df_dir / f"df_rec_clean_direction_{self.analysis_range}.parquet"

    @property
    def df_filter_temporal(self) -> Path:
        return self.df_dir / f"df_filter_temporal_{GAP_FILTER_PCTL}_{self.analysis_range}.parquet"

    @property
    def df_trips(self) -> Path:
        return self.df_dir / f"df_rec_trips_{self.analysis_range}.parquet"

    @property
    def df_enriched(self) -> Path:
        """Paso 7a — next_stop, m_next_stop, s_next_stop."""
        return self.df_dir / f"df_rec_enrich_{self.analysis_range}.parquet"

    @property
    def df_interp(self) -> Path:
        """Paso 7b — + cum_dist_m, s_next_stop_interp, quantization_error_s."""
        return self.df_dir / f"df_interp_{self.analysis_range}.parquet"

    @property
    def df_seg_feat(self) -> Path:
        """Paso 7c — + features físicas por segmento + dinámicas extendidas."""
        return self.df_dir / f"df_seg_feat_{self.analysis_range}.parquet"

    @property
    def df_full(self) -> Path:
        """Paso 7 — dataset que leen los modelos (step_07.DF_FULL_COLUMNS)."""
        return self.df_dir / f"df_rec_full_{self.analysis_range}.parquet"

    @property
    def geojson_path(self) -> Path:
        """Obsoleto: ahora se usa `polylines_json`. Conservado por retro-compat."""
        return self.polylines_json

    # ── Mapas ──
    @property
    def overview_map(self) -> Path:
        return self.maps_dir / f"route_overview_{self.analysis_range}.html"

    @property
    def stopped_map(self) -> Path:
        return self.maps_dir / f"stopped_rec_{self.analysis_range}.html"

    @property
    def direction_map(self) -> Path:
        return self.maps_dir / f"direction_map_{self.analysis_range}.html"

    # ── Reportes ──
    @property
    def gap_distribution_png(self) -> Path:
        return self.reports_dir / "gap_distribution.png"

    def ensure_dirs(self) -> None:
        """Crea todos los directorios del pipeline si no existen."""
        for d in (self.raw_records_dir, self.df_dir, self.maps_dir,
                  self.reports_dir, self.dashboards_dir, self.pipeline_dir,
                  self.audit_dir):
            d.mkdir(parents=True, exist_ok=True)


# Instancia por defecto para Ruta 1701.
PATHS = RoutePaths(
    int_route      = INT_ROUTE,
    date_range     = DATE_RANGE,
    analysis_range = ANALYSIS_RANGE,
)
TRIP_PARAMS = TripParams()
