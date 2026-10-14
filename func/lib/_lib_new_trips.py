"""
_lib_new_trips.py  ·  Nuevo método de definición de viajes (trips) — v2
=======================================================================
Consolida el método de **trips por reversión (turnaround) + día**, refinado en
la conversación de *terminal definition method refinement*. Reemplaza a
``make_dir_trips``/``turn_hex`` como definidor de viajes:

    raw (bus_code, lat, lon, speed, time)  →  build_trips()  →  df de viajes
    (un ping por fila con ``trip_id``, ``seg_id``, flags) + métricas + mapa.

Pasos (todos en ``build_trips``):
  1. **Recuperación**: segmenta SOLO por turnaround (vuelta en U recurrente) + día
     → viajes terminal-a-terminal (sin cortar por gap/break, que sólo fragmentan).
  2. **Fusión de fragmentos chicos** (``merge_short_trips``): absorbe trips con
     pocos puntos o cobertura < ¼ de la polilínea en su vecino temporal.
  3. **Cap a 1 polilínea, cortando en el turnaround más cercano** (``cap_trip_length_v2``):
     un viaje no excede L; si lo hace (2 viajes juntos), se parte en el turnaround
     interno (no a media ruta) para que las piezas terminen en terminal.
  4. **Análisis** (``analyze_trips``): perp/off-ruta (coords crudas), salida a
     **depósito** al fin del día (off-ruta tras el último ping on-ruta → ``seg_id``
     con cola ``_end``), ½-rondas, **huérfanos** (cobertura < ¼ → etiqueta
     inicio/fin de ruta + tipo inactivo/disperso), y **validación de terminal**
     (cada viaje, salvo primero/último del día y huérfanos, debe iniciar y terminar
     cerca de un turnaround/terminal).

Mapas:
  - ``render_trips_map``  : interactivo por bus (slider por viaje, gradiente de
    color temporal por viaje, off-ruta/depósito resaltados, capas de inicio/fin de
    día y depósito dominante).
  - ``orphan_map``        : HTML aparte con los puntos huérfanos.

Config: constantes a nivel de módulo (ver ``configure``). El detector de
turnarounds/terminales vive en ``_lib_terminal_discovery`` (``td``).
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import _lib_terminal_discovery as td  # noqa: E402

# ──────────────────────────────────────────────────────────────────────
# Parámetros (defaults; override con configure(**kw))
# ──────────────────────────────────────────────────────────────────────
GAP_SPLIT_MIN = 20        # gap (min) que corta en la segmentación ORIGINAL (referencia)
BREAK_MIN = 10            # dwell (min) de break mid-route (segmentación de referencia)
OFFROUTE_M = 150.0        # perp > esto = punto fuera de la polilínea (candidato a depósito)
TRIP_MARGIN = 0.05        # un trip no excede L*(1+TRIP_MARGIN); si lo hace, se corta
MIN_TRIP_PTS = 5          # mín. puntos por trip; fragmentos menores se fusionan
MIN_TRIP_COV = 0.25       # cobertura mínima por trip = 1/4 de la polilínea
HALF_LO = 0.30            # cobertura para 'mitad de ronda' (rango [HALF_LO, HALF_HI])
HALF_HI = 0.70
JOIN_FULL_FRAC = 0.90     # 2 mitades cuya unión cubre >= esto -> ronda partida
ORPHAN_SPREAD_M = 150.0   # huérfano 'inactivo' si su nube cabe en < este radio (m)
ORPHAN_DWELL_MIN = 10.0   # ... y abarca >= estos minutos (parado en un lugar)
TERMINAL_NEAR_M = 300.0   # un trip 'bien formado' inicia y termina a <= este radio de terminal
MIN_TRIP_DENSITY = 0.15   # bins_únicos / span_bins: el viaje debe RECORRER su span, no
                          # saltarlo. Rechaza glitches de GPS (~0.04) sin matar legs ralos
                          # de la dirección rápida (~0.20); denso ≈ 0.8.
# --- fix del corte de viajes (ver docs/HANDOFF_CORTE_GEOMETRICO.md) ---
# Las trazas son IDA-Y-VUELTA (poly[0]==poly[-1]): el terminal A está en los extremos
# (lin_pos 0≈1) y B en el punto medio; un round trip A→B→A recorre toda la traza y, como
# su longitud ≈ L, no lo parte el cap → sale self-loop. El fix parte el round trip en su
# visita INTERIOR al terminal lejano (B), dando legs direccionales A→B / B→A.
INCLUDE_ENDPOINT_TERMINALS = False  # (revertido: mal en out-and-back; duplica el terminal A)
ENDPOINT_MERGE_FRAC = 0.05
MERGE_TERMINAL_AWARE = False        # (revertido: colapsa el conteo por over-split en orphans)
SPLIT_ROUND_TRIPS = True            # parte cada viaje en su visita interior a un terminal
# --- proyección coherente en el tiempo (ver docs/HANDOFF_CORTE_GEOMETRICO.md § Capa 2) ---
# El mínimo global salta entre las dos ramas de la traza y ese salto se contabiliza como
# avance: infla `cov` —que es el criterio de validez— y `dist_km`. Con 0 se vuelve al
# régimen viejo, que es lo que hay que poner para reproducir una corrida anterior.
COHERENT_PROJECTION = os.environ.get('COHERENT_PROJECTION', '1').lower() not in ('0', 'false', 'no')
# --- método v2 (oct 2026; ver docs/HANDOFF_METODO_V2.md) ---
# Cada pieza tiene interruptor: con COHERENT_PROJECTION=0, CYCLIC_ORDER=v1,
# TERMINAL_RULE=dwell y COVERAGE_BY_LEG=0 se reproduce la corrida tmin publicada.
#
# Orden cíclico: v1 juzgaba la alternancia por el SIGNO del avance sobre la polilínea —y en
# una traza cerrada las dos direcciones avanzan en el mismo sentido— y medía la cobertura
# en bins ÚNICOS, que el muestreo de ~60 s no llena en las rutas cortas. Tiraba el 29–99.9 %
# de los viajes válidos (902: 20,300 → 18). v2 usa la secuencia de terminales y la
# cobertura del arco recorrido. Ver ``mark_cyclic_order``.
CYCLIC_ORDER = os.environ.get('CYCLIC_ORDER', 'v2').lower()
# Terminales de servicio (TERMINAL_RULE):
#   'dwell' — la corrida tmin publicada: las 1–2 zonas con más descansos largos (AUTO_min).
#             En 8 rutas (201, 202, 302, 401, 402, 1501, 1502, 1902) quedan a 0.02–0.12 del
#             lazo —dos sitios de descanso del mismo complejo— y las legs son vueltas casi
#             completas; ver `terminal_gap_frac` en el funnel.
#   'file'  — las de TERMINALS_FILE, un JSON {ruta: [[lat, lon], …]}: p. ej. las que escoge
#             el explorador (hermosillo-bus-trips-explorer, 2.5 → terminales_servicio.json),
#             el par que más ciclos completos de ida y vuelta produce. Una ruta que no está
#             en el archivo usa 'dwell'. Ver docs/HANDOFF_METODO_V2.md §2.2.
TERMINAL_RULE = os.environ.get('TERMINAL_RULE', 'dwell').lower()
TERMINALS_FILE = os.environ.get('TERMINALS_FILE', '')
# La cobertura mínima de un viaje válido (0.375 del lazo en modo dwell) supone legs de ½
# lazo; con dos terminales a un hueco g se usa 0.75·g. Ver build_trips. 0 = régimen viejo.
COVERAGE_BY_LEG = os.environ.get('COVERAGE_BY_LEG', '1').lower() not in ('0', 'false', 'no')
# Corte por cruce de km (TRIP_CUT=cruce; oct 2026, Comparativo_metodos_viajes.md §8 y §10):
# el viaje termina cuando el progreso continuo del bus sobre la traza pasa por el km de un
# punto de corte, no en el registro más cercano a una terminal. Los puntos son los de la ruta
# en TERMINALS_FILE (TERMINAL_RULE=file), cada uno [lat, lon, km]: con uno, el viaje es la
# vuelta; con dos, las legs A/B. Lo que pase entre dos cruces (esperas, la otra terminal,
# patios, un ramal donde dan la vuelta antes) queda dentro del viaje. Ver
# ``_build_trips_cruce``. Vacío = el corte de siempre (el ``trip_cut`` de build_trips).
TRIP_CUT = os.environ.get('TRIP_CUT', '').strip().lower()
CRUCE_VMAX = 25.0         # m/s: un avance sobre la traza mayor que esto·dt + CRUCE_SALTO_M es un salto
CRUCE_SALTO_M = 300.0
ATAJO_MAX_M = 8000.0      # un salto hacia adelante de hasta esto, tras el que el bus sigue normal
ATAJO_SIGUE = 3           # ATAJO_SIGUE registros, es un atajo (no parte el tramo; se marca)
CRUCE_INFERIDO_S = 300    # un cruce en un hueco de más de esto no se vio: se dedujo
CRUCE_KM_TOL_M = 100.0    # el km de un punto de corte debe caer a ≤ esto de su (lat, lon)
CRUCE_DIRECCIONAL = float(os.environ.get('CRUCE_DIRECCIONAL', '1'))
                          # en la proyección coherente, retroceder d sobre la traza cuesta además
                          # este peso × d: sin él, en las calles de ida y vuelta el bus cae en la
                          # otra pasada (td._project_many_coherent, `direccional`)
CRUCE_RECORRIDO_MAX = 1.5 # un viaje que recorrió (en línea recta, registro a registro) más que
                          # esto × su arco se saltó un cruce: la proyección se atascó en la otra
                          # pasada y M(t) no avanzó (20000: 7 % de vueltas dobles en marzo)

_PARAMS = ('GAP_SPLIT_MIN', 'BREAK_MIN', 'OFFROUTE_M', 'TRIP_MARGIN', 'MIN_TRIP_PTS',
           'MIN_TRIP_COV', 'HALF_LO', 'HALF_HI', 'JOIN_FULL_FRAC', 'ORPHAN_SPREAD_M',
           'ORPHAN_DWELL_MIN', 'TERMINAL_NEAR_M', 'MIN_TRIP_DENSITY',
           'INCLUDE_ENDPOINT_TERMINALS', 'ENDPOINT_MERGE_FRAC', 'MERGE_TERMINAL_AWARE',
           'SPLIT_ROUND_TRIPS', 'COHERENT_PROJECTION', 'CYCLIC_ORDER',
           'TERMINAL_RULE', 'TERMINALS_FILE', 'COVERAGE_BY_LEG',
           'TRIP_CUT', 'CRUCE_VMAX', 'CRUCE_SALTO_M', 'ATAJO_MAX_M', 'ATAJO_SIGUE',
           'CRUCE_INFERIDO_S', 'CRUCE_KM_TOL_M', 'CRUCE_RECORRIDO_MAX', 'CRUCE_DIRECCIONAL')


def method_tag(terminal_coords=None):
    """Marcador del método para la caché de viajes. El manifest invalida por parámetros y
    no por código, así que toda pieza que cambie el corte tiene que aparecer aquí (con
    terminales fijadas, también sus coordenadas; con TRIP_CUT=cruce, también su km)."""
    term = TERMINAL_RULE
    if terminal_coords:
        term += "_" + "_".join(",".join("-" if v is None else f"{float(v):.5f}" for v in p)
                               for p in terminal_coords)
    tag = (f"split{int(SPLIT_ROUND_TRIPS)}_dens{MIN_TRIP_DENSITY}_spanv1"
           f"_coh{int(COHERENT_PROJECTION)}_cyc{CYCLIC_ORDER}"
           f"_term{term}_covleg{int(COVERAGE_BY_LEG)}")
    if TRIP_CUT:
        tag += (f"_cut{TRIP_CUT}_v{CRUCE_VMAX:g}+{CRUCE_SALTO_M:g}"
                f"_atajo{ATAJO_MAX_M:g}x{ATAJO_SIGUE}_rec{CRUCE_RECORRIDO_MAX:g}"
                f"_dir{CRUCE_DIRECCIONAL:g}")
    return tag


def load_terminal_coords(route):
    """Con ``TERMINAL_RULE='file'``, las terminales de ``route`` en ``TERMINALS_FILE``
    (``[(lat, lon), …]``); None si la regla es otra o la ruta no está.

    Cada punto del archivo es ``[lat, lon]`` o ``[lat, lon, km]``. Con ``TRIP_CUT=cruce``
    se devuelve ``(lat, lon, km)`` (km None si no viene): el km decide en qué paso de la
    traza se corta cuando la traza pasa dos veces cerca del mismo punto."""
    import json
    if TERMINAL_RULE != 'file' or not TERMINALS_FILE:
        return None
    data = json.loads(open(TERMINALS_FILE, encoding='utf-8').read())
    coords = data.get(str(int(route)))
    if not coords:
        return None
    if TRIP_CUT == 'cruce':
        return [(float(p[0]), float(p[1]), float(p[2]) if len(p) > 2 and p[2] is not None else None)
                for p in coords]
    return [(float(p[0]), float(p[1])) for p in coords]


def _project_seq(d, poly, cum, L, group_col='trip_id', order_col='time'):
    """Proyecta los pings de ``d`` **como secuencias** (una por ``group_col``).

    Drop-in de ``td._project_many_to_polyline`` para los sitios donde los puntos son
    los pings de un viaje en orden: mismo 4-tuple ``(pos, perp, snap_lat, snap_lon)``.

    ⚠ El ``perp`` que devuelve es el del **mínimo global**, no el del punto elegido:
    «¿está este ping sobre la ruta?» es una pregunta sobre la geometría y no sobre la
    secuencia, así que el flag ``off`` no se mueve respecto del régimen viejo. Lo que
    cambia es ``pos`` — y con él ``bin``, ``lin_pos`` y la cobertura.

    Con ``COHERENT_PROJECTION=0`` en el entorno cae al mínimo global de siempre.
    """
    import polars as pl
    lats = d['latitude'].cast(pl.Float64).to_numpy()
    lons = d['longitude'].cast(pl.Float64).to_numpy()
    if not COHERENT_PROJECTION or group_col not in d.columns:
        return td._project_many_to_polyline(lats, lons, poly, cum)
    orden = d[order_col].to_numpy() if order_col in d.columns else None
    pos, _perp, slat, slon, perp_min = td._project_many_coherent(
        lats, lons, d[group_col].to_numpy(), poly, cum, L=L, order=orden)
    return pos, perp_min, slat, slon


def configure(**kw):
    """Sobrescribe parámetros del módulo. Ej: configure(OFFROUTE_M=120, TERMINAL_NEAR_M=250)."""
    g = globals()
    for k, v in kw.items():
        if k not in _PARAMS:
            raise KeyError(f'parámetro desconocido: {k!r} (válidos: {_PARAMS})')
        g[k] = v


def load_route_polyline(geojson_path, route_id):
    """Conveniencia: polilínea + distancias acumuladas + L. -> (poly, cum, L)."""
    poly = td.load_route_polyline(geojson_path, route_id)
    cum = td._cumulative_distances(poly)
    return poly, cum, float(cum[-1])


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────
def _dominant_zone(coords, lat0, cell_m=200.0):
    """Devuelve (lat, lon, fraccion, n) de la celda (cell_m) más usada, o None."""
    if not coords:
        return None
    la = np.array([c[0] for c in coords]); lo = np.array([c[1] for c in coords])
    cos0 = float(np.cos(np.radians(lat0)))
    cx = np.round((lo - lo.mean()) * 111320.0 * cos0 / cell_m)
    cy = np.round((la - la.mean()) * 111320.0 / cell_m)
    keys = cx * 1e6 + cy
    vals, counts = np.unique(keys, return_counts=True)
    k = vals[int(counts.argmax())]; mask = keys == k
    return float(la[mask].mean()), float(lo[mask].mean()), float(counts.max()) / len(coords), int(counts.max())


def _zone_consistency(coords, lat0, cell_m=200.0):
    """Fracción de puntos en la celda (cell_m) más densa = qué tan consistente es el lugar."""
    if not coords:
        return 0.0
    la = np.array([c[0] for c in coords]); lo = np.array([c[1] for c in coords])
    cos0 = float(np.cos(np.radians(lat0)))
    cx = np.round((lo - lo.mean()) * 111320.0 * cos0 / cell_m)
    cy = np.round((la - la.mean()) * 111320.0 / cell_m)
    _, counts = np.unique(cx * 1e6 + cy, return_counts=True)
    return float(counts.max()) / len(coords)


def cap_trip_length(dt, L, margin, lat0):
    """Cap por longitud cortando en el múltiplo exacto de L (v1; conservado para comparar)."""
    import polars as pl
    if dt.is_empty():
        return dt
    cos0 = float(np.cos(np.radians(lat0)))
    d = dt.sort(['trip_id', 'time']).with_columns([pl.col('latitude').cast(pl.Float64),
                                                   pl.col('longitude').cast(pl.Float64)])
    d = d.with_columns([(pl.col('latitude').diff().over('trip_id') * 111320).alias('_dy'),
                        (pl.col('longitude').diff().over('trip_id') * 111320 * cos0).alias('_dx')])
    d = d.with_columns((pl.col('_dx')**2 + pl.col('_dy')**2).sqrt().fill_null(0.0).alias('_st'))
    d = d.with_columns(pl.col('_st').cum_sum().over('trip_id').alias('_cum'))
    d = d.join(d.group_by('trip_id').agg(pl.col('_cum').max().alias('_tot')), on='trip_id')
    d = d.with_columns(pl.when(pl.col('_tot') > L * (1.0 + margin))
                         .then((pl.col('_cum') / L).floor().cast(pl.Int64))
                         .otherwise(0).alias('_sub'))
    d = d.with_columns((pl.col('trip_id') + pl.lit('_L') + pl.col('_sub').cast(pl.Utf8)).alias('trip_id'))
    return d.drop(['_dy', '_dx', '_st', '_cum', '_tot', '_sub'])


def cap_trip_length_v2(dt, L, margin, lat0, turn_by_bus):
    """Cap por longitud cortando en el TURNAROUND interno más cercano (v2, recomendado).
    Si un trip excede L*(1+margin), se parte en los turnarounds que contiene (para que las
    piezas terminen en terminal); si no hay turnaround interno, cae al corte por L."""
    import polars as pl
    if dt.is_empty():
        return dt
    cos0 = float(np.cos(np.radians(lat0)))
    d = dt.sort(['trip_id', 'time']).with_columns([pl.col('latitude').cast(pl.Float64),
                                                   pl.col('longitude').cast(pl.Float64)])
    d = d.with_columns([(pl.col('latitude').diff().over('trip_id') * 111320).alias('_dy'),
                        (pl.col('longitude').diff().over('trip_id') * 111320 * cos0).alias('_dx')])
    d = d.with_columns((pl.col('_dx')**2 + pl.col('_dy')**2).sqrt().fill_null(0.0).alias('_st'))
    d = d.with_columns([pl.col('_st').cum_sum().over('trip_id').alias('_cum'),
                        pl.col('time').dt.timestamp('ms').alias('_tms')])
    tot = d.group_by('trip_id').agg(pl.col('_cum').max().alias('_tot'))
    long_ids = set(tot.filter(pl.col('_tot') > L * (1 + margin))['trip_id'].to_list())
    drop = ['_dy', '_dx', '_st', '_cum', '_tms']
    if not long_ids:
        return d.with_columns((pl.col('trip_id') + pl.lit('_L0')).alias('trip_id')).drop(drop)
    parts = [d.filter(~pl.col('trip_id').is_in(list(long_ids)))
              .with_columns((pl.col('trip_id') + pl.lit('_L0')).alias('trip_id'))]
    for sub in d.filter(pl.col('trip_id').is_in(list(long_ids))).partition_by('trip_id', maintain_order=True):
        bus = sub['bus_code'][0]; tms = sub['_tms'].to_numpy(); cum = sub['_cum'].to_numpy()
        turns = turn_by_bus.get(bus, np.array([], dtype='int64'))
        internal = turns[(turns > tms[0]) & (turns < tms[-1])]
        s = np.searchsorted(internal, tms, side='right') if len(internal) else (cum / L).astype('int64')
        parts.append(sub.with_columns(pl.Series('_sub', s)).with_columns(
            (pl.col('trip_id') + pl.lit('_L') + pl.col('_sub').cast(pl.Utf8)).alias('trip_id')).drop('_sub'))
    return pl.concat(parts).drop(drop)


def merge_short_trips(dt, min_pts, min_cov_frac, nbin=100, term_coords=None, lat0=None,
                      near_m=None):
    """Absorbe trips 'chicos' (< min_pts puntos o cobertura < min_cov_frac) en su vecino
    temporal del mismo bus+día. Requiere columna 'bin' (lin_pos*100).

    **Terminal-aware** (si ``MERGE_TERMINAL_AWARE`` y se pasan ``term_coords``/``lat0``):
    en vez de acumular hasta cobertura ``min_cov_frac`` —que pega medios-lazos y crea
    self-loops— cierra el grupo cuando el fragmento **termina en un terminal DISTINTO**
    del de inicio (= se recorrió de terminal a terminal, un leg completo). Un stub que
    empieza y termina en el mismo terminal se sigue acumulando. Ver
    docs/HANDOFF_CORTE_GEOMETRICO.md."""
    import polars as pl
    if dt.is_empty():
        return dt
    info = dt.sort(['trip_id', 'time']).group_by('trip_id', maintain_order=True).agg([
        pl.col('bus_code').first().alias('bus'), pl.col('_day').first().alias('day'),
        pl.col('time').min().alias('t0'), pl.len().alias('n'), pl.col('bin').unique().alias('bins'),
        pl.col('latitude').cast(pl.Float64).first().alias('la_s'),
        pl.col('longitude').cast(pl.Float64).first().alias('lo_s'),
        pl.col('latitude').cast(pl.Float64).last().alias('la_e'),
        pl.col('longitude').cast(pl.Float64).last().alias('lo_e')])

    ta = bool(MERGE_TERMINAL_AWARE and term_coords and lat0 is not None)
    if ta:
        cos0 = float(np.cos(np.radians(lat0)))
        tla = np.array([float(t[0]) for t in term_coords])
        tlo = np.array([float(t[1]) for t in term_coords])
        nm = float(near_m) if near_m is not None else 2.0 * TERMINAL_NEAR_M

        def _term(la, lo):
            dy = (la - tla) * 111320.0; dx = (lo - tlo) * 111320.0 * cos0
            d2 = dx * dx + dy * dy; k = int(np.argmin(d2))
            return k, bool(d2[k] <= nm * nm)     # (idx terminal más cercano, ¿cerca?)

    mp = {}
    for (bus, day), grp in info.sort('t0').group_by(['bus', 'day'], maintain_order=True):
        groups = []; cur_ids = []; cur_bins = set(); cur_n = 0; cur_start = None
        for r in grp.iter_rows(named=True):
            if ta and not cur_ids:
                st, near = _term(r['la_s'], r['lo_s'])
                cur_start = st if near else None
            cur_ids.append(r['trip_id']); cur_n += r['n']; cur_bins |= set(r['bins'])
            if ta:
                close = False
                if cur_n >= min_pts:
                    en, near = _term(r['la_e'], r['lo_e'])
                    # cierra al llegar a un terminal DISTINTO del de inicio (leg completo);
                    # si el grupo no empezó en terminal, cierra al tocar cualquiera.
                    if near and (cur_start is None or en != cur_start):
                        close = True
            else:
                close = len(cur_bins) / nbin >= min_cov_frac and cur_n >= min_pts
            if close:
                groups.append(cur_ids); cur_ids = []; cur_bins = set(); cur_n = 0; cur_start = None
        if cur_ids:
            if groups: groups[-1].extend(cur_ids)
            else: groups.append(cur_ids)
        for gi, ids in enumerate(groups):
            mid = f'{bus}_M{day}_{gi}'
            for tid in ids: mp[tid] = mid
    return dt.with_columns(pl.col('trip_id').replace(mp).alias('trip_id'))


def orphan_table(dt, orph, lat0):
    """Métricas por trip HUÉRFANO (cobertura < MIN_TRIP_COV): dwell, spread, centroide,
    etiqueta inicio/fin de ruta y tipo (inactivo=parado en un sitio | disperso)."""
    import polars as pl
    o_ids = orph['trip_id'].to_list()
    if not o_ids:
        return pl.DataFrame(schema={'trip_id': pl.Utf8, 'label': pl.Utf8, 'tipo': pl.Utf8,
                                    'lat': pl.Float64, 'lon': pl.Float64, 'n': pl.Int64,
                                    'dwell_min': pl.Float64, 'spread_m': pl.Float64, 'cov': pl.Float64})
    cos0 = float(np.cos(np.radians(lat0)))
    od = dt.filter(pl.col('trip_id').is_in(o_ids)).with_columns([
        pl.col('latitude').cast(pl.Float64), pl.col('longitude').cast(pl.Float64)])
    agg = od.group_by('trip_id').agg([
        pl.col('latitude').mean().alias('lat'), pl.col('longitude').mean().alias('lon'),
        pl.col('latitude').min().alias('la0'), pl.col('latitude').max().alias('la1'),
        pl.col('longitude').min().alias('lo0'), pl.col('longitude').max().alias('lo1'),
        pl.col('time').min().alias('t0'), pl.col('time').max().alias('t1'), pl.len().alias('n')])
    agg = agg.with_columns([
        ((pl.col('t1') - pl.col('t0')).dt.total_seconds() / 60).alias('dwell_min'),
        (((pl.col('la1') - pl.col('la0')) * 111320) ** 2
         + ((pl.col('lo1') - pl.col('lo0')) * 111320 * cos0) ** 2).sqrt().alias('spread_m')])
    agg = agg.join(orph.select(['trip_id', 'rk', 'nday', 'cov']), on='trip_id')
    agg = agg.with_columns([
        pl.when(pl.col('rk') == 1).then(pl.lit('inicio'))
          .when(pl.col('rk') == pl.col('nday')).then(pl.lit('fin'))
          .otherwise(pl.lit('intermedio')).alias('label'),
        pl.when((pl.col('spread_m') < ORPHAN_SPREAD_M) & (pl.col('dwell_min') >= ORPHAN_DWELL_MIN))
          .then(pl.lit('inactivo')).otherwise(pl.lit('disperso')).alias('tipo')])
    return agg.select(['trip_id', 'label', 'tipo', 'lat', 'lon', 'n', 'dwell_min', 'spread_m', 'cov'])


def analyze_trips(df, dt, poly, _cum, L, lat0, term_coords, near_m=None, valid_cov=None):
    """Sobre un dt YA cortado: perp/off, depósito al fin del día, métricas por trip,
    ½-rondas, huérfanos y validación de terminal. -> (dt, trip_meta, stats, otab, frag).

    near_m: radio para considerar que un viaje inicia/termina en terminal (default
    ``TERMINAL_NEAR_M``; en modo dwell conviene = el radio de visita del corte).
    valid_cov: cobertura mínima de la polilínea para marcar un viaje válido (default
    ``2*MIN_TRIP_COV``; bájalo para rutas cuyo leg terminal-a-terminal sea ~½ del lazo)."""
    import polars as pl
    near_m = float(near_m) if near_m is not None else TERMINAL_NEAR_M
    valid_cov = float(valid_cov) if valid_cov is not None else 2 * MIN_TRIP_COV
    _pos, _perp, _, _ = _project_seq(dt, poly, _cum, L)
    dt = dt.with_columns([pl.Series('perp', _perp)]).with_columns((pl.col('perp') > OFFROUTE_M).alias('off'))
    # salida a DEPÓSITO al terminar el día: pings off-ruta tras el último ping ON-ruta del bus+día
    _onr = dt.filter(~pl.col('off')).group_by(['bus_code', '_day']).agg(pl.col('time').max().alias('_last_on'))
    dt = dt.join(_onr, on=['bus_code', '_day'], how='left').with_columns(
        (pl.col('off') & pl.col('_last_on').is_not_null() & (pl.col('time') > pl.col('_last_on'))).alias('depot_run')).drop('_last_on')
    dt = dt.with_columns(pl.when(pl.col('depot_run')).then(pl.col('trip_id') + pl.lit('_end'))
                           .otherwise(pl.col('trip_id')).alias('seg_id'))
    if 'bin' not in dt.columns:
        dt = dt.with_columns(pl.Series('bin', (np.clip(_pos / L, 0.0, 0.999) * 100).astype('int64')))
    frag = dt.group_by('trip_id').agg([
        pl.col('trip_id_orig').n_unique().alias('n_frag'),
        pl.col('_day').first().alias('day'), pl.col('bus_code').first().alias('bus'),
        pl.col('time').min().alias('t0'), pl.col('off').mean().alias('off_frac'),
        pl.len().alias('n'), pl.col('bin').n_unique().alias('nbin'),
        (pl.col('bin').max() - pl.col('bin').min() + 1).alias('span_bins')])
    # Cobertura = SPAN de bin (bmax−bmin), NO conteo de bins únicos: robusto al
    # submuestreo de GPS. Con cadencia rala (p.ej. 902 a 61 s ≈ 2 bins/ping) un leg
    # completo A→B abarca ~0.44 aunque toque pocos bins únicos. Ver HANDOFF.
    frag = frag.with_columns([(pl.col('span_bins') / 100).alias('cov'),
                              pl.col('t0').rank('ordinal').over(['bus', 'day']).alias('rk')])
    frag = frag.join(frag.group_by(['bus', 'day']).agg(pl.len().alias('nday')), on=['bus', 'day'])
    _bins = {r['trip_id']: set(r['bins']) for r in
             dt.group_by('trip_id').agg(pl.col('bin').unique().alias('bins')).iter_rows(named=True)}
    half_ids = set(); half_pairs = 0
    for (bus, day), grp in frag.sort('t0').group_by(['bus', 'day'], maintain_order=True):
        rr = list(grp.iter_rows(named=True))
        for i in range(len(rr) - 1):
            a, b = rr[i], rr[i + 1]
            if HALF_LO <= a['cov'] <= HALF_HI and HALF_LO <= b['cov'] <= HALF_HI:
                if len(_bins[a['trip_id']] | _bins[b['trip_id']]) / 100 >= JOIN_FULL_FRAC:
                    half_pairs += 1; half_ids.add(a['trip_id']); half_ids.add(b['trip_id'])
    orph = frag.filter(pl.col('cov') < MIN_TRIP_COV)
    o_label = {r['trip_id']: ('inicio' if r['rk'] == 1 else 'fin' if r['rk'] == r['nday'] else 'intermedio')
               for r in orph.iter_rows(named=True)}
    _cos0 = float(np.cos(np.radians(lat0)))
    def _near_term(la, lo):
        for tla, tlo in term_coords:
            dy = (la - tla) * 111320.0; dx = (lo - tlo) * 111320.0 * _cos0
            if (dx * dx + dy * dy) ** 0.5 <= near_m:
                return True
        return False
    _ends = (dt.sort(['trip_id', 'time']).group_by('trip_id').agg([
        pl.col('latitude').cast(pl.Float64).first().alias('la_s'), pl.col('longitude').cast(pl.Float64).first().alias('lo_s'),
        pl.col('latitude').cast(pl.Float64).last().alias('la_e'), pl.col('longitude').cast(pl.Float64).last().alias('lo_e')])
        .join(frag.select(['trip_id', 'rk', 'nday']), on='trip_id'))
    # Validación asimétrica de terminal: el PRIMER viaje del día puede iniciar donde sea
    # pero debe TERMINAR en terminal_like; el ÚLTIMO debe INICIAR en terminal_like (y puede
    # terminar donde sea, p.ej. depósito); los demás deben iniciar Y terminar en terminal_like.
    bad_term = set()
    for _r in _ends.iter_rows(named=True):
        if _r['trip_id'] in o_label:
            continue
        need_start = _r['rk'] > 1               # todos menos el primero del día inician en terminal
        need_end   = _r['rk'] < _r['nday']      # todos menos el último del día terminan en terminal
        bad = ((need_start and not _near_term(_r['la_s'], _r['lo_s'])) or
               (need_end and not _near_term(_r['la_e'], _r['lo_e'])))
        if bad:
            bad_term.add(_r['trip_id'])
    depot_trips = set(dt.filter(pl.col('depot_run'))['trip_id'].unique().to_list())

    def _flag(r):
        fl = []
        if r['n_frag'] >= 2: fl.append(f"recuperado×{r['n_frag']}")
        if r['trip_id'] in half_ids: fl.append('½ ronda (junta con vecino)')
        if r['trip_id'] in o_label: fl.append(f"huérfano ({o_label[r['trip_id']]} de ruta)")
        if r['trip_id'] in bad_term: fl.append('⚠ fuera de terminal')
        if r['trip_id'] in depot_trips: fl.append('→ sale a depósito (cola _end)')
        if r['rk'] == 1: fl.append('primero del día')
        if r['rk'] == r['nday']: fl.append('último del día')
        if r['off_frac'] >= 0.3: fl.append(f"off-ruta {100*r['off_frac']:.0f}%")
        return ' · '.join(fl)
    trip_meta = {r['trip_id']: _flag(r) for r in frag.iter_rows(named=True)}
    otab = orphan_table(dt, orph, lat0)
    # marca de validez para el resto del pipeline (viaje terminal-a-terminal utilizable)
    valid_ids = set(frag.filter((pl.col('cov') >= valid_cov)
                                & ((pl.col('nbin') / pl.col('span_bins')) >= MIN_TRIP_DENSITY)
                                & (~pl.col('trip_id').is_in(list(bad_term)))
                                & (~pl.col('trip_id').is_in(list(o_label.keys())))
                                & (~pl.col('trip_id').is_in(list(depot_trips))))['trip_id'].to_list())
    dt = dt.with_columns(pl.col('trip_id').is_in(list(valid_ids)).alias('trip_valid'))
    last = frag.filter(pl.col('rk') == pl.col('nday'))
    stats = {'pings': df.height, 'buses': df['bus_code'].n_unique(),
             'trips_rec': int(dt['trip_id'].n_unique()), 'trips_valid': int(len(valid_ids)),
             'merged_ge2': int(frag.filter(pl.col('n_frag') >= 2).height), 'max_frag': int(frag['n_frag'].max()),
             'min_pts': int(frag['n'].min()), 'min_cov': round(float(frag['cov'].min()), 2),
             'half_pairs': int(half_pairs), 'orphans': int(orph.height),
             'orph_inicio': sum(1 for v in o_label.values() if v == 'inicio'),
             'orph_fin': sum(1 for v in o_label.values() if v == 'fin'),
             'orph_inactivo': int(otab.filter(pl.col('tipo') == 'inactivo').height) if otab.height else 0,
             'bad_terminal': int(len(bad_term)),
             'off_pct': round(100 * float(dt['off'].mean()), 2),
             'last_off_ge30': int(last.filter(pl.col('off_frac') >= 0.3).height), 'last_n': int(last.height)}
    _sed = dt.sort(['bus_code', '_day', 'time']).group_by(['bus_code', '_day']).agg([
        pl.col('latitude').cast(pl.Float64).first().alias('slat'), pl.col('longitude').cast(pl.Float64).first().alias('slon'),
        pl.col('latitude').cast(pl.Float64).last().alias('elat'), pl.col('longitude').cast(pl.Float64).last().alias('elon')])
    stats['bus_days'] = _sed.height
    stats['start_consistency'] = round(_zone_consistency(list(zip(_sed['slat'].to_list(), _sed['slon'].to_list())), lat0), 2)
    stats['end_consistency'] = round(_zone_consistency(list(zip(_sed['elat'].to_list(), _sed['elon'].to_list())), lat0), 2)
    _dr = dt.filter(pl.col('depot_run'))
    stats['depot_run_pings'] = int(_dr.height)
    stats['depot_run_busdays'] = int(_dr.select(['bus_code', '_day']).unique().height)
    return dt, trip_meta, stats, otab, frag


def _resolve_dominance_frac(clusters, mode, grid=None):
    """Resuelve ``dominance_frac`` para el descubrimiento de terminales por dwell.

    El tipo de ``mode`` decide la semántica:
      - **float** (0.2, 0.5, …): **fracción de dominancia** directa (un cluster es
        terminal si ``n_dwells >= frac × max``). Comportamiento original.
      - **int** (1, 2, …): **número de terminales deseado** — elige el frac que deja las
        **top-N** zonas (recurrentes + on-route) por volumen de dwells. Si N ≥ nº de
        candidatos, se quedan todas (frac=0).
      - **str** ``'AUTO_min'`` / ``'full_AUTO'`` (= los viejos ``min_auto`` / ``full_auto``):
        calibración **por ruta** — barre el frac y elige el menor que deje 1–2 clusters
        dominantes. Para una sola ruta ambos modos coinciden; para la calibración
        multi-ruta usa el batch ``all_routes``.

    Devuelve ``(frac, counts)``.
    """
    import polars as pl

    # --- str: modos AUTO ---
    if isinstance(mode, str):
        if clusters is None or clusters.height == 0:
            return 0.2, {}
        cand = clusters.filter(pl.col('recurrent') & pl.col('on_route'))
        sizes = cand['n_dwells'].to_numpy().astype(float) if cand.height else np.array([])
        gmax = float(sizes.max()) if sizes.size else 0.0
        if grid is None:
            grid = [round(float(x), 2) for x in np.arange(0.0, 0.96, 0.05)]
        counts = {f: (int((sizes >= f * gmax).sum()) if gmax > 0 else 0) for f in grid}
        for f in grid:                  # AUTO_min / full_AUTO (1 ruta): menor frac con 1–2
            if counts[f] in (1, 2):
                return f, counts
        best = min(grid, key=lambda f: (abs(counts[f] - 2), f))
        return best, counts

    # --- int: nº de terminales deseado (bool cuenta como int) ---
    if isinstance(mode, (int, np.integer)) and not isinstance(mode, bool):
        if clusters is None or clusters.height == 0:
            return 0.0, None
        cand = clusters.filter(pl.col('recurrent') & pl.col('on_route'))
        sizes = sorted(cand['n_dwells'].to_numpy().astype(float).tolist(), reverse=True) if cand.height else []
        if not sizes:
            return 0.0, None
        gmax = sizes[0]
        n_want = max(1, int(mode))
        if n_want >= len(sizes) or gmax <= 0:
            return 0.0, None            # quedarse con todas las candidatas
        # piso = volumen de la N-ésima zona → quedan las top-N (más empates en la frontera)
        return float(sizes[n_want - 1] / gmax), None

    # --- float u otro numérico: fracción directa ---
    return float(mode), None


def _augment_with_polyline_endpoints(term_coords, poly, cum, L, merge_frac=None):
    """Añade los EXTREMOS de la polilínea (lin_pos 0 y 1) al set de terminales si no
    hay ya un terminal cerca. Los extremos de la traza son terminales por
    construcción; el descubrimiento por dwell los pierde cuando el retorno es rápido
    (el bus no para en el endpoint). Ver docs/HANDOFF_CORTE_GEOMETRICO.md.

    `merge_frac`: no añadir un extremo si ya existe un terminal a <= `merge_frac`
    (distancia en lin_pos, NO cíclica: los dos extremos son puntos distintos).
    """
    if not poly:
        return list(term_coords)
    mf = float(merge_frac) if merge_frac is not None else ENDPOINT_MERGE_FRAC
    existing = []
    for (la, lo) in term_coords:
        _, _, _, pos = td._project_to_polyline(la, lo, poly, cum)
        existing.append(pos / L if L else 0.0)
    out = list(term_coords)
    for ep_pos, ep in ((0.0, poly[0]), (1.0, poly[-1])):
        if not any(abs(ep_pos - e) <= mf for e in existing):
            out.append((float(ep[0]), float(ep[1])))
    return out


def _split_trips_at_interior_terminals(dt, term_coords, lat0, visit_max_m=None, verbose=False):
    """Parte cada viaje en sus visitas **interiores** a un terminal. Un round trip
    A→B→A (traza ida-y-vuelta: empieza y termina en A, pasa por B en el medio) se
    corta en el acercamiento más cercano a B → legs direccionales A→B y B→A.

    Una "visita" = tramo contiguo de pings a < ``visit_max_m`` del terminal más
    cercano; el punto de corte es el ping de distancia mínima del tramo. Solo se
    parte en visitas **interiores** (ni el 1er ni el último ping del viaje), así que
    un leg ya direccional (A→B, con B en su extremo) no se toca. El ping de corte
    queda como ÚLTIMO del leg previo; el siguiente leg arranca justo después (aún
    junto al terminal). Ver docs/HANDOFF_CORTE_GEOMETRICO.md."""
    import polars as pl
    if not term_coords or dt.is_empty():
        return dt
    visit_max = float(visit_max_m) if visit_max_m is not None else 2.0 * TERMINAL_NEAR_M
    cos0 = float(np.cos(np.radians(lat0)))
    tla = np.array([float(c[0]) for c in term_coords]); tlo = np.array([float(c[1]) for c in term_coords])
    parts = []
    n_split = 0
    for (tid,), sub in dt.sort(['trip_id', 'time']).group_by(['trip_id'], maintain_order=True):
        la = sub['latitude'].cast(pl.Float64).to_numpy()
        lo = sub['longitude'].cast(pl.Float64).to_numpy()
        n = la.shape[0]
        if n < 3:
            parts.append(sub); continue
        dy = (la[:, None] - tla[None, :]) * 111320.0
        dx = (lo[:, None] - tlo[None, :]) * 111320.0 * cos0
        dmat = np.sqrt(dx * dx + dy * dy)               # n × terminales
        dist = dmat.min(axis=1)
        nearest = dmat.argmin(axis=1)                   # terminal más cercano por ping
        # Partir SOLO round trips: inicio y fin en el MISMO terminal. Un viaje ya
        # direccional (inicio≠fin) NO se toca (evita mangle en rutas no-out-and-back).
        if nearest[0] != nearest[-1]:
            parts.append(sub); continue
        start_t = int(nearest[0])
        below = dist < visit_max
        seg = np.zeros(n, dtype='int64')
        i = 0; cur = 0
        while i < n:
            if not below[i]:
                seg[i] = cur; i += 1; continue
            j = i + 1
            while j < n and below[j]:
                j += 1
            k = i + int(np.argmin(dist[i:j]))           # acercamiento más cercano del tramo
            seg[i:j] = cur
            # nueva sub-leg solo si la visita es INTERIOR y a un terminal DISTINTO del
            # de inicio (= el giro lejano B), no un re-paso cerca del propio A.
            if 0 < k < n - 1 and int(nearest[k]) != start_t:
                cur += 1
                seg[k + 1:j] = cur
            i = j
        if cur > 0:
            n_split += 1
            parts.append(sub.with_columns(
                (pl.col('trip_id') + pl.lit('_S') + pl.Series(seg).cast(pl.Utf8)).alias('trip_id')))
        else:
            parts.append(sub)
    if verbose and n_split:
        print(f'  [split-round-trips] {n_split} viajes partidos en visitas interiores a terminal')
    return pl.concat(parts)


def _circ_gap(a, b):
    """Distancia circular entre dos posiciones lineales (fracción del lazo, 0..0.5)."""
    g = abs(float(a) - float(b)) % 1.0
    return min(g, 1.0 - g)


def _terminal_info(coords, poly, cum, L):
    """Posición sobre el lazo de cada terminal y, con dos, el hueco entre ellas (fracción
    del lazo, 0..0.5): menos de ~0.15 son dos sitios del mismo complejo."""
    if not coords or not L:
        return {'lin_pos': [], 'gap_frac': None}
    lp = [round(td._project_to_polyline(la, lo, poly, cum)[3] / L, 3) for la, lo in coords]
    return {'lin_pos': lp, 'gap_frac': round(_circ_gap(lp[0], lp[1]), 3) if len(lp) == 2 else None}


def _terminal_like_cut_events(df, poly, cum, L, dominance_frac=0.2, layover_min_min=None,
                              terminal_near_m=None, visit_max_m=None, verbose=False,
                              include_endpoints=None, terminal_coords=None,
                              return_info=False):
    """**Eventos de corte por VISITA a terminal_like** (process1_ver2 refinado).

    Las **zonas terminal_like** se descubren por **dwell** (``td.discover_terminals`` +
    dominancia ``dominance_frac``, que admite ``'AUTO_min'``/``'full_AUTO'`` =
    ``min_auto``/``full_auto``). Pero un viaje NO se corta sólo en el layover: se corta
    en **cada visita** del bus a una terminal_like, definida como el ping de
    **acercamiento más cercano** (mínimo local de la distancia a la terminal). Así cada
    viaje empieza y termina en una terminal_like aunque en esa terminal no haya layover
    (p.ej. el extremo de retorno rápido). Si en una pasada el GPS no pinga dentro de
    ``terminal_near_m``, se usa el registro más cercano de todos modos (el mínimo del
    tramo), cumpliendo "el registro hacia atrás más cercano".

    Una visita = un tramo contiguo de pings con distancia a la terminal < ``visit_max_m``
    (default 2×``terminal_near_m``); el corte cae en el ping de distancia mínima del tramo.

    Devuelve ``(cut_by_bus, term_coords, frac)`` — ``cut_by_bus[bus]`` = timestamps (ms)
    de las visitas (fronteras de viaje), análogo a ``turn_by_bus`` pero por terminal_like.
    Con ``terminal_coords`` (``TERMINAL_RULE='file'``) se usan esas terminales en vez de
    las de dwell. Con ``return_info=True`` añade un 4º elemento: la regla, la posición de
    las terminales sobre el lazo y el hueco entre ellas (``_terminal_info``).
    """
    import polars as pl
    LAYOVER = float(layover_min_min) if layover_min_min is not None else float(BREAK_MIN)
    near = float(terminal_near_m) if terminal_near_m is not None else TERMINAL_NEAR_M
    visit_max = float(visit_max_m) if visit_max_m is not None else 2.0 * near

    disc = td.discover_terminals(df, poly, layover_min_min=LAYOVER, stay_radius_m=60.0,
                                 dominance_frac=0.0)
    clusters = disc['clusters']
    frac, _counts = _resolve_dominance_frac(clusters, dominance_frac)

    term_coords = []
    if clusters is not None and clusters.height:
        on = clusters.filter(pl.col('recurrent') & pl.col('on_route'))
        gmax = float(on['n_dwells'].max() or 0)
        floor = frac * gmax
        term = on.filter(pl.col('n_dwells') >= floor)
        term_coords = list(zip(term['lat'].to_list(), term['lon'].to_list()))
    rule = 'dwell'
    if terminal_coords:
        term_coords, rule = [(float(a), float(b)) for a, b in terminal_coords], 'file'
    fix_info = {'rule': rule, **_terminal_info(term_coords, poly, cum, L)}
    if verbose:
        print(f"  [terminales] {rule}: lin_pos {fix_info['lin_pos']} · hueco {fix_info['gap_frac']}")

    # los extremos de la polilínea son terminales por construcción (el dwell los
    # pierde en el endpoint de retorno rápido) → se añaden si faltan.
    use_ep = INCLUDE_ENDPOINT_TERMINALS if include_endpoints is None else include_endpoints
    if use_ep:
        n_before = len(term_coords)
        term_coords = _augment_with_polyline_endpoints(term_coords, poly, cum, L)
        if verbose and len(term_coords) > n_before:
            print(f'  [terminal-cut] +{len(term_coords) - n_before} extremo(s) de '
                  f'polilínea añadidos como terminal (dwell no los descubrió)')

    cut_by_bus = {}
    if term_coords:
        lat0 = float(np.mean([p[0] for p in poly])); cos0 = float(np.cos(np.radians(lat0)))
        tla = np.array([c[0] for c in term_coords]); tlo = np.array([c[1] for c in term_coords])
        for (bus,), sub in df.sort(['bus_code', 'time']).group_by(['bus_code'], maintain_order=True):
            la = sub['latitude'].cast(pl.Float64).to_numpy()
            lo = sub['longitude'].cast(pl.Float64).to_numpy()
            tms = sub['time'].dt.timestamp('ms').to_numpy()
            n = la.shape[0]
            if n == 0:
                continue
            dy = (la[:, None] - tla[None, :]) * 111320.0
            dx = (lo[:, None] - tlo[None, :]) * 111320.0 * cos0
            dist = np.sqrt(dx * dx + dy * dy).min(axis=1)   # m al terminal_like más cercano
            below = dist < visit_max
            cuts = []
            i = 0
            while i < n:
                if not below[i]:
                    i += 1; continue
                j = i + 1
                while j < n and below[j]:
                    j += 1
                k = i + int(np.argmin(dist[i:j]))           # acercamiento más cercano del tramo
                cuts.append(int(tms[k]))
                i = j
            if cuts:
                cut_by_bus[bus] = np.sort(np.array(cuts, dtype='int64'))
    if verbose:
        n_ev = sum(len(v) for v in cut_by_bus.values())
        print(f'  [terminal-cut] frac={frac} · {len(term_coords)} terminal_like · '
              f'{n_ev} visitas de corte en {len(cut_by_bus)} buses '
              f'(visit_max={visit_max:.0f} m, dwell discovery≥{LAYOVER} min)')
    if return_info:
        return cut_by_bus, term_coords, frac, fix_info
    return cut_by_bus, term_coords, frac


def _cortes_en_traza(points, poly, cum, L):
    """Puntos de corte ``[(lat, lon[, km])]`` → metros sobre la traza.

    Con km, ese es el corte, y se comprueba que el punto de la traza en ese km quede a
    ≤ ``CRUCE_KM_TOL_M`` de (lat, lon): si no, la traza no es la de los cortes (otra
    geometría) y se detiene. Sin km, la proyección del punto (el paso más cercano)."""
    arr = np.asarray(poly, dtype=float)
    cum = np.asarray(cum, dtype=float)
    cos0 = float(np.cos(np.radians(arr[:, 0].mean())))
    out = []
    for p in points:
        la, lo = float(p[0]), float(p[1])
        km = p[2] if len(p) > 2 else None
        if km is None:
            out.append(float(td._project_to_polyline(la, lo, poly, cum)[3]) % L)
            continue
        m = (float(km) * 1000.0) % L
        pla, plo = float(np.interp(m, cum, arr[:, 0])), float(np.interp(m, cum, arr[:, 1]))
        dist = float(np.hypot((pla - la) * 111320.0, (plo - lo) * 111320.0 * cos0))
        if dist > CRUCE_KM_TOL_M:
            raise ValueError(
                f"punto de corte ({la:.5f}, {lo:.5f}, km {float(km):.2f}): en ese km la traza "
                f"está a {dist:.0f} m (> {CRUCE_KM_TOL_M:g}). Los cortes son de otra geometría.")
        out.append(m)
    return out


def _build_trips_cruce(df, poly, cum, L, terminal_coords, verbose=False):
    """Viajes por **cruce de km** (``TRIP_CUT=cruce``): el mismo contrato que build_trips
    → ``(df_trips, trip_meta, stats, otab, frag)``.

    1. **Tramos de servicio** por bus: se parten al cambiar de día o con más de
       ``GAP_SPLIT_MIN`` sin registros.
    2. **Proyección coherente** de cada tramo (la de ``COHERENT_PROJECTION``; con
       ``CRUCE_DIRECCIONAL``, retroceder sobre la traza cuesta, para que en una calle de ida y
       vuelta el bus no caiga en la otra pasada yendo hacia atrás) y **progreso continuo**: el avance con signo entre registros, sumado sin volver a 0 en cada vuelta;
       se usa su máximo acumulado M(t), que no retrocede con el temblor del GPS ni cuando el
       bus se regresa.
    3. **Saltos**: un avance mayor que ``CRUCE_VMAX``·dt + ``CRUCE_SALTO_M`` parte el tramo,
       salvo un **atajo**: hacia adelante, de hasta ``ATAJO_MAX_M``, y los ``ATAJO_SIGUE``
       registros siguientes avanzan normal (no es un registro suelto que vuelve). El atajo no
       parte el tramo; el viaje que lo lleva se marca (``atajo_m``: lo que se saltó).
    4. **Corte**: un viaje empieza en el primer registro después de que M(t) pase por el km
       de un punto de corte y termina antes del siguiente cruce del mismo tramo. Con un punto
       el viaje es la vuelta (``T0->T0``); con dos, las legs. Lo de antes del primer cruce y
       después del último de cada tramo es un pedazo **parcial** (``trip_valid=False``).

    Válido: completo (entre dos cruces del mismo tramo), ≥ ``MIN_TRIP_PTS`` registros,
    densidad (bins únicos / span) ≥ ``MIN_TRIP_DENSITY``, sin cola a depósito, y un recorrido
    real (la suma de las distancias en línea recta entre registros) de no más de
    ``CRUCE_RECORRIDO_MAX`` veces su arco: si recorrió más, la proyección se atascó en la otra
    pasada de la traza y el viaje lleva un cruce perdido (una vuelta doble). El orden
    cíclico lo da la construcción (``mark_cyclic_order`` no lo vuelve a juzgar) y la leg sale
    de los cortes cruzados (``cut_from``, ``cut_to``; ``assign_terminal_legs`` los usa).
    Comparativo_metodos_viajes.md §5.1 y §10."""
    import polars as pl
    if not terminal_coords:
        raise ValueError("TRIP_CUT=cruce necesita los puntos de corte de la ruta: "
                         "TERMINAL_RULE=file y la ruta en TERMINALS_FILE")
    if cum is None:
        cum = td._cumulative_distances(poly)
    if L is None:
        L = float(cum[-1])
    cortes = _cortes_en_traza(terminal_coords, poly, cum, L)
    if verbose:
        print(f"  [cruce] {df.height:,} pings · cortes en km "
              + ", ".join(f"{c / 1000:.2f}" for c in cortes) + f" (lazo {L / 1000:.2f} km)")
    d = df.sort(['bus_code', 'time'])
    n = d.height
    bus = d['bus_code'].to_numpy()
    t = d['time'].dt.timestamp('ms').to_numpy() / 1000.0
    day = d['time'].dt.date().to_numpy()
    lat = d['latitude'].cast(pl.Float64).to_numpy()
    lon = d['longitude'].cast(pl.Float64).to_numpy()
    nd = np.r_[True, (bus[1:] != bus[:-1]) | (day[1:] != day[:-1])]
    ng = np.r_[False, np.diff(t) > GAP_SPLIT_MIN * 60.0] & ~nd
    seg0 = np.cumsum(nd | ng) - 1
    if COHERENT_PROJECTION:
        pos, _pp, _sl, _so, perp = td._project_many_coherent(lat, lon, seg0, poly, cum, L=L, order=t,
                                                              direccional=CRUCE_DIRECCIONAL)
    else:
        pos, perp, _sl, _so = td._project_many_to_polyline(lat, lon, poly, cum)
    pos = np.asarray(pos, dtype=float); perp = np.asarray(perp, dtype=float)
    _cos0 = float(np.cos(np.radians(np.asarray(poly, dtype=float)[:, 0].mean())))
    paso_real = np.r_[0.0, np.hypot(np.diff(lat) * 111320.0, np.diff(lon) * 111320.0 * _cos0)]
    # saltos y atajos
    dpos = np.r_[0.0, (np.diff(pos) + L / 2) % L - L / 2]
    dts = np.r_[0.0, np.diff(t)]
    salto = (np.abs(dpos) > CRUCE_VMAX * dts + CRUCE_SALTO_M) & ~nd & ~ng
    js = np.flatnonzero(salto)
    sigue = np.ones(len(js), bool)
    for k in range(1, int(ATAJO_SIGUE) + 1):
        i = np.minimum(js + k, n - 1)
        sigue &= (js + k < n) & ~salto[i] & ~nd[i] & ~ng[i]
    atajo = np.zeros(n, bool)
    atajo[js[(dpos[js] > 0) & (dpos[js] <= ATAJO_MAX_M) & sigue]] = True
    nuevo = nd | ng | (salto & ~atajo)
    seg = np.cumsum(nuevo) - 1
    paso_real[nuevo] = 0.0
    # progreso continuo M(t), reiniciado en cada tramo
    paso = dpos.copy(); paso[nuevo] = 0.0
    ini = np.flatnonzero(nuevo); largo = np.diff(np.r_[ini, n])
    cs = np.cumsum(paso)
    U = cs - np.repeat(cs[ini] - pos[ini], largo)
    off_seg = np.repeat(np.arange(len(ini), dtype=float) * 1e9, largo)
    M = np.maximum.accumulate(U + off_seg) - off_seg
    # cruces: M(t) pasa por c + k·L dentro del mismo tramo
    mismo = np.r_[False, seg[1:] == seg[:-1]]
    ev_i, ev_c, ev_o = [], [], []
    Mprev = np.r_[M[0], M[:-1]]
    for ci, c in enumerate(cortes):
        sube = mismo & (np.r_[0.0, np.diff(np.floor((M - c) / L))] > 0)
        i = np.flatnonzero(sube)
        ev_i.append(i); ev_c.append(np.full(len(i), ci))
        ev_o.append((c - Mprev[i]) % L)               # dónde, dentro del paso, cae el cruce
    ev_i = np.concatenate(ev_i); ev_c = np.concatenate(ev_c); ev_o = np.concatenate(ev_o)
    o = np.lexsort((ev_o, ev_i)); ev_i, ev_c = ev_i[o], ev_c[o]
    if len(ev_i):                                     # dos cortes en un mismo paso: el último
        ult = np.r_[ev_i[1:] != ev_i[:-1], True]
        ev_i, ev_c = ev_i[ult], ev_c[ult]
    marca = np.zeros(n, bool); marca[ev_i] = True
    corte_en = np.full(n, -1); corte_en[ev_i] = ev_c
    # viajes: un pedazo nuevo en cada tramo y en cada cruce
    pieza = np.cumsum(nuevo | marca) - 1
    p_ini = np.flatnonzero(nuevo | marca)
    p_fin = np.r_[p_ini[1:] - 1, n - 1]
    sig = np.minimum(p_fin + 1, n - 1)
    completo = marca[p_ini] & (p_fin + 1 < n) & marca[sig] & ~nuevo[sig]
    cut_from = np.where(completo, corte_en[p_ini], -1)
    cut_to = np.where(completo, corte_en[sig], -1)
    # numeración por bus (rec_seq: pedazo; orig_seq: tramo)
    bus_p = bus[p_ini]
    nb_p = np.r_[True, bus_p[1:] != bus_p[:-1]]
    rec_seq_p = np.arange(len(p_ini)) - np.maximum.accumulate(np.where(nb_p, np.arange(len(p_ini)), 0))
    seg_ini = np.flatnonzero(nuevo); bus_s = bus[seg_ini]
    nb_s = np.r_[True, bus_s[1:] != bus_s[:-1]]
    orig_seq_s = np.arange(len(seg_ini)) - np.maximum.accumulate(np.where(nb_s, np.arange(len(seg_ini)), 0))
    rec_seq = rec_seq_p[pieza]
    orig_seq = orig_seq_s[seg]
    binv = (np.clip(pos / L, 0.0, 0.999) * 100).astype('int64')
    dt = d.with_columns([
        pl.Series('rec_seq', rec_seq.astype('int64')),
        pl.Series('orig_seq', orig_seq.astype('int64')),
        pl.Series('_day', day.astype(str)),
    ]).with_columns([
        (pl.col('bus_code') + '_C' + pl.col('rec_seq').cast(pl.Utf8)).alias('trip_id'),
        (pl.col('bus_code') + '_S' + pl.col('orig_seq').cast(pl.Utf8)).alias('trip_id_orig'),
        pl.Series('bin', binv), pl.Series('perp', perp), pl.Series('lin_pos', pos / L),
        pl.Series('prog_m', M), pl.Series('atajo', atajo),
        pl.Series('cut_from', cut_from[pieza]).cast(pl.Int64),
        pl.Series('cut_to', cut_to[pieza]).cast(pl.Int64),
    ]).with_columns([
        pl.when(pl.col('cut_from') >= 0).then(pl.col('cut_from')).otherwise(None).alias('cut_from'),
        pl.when(pl.col('cut_to') >= 0).then(pl.col('cut_to')).otherwise(None).alias('cut_to'),
        (pl.col('perp') > OFFROUTE_M).alias('off'),
    ])
    # cola a depósito (como analyze_trips): fuera de ruta tras el último registro en ruta del día
    _onr = dt.filter(~pl.col('off')).group_by(['bus_code', '_day']).agg(pl.col('time').max().alias('_last_on'))
    dt = dt.join(_onr, on=['bus_code', '_day'], how='left').with_columns(
        (pl.col('off') & pl.col('_last_on').is_not_null() & (pl.col('time') > pl.col('_last_on')))
        .alias('depot_run')).drop('_last_on')
    dt = dt.with_columns(pl.when(pl.col('depot_run')).then(pl.col('trip_id') + pl.lit('_end'))
                         .otherwise(pl.col('trip_id')).alias('seg_id'))
    # por viaje
    inferido = np.zeros(n, bool)
    inferido[ev_i] = dts[ev_i] > CRUCE_INFERIDO_S
    frag = (dt.with_columns(pl.Series('_inf', inferido), pl.Series('_real', paso_real))
            .group_by('trip_id', maintain_order=True).agg([
                pl.col('bus_code').first().alias('bus'), pl.col('_day').first().alias('day'),
                pl.col('time').min().alias('t0'), pl.col('time').max().alias('t1'),
                pl.len().alias('n'), pl.col('bin').n_unique().alias('nbin'),
                (pl.col('bin').max() - pl.col('bin').min() + 1).alias('span_bins'),
                pl.col('off').mean().alias('off_frac'), pl.col('depot_run').any().alias('depot'),
                pl.col('cut_from').first(), pl.col('cut_to').first(),
                (pl.col('prog_m').max() - pl.col('prog_m').min()).alias('avance_m'),
                pl.col('_inf').first().alias('inferido'),
                pl.col('_real').sum().alias('recorrido_m'),
                pl.col('trip_id_orig').n_unique().alias('n_frag')]))
    ats = (dt.with_columns(pl.Series('_salto_m', np.where(atajo, dpos, 0.0)))
             .group_by('trip_id').agg(pl.col('_salto_m').sum().alias('atajo_m')))
    frag = frag.join(ats, on='trip_id', how='left').with_columns([
        (pl.col('span_bins') / 100).alias('cov'),
        ((pl.col('t1') - pl.col('t0')).dt.total_seconds() / 60).alias('dur_min'),
        pl.col('cut_from').is_not_null().alias('completo')])
    # el arco entre los cortes del viaje (con un corte, la vuelta: L)
    _ck = np.asarray(cortes, dtype=float)
    _cf = frag['cut_from'].fill_null(0).to_numpy(); _ct = frag['cut_to'].fill_null(0).to_numpy()
    _arco = np.mod(_ck[_ct] - _ck[_cf], L); _arco[_arco <= 0] = L
    frag = frag.with_columns(pl.Series('arco_m', _arco)).with_columns(
        (pl.col('completo') & (pl.col('recorrido_m') > CRUCE_RECORRIDO_MAX * pl.col('arco_m')))
        .alias('cruce_perdido'))
    frag = frag.with_columns(
        (pl.col('completo') & (pl.col('n') >= MIN_TRIP_PTS)
         & ((pl.col('nbin') / pl.col('span_bins')) >= MIN_TRIP_DENSITY)
         & ~pl.col('depot') & ~pl.col('cruce_perdido')).alias('valid'))
    valid_ids = frag.filter(pl.col('valid'))['trip_id']
    dt = dt.with_columns(pl.col('trip_id').is_in(valid_ids.to_list()).alias('trip_valid'))

    def _flag(r):
        fl = []
        if not r['completo']:
            fl.append('parcial (inicio o fin de un tramo de servicio)')
        if r['cruce_perdido']:
            fl.append(f"cruce perdido: recorrió {r['recorrido_m'] / r['arco_m']:.2f} × su arco")
        if r['atajo_m']:
            fl.append(f"atajo {r['atajo_m'] / 1000:.1f} km")
        if r['inferido']:
            fl.append('cruce de salida inferido')
        if r['depot']:
            fl.append('→ sale a depósito (cola _end)')
        if r['off_frac'] >= 0.3:
            fl.append(f"off-ruta {100 * r['off_frac']:.0f}%")
        return ' · '.join(fl)
    trip_meta = {r['trip_id']: _flag(r) for r in frag.iter_rows(named=True)}
    # resumen del corte
    v = frag.filter(pl.col('valid'))
    dur = v['dur_min'].to_numpy()
    med = float(np.median(dur)) if len(dur) else float('nan')
    en_viaje = int(v['n'].sum())                 # lo que entra al df_final: los viajes válidos
    arr = np.asarray(poly, dtype=float)
    term = [(round(float(np.interp(c, cum, arr[:, 0])), 6), round(float(np.interp(c, cum, arr[:, 1])), 6))
            for c in cortes]
    legs = {}
    for a, b in v.select(['cut_from', 'cut_to']).iter_rows():
        legs[f"T{a}->T{b}"] = legs.get(f"T{a}->T{b}", 0) + 1
    cruce = {
        'cortes_km': [round(c / 1000, 3) for c in cortes], 'lazo_km': round(L / 1000, 3),
        'tramos': int(nuevo.sum()),
        'partidos_por': {'dia': int(nd.sum()), 'hueco': int(ng.sum()), 'salto': int((salto & ~atajo).sum())},
        'saltos': int(salto.sum()), 'atajos': int(atajo.sum()),
        'cobertura_pct': round(100 * en_viaje / max(n, 1), 2),
        'viajes_completos': int(frag['completo'].sum()), 'viajes_validos': int(v.height),
        'cruce_perdido': int(frag['cruce_perdido'].sum()),
        'pedazos_parciales': int((~frag['completo']).sum()),
        'viajes_por_leg': legs,
        'viajes_con_atajo_pct': round(100 * float((v['atajo_m'] > 0).mean()), 2) if v.height else None,
        'cruces_inferidos_pct': round(100 * float(inferido[ev_i].mean()), 2) if len(ev_i) else None,
        'dur_med_min': round(med, 1) if len(dur) else None,
        'dur_p10_min': round(float(np.quantile(dur, 0.1)), 1) if len(dur) else None,
        'dur_p90_min': round(float(np.quantile(dur, 0.9)), 1) if len(dur) else None,
        'raras_pct': (round(100 * float(((dur < 0.6 * med) | (dur > 1.6 * med)).mean()), 2)
                      if len(dur) and len(cortes) == 1 else None),
    }
    _sed = (dt.group_by(['bus_code', '_day'])
              .agg(pl.col('latitude').cast(pl.Float64).first().alias('slat'),
                   pl.col('longitude').cast(pl.Float64).first().alias('slon'),
                   pl.col('latitude').cast(pl.Float64).last().alias('elat'),
                   pl.col('longitude').cast(pl.Float64).last().alias('elon')))
    lat0 = float(arr[:, 0].mean())
    _dr = dt.filter(pl.col('depot_run'))
    stats = {
        'pings': df.height, 'buses': int(df['bus_code'].n_unique()), 'bus_days': int(_sed.height),
        'trips_orig': int(nuevo.sum()), 'trips_rec': int(frag.height), 'trips_valid': int(v.height),
        'merged_ge2': 0, 'max_frag': 1, 'half_pairs': 0, 'orphans': 0, 'bad_terminal': 0,
        'min_pts': int(frag['n'].min()), 'min_cov': round(float(frag['cov'].min()), 2),
        'off_pct': round(100 * float(dt['off'].mean()), 2),
        'start_consistency': round(_zone_consistency(list(zip(_sed['slat'].to_list(), _sed['slon'].to_list())), lat0), 2),
        'end_consistency': round(_zone_consistency(list(zip(_sed['elat'].to_list(), _sed['elon'].to_list())), lat0), 2),
        'depot_run_pings': int(_dr.height),
        'depot_run_busdays': int(_dr.select(['bus_code', '_day']).unique().height),
        'zones': term, 'terminals': term, 'trip_cut': 'cruce', 'dominance_frac': None,
        'dwell_terminals': 0, 'terminal_fix': {'rule': 'cruce', 'cortes_km': cruce['cortes_km']},
        'method': method_tag(terminal_coords), 'cruce': cruce,
    }
    if verbose:
        print(f"  [cruce] {cruce['tramos']:,} tramos (día {cruce['partidos_por']['dia']:,} · hueco "
              f"{cruce['partidos_por']['hueco']:,} · salto {cruce['partidos_por']['salto']:,}; "
              f"{cruce['atajos']:,} atajos no parten) · {cruce['viajes_validos']:,} viajes válidos "
              f"{legs} · cobertura {cruce['cobertura_pct']} % · mediana {cruce['dur_med_min']} min")
    return dt, trip_meta, stats, pl.DataFrame(), frag


def build_trips(df, poly, cum=None, L=None, cap='v2', trip_cut='turnaround',
                dominance_frac=0.2, layover_min_min=None, verbose=False, terminal_coords=None):
    """**Definidor de viajes** (reemplazo de make_trips). Devuelve
    (df_trips, trip_meta, stats, otab, frag).

    df_trips: un ping por fila con ``trip_id`` (viaje recuperado+capeado), ``seg_id``
    (``_end`` para la cola a depósito), ``trip_id_orig`` (segmentación de referencia),
    ``perp``, ``off``, ``depot_run``, ``bin`` (lin_pos*100) y ``trip_valid``.

    trip_cut:
      - ``'dwell'``  (process1_ver2): las terminales se descubren por **dwell**
        (``dominance_frac`` admite número o ``'AUTO_min'``/``'full_AUTO'``) y el viaje se
        corta en **cada visita** a una ``terminal_like`` (ping de acercamiento más
        cercano), de modo que **todo viaje empieza y termina en terminal_like** salvo el
        inicio del 1er viaje del día y el fin del último (que pueden caer donde sea).
        El turnaround se sigue calculando (alimenta la dirección geométrica y es fallback).
      - ``'turnaround'`` (default histórico): corta en la reversión de avance.
    cap: 'v2' (corta en el evento de corte interno más cercano, recomendado) |
         'v1' (corta en múltiplo de L).
    """
    import polars as pl
    if cum is None:
        cum = td._cumulative_distances(poly)
    if L is None:
        L = float(cum[-1])
    if TRIP_CUT == 'cruce':                       # corte por cruce de km: otro definidor
        return _build_trips_cruce(df, poly, cum, L, terminal_coords, verbose=verbose)
    lat0 = float(np.mean([p[0] for p in poly]))

    if verbose:
        print(f'  [build_trips] {df.height:,} pings · detectando turnarounds (bearing) …')
    br = td.detect_turnarounds(df, poly, source='bearing')
    tc = set(br['clusters'].filter(pl.col('is_terminal'))['cluster'].to_list())
    evt = br['events'].filter(pl.col('cluster').is_in(list(tc)))
    turn_by_bus = {b: np.sort(g['time_ms'].to_numpy()) for (b,), g in evt.group_by(['bus_code'])}


    # Señal de corte de viajes: visitas a terminal_like (process1_ver2) o turnaround.
    dwell_term_coords = []
    dom_used = None
    fix_info = None
    if trip_cut == 'dwell':
        if verbose:
            print('  [build_trips] descubriendo terminales por dwell …')
        cut_by_bus, dwell_term_coords, dom_used, fix_info = _terminal_like_cut_events(
            df, poly, cum, L, dominance_frac=dominance_frac,
            layover_min_min=layover_min_min, verbose=verbose,
            terminal_coords=terminal_coords, return_info=True)
        if not cut_by_bus:
            if verbose:
                print('  [terminal-cut] sin terminal_like → fallback a turnaround')
            cut_by_bus = turn_by_bus
            trip_cut = 'turnaround (fallback)'
    else:
        cut_by_bus = turn_by_bus

    zones = td.discover_terminal_zones(df, poly, source='bearing')['zones']
    _tlp = [td._project_to_polyline(z.get('mlat', z['lat']), z.get('mlon', z['lon']), poly, cum)[3] / L for z in zones]
    def _midroute(la, lo):
        lp = td._project_to_polyline(la, lo, poly, cum)[3] / L
        return not any(min(abs(lp - tp), 1 - abs(lp - tp)) <= 0.06 for tp in _tlp)
    if verbose:
        print('  [build_trips] stay-points de media ruta (pausas) …')
    _sl = td.detect_stay_points(df, stay_radius_m=60.0, stay_min_min=float(BREAK_MIN))
    break_by_bus = {}
    if _sl.height:
        _sl = _sl.filter(pl.struct(['lat', 'lon']).map_elements(lambda r: _midroute(r['lat'], r['lon']), return_dtype=pl.Boolean))
        for (b,), g in _sl.group_by(['bus_code']):
            break_by_bus[b] = np.sort((g['t_arr_s'].to_numpy() * 1000).astype('int64'))

    def segment_both(df_):
        parts = []
        for (bus,), sub in df_.sort(['bus_code', 'time']).group_by(['bus_code'], maintain_order=True):
            t = sub['time'].dt.timestamp('ms').to_numpy(); day = sub['time'].dt.date().to_numpy()
            # rec = señal de corte elegida (dwell o turnaround); orig = turnaround de referencia
            cuts = cut_by_bus.get(bus, np.array([], dtype='int64'))
            base_rec = np.searchsorted(cuts, t, side='right')
            turns = turn_by_bus.get(bus, np.array([], dtype='int64'))
            base_orig = np.searchsorted(turns, t, side='right')
            brks = break_by_bus.get(bus, np.array([], dtype='int64'))
            baseb = np.searchsorted(brks, t, side='right') if len(brks) else np.zeros(len(t), dtype=int)
            ed = np.zeros(len(t), dtype=int); eg = np.zeros(len(t), dtype=int); cd = 0; cg = 0
            for i in range(1, len(t)):
                newday = day[i] != day[i - 1]
                gap = (t[i] - t[i - 1]) / 60000.0 > GAP_SPLIT_MIN
                if newday: cd += 1
                if newday or gap: cg += 1
                ed[i] = cd; eg[i] = cg
            parts.append(sub.with_columns([pl.Series('rec_seq', base_rec + ed),
                                           pl.Series('orig_seq', base_orig + baseb + eg),
                                           pl.Series('_day', [str(d) for d in day])]))
        return pl.concat(parts)

    if verbose:
        print('  [build_trips] segmentando viajes por bus …')
    dtm = segment_both(df).with_columns([
        (pl.col('bus_code') + '_R' + pl.col('rec_seq').cast(pl.Utf8)).alias('trip_id'),
        (pl.col('bus_code') + '_T' + pl.col('orig_seq').cast(pl.Utf8)).alias('trip_id_orig')])
    _p0, _, _, _ = _project_seq(dtm, poly, cum, L)
    dtm = dtm.with_columns(pl.Series('bin', (np.clip(_p0 / L, 0.0, 0.999) * 100).astype('int64')))
    dtm = merge_short_trips(dtm, MIN_TRIP_PTS, MIN_TRIP_COV,
                            term_coords=(dwell_term_coords or None), lat0=lat0)
    # Traza ida-y-vuelta: parte el round trip A→B→A en su visita interior al terminal
    # lejano (B) → legs direccionales. Tras el merge (que ya rearmó fragmentos over-cut).
    if SPLIT_ROUND_TRIPS and trip_cut == 'dwell' and dwell_term_coords:
        dtm = _split_trips_at_interior_terminals(
            dtm, dwell_term_coords, lat0, visit_max_m=2.0 * TERMINAL_NEAR_M, verbose=verbose)
    if trip_cut == 'dwell' and dwell_term_coords:
        # En modo dwell la validación de terminal usa EXACTAMENTE las terminal_like del corte.
        term_coords = list(dwell_term_coords)
    else:
        term_coords = [(z.get('mlat', z['lat']), z.get('mlon', z['lon'])) for z in zones]
        _cl = br['clusters']
        if _cl is not None and _cl.height:
            _tt = _cl.filter(pl.col('is_terminal'))
            term_coords += list(zip(_tt['lat'].to_list(), _tt['lon'].to_list()))
    n_orig = int(dtm['trip_id_orig'].n_unique())
    if cap == 'v1':
        dtc = cap_trip_length(dtm, L, TRIP_MARGIN, lat0)
    else:
        # v2 corta los viajes demasiado largos en el evento de corte interno más cercano
        # (layover en modo dwell, turnaround en modo turnaround).
        dtc = cap_trip_length_v2(dtm, L, TRIP_MARGIN, lat0, cut_by_bus)
    if trip_cut == 'dwell' and dwell_term_coords:
        # validar con el mismo radio de visita del corte (acepta el registro más cercano,
        # aunque caiga a 300–600 m) y con cobertura (=SPAN de bin) de leg terminal-a-
        # terminal (~½ lazo). Con span, 0.375 admite legs completos (span 0.44–0.54) y
        # rechaza loops parciales A→medio→A (span ~0.30).
        _na, _vc = 2.0 * TERMINAL_NEAR_M, 1.5 * MIN_TRIP_COV
        # Método v2: 0.375 supone legs de ½ lazo. Con dos terminales a un hueco g < ½, la
        # leg corta mide g y nunca llega: con el mínimo global no se notaba porque los
        # saltos de rama inflaban la cobertura; con la proyección coherente se rechazan
        # todas. El umbral pasa a 0.75·g, la misma fracción (0.375 = 0.75·½).
        if COVERAGE_BY_LEG and len(dwell_term_coords) == 2:
            _lp = [td._project_to_polyline(la, lo, poly, cum)[3] / L for la, lo in dwell_term_coords]
            _vc = min(_vc, 0.75 * _circ_gap(_lp[0], _lp[1]))
    else:
        _na, _vc = None, None
    if verbose:
        print('  [build_trips] validando viajes (terminales/cobertura/huérfanos) …')
    res = list(analyze_trips(df, dtc, poly, cum, L, lat0, term_coords, near_m=_na, valid_cov=_vc))
    res[2]['trips_orig'] = n_orig
    res[2]['zones'] = [(z.get('mlat', z['lat']), z.get('mlon', z['lon'])) for z in zones]
    # `zones` son TODAS las zonas de reversión (bearing), independientes de
    # dominance_frac. Las terminales que definen las LEGS deben ser el set
    # seleccionado por dominancia (top-N vía dominance_frac/TERMINALES): en modo
    # dwell = las terminal_like del corte; si no (turnaround), caen en `zones`.
    # Antes assign_terminal_legs usaba `zones` → salían N terminales aunque se
    # pidiera 1 (p. ej. ruta 102: TERMINALES=1 pero 4 zonas → 15 legs).
    res[2]['terminals'] = (list(dwell_term_coords) if dwell_term_coords
                           else res[2]['zones'])
    res[2]['trip_cut'] = trip_cut
    res[2]['dominance_frac'] = dom_used
    res[2]['dwell_terminals'] = len(dwell_term_coords)
    res[2]['terminal_fix'] = fix_info
    res[2]['method'] = method_tag(terminal_coords)
    return tuple(res)


# ──────────────────────────────────────────────────────────────────────
# Caché de build_trips (+ dirección geométrica + orden cíclico)
# ──────────────────────────────────────────────────────────────────────
# build_trips es determinista dados los mismos pings y parámetros, así que su
# salida (con las columnas de 3a ya plegadas en trip_valid) puede persistirse
# y reusarse. Clave de validez: manifest.json con los parámetros + nº de pings
# de entrada; si algo difiere, el caché se descarta y se recalcula. Reusar el
# caché garantiza además trip_id idénticos a los del caché del enrich generado
# en la misma sesión (evita el re-enrich forzado del runner).

def _normalize_cache_params(params: dict) -> dict:
    """Round-trip por JSON para comparar igual lo guardado y lo pedido
    (tuplas→listas, floats numpy→float, etc.)."""
    import json
    return json.loads(json.dumps(params, sort_keys=True, default=float))


def save_trips_cache(cache_dir, params: dict, df_trips, trip_meta: dict,
                     stats: dict, otab, frag, verbose: bool = False) -> None:
    """Persiste la salida de build_trips en ``cache_dir``. DataFrames vacíos
    (otab/frag sin filas) no se escriben (convención del repo): el manifest
    registra su altura y load los reconstruye vacíos."""
    import json
    from pathlib import Path
    import polars as pl
    cache_dir = Path(cache_dir)
    if df_trips.height == 0:
        return  # sin viajes no hay nada que cachear (el pipeline fallará aguas abajo)
    cache_dir.mkdir(parents=True, exist_ok=True)
    df_trips.write_parquet(cache_dir / "df_trips.parquet")
    heights = {"otab": int(otab.height), "frag": int(frag.height)}
    for name, d in (("otab", otab), ("frag", frag)):
        if d.height:
            d.write_parquet(cache_dir / f"{name}.parquet")
    with open(cache_dir / "trip_meta.json", "w", encoding="utf-8") as f:
        json.dump(trip_meta, f, ensure_ascii=False)
    with open(cache_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, default=float)
    with open(cache_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump({"params": _normalize_cache_params(params), "heights": heights},
                  f, ensure_ascii=False, indent=2)
    if verbose:
        print(f"  💾 caché de viajes guardado: {cache_dir.name} "
              f"({df_trips.height:,} pings · {df_trips['trip_id'].n_unique()} viajes)")


def load_trips_cache(cache_dir, params: dict, verbose: bool = False):
    """Carga la salida cacheada de build_trips si existe Y los parámetros
    coinciden. Devuelve (df_trips, trip_meta, stats, otab, frag) o None."""
    import json
    from pathlib import Path
    import polars as pl
    cache_dir = Path(cache_dir)
    manifest_p = cache_dir / "manifest.json"
    if not manifest_p.is_file():
        return None
    try:
        with open(manifest_p, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        if manifest.get("params") != _normalize_cache_params(params):
            if verbose:
                print("  ℹ caché de viajes existe pero con otros parámetros → recalculo")
            return None
        df_trips = pl.read_parquet(cache_dir / "df_trips.parquet")
        heights = manifest.get("heights", {})
        out = {}
        for name in ("otab", "frag"):
            p = cache_dir / f"{name}.parquet"
            if heights.get(name, 0) > 0:
                if not p.is_file():
                    return None   # purgado/incompleto → recalcular
                out[name] = pl.read_parquet(p)
            else:
                out[name] = pl.DataFrame()
        with open(cache_dir / "trip_meta.json", "r", encoding="utf-8") as f:
            trip_meta = json.load(f)
        with open(cache_dir / "stats.json", "r", encoding="utf-8") as f:
            stats = json.load(f)
    except Exception as e:
        if verbose:
            print(f"  ⚠ caché de viajes ilegible ({e}) → recalculo")
        return None
    if df_trips.height == 0:
        return None
    if verbose:
        print(f"  ✔ viajes reusados de caché: {cache_dir.name} "
              f"({df_trips.height:,} pings · {df_trips['trip_id'].n_unique()} viajes)")
    return df_trips, trip_meta, stats, out["otab"], out["frag"]


# ──────────────────────────────────────────────────────────────────────
# Mapas
# ──────────────────────────────────────────────────────────────────────
def render_trips_map(dt, poly, out_html, route, trip_meta=None, square_col=None,
                     full_props=False, square2_col=None):
    """Mapa interactivo por bus: slider por viaje, gradiente de color temporal por viaje,
    off-ruta/depósito resaltados, capas de inicio/fin de día y depósito dominante.

    ``square_col``: nombre de una columna booleana de ``dt``; los pings donde sea True se
    resaltan con un **cuadrado rojo** encima del marcador normal (p.ej. los registros de
    fin de día 'rumbo a depósito'). El mapa es idéntico al normal salvo ese resaltado.

    ``full_props``: si True, cada ping embebe **todas sus columnas** y al hacer **click**
    se abre un popup con todas las propiedades (key: value). El tooltip (hover) sigue
    mostrando el resumen corto."""
    import polars as pl
    import folium, json as _json
    LAT0 = float(np.mean([p[0] for p in poly])); LON0 = float(np.mean([p[1] for p in poly]))
    ROUTE = route
    has_sq = bool(square_col) and (square_col in dt.columns)
    has_sq2 = bool(square2_col) and (square2_col in dt.columns)   # 2º resaltado (naranja, p.ej. warmup)

    dtd = dt.with_columns(pl.col('time').dt.date().cast(pl.Utf8).alias('d')).sort(['bus_code', 'time'])
    bus0 = (dtd.group_by('bus_code').agg(pl.col('trip_id').n_unique().alias('nt'))
               .sort('nt', descending=True)['bus_code'][0])
    db = dtd.filter(pl.col('bus_code') == bus0)
    days_ordered = db['d'].unique(maintain_order=True).to_list()[:30]
    db = db.filter(pl.col('d').is_in(days_ordered))

    palette = ['#1D9E75', '#D85A30', '#185FA5', '#BA7517', '#7F77DD', '#D4537E', '#639922', '#0F6E56', '#993C1D', '#534AB7']
    all_tids = db['trip_id'].unique(maintain_order=True).to_list()
    cmap = {tid: palette[i % len(palette)] for i, tid in enumerate(all_tids)}

    DATA, TRIP_LABELS = {}, {}
    for day in days_ordered:
        sub = db.filter(pl.col('d') == day)
        DATA[day], TRIP_LABELS[day] = {}, []
        for t_idx, tid in enumerate(sub['trip_id'].unique(maintain_order=True).to_list()):
            g = sub.filter(pl.col('trip_id') == tid)
            _sqv = g[square_col].to_list() if has_sq else [False] * g.height
            _sq2v = g[square2_col].to_list() if has_sq2 else [False] * g.height
            pts = [{'lat': float(la), 'lon': float(lo), 'tid': tid, 'time': t.strftime('%H:%M:%S'),
                    'off': bool(o), 'perp': round(float(pp), 0), 'dep': bool(dp), 'seg': sg,
                    'sq': bool(sq), 'sq2': bool(sq2)}
                   for la, lo, t, o, pp, dp, sg, sq, sq2 in zip(g['latitude'], g['longitude'], g['time'],
                                                               g['off'], g['perp'], g['depot_run'], g['seg_id'], _sqv, _sq2v)]
            if full_props:   # adjunta TODAS las columnas del ping (para el popup al click)
                _allrows = g.to_dicts()
                for _k, _d in enumerate(pts):
                    _d['all'] = {kk: ('' if vv is None else str(vv)) for kk, vv in _allrows[_k].items()}
            DATA[day][str(t_idx)] = pts
            TRIP_LABELS[day].append({'idx': t_idx, 'tid': tid,
                't_start': pts[0]['time'], 't_end': pts[-1]['time'], 'n_pts': len(pts),
                'flags': (trip_meta or {}).get(tid, '')})

    max_trips = max((len(v) for v in TRIP_LABELS.values()), default=1)
    day0 = max(days_ordered, key=lambda d: len(TRIP_LABELS[d]))
    default_day = day0

    m2 = folium.Map(location=[LAT0, LON0], zoom_start=12, tiles='cartodbpositron')
    folium.PolyLine([(p[0], p[1]) for p in poly], color='#ccc', weight=2, opacity=0.6).add_to(m2)

    css = """
    <style>
    #vbt-panel{position:fixed;top:10px;right:10px;z-index:9999;background:rgba(15,17,23,.93);
      color:#e0e0e0;font-family:monospace;font-size:12px;padding:12px 16px;border-radius:10px;
      border:1px solid rgba(255,255,255,.12);max-height:92vh;overflow-y:auto;width:250px;
      box-shadow:0 4px 24px rgba(0,0,0,.5);}
    #vbt-panel h3{margin:0 0 8px 0;font-size:13px;color:#1D9E75;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
    .sec-label{font-size:10px;color:#888;text-transform:uppercase;letter-spacing:.06em;margin:10px 0 4px 0;display:block;}
    #vbt-slider{width:100%;accent-color:#1D9E75;cursor:pointer;}
    #vbt-trip-label{text-align:center;font-size:11px;font-weight:bold;color:#f5f5f5;margin:4px 0 2px 0;}
    #vbt-trip-sub{text-align:center;font-size:10px;color:#888;margin-bottom:2px;}
    .vbt-day-row{display:flex;align-items:center;gap:6px;margin:3px 0;cursor:pointer;padding:2px 4px;border-radius:4px;}
    .vbt-day-row:hover{background:rgba(255,255,255,.06);}
    .vbt-day-row input{cursor:pointer;accent-color:#1D9E75;}
    .vbt-day-row label{cursor:pointer;font-size:11px;color:#ccc;}
    #vbt-legend{margin-top:6px;max-height:160px;overflow-y:auto;}
    .vbt-leg-row{display:flex;align-items:center;gap:5px;margin:2px 0;font-size:11px;}
    .vbt-swatch{width:10px;height:10px;border-radius:50%;flex-shrink:0;}
    #vbt-stats{margin-top:8px;font-size:10px;color:#888;line-height:1.6;}
    #vbt-grad{height:8px;border-radius:4px;margin:6px 0 1px 0;background:linear-gradient(90deg,rgb(68,1,84),rgb(33,144,140),rgb(253,231,37));}
    #vbt-grad-lbl{display:flex;justify-content:space-between;font-size:9px;color:#999;}
    </style>
    """

    html_panel = (
        "<div id='vbt-panel'>"
        "<h3>\U0001f68c " + str(bus0) + " — ruta " + str(ROUTE) + "</h3>"
        "<span class='sec-label'>Viaje en el día</span>"
        "<input id='vbt-slider' type='range' min='0' max='" + str(max_trips - 1) + "' value='0' step='1'>"
        "<div id='vbt-trip-label'>—</div>"
        "<div id='vbt-trip-sub'>—</div>"
        "<div id='vbt-grad'></div>"
        "<div id='vbt-grad-lbl'><span>temprano</span><span>tarde</span></div>"
        "<span class='sec-label'>Días</span>"
        "<div id='vbt-days'></div>"
        "<span class='sec-label'>Viajes del día</span>"
        "<div id='vbt-legend'></div>"
        "<div id='vbt-stats'></div>"
        "</div>"
    )

    js_tpl = """
    <script>
    (function(){
      var DATA=%s, COLORS=%s, DAYS=%s, TRIP_LABELS=%s, DEFAULT_DAY=%s, MAP_ID=%s;
      var activeDays={}; DAYS.forEach(function(d){activeDays[d]=(d===DEFAULT_DAY);});
      var currentIdx=0, layer=null;
      function getMap(){
        /* usa el nombre exacto del mapa Folium; fallback a búsqueda por _leaflet_id */
        if(MAP_ID && window[MAP_ID] && window[MAP_ID].addLayer) return window[MAP_ID];
        for(var k in window)
          if(window[k] && window[k]._leaflet_id!==undefined && window[k].addLayer
             && typeof window[k].getBounds === 'function')   /* solo L.Map tiene getBounds */
            return window[k];
        return null;
      }
      function gradient(t){var s=[[68,1,84],[33,144,140],[253,231,37]];var i=t<0.5?0:1;var f=t<0.5?t/0.5:(t-0.5)/0.5;var a=s[i],b=s[i+1];return 'rgb('+Math.round(a[0]+(b[0]-a[0])*f)+','+Math.round(a[1]+(b[1]-a[1])*f)+','+Math.round(a[2]+(b[2]-a[2])*f)+')';}
      function updateSliderRange(){
        var maxT=0;
        DAYS.forEach(function(d){if(!activeDays[d])return; var lbs=TRIP_LABELS[d]; if(lbs&&lbs.length>maxT)maxT=lbs.length;});
        if(maxT===0)maxT=1;
        var sl=document.getElementById('vbt-slider'); sl.max=maxT-1;
        if(currentIdx>=maxT)currentIdx=0; sl.value=currentIdx;
      }
      function render(){
        var lmap=getMap(); if(!lmap){setTimeout(render,300);return;}
        if(layer){lmap.removeLayer(layer);} layer=L.layerGroup();
        var nPts=0, firstMeta=null;
        DAYS.forEach(function(day){
          if(!activeDays[day])return;
          var dayData=DATA[day], labels=TRIP_LABELS[day];
          if(!dayData||!labels)return;
          var meta=labels[currentIdx]; if(!meta)return;
          if(!firstMeta)firstMeta=meta;
          var pts=dayData[String(currentIdx)]; if(!pts)return;
          pts.forEach(function(p, ii){
            nPts++;
            var frac = pts.length>1 ? ii/(pts.length-1) : 0;
            var fc = gradient(frac);
            var col = p.dep ? '#FF7F0E' : fc;
            var stroke = (p.dep||p.off) ? '#000' : fc;
            var tip='Viaje: '+p.tid+'<br>Hora: '+p.time+'<br>orig: '+(+p.lat).toFixed(6)+', '+(+p.lon).toFixed(6)+'<br>perp: '+p.perp+' m'+(p.off?'  &#9888; OFF-RUTA':'')+(p.dep?('<br>&#11088; DEPÓSITO: '+p.seg):'');
            var popup=null;
            if(p.all){var ph="<div style='max-height:260px;overflow:auto;font-size:11px;line-height:1.5'>";for(var k in p.all){ph+="<b>"+k+"</b>: "+p.all[k]+"<br>";}popup=ph+"</div>";}
            var mk=L.circleMarker([p.lat,p.lon],{radius:p.dep?5:(p.off?4:3),color:stroke,fillColor:col,fillOpacity:.9,opacity:.95,weight:(p.dep||p.off)?2:1}).bindTooltip(tip);
            if(popup)mk.bindPopup(popup,{maxWidth:360});
            mk.addTo(layer);
            if(p.sq){
              var sm=L.marker([p.lat,p.lon],{icon:L.divIcon({className:'',iconSize:[11,11],iconAnchor:[6,6],html:"<div style='width:9px;height:9px;background:#E11;border:1px solid #000;'></div>"})})
                .bindTooltip('&#128230; fin de día &#8594; depósito<br>'+p.tid+'<br>'+p.time);
              if(popup)sm.bindPopup(popup,{maxWidth:360});
              sm.addTo(layer);
            }
            if(p.sq2){
              var sw=L.marker([p.lat,p.lon],{icon:L.divIcon({className:'',iconSize:[11,11],iconAnchor:[6,6],html:"<div style='width:9px;height:9px;background:#FF8C00;border:1px solid #000;'></div>"})})
                .bindTooltip('&#9728; calentamiento (inicio día)<br>'+p.tid+'<br>'+p.time);
              if(popup)sw.bindPopup(popup,{maxWidth:360});
              sw.addTo(layer);
            }
          });
        });
        layer.addTo(lmap);
        var lbl=document.getElementById('vbt-trip-label'), sub=document.getElementById('vbt-trip-sub');
        if(firstMeta){
          lbl.textContent=firstMeta.tid;
          sub.textContent=firstMeta.t_start+' – '+firstMeta.t_end+'  ('+firstMeta.n_pts+' pts)'+(firstMeta.flags?(' · '+firstMeta.flags):'');
        } else { lbl.textContent='— sin viaje —'; sub.textContent=''; }
        var legDiv=document.getElementById('vbt-legend'); legDiv.innerHTML='';
        DAYS.forEach(function(day){
          if(!activeDays[day])return;
          var lbs=TRIP_LABELS[day]; if(!lbs)return;
          lbs.forEach(function(meta){
            var row=document.createElement('div'); row.className='vbt-leg-row';
            if(meta.idx===currentIdx){row.style.background='rgba(29,158,117,.15)';row.style.borderRadius='3px';}
            var sw=document.createElement('div'); sw.className='vbt-swatch'; sw.style.background=COLORS[meta.tid]||'#888';
            var sp=document.createElement('span'); sp.style.color=meta.idx===currentIdx?'#f5f5f5':'#aaa';
            sp.textContent=meta.tid+' ('+meta.t_start.slice(0,5)+')';
            row.appendChild(sw); row.appendChild(sp); legDiv.appendChild(row);
          });
        });
        document.getElementById('vbt-stats').innerHTML='Viaje '+(currentIdx+1)+' de '+(parseInt(document.getElementById('vbt-slider').max)+1)+'<br>Puntos: <b>'+nPts+'</b>';
      }
      document.getElementById('vbt-slider').addEventListener('input',function(){currentIdx=parseInt(this.value);render();});
      var daysDiv=document.getElementById('vbt-days');
      DAYS.forEach(function(day){
        var row=document.createElement('div'); row.className='vbt-day-row';
        var cb=document.createElement('input'); cb.type='checkbox'; cb.id='vbt-d-'+day; cb.checked=activeDays[day];
        cb.addEventListener('change',function(){activeDays[day]=cb.checked;updateSliderRange();render();});
        var lbl=document.createElement('label'); lbl.htmlFor='vbt-d-'+day; lbl.textContent=day;
        row.appendChild(cb); row.appendChild(lbl); daysDiv.appendChild(row);
      });
      setTimeout(function(){updateSliderRange();render();},INIT_DELAY);
    })();
    </script>
    """
    _data_json   = _json.dumps(DATA)
    _colors_json = _json.dumps(cmap)
    _days_json   = _json.dumps(days_ordered)
    _labels_json = _json.dumps(TRIP_LABELS)
    _day0_json   = _json.dumps(default_day)
    _mapid_json  = _json.dumps(m2.get_name())
    # Timeout proporcional al tamaño del DATA: 600ms base + 1ms por KB (máx 4s)
    _init_delay  = min(4000, 600 + len(_data_json) // 1000)
    js_filled = (js_tpl
                 .replace('INIT_DELAY', str(_init_delay))
                 % (_data_json, _colors_json, _days_json, _labels_json, _day0_json, _mapid_json))
    m2.get_root().html.add_child(folium.Element(css + html_panel + js_filled))
    out2 = out_html; out2.parent.mkdir(parents=True, exist_ok=True)
    _sed = (dt.sort(['bus_code', '_day', 'time']).group_by(['bus_code', '_day']).agg([
        pl.col('latitude').cast(pl.Float64).first().alias('slat'), pl.col('longitude').cast(pl.Float64).first().alias('slon'),
        pl.col('latitude').cast(pl.Float64).last().alias('elat'), pl.col('longitude').cast(pl.Float64).last().alias('elon')]))
    _fgs = folium.FeatureGroup(name='inicio de día', show=False)
    _fge = folium.FeatureGroup(name='fin de día', show=False)
    for _r in _sed.iter_rows(named=True):
        folium.CircleMarker([_r['slat'], _r['slon']], radius=4, color='#1D9E75', weight=1, fill=True,
                            fill_opacity=0.55, popup=f"inicio · {_r['bus_code']} · {_r['_day']}").add_to(_fgs)
        folium.CircleMarker([_r['elat'], _r['elon']], radius=4, color='#D62728', weight=1, fill=True,
                            fill_opacity=0.55, popup=f"fin · {_r['bus_code']} · {_r['_day']}").add_to(_fge)
    for _co, _fg, _cc, _lb in [
        (list(zip(_sed['slat'].to_list(), _sed['slon'].to_list())), _fgs, '#1D9E75', 'inicio'),
        (list(zip(_sed['elat'].to_list(), _sed['elon'].to_list())), _fge, '#D62728', 'fin')]:
        _dz = _dominant_zone(_co, LAT0)
        if _dz:
            folium.Circle([_dz[0], _dz[1]], radius=200, color=_cc, weight=3, fill=True, fill_opacity=0.10,
                          popup=f"zona dominante {_lb}: {100*_dz[2]:.0f}% de bus-días ({_dz[3]})").add_to(_fg)
            folium.map.Marker([_dz[0], _dz[1]], icon=folium.DivIcon(
                html=f'<div style="font-weight:700;color:{_cc};white-space:nowrap">{_lb} {100*_dz[2]:.0f}%</div>')).add_to(_fg)
    _fgs.add_to(m2); _fge.add_to(m2)
    _dr = dt.filter(pl.col('depot_run'))
    _fgd = folium.FeatureGroup(name='salida a depósito (fin de día)', show=False)
    for _r in _dr.iter_rows(named=True):
        folium.CircleMarker([float(_r['latitude']), float(_r['longitude'])], radius=3, color='#FF7F0E', weight=1,
                            fill=True, fill_opacity=0.7,
                            popup=f"depósito · {_r['bus_code']} · {_r['_day']} · perp {_r['perp']:.0f} m").add_to(_fgd)
    if _dr.height:
        _de = _dr.sort(['bus_code', '_day', 'time']).group_by(['bus_code', '_day']).agg([
            pl.col('latitude').cast(pl.Float64).last().alias('dlat'), pl.col('longitude').cast(pl.Float64).last().alias('dlon')])
        _dz = _dominant_zone(list(zip(_de['dlat'].to_list(), _de['dlon'].to_list())), LAT0)
        if _dz:
            folium.Circle([_dz[0], _dz[1]], radius=200, color='#FF7F0E', weight=3, fill=True, fill_opacity=0.12,
                          popup=f"depósito dominante: {100*_dz[2]:.0f}% de los fines de día ({_dz[3]})").add_to(_fgd)
    _fgd.add_to(m2)
    folium.LayerControl(collapsed=False, position='topleft').add_to(m2)
    m2.save(str(out2))
    return out2


def assign_geometric_direction(df_trips, poly, cum=None, L=None):
    """Asigna ``direction`` (0/1) por viaje = **signo del avance sobre la polilínea**
    (geometría anclada en las terminales geográficas/extremos de la ruta). Es la
    identificación de las 2 direcciones, *previa* a los filtros por terminal_like.

    Añade ``lin_pos`` (0..1, posición lineal por ping) y ``direction`` (por viaje:
    1 si el viaje avanza en sentido decreciente de la posición lineal, 0 si creciente).
    """
    import polars as pl
    if cum is None:
        cum = td._cumulative_distances(poly)
    if L is None:
        L = float(cum[-1])
    d = df_trips.with_columns([pl.col('latitude').cast(pl.Float64), pl.col('longitude').cast(pl.Float64)])
    if not ('cut_from' in d.columns and 'lin_pos' in d.columns):
        # (con TRIP_CUT=cruce, lin_pos ya viene de la proyección coherente por tramo)
        pos, _, _, _ = _project_seq(d, poly, cum, L)
        d = d.with_columns(pl.Series('lin_pos', (pos / L)))
    dirmap = (d.sort(['trip_id', 'time']).group_by('trip_id', maintain_order=True)
                .agg((pl.col('lin_pos').last() - pl.col('lin_pos').first()).alias('_dp'))
                .with_columns((pl.col('_dp') < 0).cast(pl.Int64).alias('direction')))
    return d.join(dirmap.select(['trip_id', 'direction']), on='trip_id', how='left')


def mark_cyclic_order(df_trips, min_leg_cov=0.4, direction_col='direction', terminals=None,
                      version=None):
    """Limpieza por **orden cíclico** (por viaje). Marca ``cyclic_ok=True`` el viaje que:
      1. es una **leg completa**, no un fragmento; y
      2. sigue el **ciclo** del bus en el día: dos legs seguidas en la misma dirección =
         turnaround perdido → la segunda rompe el ciclo.

    ``version`` (default ``CYCLIC_ORDER``):

    - ``'v1'`` — la de la corrida tmin publicada. Completa = ``bin.n_unique()/100 >=
      min_leg_cov``; ciclo = alterna ``direction`` (signo del avance, ver
      ``assign_geometric_direction``). Dos defectos, medidos sobre los cachés tmin de las
      27 rutas: (a) en una traza cerrada las dos direcciones avanzan en el MISMO sentido,
      así que el signo casi no alterna (viajes seguidos con la misma «dirección»: 44–96 %
      salvo 101/102/601, donde alterna porque la proyección salta de rama); (b) los bins
      únicos no se llenan con un ping por minuto en las rutas cortas —es la razón por la
      que build_trips pasó a medir la cobertura por span— (902: 0.1 % de los viajes con
      ``cov >= 0.4``, contra 91 % por span). Resultado: tiraba el 29–99.9 % de los viajes
      válidos, más en las rutas cortas (902: 20,300 → 18; 1602: 17,148 → 408).
    - ``'v2'`` — completa = **válida según build_trips**, cuya cobertura mínima ya se mide
      sobre la leg (``COVERAGE_BY_LEG``) y con la proyección coherente; ciclo = la leg
      arranca en la terminal (la más cercana a su primer ping) donde terminó la leg
      válida anterior del mismo (bus, día). No usa la posición lineal de las terminales:
      un sitio de descanso sobre una calle de ida y vuelta tiene DOS posiciones en el lazo
      y el arco esperado salía mal (201 en marzo: 908 válidos → 177). Necesita
      ``terminals`` y la columna ``trip_valid``; sin terminales cae a ``'v1'``.
    - ``'v2-sin-ciclo'`` — sólo la condición 1 (para medir cuánto quita el ciclo).

    Devuelve ``df_trips`` con columna booleana ``cyclic_ok``.
    """
    import polars as pl
    if 'cut_from' in df_trips.columns:
        # corte por cruce de km (TRIP_CUT=cruce): cada viaje va de un cruce al siguiente del
        # mismo tramo, así que el ciclo se cumple por construcción
        return df_trips.with_columns(pl.col('trip_valid').alias('cyclic_ok'))
    version = (version or CYCLIC_ORDER).lower()
    if version != 'v1' and terminals:
        return _mark_cyclic_order_v2(df_trips, terminals, continuity=(version != 'v2-sin-ciclo'))
    info = df_trips.with_columns(pl.col('time').dt.date().alias('_cd')).group_by('trip_id').agg([
        pl.col('bus_code').first().alias('bus'), pl.col('_cd').first().alias('day'),
        pl.col('time').min().alias('t0'), (pl.col('bin').n_unique() / 100.0).alias('cov'),
        pl.col(direction_col).first().alias('dir')])
    ok_ids = []
    for _key, grp in info.sort('t0').group_by(['bus', 'day'], maintain_order=True):
        prev = None
        for r in grp.iter_rows(named=True):
            if r['cov'] < min_leg_cov:
                continue                          # fragmento: no es leg completa
            if prev is None or r['dir'] != prev:  # alterna -> leg válida del ciclo
                ok_ids.append(r['trip_id']); prev = r['dir']
            # misma dirección que la leg anterior -> rompe el ciclo: se omite (prev no cambia)
    return df_trips.with_columns(pl.col('trip_id').is_in(ok_ids).alias('cyclic_ok'))


def _mark_cyclic_order_v2(df_trips, terminals, continuity=True):
    """``mark_cyclic_order`` versión ``'v2'``: ver allí."""
    import polars as pl
    lat0 = float(np.mean([float(t[0]) for t in terminals]))
    cos0 = float(np.cos(np.radians(lat0)))
    tla = np.array([float(t[0]) for t in terminals]); tlo = np.array([float(t[1]) for t in terminals])

    def _nearest(la, lo):
        return int(np.argmin(((la - tla) * 111320.0) ** 2 + ((lo - tlo) * 111320.0 * cos0) ** 2))

    info = (df_trips.sort(['trip_id', 'time'])
            .group_by('trip_id', maintain_order=True)
            .agg([pl.col('bus_code').first().alias('bus'),
                  pl.col('time').first().dt.date().alias('day'),
                  pl.col('time').first().alias('t0'),
                  pl.col('trip_valid').first().alias('valid'),
                  pl.col('latitude').cast(pl.Float64).first().alias('la_s'),
                  pl.col('longitude').cast(pl.Float64).first().alias('lo_s'),
                  pl.col('latitude').cast(pl.Float64).last().alias('la_e'),
                  pl.col('longitude').cast(pl.Float64).last().alias('lo_e')]))
    ok_ids = []
    for _key, grp in info.filter(pl.col('valid')).sort('t0').group_by(['bus', 'day'], maintain_order=True):
        prev_end = None
        for r in grp.iter_rows(named=True):
            s, e = _nearest(r['la_s'], r['lo_s']), _nearest(r['la_e'], r['lo_e'])
            if not continuity or prev_end is None or s == prev_end:
                ok_ids.append(r['trip_id']); prev_end = e
            # arranca en otra terminal que donde acabó la anterior → turnaround perdido
    return df_trips.with_columns(pl.col('trip_id').is_in(ok_ids).alias('cyclic_ok'))


def assign_terminal_legs(df_final, terminals, lat0=None):
    """Asigna a cada viaje su **terminal de inicio/fin** y su **leg** (segmento
    dirigido terminal→terminal). Reemplaza a la vieja noción de ``direction``:
    en vez de 0/1, la unidad de análisis es la pareja de terminales recorrida.

    Para una ruta de 2 terminales esto reproduce las 2 direcciones (T0→T1 y
    T1→T0); para N>2 terminales aparecen tantas legs como parejas observadas.

    Parameters
    ----------
    df_final : pl.DataFrame
        Debe tener ``trip_id``, ``time``, ``latitude``, ``longitude``.
    terminals : list[(lat, lon)]
        Coordenadas de las terminales descubiertas (p.ej. ``stats['zones']``).
    lat0 : float | None
        Latitud de referencia para la métrica local (default: media del df).

    Returns
    -------
    pl.DataFrame
        ``df_final`` con columnas añadidas:
          - ``start_terminal`` (Int64) — índice del terminal más cercano al 1er ping
          - ``end_terminal``   (Int64) — índice del terminal más cercano al último ping
          - ``leg``            (Utf8)  — ``"T{start}->T{end}"`` (clave de análisis)
    """
    import polars as pl
    if 'cut_from' in df_final.columns and 'cut_to' in df_final.columns:
        # corte por cruce de km: la leg es el par de cortes cruzados (con uno, T0->T0 = la
        # vuelta), no la terminal más cercana al primer y al último registro
        return df_final.with_columns([
            pl.col('cut_from').cast(pl.Int64).alias('start_terminal'),
            pl.col('cut_to').cast(pl.Int64).alias('end_terminal'),
            pl.format('T{}->T{}', pl.col('cut_from'), pl.col('cut_to')).alias('leg')])
    if not terminals:
        # Sin terminales descubiertas: todo el tráfico cae en una sola leg.
        return df_final.with_columns([
            pl.lit(0, dtype=pl.Int64).alias('start_terminal'),
            pl.lit(0, dtype=pl.Int64).alias('end_terminal'),
            pl.lit('T0->T0').alias('leg')])
    if lat0 is None:
        lat0 = float(df_final['latitude'].cast(pl.Float64).mean())
    cos0 = float(np.cos(np.radians(lat0)))
    tla = np.array([float(t[0]) for t in terminals])
    tlo = np.array([float(t[1]) for t in terminals])

    def _nearest(la, lo):
        dy = (la - tla) * 111320.0
        dx = (lo - tlo) * 111320.0 * cos0
        return int(np.argmin(dx * dx + dy * dy))

    ends = (df_final.sort(['trip_id', 'time']).group_by('trip_id', maintain_order=True).agg([
        pl.col('latitude').cast(pl.Float64).first().alias('la_s'),
        pl.col('longitude').cast(pl.Float64).first().alias('lo_s'),
        pl.col('latitude').cast(pl.Float64).last().alias('la_e'),
        pl.col('longitude').cast(pl.Float64).last().alias('lo_e')]))
    rows = []
    for r in ends.iter_rows(named=True):
        st = _nearest(r['la_s'], r['lo_s']); en = _nearest(r['la_e'], r['lo_e'])
        rows.append({'trip_id': r['trip_id'], 'start_terminal': st,
                     'end_terminal': en, 'leg': f'T{st}->T{en}'})
    legmap = pl.DataFrame(rows, schema={'trip_id': df_final['trip_id'].dtype,
                                        'start_terminal': pl.Int64,
                                        'end_terminal': pl.Int64, 'leg': pl.Utf8})
    return df_final.join(legmap, on='trip_id', how='left')


def render_terminal_dashboard(df_final, terminals, out_html, route, valid_only=True):
    """**Dashboard de viajes por terminal** (HTML autocontenido, Plotly inline).

    Reemplaza al dashboard viejo basado en ``direction``: agrupa por ``leg``
    (terminal→terminal) en lugar de 0/1, de modo que soporta rutas con más de 2
    terminales (un viaje que *inicia en la terminal X* se contabiliza en su leg).

    Requiere que ``df_final`` tenga ``leg`` (ver :func:`assign_terminal_legs`),
    ``trip_id``, ``bus_code``, ``time`` y ``cum_dist_m``. Si falta ``leg`` se
    asigna al vuelo con ``terminals``.

    Gráficas:
      g1 — nº de viajes por leg (terminal de inicio)
      g2 — distribución de duración (min) por leg
      g3 — distribución de longitud recorrida (km) por leg
      g4 — viajes por hora del día y leg
    """
    import polars as pl
    import plotly.graph_objects as go
    import plotly.io as pio

    df = df_final
    if 'leg' not in df.columns:
        df = assign_terminal_legs(df, terminals)
    if valid_only and 'trip_valid' in df.columns:
        df = df.filter(pl.col('trip_valid'))

    has_dist = 'cum_dist_m' in df.columns
    agg = [pl.col('leg').first().alias('leg'),
           pl.col('bus_code').first().alias('bus'),
           pl.col('time').min().alias('t0'), pl.col('time').max().alias('t1'),
           pl.len().alias('n_pts')]
    if has_dist:
        agg.append(pl.col('cum_dist_m').max().alias('len_m'))
    per_trip = (df.group_by('trip_id').agg(agg)
                  .with_columns([
                      ((pl.col('t1') - pl.col('t0')).dt.total_seconds() / 60).alias('dur_min'),
                      pl.col('t0').dt.hour().alias('hour')]))
    legs = sorted(per_trip['leg'].unique().to_list())
    palette = ['#1D9E75', '#D85A30', '#185FA5', '#BA7517', '#7F77DD',
               '#D4537E', '#639922', '#0F6E56', '#993C1D', '#534AB7']
    lcol = {lg: palette[i % len(palette)] for i, lg in enumerate(legs)}

    DARK = dict(paper_bgcolor='#262626', plot_bgcolor='#262626',
                font=dict(color='#ddd'), legend=dict(font=dict(color='#ccc')),
                xaxis=dict(gridcolor='#333', tickfont=dict(color='#aaa')),
                yaxis=dict(gridcolor='#333', tickfont=dict(color='#aaa')),
                margin=dict(t=50, b=50, l=60, r=20))

    # g1 — nº de viajes por leg
    counts = (per_trip.group_by('leg').len().sort('leg'))
    fig1 = go.Figure(go.Bar(x=counts['leg'].to_list(), y=counts['len'].to_list(),
                            marker_color=[lcol[l] for l in counts['leg'].to_list()]))
    fig1.update_layout(title='Viajes por leg (terminal de inicio→fin)',
                       xaxis_title='leg', yaxis_title='nº de viajes', **DARK)

    # g2 — duración por leg (histograma superpuesto)
    fig2 = go.Figure()
    for lg in legs:
        v = per_trip.filter(pl.col('leg') == lg)['dur_min'].to_numpy()
        v = v[(v > 0) & (v < 300)]
        fig2.add_trace(go.Histogram(x=v, name=lg, opacity=0.6,
                                    marker_color=lcol[lg], nbinsx=40))
    fig2.update_layout(title='Distribución de duración del viaje por leg',
                       barmode='overlay', xaxis_title='duración (min)',
                       yaxis_title='frecuencia', **DARK)

    # g3 — longitud recorrida por leg
    fig3 = go.Figure()
    if has_dist:
        for lg in legs:
            v = per_trip.filter(pl.col('leg') == lg)['len_m'].to_numpy() / 1000.0
            v = v[v > 0]
            fig3.add_trace(go.Histogram(x=v, name=lg, opacity=0.6,
                                        marker_color=lcol[lg], nbinsx=40))
        fig3.update_layout(title='Longitud recorrida por viaje y leg',
                           barmode='overlay', xaxis_title='km recorridos',
                           yaxis_title='frecuencia', **DARK)
    else:
        fig3.update_layout(title='Longitud recorrida (sin cum_dist_m)', **DARK)

    # g4 — viajes por hora del día y leg
    fig4 = go.Figure()
    for lg in legs:
        h = (per_trip.filter(pl.col('leg') == lg).group_by('hour').len().sort('hour'))
        fig4.add_trace(go.Scatter(x=h['hour'].to_list(), y=h['len'].to_list(),
                                  mode='lines+markers', name=lg,
                                  line=dict(color=lcol[lg], width=2)))
    fig4.update_layout(title='Viajes por hora del día y leg',
                       xaxis_title='hora', yaxis_title='nº de viajes', **DARK)

    # Tarjetas de stats + tabla por leg
    n_term = len(terminals) if terminals else per_trip['leg'].n_unique()
    cards = [('terminales', f'{n_term}'), ('legs', f'{len(legs)}'),
             ('viajes', f'{per_trip.height:,}'),
             ('buses', f"{per_trip['bus'].n_unique():,}"),
             ('duración mediana', f"{float(per_trip['dur_min'].median() or 0):.1f} min")]
    cards_html = ''.join(
        f'<div class="stat"><div class="v">{v}</div><div class="l">{l}</div></div>'
        for l, v in cards)

    trows = ''
    for lg in legs:
        sub = per_trip.filter(pl.col('leg') == lg)
        dur = sub['dur_min'].to_numpy(); dur = dur[(dur > 0) & (dur < 300)]
        km = (sub['len_m'].to_numpy() / 1000.0) if has_dist else np.array([np.nan])
        trows += (
            f"<tr><td>{lg}</td><td>{sub.height:,}</td>"
            f"<td>{sub['bus'].n_unique():,}</td>"
            f"<td>{(np.median(dur) if len(dur) else float('nan')):.1f}</td>"
            f"<td>{(np.median(km) if has_dist else float('nan')):.2f}</td></tr>")
    table = ('<table><thead><tr><th>leg</th><th>viajes</th><th>buses</th>'
             '<th>dur. mediana (min)</th><th>long. mediana (km)</th></tr></thead>'
             f'<tbody>{trows}</tbody></table>')

    div1 = pio.to_html(fig1, full_html=False, include_plotlyjs=True)
    div2 = pio.to_html(fig2, full_html=False, include_plotlyjs=False)
    div3 = pio.to_html(fig3, full_html=False, include_plotlyjs=False)
    div4 = pio.to_html(fig4, full_html=False, include_plotlyjs=False)

    html = f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8">
<title>Dashboard de viajes por terminal · ruta {route}</title>
<style>
  body {{ font-family:-apple-system,Segoe UI,sans-serif; background:#1b1b1b; color:#ddd; margin:0; padding:20px; }}
  h1 {{ color:#fff; }}
  .card {{ background:#262626; border-radius:6px; padding:20px; margin-bottom:24px; border:1px solid #333; }}
  .card h3 {{ margin-top:0; color:#fff; }}
  .stats {{ display:flex; gap:24px; flex-wrap:wrap; margin-bottom:18px; }}
  .stat {{ background:#1f1f1f; padding:10px 16px; border-radius:4px; border-left:3px solid #1D9E75; min-width:140px; }}
  .stat .v {{ font-size:22px; font-weight:bold; color:#fff; }}
  .stat .l {{ font-size:11px; color:#888; text-transform:uppercase; }}
  table {{ width:100%; border-collapse:collapse; font-size:13px; }}
  th, td {{ padding:6px 10px; text-align:right; border-bottom:1px solid #333; }}
  th {{ color:#aaa; font-weight:normal; }}
  td:first-child, th:first-child {{ text-align:left; }}
</style></head><body>
<h1>Viajes por terminal — ruta {route}</h1>
<p style="color:#888">Unidad de análisis: <b>leg</b> (terminal de inicio → terminal de fin), no la vieja <i>direction</i> 0/1.</p>
<div class="stats">{cards_html}</div>
<div class="card">{div1}</div>
<div class="card">{div2}</div>
<div class="card">{div3}</div>
<div class="card">{div4}</div>
<div class="card"><h3>Resumen por leg</h3>{table}</div>
</body></html>"""

    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding='utf-8')
    return out_html


def _segment_profile(df, seg_window_m):
    """Perfil por ventana de distancia fija (= la grid de los métodos de predicción).
    Para cada viaje interpola el tiempo acumulado sobre 0, W, 2W, … (igual que
    build_trajectory_matrix / el builder grid del LSTM) y de ahí saca, por ventana:
    tiempo (s) en cruzarla y velocidad efectiva (km/h). Devuelve un dict por leg con
    arrays alineados por índice de segmento. Trabaja en numpy; loop por viaje."""
    import polars as pl
    W = float(seg_window_m)
    rows_leg = {}     # leg -> {seg_idx: [list of dt_s]}
    base = (df.select(['trip_id', 'leg', 'time', 'cum_dist_m'])
              .sort(['trip_id', 'time']))
    for tid, sub in base.partition_by('trip_id', as_dict=True).items():
        d = sub['cum_dist_m'].cast(pl.Float64).to_numpy()
        tser = sub['time']
        # segundos desde el inicio del viaje
        t = (tser.cast(pl.Int64).to_numpy() - int(tser.cast(pl.Int64).min()))
        tdt = str(tser.dtype)
        div = 1_000_000 if 'us' in tdt else (1_000 if 'ms' in tdt else 1)
        t = t.astype(np.float64) / div
        if d.size < 3:
            continue
        # monotonicidad estricta de la distancia
        _, first = np.unique(d, return_index=True)
        d = d[first]; t = t[first]
        if not np.all(np.diff(d) >= 0) or d.max() < W:
            continue
        n_cells = int(d.max() // W)
        if n_cells < 1:
            continue
        grid = np.arange(0, n_cells + 1) * W
        tg = np.interp(grid, d, t)
        dt = np.diff(tg)                       # tiempo por ventana (s)
        leg = sub['leg'][0]
        acc = rows_leg.setdefault(leg, {})
        for s, dts in enumerate(dt):
            if dts > 0:
                acc.setdefault(s, []).append(float(dts))
    out = {}
    for leg, segs in rows_leg.items():
        idx = sorted(segs)
        dist_m = [int(s * W) for s in idx]
        t_med = [float(np.median(segs[s])) for s in idx]
        spd = [round((W / tm) * 3.6, 1) if tm > 0 else None for tm in t_med]
        out[leg] = {'dist_m': dist_m, 't_med_s': [round(x, 1) for x in t_med],
                    'speed_kmh': spd, 'n': [len(segs[s]) for s in idx]}
    return out


def render_data_dashboard(df_final, terminals, out_html, route,
                          seg_window_m=100, valid_only=True):
    """**Dashboard de datos** (el que antes era ``dashboard_AB``, ahora ``dashboard_data``),
    adaptado a la **lógica de terminales**: las vistas se agrupan por ``leg``
    (terminal→terminal), no por ``direction`` A/B.

    Reproduce las estadísticas del dashboard viejo y añade el análisis por
    **segmentos de distancia fija** = la ventana que usan los métodos de predicción
    (``seg_window_m``; por defecto 100 m = ``grid_step_m`` de baselines y LSTM):
      - KPIs globales (viajes, buses, terminales, legs, km totales, vel. media)
      - **Resumen de camiones** (por bus: viajes, km, horas activas, vel. media)
      - Viajes por hora del día
      - **Velocidad media por segmento** (km/h) y **tiempo por segmento** (s) por leg
      - Mapa de velocidad (pings coloreados por velocidad instantánea)

    Requiere en ``df_final``: ``bus_code``, ``trip_id``, ``time``, ``latitude``,
    ``longitude``, ``cum_dist_m`` y ``leg`` (si falta ``leg`` se asigna con
    ``terminals``).
    """
    import polars as pl
    import plotly.graph_objects as go
    import plotly.io as pio

    df = df_final
    if 'leg' not in df.columns:
        df = assign_terminal_legs(df, terminals)
    if valid_only and 'trip_valid' in df.columns:
        df = df.filter(pl.col('trip_valid'))
    df = df.with_columns([pl.col('latitude').cast(pl.Float64),
                          pl.col('longitude').cast(pl.Float64),
                          pl.col('cum_dist_m').cast(pl.Float64)])

    lat0 = float(df['latitude'].mean()); cos0 = float(np.cos(np.radians(lat0)))

    # ── velocidad instantánea por ping (haversine / dt) ──────────────────
    df = df.sort(['trip_id', 'time']).with_columns([
        (pl.col('latitude').diff().over('trip_id') * 111320.0).alias('_dy'),
        (pl.col('longitude').diff().over('trip_id') * 111320.0 * cos0).alias('_dx'),
        (pl.col('time').diff().over('trip_id').dt.total_seconds()).alias('_dt')])
    df = df.with_columns(
        pl.when(pl.col('_dt') > 0)
          .then(((pl.col('_dx')**2 + pl.col('_dy')**2).sqrt() / pl.col('_dt')) * 3.6)
          .otherwise(None).alias('speed_kmh'))

    # ── tabla por viaje ──────────────────────────────────────────────────
    per_trip = (df.group_by('trip_id').agg([
        pl.col('leg').first().alias('leg'), pl.col('bus_code').first().alias('bus'),
        pl.col('time').min().alias('t0'), pl.col('time').max().alias('t1'),
        pl.col('cum_dist_m').max().alias('len_m')])
        .with_columns([
            ((pl.col('t1') - pl.col('t0')).dt.total_seconds() / 60).alias('dur_min'),
            pl.col('t0').dt.hour().alias('hour')]))
    legs = sorted(per_trip['leg'].unique().to_list())
    palette = ['#1D9E75', '#D85A30', '#185FA5', '#BA7517', '#7F77DD',
               '#D4537E', '#639922', '#0F6E56', '#993C1D', '#534AB7']
    lcol = {lg: palette[i % len(palette)] for i, lg in enumerate(legs)}

    total_km = float(per_trip['len_m'].sum() or 0) / 1000.0
    total_h = float(per_trip['dur_min'].sum() or 0) / 60.0
    avg_speed = (total_km / total_h) if total_h > 0 else float('nan')

    DARK = dict(paper_bgcolor='#262626', plot_bgcolor='#262626',
                font=dict(color='#ddd'), legend=dict(font=dict(color='#ccc')),
                xaxis=dict(gridcolor='#333', tickfont=dict(color='#aaa')),
                yaxis=dict(gridcolor='#333', tickfont=dict(color='#aaa')),
                margin=dict(t=50, b=50, l=60, r=20))

    # ── g_hour — viajes por hora ─────────────────────────────────────────
    fig_h = go.Figure()
    for lg in legs:
        h = (per_trip.filter(pl.col('leg') == lg).group_by('hour').len().sort('hour'))
        fig_h.add_trace(go.Bar(x=h['hour'].to_list(), y=h['len'].to_list(),
                               name=lg, marker_color=lcol[lg]))
    fig_h.update_layout(title='Viajes por hora del día y leg', barmode='stack',
                        xaxis_title='hora', yaxis_title='nº de viajes', **DARK)

    # ── perfil por segmento (ventana = grid de predicción) ───────────────
    prof = _segment_profile(df, seg_window_m)
    fig_spd = go.Figure(); fig_tim = go.Figure()
    for lg in legs:
        p = prof.get(lg)
        if not p:
            continue
        fig_spd.add_trace(go.Scatter(x=p['dist_m'], y=p['speed_kmh'], mode='lines',
                                     name=lg, line=dict(color=lcol[lg], width=2)))
        fig_tim.add_trace(go.Scatter(x=p['dist_m'], y=p['t_med_s'], mode='lines',
                                     name=lg, line=dict(color=lcol[lg], width=2)))
    fig_spd.update_layout(
        title=f'Velocidad media por segmento de {seg_window_m} m (km/h)',
        xaxis_title='distancia sobre el viaje (m)', yaxis_title='velocidad (km/h)', **DARK)
    fig_tim.update_layout(
        title=f'Tiempo medio por segmento de {seg_window_m} m (s)',
        xaxis_title='distancia sobre el viaje (m)', yaxis_title='tiempo (s)', **DARK)

    # ── mapa de velocidad (pings coloreados, sin tiles → autocontenido) ──
    smp = df.filter(pl.col('speed_kmh').is_not_null())
    if smp.height > 25000:
        smp = smp.sample(25000, seed=42)
    spd_v = np.clip(smp['speed_kmh'].to_numpy(), 0, 80)
    fig_map = go.Figure(go.Scattergl(
        x=smp['longitude'].to_numpy(), y=smp['latitude'].to_numpy(), mode='markers',
        marker=dict(size=4, color=spd_v, colorscale='Turbo', cmin=0, cmax=80,
                    colorbar=dict(title='km/h'), opacity=0.6)))
    fig_map.update_layout(title='Mapa de velocidad instantánea (color = km/h)',
                          xaxis_title='lon', yaxis_title='lat',
                          yaxis=dict(scaleanchor='x', scaleratio=1 / cos0,
                                     gridcolor='#333', tickfont=dict(color='#aaa')),
                          **{k: v for k, v in DARK.items() if k != 'yaxis'})

    # ── KPIs + tabla de camiones + tabla por segmento/leg ────────────────
    n_term = len(terminals) if terminals else per_trip['leg'].n_unique()
    cards = [('viajes', f'{per_trip.height:,}'),
             ('camiones', f"{per_trip['bus'].n_unique():,}"),
             ('terminales', f'{n_term}'), ('legs', f'{len(legs)}'),
             ('km totales', f'{total_km:,.0f}'),
             ('duración mediana', f"{float(per_trip['dur_min'].median() or 0):.1f} min"),
             ('velocidad media', f'{avg_speed:.1f} km/h')]
    cards_html = ''.join(
        f'<div class="stat"><div class="v">{v}</div><div class="l">{l}</div></div>'
        for l, v in cards)

    # resumen por camión (bus)
    bus_t = (per_trip.group_by('bus').agg([
        pl.len().alias('viajes'), (pl.col('len_m').sum() / 1000).alias('km'),
        (pl.col('dur_min').sum() / 60).alias('horas'),
        pl.col('leg').n_unique().alias('legs')]).sort('km', descending=True))
    brows = ''.join(
        f"<tr><td>{r['bus']}</td><td>{r['viajes']:,}</td><td>{r['km']:.0f}</td>"
        f"<td>{r['horas']:.1f}</td>"
        f"<td>{((r['km']/r['horas']) if r['horas'] else float('nan')):.1f}</td>"
        f"<td>{r['legs']}</td></tr>"
        for r in bus_t.iter_rows(named=True))
    bus_table = ('<table><thead><tr><th>camión</th><th>viajes</th><th>km</th>'
                 '<th>horas activas</th><th>vel. media (km/h)</th><th>legs</th>'
                 f'</tr></thead><tbody>{brows}</tbody></table>')

    # resumen por segmento/leg (vel. media y tiempo medio del leg completo)
    srows = ''
    for lg in legs:
        p = prof.get(lg)
        if not p:
            continue
        sp = [x for x in p['speed_kmh'] if x is not None]
        srows += (f"<tr><td>{lg}</td><td>{len(p['dist_m'])}</td>"
                  f"<td>{(np.median(sp) if sp else float('nan')):.1f}</td>"
                  f"<td>{(np.median(p['t_med_s']) if p['t_med_s'] else float('nan')):.1f}</td>"
                  f"<td>{(max(p['dist_m'])/1000 if p['dist_m'] else 0):.1f}</td></tr>")
    seg_table = ('<table><thead><tr><th>leg</th>'
                 f'<th>nº segmentos ({seg_window_m} m)</th><th>vel. mediana (km/h)</th>'
                 '<th>tiempo mediano/seg (s)</th><th>alcance (km)</th>'
                 f'</tr></thead><tbody>{srows}</tbody></table>')

    div_h = pio.to_html(fig_h, full_html=False, include_plotlyjs=True)
    div_spd = pio.to_html(fig_spd, full_html=False, include_plotlyjs=False)
    div_tim = pio.to_html(fig_tim, full_html=False, include_plotlyjs=False)
    div_map = pio.to_html(fig_map, full_html=False, include_plotlyjs=False)

    html = f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8">
<title>Dashboard de datos · ruta {route}</title>
<style>
  body {{ font-family:-apple-system,Segoe UI,sans-serif; background:#1b1b1b; color:#ddd; margin:0; padding:20px; }}
  h1 {{ color:#fff; }} h3 {{ color:#fff; }}
  .card {{ background:#262626; border-radius:6px; padding:20px; margin-bottom:24px; border:1px solid #333; }}
  .stats {{ display:flex; gap:24px; flex-wrap:wrap; margin-bottom:18px; }}
  .stat {{ background:#1f1f1f; padding:10px 16px; border-radius:4px; border-left:3px solid #185FA5; min-width:130px; }}
  .stat .v {{ font-size:22px; font-weight:bold; color:#fff; }}
  .stat .l {{ font-size:11px; color:#888; text-transform:uppercase; }}
  table {{ width:100%; border-collapse:collapse; font-size:13px; }}
  th, td {{ padding:6px 10px; text-align:right; border-bottom:1px solid #333; }}
  th {{ color:#aaa; font-weight:normal; }}
  td:first-child, th:first-child {{ text-align:left; }}
  .scroll {{ max-height:360px; overflow-y:auto; }}
</style></head><body>
<h1>Dashboard de datos — ruta {route}</h1>
<p style="color:#888">Antes <i>dashboard_AB</i>. Vistas agrupadas por <b>leg</b>
(terminal→terminal). Segmentos = ventana de <b>{seg_window_m} m</b> (la misma grid de
baselines y LSTM).</p>
<div class="stats">{cards_html}</div>
<div class="card"><h3>Análisis por segmento — velocidad</h3>{div_spd}</div>
<div class="card"><h3>Análisis por segmento — tiempo</h3>{div_tim}</div>
<div class="card"><h3>Resumen por leg (segmentos de {seg_window_m} m)</h3>{seg_table}</div>
<div class="card"><h3>Viajes por hora</h3>{div_h}</div>
<div class="card"><h3>Mapa de velocidad</h3>{div_map}</div>
<div class="card"><h3>Resumen de camiones</h3><div class="scroll">{bus_table}</div></div>
</body></html>"""

    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding='utf-8')
    return out_html


def prepare_trips_for_enrich(df_trips, poly, cum=None, L=None):
    """Adapta el ``df_trips`` del método nuevo al formato que espera el **pipeline
    original** (``step_07`` enrich + auditorías): añade ``snap_lat``/``snap_lon``
    (proyección a la polilínea) y ``direction`` (0/1 geométrica por viaje, signo del
    avance sobre la ruta) si falta. NO filtra: conserva todos los pings y ``trip_valid``
    (step_07 filtra los válidos internamente).

    Columnas garantizadas a la salida: las de ``df_trips`` + ``snap_lat``, ``snap_lon``,
    ``snap_dist_m`` (distancia perpendicular del ping crudo a la polilínea = distancia al
    snap; la usa la sección B del audit de integridad) y ``direction`` — suficientes para
    ``enrich_trip_segments`` y los audits, que usan ``bus_code, speed, angle, time,
    snap_lat, snap_lon, snap_dist_m, trip_id, direction, trip_valid``.
    """
    import polars as pl
    if cum is None:
        cum = td._cumulative_distances(poly)
    if L is None:
        L = float(cum[-1])
    d = df_trips.with_columns([pl.col('latitude').cast(pl.Float64),
                               pl.col('longitude').cast(pl.Float64)])
    pos, _perp, slat, slon = td._project_many_to_polyline(
        d['latitude'].to_numpy(), d['longitude'].to_numpy(), poly, cum)
    d = d.with_columns([pl.Series('snap_lat', slat), pl.Series('snap_lon', slon),
                        pl.Series('snap_dist_m', np.round(_perp, 2)), pl.Series('_pos_m', pos)])
    if 'direction' not in d.columns:
        dirmap = (d.sort(['trip_id', 'time']).group_by('trip_id', maintain_order=True)
                   .agg((pl.col('_pos_m').last() - pl.col('_pos_m').first()).alias('_dp'))
                   .with_columns((pl.col('_dp') < 0).cast(pl.Int64).alias('direction')))
        d = d.join(dirmap.select(['trip_id', 'direction']), on='trip_id', how='left')
    return d.drop('_pos_m')


def build_legacy_features(df, df_segments, poly, cum=None, L=None,
                          route_prefix=None, valid_only=True, verbose=False):
    """Corre el **feature engineering del pipeline ORIGINAL** sobre los viajes del
    método nuevo, para poder regenerar el dashboard clásico (``dashboard_AB``).

    Lo único que cambia respecto al pipeline viejo es de dónde salen ``trip_id`` y
    ``direction`` (ahora del método nuevo de viajes/terminales). El resto es idéntico:
      1. **Snapea** cada ping a la polilínea de la ruta (snap_lat/snap_lon → lat/lon).
      2. ``enrich_trip_segments`` → ``next_stop``, ``m_next_stop``, ``s_next_stop``.
      3. ``compute_three_features`` → ``cum_dist_m``, ``s_next_stop_interp``,
         ``link_travel_time``, ``segment_length_m``, ``progress_in_segment``,
         ``hour_of_day``, ``day_of_week``.

    Parameters
    ----------
    df : pl.DataFrame
        Viajes del método nuevo (df_trips o df_final). Requiere ``bus_code``,
        ``trip_id``, ``time``, ``latitude``, ``longitude`` y ``trip_valid``;
        ``speed``/``angle`` y ``direction`` se usan si están (``direction`` se
        deriva si falta). ``leg`` se conserva si está.
    df_segments : pl.DataFrame
        Geometría de segmentos de la ruta (``stop``, ``polyline``).
    poly, cum, L : polilínea + distancias acumuladas (para snapear y derivar dirección).
    route_prefix : str | None
        Prefijo de ruta para el código de parada (p.ej. ``"1701"`` → ``"1701-FQAE"``).

    Returns
    -------
    (df_enriched, df_pipeline) — ambos listos para ``make_trips_dashboard``.
    """
    import polars as pl
    from _lib_features import enrich_trip_segments
    from _lib_dashboard import compute_three_features

    if cum is None:
        cum = td._cumulative_distances(poly)
    if L is None:
        L = float(cum[-1])

    d = df.filter(pl.col('trip_valid')) if (valid_only and 'trip_valid' in df.columns) else df
    d = d.with_columns([pl.col('latitude').cast(pl.Float64), pl.col('longitude').cast(pl.Float64)])

    # 1) snap a la polilínea (igual que el pipeline viejo, que enriquecía sobre snap_*)
    pos, _perp, slat, slon = td._project_many_to_polyline(
        d['latitude'].to_numpy(), d['longitude'].to_numpy(), poly, cum)
    d = d.with_columns([pl.Series('snap_lat', slat), pl.Series('snap_lon', slon),
                        pl.Series('_pos_m', pos)])

    # direction (0/1) — del método nuevo si existe; si no, signo del avance sobre la ruta
    if 'direction' not in d.columns:
        dirmap = (d.sort(['trip_id', 'time']).group_by('trip_id', maintain_order=True)
                   .agg((pl.col('_pos_m').last() - pl.col('_pos_m').first()).alias('_dp'))
                   .with_columns((pl.col('_dp') < 0).cast(pl.Int64).alias('direction')))
        d = d.join(dirmap.select(['trip_id', 'direction']), on='trip_id', how='left')

    if 'speed' not in d.columns:
        d = d.with_columns(pl.lit(None, dtype=pl.Float64).alias('speed'))
    if 'angle' not in d.columns:
        d = d.with_columns(pl.lit(None, dtype=pl.Float64).alias('angle'))

    keep = ['bus_code', 'speed', 'angle', 'time', 'trip_id', 'direction', 'trip_valid']
    if 'leg' in d.columns:
        keep.append('leg')
    # enrich opera sobre las coords SNAPEADAS (renombradas a latitude/longitude)
    df_in = (d.select(keep + ['snap_lat', 'snap_lon'])
              .rename({'snap_lat': 'latitude', 'snap_lon': 'longitude'}))

    if verbose:
        print(f'  [legacy] {df_in.height:,} pings · {df_in["trip_id"].n_unique()} viajes · '
              f'dir 0/1: {df_in.group_by("direction").len().sort("direction").to_dicts()}')

    df_enriched = enrich_trip_segments(df_in, df_segments, route_prefix=route_prefix, verbose=verbose)
    df_pipeline = compute_three_features(df_enriched, df_segments, route_prefix=route_prefix, verbose=verbose)
    return df_enriched, df_pipeline


def render_legacy_dashboard(df, df_segments, out_html, route, poly, cum=None, L=None,
                            route_prefix=None, step_km=0.1,
                            end_stop_label_a='Dirección A', end_stop_label_b='Dirección B',
                            valid_only=True, verbose=False):
    """**dashboard_original_legacy**: el dashboard clásico (``make_trips_dashboard`` /
    antiguo ``dashboard_AB``) regenerado sobre los viajes del método nuevo.

    Snapea + enriquece con las MISMAS features del pipeline original
    (:func:`build_legacy_features`) y produce el HTML con las vistas de siempre:
    Dir A / B / AB, Segmentos (link_travel_time + s_next_stop_interp), Mapa
    (segmentos por velocidad efectiva) y Reporte (KPIs, tabla de buses, horario).

    ``step_km`` = tamaño del tramo de la grilla espacial del dashboard; por defecto
    0.1 km (100 m) para coincidir con la ventana de los métodos de predicción.

    Returns
    -------
    (out_html, df_pipeline)
    """
    from _lib_dashboard import make_trips_dashboard
    _df_enriched, df_pipeline = build_legacy_features(
        df, df_segments, poly, cum=cum, L=L, route_prefix=route_prefix,
        valid_only=valid_only, verbose=verbose)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    make_trips_dashboard(
        df_pipeline, output_path=str(out_html),
        df_segments_route=df_segments, route_prefix=route_prefix, step_km=step_km,
        end_stop_label_a=end_stop_label_a, end_stop_label_b=end_stop_label_b,
        verbose=verbose)
    return out_html, df_pipeline


def orphan_map(otab, poly, out_html, route, lat0, lon0):
    """HTML aparte con los puntos huérfanos: color por inicio/fin/intermedio de ruta,
    círculo = footprint (spread) para los 'inactivo' (parado en un sitio)."""
    import folium
    COL = {'inicio': '#1D9E75', 'fin': '#D62728', 'intermedio': '#888'}
    m = folium.Map(location=[lat0, lon0], zoom_start=12, tiles='cartodbpositron')
    folium.PolyLine([(p[0], p[1]) for p in poly], color='#bbb', weight=2, opacity=0.5).add_to(m)
    grp = {k: folium.FeatureGroup(name=f'{k} de ruta', show=True) for k in ('inicio', 'fin', 'intermedio')}
    g_in = folium.FeatureGroup(name='inactivo (dwell en sitio)', show=True)
    for r in (otab.iter_rows(named=True) if otab.height else []):
        col = COL.get(r['label'], '#888')
        pop = (f"{r['label']} de ruta · {r['tipo']}<br>{r['n']} pts · dwell {r['dwell_min']:.0f} min · "
               f"spread {r['spread_m']:.0f} m · cov {r['cov']:.2f}<br>{r['lat']:.6f}, {r['lon']:.6f}")
        folium.CircleMarker([r['lat'], r['lon']], radius=5, color=col, weight=2, fill=True,
                            fill_opacity=0.7, popup=pop).add_to(grp[r['label']])
        if r['tipo'] == 'inactivo':
            folium.Circle([r['lat'], r['lon']], radius=min(max(float(r['spread_m']), 20.0), ORPHAN_SPREAD_M),
                          color='#222', weight=1, fill=True, fill_opacity=0.15).add_to(g_in)
    for g in list(grp.values()) + [g_in]:
        g.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    out_html.parent.mkdir(parents=True, exist_ok=True); m.save(str(out_html))
    return out_html


def split_trips_on_gap(df_trips, gap_min, suffix='_G'):
    """**Split por gap temporal** (opción b): parte cada ``trip_id`` en sus huecos
    internos > ``gap_min`` minutos. Reemplaza el ``GAP_SPLIT`` que el corte por
    turnaround/dwell omite: un 'viaje' que abarca un estacionamiento/turno largo se
    convierte en piezas temporalmente coherentes (cada hueco > ``gap_min`` es una
    frontera de viaje, no un agujero dentro de un viaje).

    Devuelve ``df_trips`` con ``trip_id`` re-sufijado (``_G0``, ``_G1`` …; las piezas
    quedan listas para reclasificar por cobertura) y conserva el id previo en
    ``trip_id_pregap``. NO fusiona ni filtra.
    """
    import polars as pl
    if df_trips.is_empty():
        return df_trips
    d = df_trips.sort(['trip_id', 'time'])
    d = d.with_columns(
        (pl.col('time').diff().over('trip_id').dt.total_seconds() / 60.0).alias('_gap'))
    d = d.with_columns(
        (pl.col('_gap') > gap_min).fill_null(False).cast(pl.Int64)
         .cum_sum().over('trip_id').alias('_piece'))
    d = d.with_columns([
        pl.col('trip_id').alias('trip_id_pregap'),
        (pl.col('trip_id') + pl.lit(suffix) + pl.col('_piece').cast(pl.Utf8)).alias('trip_id')])
    return d.drop(['_gap', '_piece'])


def trim_first_trip_warmup(df_trips, stop_speed=2.0, min_dwell_min=5.0, lead_max_pts=15,
                           skip_after=0):
    """Recorta el **calentamiento del primer viaje de cada (bus, día)**.

    Patrón objetivo: el primer viaje arranca con *unos pocos* puntos en movimiento,
    seguidos de un **cluster de espera** (muchos pings con ``speed <= stop_speed``) donde
    el bus aguarda antes de arrancar; el viaje "real" empieza cuando **sale del cluster**.
    Esta función marca como ``warmup=True`` esos pings (el cluster de espera **y los
    anteriores a él**), de modo que el viaje cuente desde la salida del cluster.

    ``skip_after``: además del cluster, ignora los **primeros ``skip_after`` pings tras la
    espera** (sólo si hubo espera). Análogo de inicio del filtro de fin de día: hay casos en
    que el bus, tras el dwell, sigue parado/merodeando **cerca del inicio** antes de arrancar
    de verdad, y esos pings no son viaje todavía.

    Detección por (bus, día), sólo sobre el **primer** ``trip_id`` del día (por hora):
      - se busca la **primera corrida de pings parados** (``speed <= stop_speed``) que
        empiece dentro de los primeros ``lead_max_pts`` pings y dure ``>= min_dwell_min``
        minutos (= el cluster de espera; un alto de semáforo corto no califica);
      - se marca ``warmup`` todo hasta el último ping de esa corrida (inclusive);
      - el primer ping después es el arranque real del viaje.
    Si el primer viaje arranca con movimiento continuo (comportamiento normal), no hay
    cluster que califique y no se marca nada. Devuelve ``df_trips`` con columna ``warmup``.
    """
    import polars as pl
    if df_trips.is_empty():
        return df_trips.with_columns(pl.lit(False).alias('warmup'))
    d = df_trips.with_columns(pl.col('time').dt.date().alias('_wday')).sort(['bus_code', '_wday', 'time'])
    first = d.group_by(['bus_code', '_wday']).agg(
        pl.col('trip_id').sort_by('time').first().alias('_ftid'))
    d = d.join(first, on=['bus_code', '_wday'], how='left')

    cuts = []
    fp = d.filter(pl.col('trip_id') == pl.col('_ftid'))
    for key, g in fp.group_by(['bus_code', '_wday'], maintain_order=True):
        g = g.sort('time')
        sp = g['speed'].fill_null(0.0).to_numpy()
        tms = g['time'].dt.timestamp('ms').to_numpy()
        n = len(sp); stat = sp <= stop_speed
        cut_ms = None; i = 0
        while i < n:
            if stat[i]:
                j = i
                while j + 1 < n and stat[j + 1]:
                    j += 1
                if i <= lead_max_pts and (tms[j] - tms[i]) / 60000.0 >= min_dwell_min:
                    cut_idx = min(j + int(skip_after), n - 1)   # + buffer tras la espera
                    cut_ms = int(tms[cut_idx]); break
                i = j + 1
            else:
                i += 1
        if cut_ms is not None:
            cuts.append({'bus_code': key[0], '_wday': key[1], '_cut_ms': cut_ms})

    if cuts:
        cut_df = pl.DataFrame(cuts).with_columns(pl.col('_wday').cast(d.schema['_wday']))
        d = d.join(cut_df, on=['bus_code', '_wday'], how='left')
        d = d.with_columns(
            ((pl.col('trip_id') == pl.col('_ftid')) & pl.col('_cut_ms').is_not_null()
             & (pl.col('time').dt.timestamp('ms') <= pl.col('_cut_ms'))).alias('warmup')).drop('_cut_ms')
    else:
        d = d.with_columns(pl.lit(False).alias('warmup'))
    return d.drop(['_wday', '_ftid'])


def mark_start_dwell(df_trips, stop_speed=2.0, min_dwell_min=2.0, lead_max_pts=15):
    """Marca ``start_dwell`` = la **espera inicial de CADA viaje** en su terminal_like antes
    de arrancar (generaliza el warmup del primer viaje del día a todos los viajes).

    Es común que el bus llegue a la terminal_like y se quede quieto un buen momento antes de
    iniciar el siguiente "viaje". Esta función identifica esa espera por trip_id: la primera
    corrida de pings parados (``speed <= stop_speed``) que empiece dentro de los primeros
    ``lead_max_pts`` pings del viaje y dure ``>= min_dwell_min`` minutos; marca ``start_dwell``
    todos los pings hasta el final de esa corrida (inclusive). El viaje "arranca" en el primer
    ping con movimiento después. Si el viaje arranca de inmediato (sin espera notable), no se
    marca nada. Devuelve ``df_trips`` con columna booleana ``start_dwell``.
    """
    import polars as pl
    if df_trips.is_empty():
        return df_trips.with_columns(pl.lit(False).alias('start_dwell'))
    cuts = []
    for key, g in df_trips.sort(['trip_id', 'time']).group_by(['trip_id'], maintain_order=True):
        sp = g['speed'].fill_null(0.0).to_numpy()
        tms = g['time'].dt.timestamp('ms').to_numpy()
        n = len(sp); stat = sp <= stop_speed
        cut_ms = None; i = 0
        while i < n:
            if stat[i]:
                j = i
                while j + 1 < n and stat[j + 1]:
                    j += 1
                if i <= lead_max_pts and (tms[j] - tms[i]) / 60000.0 >= min_dwell_min:
                    cut_ms = int(tms[j]); break
                i = j + 1
            else:
                i += 1
        if cut_ms is not None:
            cuts.append({'trip_id': key[0], '_cut_ms': cut_ms})
    if cuts:
        cut_df = pl.DataFrame(cuts).with_columns(pl.col('trip_id').cast(df_trips.schema['trip_id']))
        d = df_trips.join(cut_df, on='trip_id', how='left')
        d = d.with_columns(
            (pl.col('_cut_ms').is_not_null()
             & (pl.col('time').dt.timestamp('ms') <= pl.col('_cut_ms'))).alias('start_dwell')).drop('_cut_ms')
    else:
        d = df_trips.with_columns(pl.lit(False).alias('start_dwell'))
    return d


def _gap_hist_svg(after_vals, before_vals, gap_soft, gap_hard, nbins=30, W=760, H=190, pad=26):
    """Histograma SVG (escala log en altura) del hueco máximo por viaje: serie ANTES
    del split (gris, cola larga) vs DESPUÉS (azul/rojo, recortada). Líneas en los
    umbrales soft/hard. Evidencia visual de que el split colapsa la cola."""
    import math
    av = [v for v in after_vals if v is not None]
    bv = [v for v in before_vals if v is not None] if before_vals else []
    gmax = max([v for v in av + bv] + [1.0])
    edges = [gmax * k / nbins for k in range(nbins + 1)]

    def _counts(vals):
        c = [0] * nbins
        for v in vals:
            b = min(nbins - 1, int(v / gmax * nbins)) if gmax > 0 else 0
            c[b] += 1
        return c
    ca, cb = _counts(av), _counts(bv)
    cmax = max(ca + cb + [1])
    def _h(c):
        return (H - 2 * pad) * (math.log(c + 1) / math.log(cmax + 1)) if c else 0.0
    bw = (W - 2 * pad) / nbins
    soft_x = pad + (gap_soft / gmax) * (W - 2 * pad)
    hard_x = pad + (gap_hard / gmax) * (W - 2 * pad)
    bars = ''
    for i in range(nbins):
        x = pad + i * bw
        if bv:                                          # 'antes' al fondo (gris)
            hb = _h(cb[i])
            bars += (f"<rect x='{x:.1f}' y='{H-pad-hb:.1f}' width='{bw-1:.1f}' height='{hb:.1f}' "
                     f"fill='#c9ced3'><title>antes {edges[i]:.0f}-{edges[i+1]:.0f} min: {cb[i]}</title></rect>")
        ha = _h(ca[i])                                  # 'después' al frente
        col = '#D62728' if edges[i] >= gap_hard else ('#BA7517' if edges[i] >= gap_soft else '#4292c6')
        bars += (f"<rect x='{x+bw*0.18:.1f}' y='{H-pad-ha:.1f}' width='{bw*0.64:.1f}' height='{ha:.1f}' "
                 f"fill='{col}' opacity='0.92'><title>después {edges[i]:.0f}-{edges[i+1]:.0f} min: {ca[i]}</title></rect>")
    def _vline(x, lab, col):
        return (f"<line x1='{x:.1f}' y1='{pad}' x2='{x:.1f}' y2='{H-pad}' stroke='{col}' "
                f"stroke-dasharray='4 3' stroke-width='1.3'/>"
                f"<text x='{x+3:.1f}' y='{pad+10}' fill='{col}' font-size='10'>{lab}</text>")
    legend = ("<g font-size='10'>"
              "<rect x='600' y='8' width='10' height='10' fill='#c9ced3'/><text x='614' y='17' fill='#666'>antes</text>"
              "<rect x='660' y='8' width='10' height='10' fill='#4292c6'/><text x='674' y='17' fill='#666'>después</text></g>")
    return (f"<svg viewBox='0 0 {W} {H}' width='100%' style='max-width:{W}px'>{bars}"
            f"{_vline(soft_x, f'soft {gap_soft:.0f}m', '#BA7517')}{_vline(hard_x, f'hard {gap_hard:.0f}m', '#D62728')}"
            f"{legend}"
            f"<text x='{pad}' y='{H-7}' fill='#888' font-size='10'>0 min</text>"
            f"<text x='{W-pad:.0f}' y='{H-7}' fill='#888' font-size='10' text-anchor='end'>{gmax:.0f} min · altura log</text>"
            f"</svg>")


def render_gap_report(per_trip, out_html, route, split_info=None, gap_soft=10.0, top_n=300):
    """Reporte HTML (autocontenido) del **análisis de gaps + split por gap (opción b)**.

    ``per_trip`` (por PIEZA tras el split) debe tener: ``trip_id, n_pts, cov,
    max_gap_min, n_gaps_gt, is_fragment, has_gap, invalid``. ``split_info`` (dict) trae
    el antes/después para evaluar el método: ``gap_hard, min_frag_cov, trips_before,
    pieces_after, pieces_added, monsters_before, monsters_after, fragments, gap_before``.
    """
    import polars as pl
    si = split_info or {}
    gap_hard = float(si.get('gap_hard', 45.0))
    min_frag_cov = float(si.get('min_frag_cov', 0.25))
    n_pieces = per_trip.height
    n_frag   = int(per_trip.filter(pl.col('invalid')).height)
    n_gap    = int(per_trip.filter(pl.col('has_gap') & ~pl.col('invalid')).height)

    def _card(label, value, sub='', accent='#185FA5'):
        return (f"<div class='card'><div class='cv' style='color:{accent}'>{value}</div>"
                f"<div class='cl'>{label}</div><div class='cs'>{sub}</div></div>")
    cards = (
        _card('Piezas', f'{n_pieces:,}', f'tras split (antes {si.get("trips_before","?"):,})'
              if 'trips_before' in si else 'tras split', '#185FA5') +
        _card('Partidas por gap', f'+{si.get("pieces_added", 0):,}', f'huecos &gt; {gap_hard:.0f} min', '#0F6E56') +
        _card('Huecos &gt;60 min', f'{si.get("monsters_before","?")} → {si.get("monsters_after","?")}',
              'antes → después del split', '#D62728') +
        _card('Fragmentos', f'{n_frag:,}', f'cobertura &lt; {min_frag_cov:.0%} → inválidos', '#BA7517') +
        _card('Con hueco', f'{n_gap:,}', f'{gap_soft:.0f}–{gap_hard:.0f} min (válidos)', '#4292c6'))

    hist = _gap_hist_svg(per_trip['max_gap_min'].to_list(), si.get('gap_before'), gap_soft, gap_hard)

    # tabla: inválidos (fragmentos) primero, luego piezas con hueco residual
    tbl = (per_trip.filter(pl.col('invalid') | pl.col('has_gap'))
           .sort(['invalid', 'max_gap_min'], descending=True).head(top_n))
    def _badge(b, txt, col):
        return f"<span class='bdg' style='background:{col}'>{txt}</span>" if b else ''
    rows = ''
    for r in tbl.iter_rows(named=True):
        flags = (_badge(r['invalid'], 'INVÁLIDO (fragmento)', '#D62728') +
                 _badge(r['has_gap'] and not r['invalid'], 'hueco', '#4292c6'))
        tr_cls = " class='sg'" if r['invalid'] else ''
        rows += (f"<tr{tr_cls}><td>{r['trip_id']}</td><td>{r['n_pts']}</td><td>{r['cov']:.2f}</td>"
                 f"<td>{r['max_gap_min']:.1f}</td><td>{r['n_gaps_gt']}</td><td>{flags}</td></tr>")

    html = f"""<!DOCTYPE html><html><head><meta charset='utf-8'>
<title>Reporte de gaps · ruta {route}</title><style>
body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#f4f6f8;color:#1a1a1a}}
.wrap{{max-width:920px;margin:0 auto;padding:24px}}
h1{{font-size:20px;margin:0 0 4px}} .sub{{color:#666;font-size:13px;margin-bottom:18px}}
.cards{{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:20px}}
.card{{background:#fff;border:1px solid #e2e6ea;border-radius:10px;padding:12px 16px;min-width:120px;flex:1}}
.cv{{font-size:22px;font-weight:700}} .cl{{font-size:12px;color:#444;margin-top:2px}}
.cs{{font-size:11px;color:#888;margin-top:2px}}
.panel{{background:#fff;border:1px solid #e2e6ea;border-radius:10px;padding:16px;margin-bottom:20px}}
.panel h2{{font-size:14px;margin:0 0 10px;color:#333}}
table{{width:100%;border-collapse:collapse;font-size:12px}}
th,td{{text-align:left;padding:6px 10px;border-bottom:1px solid #eef1f3}}
th{{color:#666;font-weight:600;text-transform:uppercase;font-size:10px;letter-spacing:.04em}}
tr.sg{{background:#fdecec}} td:nth-child(n+2):nth-child(-n+5){{font-variant-numeric:tabular-nums}}
.bdg{{display:inline-block;color:#fff;font-size:10px;padding:1px 6px;border-radius:8px;margin-right:4px}}
</style></head><body><div class='wrap'>
<h1>Análisis de gaps temporales — ruta {route}</h1>
<div class='sub'><b>Opción b (split por gap):</b> cada hueco temporal <b>&gt; {gap_hard:.0f} min</b> parte el
viaje en piezas coherentes; luego una pieza con <b>cobertura &lt; {min_frag_cov:.0%}</b> de la polilínea
se marca inválida (fragmento) y se separa. Así desaparecen los 'viajes' que abarcaban horas de hueco.</div>
<div class='cards'>{cards}</div>
<div class='panel'><h2>Hueco máximo por viaje — antes (gris) vs después del split (color)</h2>{hist}</div>
<div class='panel'><h2>Piezas inválidas / con hueco residual (top {top_n})</h2>
<table><thead><tr><th>trip_id</th><th>n_pts</th><th>cov</th><th>max_gap (min)</th><th>n_gaps&gt;{gap_soft:.0f}</th><th>flags</th></tr></thead>
<tbody>{rows or "<tr><td colspan=6><i>ninguno</i></td></tr>"}</tbody></table></div>
</div></body></html>"""
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding='utf-8')
    return out_html


# ──────────────────────────────────────────────────────────────────────
# Reporte + mapa del paso SNAP + DIRECCIÓN (sobre la salida de
# prepare_trips_for_enrich): calidad de la proyección a la polilínea
# (snap_dist_m) y balance de las 2 direcciones geométricas.
# ──────────────────────────────────────────────────────────────────────

def _snap_hist_svg(vals, thr_m=None, nbins=40, W=760, H=190, pad=26, clip_q=0.99):
    """Histograma SVG (altura log) de ``snap_dist_m``. Recorta la cola al percentil
    ``clip_q`` para que los outliers no aplasten el dibujo y traza una línea en el
    umbral ``thr_m`` (off-ruta)."""
    import math
    v = sorted(x for x in vals if x is not None and not (isinstance(x, float) and x != x))
    if not v:
        return "<svg viewBox='0 0 760 190'></svg>"
    vmax = max(v[min(len(v) - 1, int(clip_q * len(v)))], 1.0)
    counts = [0] * nbins
    for x in v:
        b = min(nbins - 1, int(x / vmax * nbins)) if vmax > 0 else 0
        counts[b] += 1
    cmax = max(counts + [1])
    bw = (W - 2 * pad) / nbins
    bars = ''
    for i, c in enumerate(counts):
        h = (H - 2 * pad) * (math.log(c + 1) / math.log(cmax + 1)) if c else 0.0
        x = pad + i * bw
        edge0, edge1 = vmax * i / nbins, vmax * (i + 1) / nbins
        col = '#D62728' if (thr_m is not None and edge0 >= thr_m) else '#4292c6'
        bars += (f"<rect x='{x+bw*0.1:.1f}' y='{H-pad-h:.1f}' width='{bw*0.8:.1f}' height='{h:.1f}' "
                 f"fill='{col}' opacity='0.9'><title>{edge0:.0f}-{edge1:.0f} m: {c}</title></rect>")
    thr_svg = ''
    if thr_m is not None and thr_m < vmax:
        tx = pad + (thr_m / vmax) * (W - 2 * pad)
        thr_svg = (f"<line x1='{tx:.1f}' y1='{pad}' x2='{tx:.1f}' y2='{H-pad}' stroke='#D62728' "
                   f"stroke-dasharray='4 3' stroke-width='1.3'/>"
                   f"<text x='{tx+3:.1f}' y='{pad+10}' fill='#D62728' font-size='10'>off-ruta {thr_m:.0f}m</text>")
    return (f"<svg viewBox='0 0 {W} {H}' width='100%' style='max-width:{W}px'>{bars}{thr_svg}"
            f"<text x='{pad}' y='{H-7}' fill='#888' font-size='10'>0 m</text>"
            f"<text x='{W-pad:.0f}' y='{H-7}' fill='#888' font-size='10' text-anchor='end'>"
            f"{vmax:.0f} m (p{int(clip_q*100)}) · altura log</text></svg>")


def render_snap_report(df_enrich, out_html, route, off_m=None, top_n=300):
    """Reporte HTML (autocontenido) del paso **snap + dirección**
    (``prepare_trips_for_enrich``).

    ``df_enrich`` debe traer: ``snap_dist_m``, ``direction``, ``trip_id``,
    ``bus_code`` (y opcionalmente ``trip_valid``). Resume la **calidad del snap** a la
    polilínea (qué tan lejos cae cada ping crudo de la ruta) y el **balance de las 2
    direcciones** geométricas, además de listar los viajes peor snapeados (candidatos a
    off-ruta / depósito / mal proyectados)."""
    import polars as pl
    off_m = float(off_m if off_m is not None else OFFROUTE_M)
    d = df_enrich
    n_ping = d.height
    n_trip = d['trip_id'].n_unique()
    n_bus  = d['bus_code'].n_unique()

    sd = d['snap_dist_m'].cast(pl.Float64).drop_nulls()
    med = float(sd.median() or 0.0); p90 = float(sd.quantile(0.90) or 0.0)
    p99 = float(sd.quantile(0.99) or 0.0); mx = float(sd.max() or 0.0)
    _w = lambda t: (d.filter(pl.col('snap_dist_m') <= t).height / n_ping * 100.0) if n_ping else 0.0
    pct_off = (d.filter(pl.col('snap_dist_m') > off_m).height / n_ping * 100.0) if n_ping else 0.0

    def _dir_counts(df):
        g = (df.group_by('direction')
               .agg([pl.len().alias('p'), pl.col('trip_id').n_unique().alias('t')])
               .sort('direction'))
        return {(None if r['direction'] is None else int(r['direction'])): (int(r['p']), int(r['t']))
                for r in g.iter_rows(named=True)}
    dc_all = _dir_counts(d)
    has_valid = 'trip_valid' in d.columns
    dc_val = _dir_counts(d.filter(pl.col('trip_valid'))) if has_valid else {}

    def _card(label, value, sub='', accent='#185FA5'):
        return (f"<div class='card'><div class='cv' style='color:{accent}'>{value}</div>"
                f"<div class='cl'>{label}</div><div class='cs'>{sub}</div></div>")
    cards = (
        _card('Pings', f'{n_ping:,}', f'{n_trip:,} viajes · {n_bus} buses', '#185FA5') +
        _card('snap_dist mediana', f'{med:.0f} m', f'p90 {p90:.0f} · p99 {p99:.0f} m', '#0F6E56') +
        _card('snap_dist máx', f'{mx:.0f} m', 'ping más lejano de la ruta', '#BA7517') +
        _card('On-ruta', f'{_w(off_m):.1f}%', f'snap ≤ {off_m:.0f} m', '#1D9E75') +
        _card('Off-ruta', f'{pct_off:.1f}%', f'snap &gt; {off_m:.0f} m (depósito/ruido)', '#D62728'))

    hist = _snap_hist_svg(sd.to_list(), thr_m=off_m)

    # Panel de dirección: pings y viajes por dir (todos y, si existe, válidos).
    def _dirrow(k):
        pa, ta = dc_all.get(k, (0, 0))
        pv, tv = dc_val.get(k, (0, 0))
        kd = 'sin dir' if k is None else f'dir {k}'
        valcell = f"<td>{pv:,}</td><td>{tv:,}</td>" if has_valid else ''
        return f"<tr><td>{kd}</td><td>{pa:,}</td><td>{ta:,}</td>{valcell}</tr>"
    dir_keys = sorted([k for k in dc_all if k is not None]) + ([None] if None in dc_all else [])
    valhead = "<th>pings (válidos)</th><th>viajes (válidos)</th>" if has_valid else ''
    dir_table = (
        "<table><thead><tr><th>dirección</th><th>pings</th><th>viajes</th>"
        f"{valhead}</tr></thead><tbody>"
        + ''.join(_dirrow(k) for k in dir_keys) + "</tbody></table>")

    # Viajes peor snapeados (mediana de snap_dist desc).
    valid_expr = pl.col('trip_valid').first().alias('valid') if has_valid \
        else pl.lit(True).alias('valid')
    per_trip = (d.group_by('trip_id').agg([
                    pl.col('bus_code').first().alias('bus'),
                    pl.col('direction').first().alias('dir'),
                    pl.len().alias('n_pts'),
                    pl.col('snap_dist_m').median().alias('sd_med'),
                    pl.col('snap_dist_m').max().alias('sd_max'),
                    (pl.col('snap_dist_m') > off_m).mean().alias('frac_off'),
                    valid_expr,
                ])
                .sort('sd_med', descending=True)
                .head(top_n))
    rows = ''
    for r in per_trip.iter_rows(named=True):
        far = (r['sd_med'] or 0) > off_m
        badge = "<span class='bdg' style='background:#D62728'>off-ruta</span>" if far else ''
        inval = "<span class='bdg' style='background:#999'>inválido</span>" if (has_valid and not r['valid']) else ''
        tr_cls = " class='sg'" if far else ''
        rows += (f"<tr{tr_cls}><td>{r['trip_id']}</td><td>{r['bus']}</td><td>{r['dir']}</td>"
                 f"<td>{r['n_pts']}</td><td>{(r['sd_med'] or 0):.0f}</td><td>{(r['sd_max'] or 0):.0f}</td>"
                 f"<td>{100*(r['frac_off'] or 0):.0f}%</td><td>{badge}{inval}</td></tr>")

    html = f"""<!DOCTYPE html><html><head><meta charset='utf-8'>
<title>Reporte snap + dirección · ruta {route}</title><style>
body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#f4f6f8;color:#1a1a1a}}
.wrap{{max-width:920px;margin:0 auto;padding:24px}}
h1{{font-size:20px;margin:0 0 4px}} .sub{{color:#666;font-size:13px;margin-bottom:18px}}
.cards{{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:20px}}
.card{{background:#fff;border:1px solid #e2e6ea;border-radius:10px;padding:12px 16px;min-width:120px;flex:1}}
.cv{{font-size:22px;font-weight:700}} .cl{{font-size:12px;color:#444;margin-top:2px}}
.cs{{font-size:11px;color:#888;margin-top:2px}}
.panel{{background:#fff;border:1px solid #e2e6ea;border-radius:10px;padding:16px;margin-bottom:20px}}
.panel h2{{font-size:14px;margin:0 0 10px;color:#333}}
table{{width:100%;border-collapse:collapse;font-size:12px}}
th,td{{text-align:left;padding:6px 10px;border-bottom:1px solid #eef1f3}}
th{{color:#666;font-weight:600;text-transform:uppercase;font-size:10px;letter-spacing:.04em}}
tr.sg{{background:#fdecec}} td{{font-variant-numeric:tabular-nums}}
.bdg{{display:inline-block;color:#fff;font-size:10px;padding:1px 6px;border-radius:8px;margin-right:4px}}
</style></head><body><div class='wrap'>
<h1>Snap a la polilínea + dirección geométrica — ruta {route}</h1>
<div class='sub'>Tras <b>prepare_trips_for_enrich</b>: cada ping crudo se proyecta a la polilínea
(<b>snap_dist_m</b> = distancia perpendicular) y se asigna una de las 2 <b>direcciones</b> geométricas por viaje.
Pings con snap_dist <b>&gt; {off_m:.0f} m</b> caen lejos de la ruta (depósito, ruido GPS o tramos no mapeados).</div>
<div class='cards'>{cards}</div>
<div class='panel'><h2>Distribución de snap_dist_m (distancia ping → ruta)</h2>{hist}</div>
<div class='panel'><h2>Balance de direcciones</h2>{dir_table}</div>
<div class='panel'><h2>Viajes peor snapeados (mediana snap_dist, top {top_n})</h2>
<table><thead><tr><th>trip_id</th><th>bus</th><th>dir</th><th>n_pts</th><th>snap med (m)</th><th>snap máx (m)</th><th>% off-ruta</th><th>flags</th></tr></thead>
<tbody>{rows or "<tr><td colspan=8><i>ninguno</i></td></tr>"}</tbody></table></div>
</div></body></html>"""
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding='utf-8')
    return out_html


def render_snap_map(df_enrich, poly, out_html, route, off_m=None, max_pts=6000):
    """Mapa folium del paso **snap + dirección** para UN bus (el de más viajes):

      * puntos **snapeados** (sobre la polilínea) coloreados por **dirección** (0/1),
      * **conectores** raw→snap (línea fina del ping crudo a su proyección; sólo si el
        desplazamiento es notable, para no saturar),
      * resaltado en **rojo** de los pings con ``snap_dist_m > off_m`` (off-ruta).

    Capas conmutables con ``LayerControl``. Análogo en espíritu al mapa
    ``new_trips_filtered`` pero enfocado en lo que añade este paso."""
    import polars as pl, folium
    off_m = float(off_m if off_m is not None else OFFROUTE_M)
    LAT0 = float(np.mean([p[0] for p in poly])); LON0 = float(np.mean([p[1] for p in poly]))

    # Bus con más viajes (mismo criterio que render_trips_map).
    bus0 = (df_enrich.group_by('bus_code').agg(pl.col('trip_id').n_unique().alias('nt'))
            .sort('nt', descending=True)['bus_code'][0])
    d = df_enrich.filter(pl.col('bus_code') == bus0).sort('time')
    # Cap por tamaño: si hay demasiados pings, conserva los primeros días completos.
    if d.height > max_pts:
        d = d.with_columns(pl.col('time').dt.date().cast(pl.Utf8).alias('_d'))
        keep, tot = [], 0
        for dd in d['_d'].unique(maintain_order=True).to_list():
            n = d.filter(pl.col('_d') == dd).height
            if tot + n > max_pts and keep:
                break
            keep.append(dd); tot += n
        d = d.filter(pl.col('_d').is_in(keep)).drop('_d')

    m = folium.Map(location=[LAT0, LON0], zoom_start=12, tiles='cartodbpositron')
    folium.PolyLine([(p[0], p[1]) for p in poly], color='#bbb', weight=2, opacity=0.7).add_to(m)

    DIRCOL = {0: '#185FA5', 1: '#D85A30'}
    fg0 = folium.FeatureGroup(name='Dirección 0 (snap)', show=True)
    fg1 = folium.FeatureGroup(name='Dirección 1 (snap)', show=True)
    fgc = folium.FeatureGroup(name='Conectores raw-&gt;snap', show=True)
    fgf = folium.FeatureGroup(name=f'Snap lejano (&gt;{off_m:.0f} m)', show=True)

    for r in d.to_dicts():
        la, lo = r['latitude'], r['longitude']
        sla, slo = r['snap_lat'], r['snap_lon']
        sd = r.get('snap_dist_m') or 0.0
        dr = r.get('direction')
        col = DIRCOL.get(int(dr) if dr is not None else -1, '#888')
        tstr = r['time'].strftime('%m-%d %H:%M:%S') if r.get('time') is not None else '?'
        tip = f"{r['trip_id']} · {tstr} · dir {dr} · snap {sd:.0f} m"
        folium.CircleMarker([sla, slo], radius=2.5, color=col, fill=True, fill_opacity=0.85,
                            weight=0, tooltip=tip).add_to(fg1 if dr == 1 else fg0)
        if sd and sd > 8:   # conector sólo si el ping se movió algo al proyectar
            ccol = '#D62728' if sd > off_m else '#999'
            folium.PolyLine([(la, lo), (sla, slo)], color=ccol, weight=1, opacity=0.5).add_to(fgc)
        if sd and sd > off_m:
            folium.CircleMarker([la, lo], radius=3.5, color='#D62728', fill=True, fill_opacity=0.9,
                                weight=1, tooltip='RAW off-ruta · ' + tip).add_to(fgf)

    for fg in (fg0, fg1, fgc, fgf):
        fg.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    legend = ("<div style='position:fixed;bottom:18px;left:18px;z-index:9999;background:rgba(15,17,23,.9);"
              "color:#e0e0e0;font:12px monospace;padding:10px 14px;border-radius:8px;"
              "box-shadow:0 4px 24px rgba(0,0,0,.5)'>"
              f"<b>Snap + dirección · bus {bus0} · ruta {route}</b><br>"
              "<span style='color:#185FA5'>&#9679;</span> dir 0 &nbsp;"
              "<span style='color:#D85A30'>&#9679;</span> dir 1<br>"
              f"<span style='color:#D62728'>&#9679;</span> snap &gt; {off_m:.0f} m (off-ruta)</div>")
    m.get_root().html.add_child(folium.Element(legend))
    out_html.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_html))
    return out_html
