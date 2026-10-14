"""
_lib_terminal_discovery.py  ·  DRAFT / EXPLORATORIO
===================================================
Descubrimiento **data-driven** del número de terminales (N) de una ruta, sin
asumir N=2 como hace ``_lib_turn_hex`` por construcción, y **sin usar señales
derivadas de la propia definición de terminales** (ver nota de fuga abajo).

Idea (ver ``docs/HANDOFF_DESCUBRIMIENTO_N_TERMINALES.md``):

    candidatos por trayectoria  →  clustering espacial  →  filtro por
    discriminantes que la densidad H3 NO aplica  →  N = clusters que sobreviven.

Dos detectores complementarios
------------------------------
1. **Layover** (``discover_terminals``): stay-points largos por trayectoria
   (Li & Zheng 2008) → DBSCAN → filtro de recurrencia/on-route. Cuenta
   terminales *de descanso*. Usa sólo posición + tiempo.
2. **Reversión de avance** (``detect_turnarounds``): dónde se **invierte el
   sentido de avance del bus sobre la polilínea** de forma recurrente. Cuenta
   *fronteras de viaje* aunque no haya layover largo. Es el discriminante
   correcto para una ruta loop S¹.

⚠ NOTA DE FUGA (leakage) — por qué NO se usa ``direction``
----------------------------------------------------------
La columna ``direction`` de ``df_route_direction`` se calcula en ``step_04``
mediante ``_split_circular_by_end_stops``, que **recibe las terminales como
input**. Usarla para *descubrir* terminales sería circular (validar la
conclusión con un dato que la asume). El detector de reversión usa, en su lugar,
la **posición proyectada sobre la polilínea cruda** (geometría externa del
GeoJSON, independiente de cualquier terminal) y el signo de su variación en el
tiempo. El raw sólo tiene ``bus_code, lat, lon, speed, angle, time``.

(Para auditar/comparar, ``detect_turnarounds(source="direction")`` reproduce el
cálculo con la columna ``direction`` — pero está marcado como circular y es sólo
para cross-check, nunca para el conteo final.)

Dependencias
------------
Sólo ``numpy`` + ``_lib_geo._haversine_m``. No requiere sklearn / shapely /
pyproj. DBSCAN, proyección métrica y point-to-polyline se implementan aquí.

Estado: DRAFT. No lo importa el pipeline productivo. Sólo lo usa
``notebooks/test_n_terminals.ipynb``.
"""

from __future__ import annotations

import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from _lib_geo import _haversine_m  # noqa: E402


# ──────────────────────────────────────────────────────────────────────
# Polilínea: carga + helpers de proyección
# ──────────────────────────────────────────────────────────────────────

def load_route_polyline(geojson_path: str, route_id) -> List[Tuple[float, float]]:
    """Carga la polilínea (ciclo completo) de ``route_id`` → ``[(lat, lon), ...]``."""
    with open(geojson_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    route_key = str(route_id)

    def _coords(geom: dict) -> List[Tuple[float, float]]:
        return [(c[1], c[0]) for c in geom["coordinates"]]  # GeoJSON [lon,lat] → (lat,lon)

    def _feat_route(feat: dict) -> Optional[str]:
        props = feat.get("properties", {}) or {}
        for k in ("id", "line", "route", "route_id"):
            if props.get(k) is not None:
                return str(props[k])
        return None

    feats = data.get("features", []) or []
    matched = [f for f in feats
               if f.get("geometry", {}).get("type") == "LineString"
               and _feat_route(f) == route_key]
    if not matched:
        raise ValueError(
            f"No se encontró LineString id={route_key!r} en {geojson_path}. "
            f"IDs: {sorted({_feat_route(f) for f in feats if _feat_route(f)})[:20]}")
    coords: List[Tuple[float, float]] = []
    for f in matched:
        coords.extend(_coords(f["geometry"]))
    return coords


def _cumulative_distances(coords: List[Tuple[float, float]]) -> np.ndarray:
    cum = [0.0]
    for i in range(1, len(coords)):
        cum.append(cum[-1] + _haversine_m(coords[i - 1][0], coords[i - 1][1],
                                          coords[i][0], coords[i][1]))
    return np.array(cum)


def _project_to_polyline(p_lat, p_lon, coords, cum) -> Tuple[float, float, float, float]:
    """Un punto → (snap_lat, snap_lon, perp_m, pos_m). Copia de _lib_map_match."""
    best_perp, best_lat, best_lon, best_pos = np.inf, coords[0][0], coords[0][1], 0.0
    for i in range(len(coords) - 1):
        lat1, lon1 = coords[i]; lat2, lon2 = coords[i + 1]
        seg_len = cum[i + 1] - cum[i]
        if seg_len < 1e-9:
            continue
        cos_mid = math.cos(math.radians((lat1 + lat2) / 2.0))
        dx_s = (lon2 - lon1) * cos_mid; dy_s = lat2 - lat1
        seg_sq = dx_s ** 2 + dy_s ** 2
        if seg_sq < 1e-20:
            continue
        dx_p = (p_lon - lon1) * cos_mid; dy_p = p_lat - lat1
        t = max(0.0, min(1.0, (dx_p * dx_s + dy_p * dy_s) / seg_sq))
        proj_lat = lat1 + t * (lat2 - lat1); proj_lon = lon1 + t * (lon2 - lon1)
        perp = math.sqrt(((p_lon - proj_lon) * cos_mid * 111_320) ** 2
                         + ((p_lat - proj_lat) * 110_540) ** 2)
        if perp < best_perp:
            best_perp, best_lat, best_lon, best_pos = perp, proj_lat, proj_lon, cum[i] + t * seg_len
    return best_lat, best_lon, best_perp, best_pos


def _project_many_to_polyline(plats: np.ndarray, plons: np.ndarray,
                              coords: List[Tuple[float, float]], cum: np.ndarray
                              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Versión vectorizada: proyecta N puntos sobre la polilínea de M vértices.
    Loop sobre M-1 segmentos (no sobre N puntos) → O(N·M) en numpy, memoria O(N).
    Devuelve (pos_m, perp_m, snap_lat, snap_lon), cada uno shape (N,).
    """
    n = plats.shape[0]
    best_perp = np.full(n, np.inf)
    best_pos = np.zeros(n); best_lat = np.zeros(n); best_lon = np.zeros(n)
    arr = np.asarray(coords, dtype=float)
    for i in range(len(coords) - 1):
        lat1, lon1 = arr[i]; lat2, lon2 = arr[i + 1]
        seg_len = cum[i + 1] - cum[i]
        if seg_len < 1e-9:
            continue
        cos_mid = math.cos(math.radians((lat1 + lat2) / 2.0))
        dx_s = (lon2 - lon1) * cos_mid; dy_s = lat2 - lat1
        seg_sq = dx_s ** 2 + dy_s ** 2
        if seg_sq < 1e-20:
            continue
        dx_p = (plons - lon1) * cos_mid; dy_p = plats - lat1
        t = np.clip((dx_p * dx_s + dy_p * dy_s) / seg_sq, 0.0, 1.0)
        proj_lat = lat1 + t * (lat2 - lat1); proj_lon = lon1 + t * (lon2 - lon1)
        perp = np.sqrt(((plons - proj_lon) * cos_mid * 111_320) ** 2
                       + ((plats - proj_lat) * 110_540) ** 2)
        upd = perp < best_perp
        best_perp = np.where(upd, perp, best_perp)
        best_pos = np.where(upd, cum[i] + t * seg_len, best_pos)
        best_lat = np.where(upd, proj_lat, best_lat)
        best_lon = np.where(upd, proj_lon, best_lon)
    return best_pos, best_perp, best_lat, best_lon


# ──────────────────────────────────────────────────────────────────────
# Proyección COHERENTE EN EL TIEMPO  (sep-2026)
#
# ⚠ Por qué existe: ``_project_many_to_polyline`` toma el **mínimo global**, y en
# una traza de ida y vuelta las dos ramas van por la misma calle, separadas por
# unos metros. Basta que un ping caiga unos centímetros del otro lado para que se
# proyecte en la rama contraria y, como las dos difieren en 30-50 % de la posición
# lineal, ese salto se contabiliza después como **avance**.
#
# No es la costura del lazo (ese salto es de ~100 % y ya se corrige desenrollando):
# es un salto de MEDIA traza, justo por debajo del umbral ``L/2`` con el que se
# detecta la costura, y por eso pasaba inadvertido.
#
# Medido en la 601 (semana 2-8 sep 2024, 1,383 viajes): el **84 %** de los viajes
# tenía al menos un salto, la distancia declarada era **1.79x** la recorrida en la
# mediana, y **527 viajes marcados válidos** la tenían inflada más de 1.5x. Como la
# cobertura se mide sobre esta posición y la cobertura ES el criterio de validez,
# el filtro dejaba pasar viajes que no lo merecían.
#
# El arreglo no elige el punto más cercano sino la **secuencia** coherente con el
# movimiento: candidatos = mínimos locales; elección por Viterbi con emisión =
# distancia a la traza y transición = |‖Δposición circular‖ − desplazamiento real|.
# Un salto de rama implica avanzar 30 km habiendo recorrido 300 m: paga ~30,000 de
# transición para ahorrar unos metros de emisión, así que queda descartado.
#
# ⚠ **Sólo aplica a SECUENCIAS** — los pings de un viaje, en orden. Para puntos
# sueltos (centroides de cluster, coordenadas de terminal) no hay secuencia que
# seguir y el mínimo global sigue siendo lo correcto: ``_project_to_polyline``.
#
# Diagnóstico y medición: ``docs/HANDOFF_CORTE_GEOMETRICO.md`` § Capa 2.
# ──────────────────────────────────────────────────────────────────────

COHERENT_K = int(os.environ.get('COHERENT_K', '4'))              # candidatos por ping
COHERENT_TOL_M = float(os.environ.get('COHERENT_TOL_M', '400'))  # descarta el candidato
                                                # que esté a más de esto del mejor
_M_LAT, _M_LON = 110_540.0, 111_320.0    # m/grado — los mismos de _project_many_*
_BIG = 1e9                               # candidato inexistente. No un np.inf: entra
                                         # en las sumas del DP, y inf-inf es NaN


def _xy_local_flat(lats, lons, lat0: float) -> np.ndarray:
    """(lat, lon) → metros. Equirectangular local con un solo ``cos(lat0)``: a escala
    de ciudad la diferencia contra el ``cos`` por segmento es de centímetros."""
    cos0 = math.cos(math.radians(lat0))
    return np.column_stack([np.asarray(lons, dtype=float) * _M_LON * cos0,
                            np.asarray(lats, dtype=float) * _M_LAT])


def _coords_at_positions(pos: np.ndarray, arr: np.ndarray, cum: np.ndarray
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """Posición lineal (m) sobre la polilínea → (snap_lat, snap_lon)."""
    j = np.clip(np.searchsorted(cum, pos, side='right') - 1, 0, len(arr) - 2)
    t = ((pos - cum[j]) / np.maximum(cum[j + 1] - cum[j], 1e-9))[:, None]
    out = arr[j] + t * (arr[j + 1] - arr[j])
    return out[:, 0], out[:, 1]


def _polyline_candidates(plats: np.ndarray, plons: np.ndarray, arr: np.ndarray,
                         cum: np.ndarray, lat0: float, k: int = 4,
                         tol_m: float = 400.0, block: int = 1200
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """Hasta ``k`` posiciones candidatas por punto: los **mínimos locales** de la
    distancia a la polilínea, no sólo el global.

    → ``(cand_pos, cand_perp)``, ambos ``(N, k)`` y ordenados por distancia
    creciente, así que la columna 0 **es** el mínimo global. Los huecos —menos de
    ``k`` mínimos locales, o un candidato a más de ``tol_m`` del mejor— van a
    ``_BIG``.

    Va por bloques de filas porque la matriz intermedia es ``(bloque, M-1)``: con
    M ~ 900 vértices, 1,200 filas son ~10 MB por temporal.
    """
    V = _xy_local_flat(arr[:, 0], arr[:, 1], lat0)
    Pt = _xy_local_flat(plats, plons, lat0)
    A, S = V[:-1], V[1:] - V[:-1]
    l2 = S[:, 0] ** 2 + S[:, 1] ** 2
    l2 = np.where(l2 < 1e-12, np.inf, l2)          # segmento degenerado: nunca gana
    dcum = cum[1:] - cum[:-1]
    n = Pt.shape[0]
    cand_pos = np.zeros((n, k)); cand_perp = np.full((n, k), _BIG)
    for s in range(0, n, block):
        e = min(n, s + block)
        wx = Pt[s:e, 0, None] - A[None, :, 0]
        wy = Pt[s:e, 1, None] - A[None, :, 1]
        t = np.clip((wx * S[None, :, 0] + wy * S[None, :, 1]) / l2[None, :], 0.0, 1.0)
        dx = wx - t * S[None, :, 0]; dy = wy - t * S[None, :, 1]
        d = np.sqrt(dx * dx + dy * dy)
        pos = cum[:-1][None, :] + t * dcum[None, :]
        dl = np.concatenate([d[:, :1], d[:, :-1]], axis=1)
        dr = np.concatenate([d[:, 1:], d[:, -1:]], axis=1)
        dm = np.where((d <= dl) & (d <= dr), d, _BIG)          # mínimos locales
        kk = min(k, dm.shape[1])
        idx = np.argpartition(dm, kth=kk - 1, axis=1)[:, :kk]
        r = np.arange(e - s)[:, None]
        dsel, psel = dm[r, idx], pos[r, idx]
        o = np.argsort(dsel, axis=1)
        dsel = np.take_along_axis(dsel, o, 1); psel = np.take_along_axis(psel, o, 1)
        dsel = np.where(dsel > dsel[:, :1] + tol_m, _BIG, dsel)
        cand_perp[s:e, :kk], cand_pos[s:e, :kk] = dsel, psel
    return cand_pos, cand_perp


def _project_many_coherent(plats: np.ndarray, plons: np.ndarray, groups: np.ndarray,
                           coords, cum: np.ndarray, L: Optional[float] = None,
                           order: Optional[np.ndarray] = None,
                           k: Optional[int] = None, tol_m: Optional[float] = None,
                           block: int = 1200, direccional: float = 0.0
                           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                      np.ndarray, np.ndarray]:
    """Proyección coherente en el tiempo. Sustituye a ``_project_many_to_polyline``
    allí donde los puntos son una **secuencia** (los pings de un viaje).

    ``direccional`` (por defecto 0: lo de siempre). La transición cuesta
    | |avance| − distancia recorrida |: retroceder 700 m sobre la traza cuesta lo mismo que
    avanzarlos. En una calle que la traza recorre de ida y de vuelta, el bus puede quedar
    entonces en la otra pasada, yendo hacia atrás, toda una media vuelta (20000, de ida y vuelta:
    7 % de vueltas dobles en marzo). Con un peso w > 0, retroceder d cuesta además w·d.
    - Una media vuelta en la pasada equivocada suma w × varios km, y cambiar de pasada una vez
      cuesta menos.
    - Un tramo corto que los buses de verdad recorren al revés del dibujo (801, km 16–20) suma
      poco, y no se convierte en saltos.

    Marzo, con el corte por cruce, w = 1:
    - 20000: de 41 vueltas dobles a 1, y de 77.7 a 86.8 % de cobertura;
    - 902: de 25 vueltas dobles a 2;
    - 801: 113 tramos partidos (111 sin peso).

    Una primera versión que cobraba al retroceso d + la distancia recorrida (el doble)
    partía la 801 en 230 tramos. Lo usa el corte por cruce de km
    (``_lib_new_trips._build_trips_cruce``, ``CRUCE_DIRECCIONAL``).

    ``groups``  etiqueta de secuencia por punto (``trip_id``). Las filas NO tienen
                que venir agrupadas ni ordenadas: se ordena aquí y se devuelve en el
                orden de entrada.
    ``order``   criterio de orden dentro del grupo (``time``). None = respeta el
                orden de las filas.
    ``L``       largo del lazo, para la distancia circular. Default ``cum[-1]``.

    → ``(pos_m, perp_m, snap_lat, snap_lon, perp_min_m)``, cada uno ``(N,)``.

    ⚠ Devuelve **dos** perpendiculares y no son intercambiables:

    - ``perp_m``     distancia al punto ELEGIDO — la que corresponde a ``pos_m``.
    - ``perp_min_m`` distancia al mínimo global — la de siempre. No es idéntica a la
      de ``_project_many_to_polyline`` porque aquélla usa un ``cos`` por segmento y
      ésta uno solo, en ``lat0``: medido en la 601, **2.2 cm** de diferencia máxima
      y **ningún** flag ``off`` que cambie (``func/test_coherent_projection.py``).

    Para «¿está este ping sobre la ruta?» (``off``, depósito) usa ``perp_min_m``: es
    una pregunta sobre la geometría y no sobre la secuencia, y así ese flag no se
    mueve respecto del régimen viejo. Para «¿dónde va el bus?» (``pos``, ``bin``,
    ``lin_pos``, cobertura) usa ``pos_m``.
    """
    k = COHERENT_K if k is None else int(k)
    tol_m = COHERENT_TOL_M if tol_m is None else float(tol_m)
    arr = np.asarray(coords, dtype=float)
    cum = np.asarray(cum, dtype=float)
    plats = np.asarray(plats, dtype=float); plons = np.asarray(plons, dtype=float)
    n = plats.shape[0]
    if n == 0:
        z = np.zeros(0)
        return z, z, z, z, z
    L = float(cum[-1]) if L is None else float(L)
    lat0 = float(arr[:, 0].mean())
    cand_pos, cand_perp = _polyline_candidates(plats, plons, arr, cum, lat0,
                                               k=k, tol_m=tol_m, block=block)
    k = cand_pos.shape[1]

    # orden: grupos contiguos y, dentro de cada grupo, el criterio pedido
    codes = np.unique(np.asarray(groups), return_inverse=True)[1]
    sec = np.arange(n) if order is None else np.asarray(order)
    idx = np.lexsort((sec, codes))
    g = codes[idx]
    cx, cp = cand_pos[idx], cand_perp[idx]
    xy = _xy_local_flat(plats, plons, lat0)[idx]

    ini = np.flatnonzero(np.concatenate([[True], g[1:] != g[:-1]]))
    largo = np.diff(np.append(ini, n))
    dp = cp[ini].copy()                          # (G, k) coste acumulado
    atras = np.zeros((n, k), dtype=np.int8)      # backpointer por fila

    # Viterbi vectorizado SOBRE LOS GRUPOS: un paso de numpy por posición dentro del
    # viaje (cientos) en vez de uno por ping (millones).
    for j in range(1, int(largo.max())):
        act = np.flatnonzero(largo > j)
        fil = ini[act] + j; pre = fil - 1
        dpos = cx[fil][:, :, None] - cx[pre][:, None, :]
        dpos = (dpos + L / 2) % L - L / 2                      # avance circular con signo
        atras_m = np.maximum(-dpos, 0.0) if direccional else 0.0
        dpos = np.abs(dpos)                                    # distancia circular
        paso = np.hypot(xy[fil, 0] - xy[pre, 0], xy[fil, 1] - xy[pre, 1])
        tot = (dp[act][:, None, :] + np.abs(dpos - paso[:, None, None])
               + cp[fil][:, :, None] + float(direccional) * atras_m)
        bp = np.argmin(tot, axis=2)
        atras[fil] = bp.astype(np.int8)
        dp[act] = np.take_along_axis(tot, bp[:, :, None], axis=2)[:, :, 0]

    # backtracking, también por posición: cada grupo entra en su propio último paso
    elegido = np.zeros(n, dtype=np.int64)
    c = np.argmin(dp, axis=1)
    for j in range(int(largo.max()) - 1, -1, -1):
        act = np.flatnonzero(largo > j)
        fil = ini[act] + j
        elegido[fil] = c[act]
        c[act] = atras[fil, c[act]]

    sel = np.empty(n, dtype=np.int64); sel[idx] = elegido
    r = np.arange(n)
    pos = cand_pos[r, sel]; perp = cand_perp[r, sel]
    slat, slon = _coords_at_positions(pos, arr, cum)
    return pos, perp, slat, slon, cand_perp[:, 0]


# ──────────────────────────────────────────────────────────────────────
# Proyección métrica local (reemplaza pyproj/UTM) + DBSCAN propio
# ──────────────────────────────────────────────────────────────────────

def _local_xy(lats, lons, lat0, lon0) -> np.ndarray:
    cos0 = math.cos(math.radians(lat0))
    x = (lons - lon0) * (math.pi / 180.0) * 6_371_000.0 * cos0
    y = (lats - lat0) * (math.pi / 180.0) * 6_371_000.0
    return np.column_stack([x, y])


def _dbscan(xy: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """
    DBSCAN sobre puntos 2D en metros, con **búsqueda de vecinos por grid
    espacial** (celda = eps). Coste ~O(n) en densidad uniforme y memoria O(n),
    en vez de la matriz de distancias O(n²) — necesario para escalar de la
    semana de prueba (miles de eventos) a ene–ago completo (decenas de miles)
    sin agotar memoria. Devuelve labels (−1 = ruido).
    """
    n = xy.shape[0]
    labels = np.full(n, -1, dtype=int)
    if n == 0:
        return labels

    # Hash de cada punto a una celda de lado eps. Los puntos a <= eps están
    # garantizados dentro de la vecindad 3×3 de celdas.
    cells = np.floor(xy / eps).astype(np.int64)
    grid: Dict[Tuple[int, int], List[int]] = {}
    for i in range(n):
        grid.setdefault((int(cells[i, 0]), int(cells[i, 1])), []).append(i)

    eps2 = eps * eps

    def region_query(i: int) -> List[int]:
        cx, cy = int(cells[i, 0]), int(cells[i, 1])
        out = []
        xi, yi = xy[i, 0], xy[i, 1]
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in grid.get((cx + dx, cy + dy), ()):
                    ddx = xy[j, 0] - xi; ddy = xy[j, 1] - yi
                    if ddx * ddx + ddy * ddy <= eps2:
                        out.append(j)
        return out

    cid = -1
    visited = np.zeros(n, dtype=bool)
    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        nbrs = region_query(i)
        if len(nbrs) < min_samples:
            continue  # ruido (puede re-etiquetarse como borde)
        cid += 1
        labels[i] = cid
        seeds = list(nbrs); k = 0
        in_seeds = set(nbrs)
        while k < len(seeds):
            j = seeds[k]; k += 1
            if not visited[j]:
                visited[j] = True
                jn = region_query(j)
                if len(jn) >= min_samples:
                    for nb in jn:
                        if nb not in in_seeds:
                            in_seeds.add(nb); seeds.append(nb)
            if labels[j] == -1:
                labels[j] = cid
    return labels


def _runs(sign: np.ndarray) -> List[Tuple[int, int, int]]:
    """Run-length encoding de un array de ±1 (ignora 0). Devuelve (start,end,val)."""
    runs = []
    n = len(sign); i = 0
    while i < n:
        if sign[i] == 0:
            i += 1; continue
        j = i + 1
        while j < n and (sign[j] == sign[i] or sign[j] == 0):
            j += 1
        runs.append((i, j, int(sign[i])))
        i = j
    return runs


# ──────────────────────────────────────────────────────────────────────
# PASO 1 · Stay-points por trayectoria (Li & Zheng 2008) — sólo pos+tiempo
# ──────────────────────────────────────────────────────────────────────

def _stay_points_one_bus(lat, lon, t_s, D_max, T_min_s) -> List[Dict[str, Any]]:
    n = len(lat); out = []; i = 0
    while i < n:
        j = i + 1
        while j < n and _haversine_m(lat[i], lon[i], lat[j], lon[j]) <= D_max:
            j += 1
        dwell = t_s[j - 1] - t_s[i]
        if dwell >= T_min_s:
            out.append({"lat": float(lat[i:j].mean()), "lon": float(lon[i:j].mean()),
                        "t_arr_s": float(t_s[i]), "t_dep_s": float(t_s[j - 1]),
                        "dwell_min": float(dwell / 60.0), "n_pts": int(j - i)})
            i = j
        else:
            i += 1
    return out


def detect_stay_points(df_dir, *, stay_radius_m=60.0, stay_min_min=1.0, verbose=False):
    """Stay-points por bus (sólo posición + tiempo, sin ``direction``)."""
    import polars as pl
    df = df_dir.select(["bus_code", "time", "latitude", "longitude"]).with_columns([
        pl.col("latitude").cast(pl.Float64), pl.col("longitude").cast(pl.Float64),
    ]).sort(["bus_code", "time"])
    rows = []; T_min_s = stay_min_min * 60.0
    for (bus,), sub in df.group_by(["bus_code"], maintain_order=True):
        lat = sub["latitude"].to_numpy(); lon = sub["longitude"].to_numpy()
        t_s = sub["time"].dt.timestamp("ms").to_numpy() / 1000.0
        for sp in _stay_points_one_bus(lat, lon, t_s, stay_radius_m, T_min_s):
            sp["bus_code"] = bus; rows.append(sp)
    sps = pl.DataFrame(rows) if rows else pl.DataFrame(
        schema={"lat": pl.Float64, "lon": pl.Float64, "t_arr_s": pl.Float64,
                "t_dep_s": pl.Float64, "dwell_min": pl.Float64, "n_pts": pl.Int64,
                "bus_code": pl.String})
    if sps.height:
        sps = sps.with_columns(
            (pl.col("t_arr_s") * 1000).cast(pl.Datetime("ms")).dt.date().alias("date"))
    if verbose:
        print(f"  [stay-points] {sps.height} dwells >= {stay_min_min} min "
              f"(radio {stay_radius_m} m) sobre {df['bus_code'].n_unique()} buses")
    return sps


# ──────────────────────────────────────────────────────────────────────
# Detección por REVERSIÓN DE AVANCE — señal RAW (posición sobre la polilínea)
# ──────────────────────────────────────────────────────────────────────

def detect_turnarounds(
    df_dir,
    polyline_coords: List[Tuple[float, float]],
    *,
    source: str = "bearing",        # "bearing" (RAW, recomendado) | "direction" (circular, cross-check)
    win: int = 4,
    min_disp_m: float = 60.0,
    rev_thr: float = 0.3,
    min_run: int = 3,
    eps_m: float = 150.0,
    min_samples: int = 3,
    min_buses: int = 3,
    min_days: int = 2,
    onroute_max_perp_m: float = 80.0,
    dominance_frac: float = 0.2,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Detecta terminales como **puntos de reversión del movimiento** del bus
    (vueltas en U ~180°), recurrentes y on-route.

    source="bearing" (default): señal RAW e independiente de terminales. Usa
    sólo lat/lon crudos. En cada ping compara el vector de desplazamiento neto de
    la ventana ANTERIOR (p[i]-p[i-win]) con la POSTERIOR (p[i+win]-p[i]); si son
    casi anti-paralelos (rev = -cos θ >= rev_thr) hay vuelta en U. La ventana de
    ``win`` pings da robustez al jitter por-ping. (No se usa la polilínea de 40 km
    para esto: esa polilínea codifica el viaje redondo como monótono y NO tiene
    reversión — sólo sirve para medir perp/on-route y para snapear el evento.)

    source="direction": usa la columna ``direction`` (⚠ CIRCULAR — derivada de la
    definición de terminales). Sólo para auditar/comparar, nunca para el conteo.

    Devuelve dict con:
      - "n_terminals" : nº de clusters que pasan (recurrente + on-route + dominante).
      - "clusters"    : pl.DataFrame con TODOS los clusters candidatos + flags.
      - "events"      : pl.DataFrame con cada evento de reversión (para mapear).
      - "source"      : la señal usada.
    """
    import polars as pl

    needed = ["bus_code", "time", "latitude", "longitude"]
    if source == "direction":
        needed += ["direction", "snap_lat", "snap_lon"]
    df = df_dir.select(needed).with_columns([
        pl.col("latitude").cast(pl.Float64), pl.col("longitude").cast(pl.Float64),
    ])

    cum = _cumulative_distances(polyline_coords)
    L = float(cum[-1])

    # snap de TODOS los pings sobre la polilínea (para perp + ubicar el evento on-route)
    _pos, _perp, slat_all, slon_all = _project_many_to_polyline(
        df["latitude"].to_numpy(), df["longitude"].to_numpy(), polyline_coords, cum)
    df = df.with_columns([pl.Series("snap_lat_p", slat_all), pl.Series("snap_lon_p", slon_all)])
    if source == "direction":
        df = df.with_columns(pl.col("direction").cast(pl.Int64))

    df = df.sort(["bus_code", "time"])
    lat0 = float(df["latitude"].mean()); lon0 = float(df["longitude"].mean())

    events = []
    for (bus,), sub in df.group_by(["bus_code"], maintain_order=True):
        # Clustering sobre coords CRUDAS (la vuelta en U ocurre en un lugar físico
        # concreto; el snap a la polilínea de 40 km distorsiona cerca del wrap A).
        rlat = sub["latitude"].to_numpy(); rlon = sub["longitude"].to_numpy()
        dates = sub["time"].dt.date().to_numpy()
        t_ms = sub["time"].dt.timestamp("ms").to_numpy()   # epoch ms (para segmentar viajes)

        if source == "bearing":
            xy = _local_xy(rlat, rlon, lat0, lon0)
            x, y = xy[:, 0], xy[:, 1]
            n = len(x)
            rev = np.full(n, -np.inf)
            for i in range(win, n - win):
                vin_x = x[i] - x[i - win]; vin_y = y[i] - y[i - win]
                vout_x = x[i + win] - x[i]; vout_y = y[i + win] - y[i]
                nin = math.hypot(vin_x, vin_y); nout = math.hypot(vout_x, vout_y)
                if nin < min_disp_m or nout < min_disp_m:
                    continue
                rev[i] = -(vin_x * vout_x + vin_y * vout_y) / (nin * nout)
            # eventos = picos: runs contiguos con rev>=thr → argmax de cada run
            above = rev >= rev_thr
            i = 0
            while i < n:
                if not above[i]:
                    i += 1; continue
                j = i + 1
                while j < n and above[j]:
                    j += 1
                idx = i + int(np.argmax(rev[i:j]))
                events.append({"bus_code": bus, "lat": float(rlat[idx]),
                               "lon": float(rlon[idx]), "date": str(dates[idx]),
                               "time_ms": int(t_ms[idx])})
                i = j

        else:  # direction → ±1, reversión por run-length
            dval = sub["direction"].to_numpy()
            sign = np.where(dval == 0, 1, -1).astype(int)
            runs = _runs(sign)
            for k in range(1, len(runs)):
                prev_s, prev_e, prev_d = runs[k - 1]
                cur_s, cur_e, cur_d = runs[k]
                if prev_d != cur_d and (prev_e - prev_s) >= min_run and (cur_e - cur_s) >= min_run:
                    idx = prev_e - 1
                    events.append({"bus_code": bus, "lat": float(rlat[idx]),
                                   "lon": float(rlon[idx]), "date": str(dates[idx]),
                                   "time_ms": int(t_ms[idx])})

    if not events:
        return {"n_terminals": 0, "clusters": pl.DataFrame(), "events": pl.DataFrame(),
                "source": source}

    ev = pl.DataFrame(events)
    lats = ev["lat"].to_numpy(); lons = ev["lon"].to_numpy()
    labels = _dbscan(_local_xy(lats, lons, float(lats.mean()), float(lons.mean())),
                     eps=eps_m, min_samples=min_samples)
    ev = ev.with_columns(pl.Series("cluster", labels))

    max_rev = max((int((labels == cid).sum()) for cid in set(labels) if cid != -1), default=0)
    rev_floor = dominance_frac * max_rev
    rows = []
    for cid in sorted(set(labels)):
        if cid == -1:
            continue
        c = ev.filter(pl.col("cluster") == cid)
        clat = float(c["lat"].mean()); clon = float(c["lon"].mean())
        _, _, perp, pos_m = _project_to_polyline(clat, clon, polyline_coords, cum)
        n_buses = c["bus_code"].n_unique(); n_days = c["date"].n_unique()
        recurrent = (n_buses >= min_buses) and (n_days >= min_days)
        on_route = perp <= onroute_max_perp_m
        dominant = c.height >= rev_floor
        rows.append({"cluster": int(cid), "n_reversals": c.height,
                     "lat": round(clat, 6), "lon": round(clon, 6),
                     "n_buses": int(n_buses), "n_days": int(n_days),
                     "perp_m": round(float(perp), 1),
                     "lin_pos": round(pos_m / L, 3) if L else float("nan"),
                     "recurrent": recurrent, "on_route": on_route, "dominant": dominant,
                     "is_terminal": bool(recurrent and on_route and dominant)})
    if not rows:
        # DBSCAN marcó todos los eventos como ruido (ningún cluster sobrevivió):
        # no hay terminales por turnaround. Sin este guard, pl.DataFrame([]).sort(...)
        # revienta con ColumnNotFoundError. (Detectado en la 901.)
        return {"n_terminals": 0, "clusters": pl.DataFrame(), "events": ev,
                "source": source}
    clusters = pl.DataFrame(rows).sort("n_reversals", descending=True)
    n_term = int(clusters.filter(pl.col("is_terminal")).height)
    if verbose:
        print(f"  [turnaround/{source}] {ev.height} eventos → {clusters.height} clusters "
              f"candidatos → {n_term} terminales (recurrente+on-route+dominante)")
    return {"n_terminals": n_term, "clusters": clusters, "events": ev, "source": source}


def consolidate_terminal_zones(clusters, polyline_length_m: float,
                               zone_merge_m: float = 3000.0) -> List[Dict[str, Any]]:
    """
    Agrupa los clusters ``is_terminal`` en **zonas terminales** por proximidad a
    lo largo de la RUTA (posición lineal con wrap del lazo), no por distancia
    euclidiana. Un terminal complejo (p.ej. un circuito de varias vueltas cerca
    de la costura del lazo) produce varios clusters físicamente separados pero
    contiguos sobre la ruta; esta consolidación los une en una sola zona.

    Devuelve lista de zonas: {n_clusters, n_reversals, lin_pos_min/max, clusters[]}.
    Single-linkage con umbral ``zone_merge_m`` sobre la distancia circular en
    metros (= lin_pos × L, con wrap en L).
    """
    import polars as pl
    if clusters is None or clusters.height == 0:
        return []
    terms = clusters.filter(pl.col("is_terminal")).sort("lin_pos").to_dicts()
    if not terms:
        return []
    L = polyline_length_m
    pos = [t["lin_pos"] * L for t in terms]

    # unión por single-linkage circular
    parents = list(range(len(terms)))
    def find(a):
        while parents[a] != a:
            parents[a] = parents[parents[a]]; a = parents[a]
        return a
    def union(a, b):
        parents[find(a)] = find(b)
    for a in range(len(terms)):
        for b in range(a + 1, len(terms)):
            d = abs(pos[a] - pos[b]); d = min(d, L - d)   # circular
            if d <= zone_merge_m:
                union(a, b)

    zones: Dict[int, List[int]] = {}
    for i in range(len(terms)):
        zones.setdefault(find(i), []).append(i)

    out = []
    for members in zones.values():
        ms = [terms[i] for i in members]
        out.append({
            "n_clusters": len(ms),
            "n_reversals": sum(m["n_reversals"] for m in ms),
            "lin_pos_min": round(min(m["lin_pos"] for m in ms), 3),
            "lin_pos_max": round(max(m["lin_pos"] for m in ms), 3),
            "lat": round(sum(m["lat"] for m in ms) / len(ms), 6),
            "lon": round(sum(m["lon"] for m in ms) / len(ms), 6),
            "clusters": [m["cluster"] for m in ms],
        })
    out.sort(key=lambda z: z["n_reversals"], reverse=True)
    return out


def _densest_event_point(zlat, zlon, ev_lat, ev_lon,
                         *, search_m: float = 2000.0, cell_m: float = 60.0):
    """
    Punto representativo de una zona = **el lugar más denso de eventos** cercano
    a su centroide (el hotspot del heatmap), no la media de coordenadas.

    La media de puntos dispersos (p.ej. los 3 sub-clusters de la terminal A en la
    costura del lazo, o el mega-cluster del agregado) cae *entre* ellos —
    típicamente fuera de la traza. Aquí, en cambio, se busca el evento modal:
    se toman los eventos dentro de ``search_m`` del centroide, se rejilla a
    ``cell_m`` y se devuelve la media de la celda con más eventos. Así el marcador
    cae siempre sobre la concentración real.
    """
    import numpy as np
    if ev_lat is None or len(ev_lat) == 0:
        return zlat, zlon
    dlat = search_m / 111_000.0
    dlon = search_m / (111_000.0 * max(0.1, math.cos(math.radians(zlat))))
    mask = (np.abs(ev_lat - zlat) <= dlat) & (np.abs(ev_lon - zlon) <= dlon)
    nlat, nlon = ev_lat[mask], ev_lon[mask]
    if nlat.size == 0:
        return zlat, zlon
    xy = _local_xy(nlat, nlon, zlat, zlon)
    cells = np.round(xy / cell_m).astype(np.int64)
    keys = cells[:, 0] * 1_000_000 + cells[:, 1]
    uniq, inv, counts = np.unique(keys, return_inverse=True, return_counts=True)
    sel = inv == int(counts.argmax())
    return float(nlat[sel].mean()), float(nlon[sel].mean())


def _assign_zone_markers(zones, events_df, search_m: float = 2000.0):
    """Añade ``mlat``/``mlon`` (hotspot de eventos) a cada zona, in-place."""
    import numpy as np
    ev_lat = events_df["lat"].to_numpy() if events_df is not None and events_df.height else np.array([])
    ev_lon = events_df["lon"].to_numpy() if events_df is not None and events_df.height else np.array([])
    for z in zones:
        z["mlat"], z["mlon"] = _densest_event_point(z["lat"], z["lon"], ev_lat, ev_lon,
                                                     search_m=search_m)
    return zones


# ──────────────────────────────────────────────────────────────────────
# Descubrimiento de zonas ROBUSTO A LA ESCALA (directo vs consenso por período)
# ──────────────────────────────────────────────────────────────────────

def discover_terminal_zones(
    df_dir,
    polyline_coords: List[Tuple[float, float]],
    *,
    span_days_threshold: int = 45,
    period_fmt: str = "%Y-%m",
    min_period_frac: float = 0.5,
    consensus_eps_m: float = 800.0,
    zone_merge_m: float = 3000.0,
    period_min_days: int = 2,
    verbose: bool = False,
    **turnaround_kw,
) -> Dict[str, Any]:
    """
    Descubre zonas terminales de forma **robusta a la escala temporal**.

    Problema: ``detect_turnarounds`` directo sobre muchos meses sobre-fusiona —
    el piso de ruido de reversiones (acumulado del map-match) hace que DBSCAN
    encadene casi todos los eventos en un mega-cluster cuyo centroide cae
    off-route, dando N=0. Por período corto la densidad es manejable.

    Estrategia:
      - **span ≤ ``span_days_threshold``**: detección directa
        (``detect_turnarounds`` + ``consolidate_terminal_zones``).
      - **span > ``span_days_threshold``**: por período (``period_fmt``, mes por
        defecto) + **consenso**: una zona es terminal si aparece en
        ≥ ``min_period_frac`` de los períodos, agrupando los centroides por
        cercanía física (``consensus_eps_m``) — robusto al salto de etiqueta
        lin_pos en la costura del lazo.

    Devuelve dict con: ``mode`` ('direct'|'consensus'), ``N``, ``zones`` (lista
    uniforme {lat, lon, n_rev, n_periods, periods}), ``events`` (para mapear),
    ``candidates`` (clusters crudos), ``per_period`` (centroides mensuales) y
    ``span_days``.
    """
    import polars as pl

    cum = _cumulative_distances(polyline_coords)
    Ltot = float(cum[-1])
    span = (df_dir["time"].max() - df_dir["time"].min()).days

    # Eventos globales (siempre, para el mapa). Sobre datos largos esto
    # sobre-fusiona, pero los EVENTOS en sí sirven para visualizar densidad.
    br = detect_turnarounds(df_dir, polyline_coords, **turnaround_kw)

    if span <= span_days_threshold:
        zr = consolidate_terminal_zones(br["clusters"], Ltot, zone_merge_m=zone_merge_m)
        zones = [{"lat": z["lat"], "lon": z["lon"], "n_rev": z["n_reversals"],
                  "n_periods": 1, "periods": ["(directo)"]} for z in zr]
        _assign_zone_markers(zones, br["events"])
        if verbose:
            print(f"  [discover] modo DIRECTO (span={span}d ≤ {span_days_threshold}) "
                  f"→ N={len(zones)}")
        return {"mode": "direct", "N": len(zones), "zones": zones,
                "events": br["events"], "candidates": br["clusters"],
                "per_period": [], "span_days": span}

    # ── Consenso por período ──
    dfp = df_dir.with_columns(pl.col("time").dt.strftime(period_fmt).alias("__p"))
    periods = sorted(dfp["__p"].unique().to_list())
    kw_p = {**turnaround_kw}
    kw_p.setdefault("min_days", period_min_days)
    mz: List[Dict[str, Any]] = []
    for pm in periods:
        sub = dfp.filter(pl.col("__p") == pm).drop("__p")
        zs = consolidate_terminal_zones(
            detect_turnarounds(sub, polyline_coords, **kw_p)["clusters"],
            Ltot, zone_merge_m=zone_merge_m)
        for z in zs:
            mz.append({"lat": z["lat"], "lon": z["lon"], "period": pm,
                       "n_rev": z["n_reversals"]})

    zones: List[Dict[str, Any]] = []
    if mz:
        import numpy as np
        la = np.array([m["lat"] for m in mz]); lo = np.array([m["lon"] for m in mz])
        lab = _dbscan(_local_xy(la, lo, float(la.mean()), float(lo.mean())),
                      eps=consensus_eps_m, min_samples=1)
        min_periods = max(1, int(math.ceil(min_period_frac * len(periods))))
        for cid in sorted(set(lab)):
            mem = [mz[i] for i in range(len(lab)) if lab[i] == cid]
            nper = len(set(m["period"] for m in mem))
            if nper >= min_periods:
                zones.append({
                    "lat": float(np.mean([m["lat"] for m in mem])),
                    "lon": float(np.mean([m["lon"] for m in mem])),
                    "n_rev": int(sum(m["n_rev"] for m in mem)),
                    "n_periods": nper, "periods": sorted(set(m["period"] for m in mem))})
        zones.sort(key=lambda z: z["n_rev"], reverse=True)

    _assign_zone_markers(zones, br["events"])
    if verbose:
        mp = max(1, int(math.ceil(min_period_frac * len(periods))))
        print(f"  [discover] modo CONSENSO (span={span}d, {len(periods)} períodos, "
              f"≥{mp} para terminal) → N={len(zones)}")
    return {"mode": "consensus", "N": len(zones), "zones": zones,
            "events": br["events"], "candidates": br["clusters"],
            "per_period": mz, "span_days": span}


# ──────────────────────────────────────────────────────────────────────
# PASOS 2-3 · Descubrimiento por LAYOVER (clustering + filtro) → terminales de descanso
# ──────────────────────────────────────────────────────────────────────

def discover_terminals(
    df_dir,
    polyline_coords: List[Tuple[float, float]],
    *,
    layover_min_min: float = 15.0,
    stay_radius_m: float = 60.0,
    stay_min_min: float = 1.0,
    eps_m: float = 80.0,
    min_samples: int = 3,
    min_buses: int = 3,
    min_days: int = 2,
    onroute_max_perp_m: float = 50.0,
    dominance_frac: float = 0.2,
    turnaround_coords: Optional[List[Tuple[float, float]]] = None,
    turnaround_match_m: float = 200.0,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Descubre terminales **de layover** (dwell largo recurrente on-route).

    Clasificación de cada cluster (sin usar ``direction``):
      - no recurrente            → ``noise``
      - no on-route              → ``depot``
      - on-route pero **no dominante** (``n_dwells < dominance_frac × máx``)
                                 → ``minor_dwell`` (parada de bajo volumen, no terminal)
      - on-route + dominante + coincide con un turnaround (si ``turnaround_coords``)
                                 → ``terminal``
      - on-route + dominante + NO coincide → ``relief_point`` (parada larga a media ruta)
      Si ``turnaround_coords`` es None, los on-route dominantes se marcan
      ``terminal_like`` (sin poder separar relief).

    El filtro de **dominancia** (``dominance_frac``, análogo al de
    ``detect_turnarounds``) evita que un cluster de bajo volumen se confunda con la
    terminal dominante: cada cluster trae ``n_dwells`` (volumen), ``dwell_share``
    (n_dwells / máx) y ``dominant`` (bool). Sube ``dominance_frac`` para ser más
    estricto; ``0.0`` lo desactiva (comportamiento previo).

    Devuelve dict con "N", "clusters", "stay_points", "long_dwells", listas por clase.
    """
    import polars as pl

    sps = detect_stay_points(df_dir, stay_radius_m=stay_radius_m,
                             stay_min_min=stay_min_min, verbose=verbose)
    long_dwells = sps.filter(pl.col("dwell_min") >= layover_min_min)
    if verbose:
        print(f"  [discovery] dwells largos (>= {layover_min_min} min): "
              f"{long_dwells.height} de {sps.height}")

    base = {"stay_points": sps, "long_dwells": long_dwells,
            "params": dict(layover_min_min=layover_min_min, stay_radius_m=stay_radius_m,
                           eps_m=eps_m, min_buses=min_buses, min_days=min_days,
                           onroute_max_perp_m=onroute_max_perp_m)}
    if long_dwells.height == 0:
        return {**base, "N": 0, "clusters": pl.DataFrame(),
                "terminals": [], "relief_points": [], "depots": []}

    lats = long_dwells["lat"].to_numpy(); lons = long_dwells["lon"].to_numpy()
    labels = _dbscan(_local_xy(lats, lons, float(lats.mean()), float(lons.mean())),
                     eps=eps_m, min_samples=min_samples)
    long_dwells = long_dwells.with_columns(pl.Series("cluster", labels))
    if verbose:
        print(f"  [discovery] DBSCAN: {labels.max() + 1 if labels.size else 0} clusters "
              f"(+{int((labels == -1).sum())} ruido) sobre {long_dwells.height} dwells largos")

    cum = _cumulative_distances(polyline_coords); total_len = float(cum[-1])

    def _near_turnaround(clat, clon):
        if not turnaround_coords:
            return None
        return any(_haversine_m(clat, clon, tlat, tlon) <= turnaround_match_m
                   for tlat, tlon in turnaround_coords)

    # Volumen máximo entre clusters → piso de dominancia (como en detect_turnarounds)
    _sizes = {cid: int((labels == cid).sum()) for cid in set(labels) if cid != -1}
    max_dwells = max(_sizes.values(), default=0)
    dwell_floor = dominance_frac * max_dwells

    rows = []
    for cid in sorted(set(labels)):
        if cid == -1:
            continue
        c = long_dwells.filter(pl.col("cluster") == cid)
        clat = float(c["lat"].mean()); clon = float(c["lon"].mean())
        _, _, perp, pos_m = _project_to_polyline(clat, clon, polyline_coords, cum)
        n_buses = c["bus_code"].n_unique(); n_days = c["date"].n_unique()
        recurrent = (n_buses >= min_buses) and (n_days >= min_days)
        on_route = perp <= onroute_max_perp_m
        dominant = c.height >= dwell_floor
        dwell_share = (c.height / max_dwells) if max_dwells else 0.0
        at_turn = _near_turnaround(clat, clon)
        if not recurrent:
            klass = "noise"
        elif not on_route:
            klass = "depot"
        elif not dominant:
            klass = "minor_dwell"     # recurrente y on-route pero bajo volumen
        elif at_turn is True:
            klass = "terminal"
        elif at_turn is False:
            klass = "relief_point"
        else:
            klass = "terminal_like"   # sin info de turnaround
        rows.append({"cluster": int(cid), "n_dwells": c.height,
                     "dwell_share": round(float(dwell_share), 3), "dominant": dominant,
                     "lat": round(clat, 6), "lon": round(clon, 6),
                     "n_buses": int(n_buses), "n_days": int(n_days),
                     "perp_m": round(float(perp), 1),
                     "lin_pos": round(pos_m / total_len, 3) if total_len else float("nan"),
                     "median_dwell_min": round(float(c["dwell_min"].median()), 1),
                     "recurrent": recurrent, "on_route": on_route,
                     "at_turnaround": at_turn, "class": klass})
    clusters = pl.DataFrame(rows).sort("n_dwells", descending=True)
    terms = clusters.filter(pl.col("class").is_in(["terminal", "terminal_like"])).to_dicts()
    relief = clusters.filter(pl.col("class") == "relief_point").to_dicts()
    depots = clusters.filter(pl.col("class") == "depot").to_dicts()
    n_minor = int(clusters.filter(pl.col("class") == "minor_dwell").height)
    if verbose:
        print(f"  [discovery] N(layover) = {len(terms)} | {len(relief)} relief | "
              f"{len(depots)} depósitos | {n_minor} minor_dwell (bajo volumen, "
              f"dominance_frac={dominance_frac})")
    return {**base, "N": len(terms), "clusters": clusters,
            "terminals": terms, "relief_points": relief, "depots": depots}
