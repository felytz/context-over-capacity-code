"""
_lib_geo.py
===========
Helpers geométricos compartidos del pipeline.

Consolida `_haversine_m`, que vivía duplicada en 8 sitios (mismo R = 6 371 000 m,
con variantes menores de nomenclatura/clamp): `_lib_dashboard`, `_lib_features`
(×3), `_lib_map_match`, `_lib_plot`, `_lib_turn_hex`, `audits/integrity_audit` y
`steps/step_03_stopped_records_audit`. Esta versión canónica unifica esas variantes
y añade dos salvaguardas que solo tenían algunas copias:

- **coerción `float()`** de las entradas — las columnas de paradas pueden venir como
  `decimal.Decimal` desde la BD (la copia de `step_03` lo hacía explícitamente);
- **clamp `max(0.0, a)`** antes de la raíz — evita `NaN` por error de punto flotante
  cuando los dos puntos coinciden y `a` queda en ~-1e-17 (las copias de
  `_lib_map_match` e `integrity_audit` no lo tenían).

Para todas las entradas válidas el resultado es numéricamente idéntico a las 8
copias previas (diferencias < 1e-9 m); las salvaguardas solo cambian el
comportamiento en los bordes degenerados (Decimal / puntos coincidentes).

Importable por nombre simple (`from _lib_geo import _haversine_m`) en cuanto
`func/lib/` esté en `sys.path` — lo que ya ocurre al importar `config`, igual que
el resto de los `_lib_*`.
"""

from __future__ import annotations

import ast
import json
import math
from typing import List, Tuple

# Radio terrestre medio (m). Valor canónico usado en todo el pipeline.
EARTH_RADIUS_M = 6_371_000.0


def _haversine_m(lat1, lon1, lat2, lon2) -> float:
    """Distancia haversine en metros entre dos puntos lat/lon (grados decimales).

    Acepta ``float`` / ``int`` / ``str`` numérico / ``decimal.Decimal`` (se coercen
    con ``float()``). Devuelve ``float``.
    """
    lat1 = float(lat1); lon1 = float(lon1)
    lat2 = float(lat2); lon2 = float(lon2)
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2)
    return EARTH_RADIUS_M * 2 * math.asin(math.sqrt(max(0.0, a)))


def _parse_polyline_struct(raw) -> List[Tuple[float, float]]:
    """Parsea el campo ``polyline`` (de df_segments / df_segments_route) a una
    lista de ``(lat, lon)``. Maneja los formatos que aparecen en el pipeline:

      - ``list[struct[{'latitude','longitude'}]]``  (lo más común al leer parquet)
      - ``list[{'lat','lon'}]`` / ``{'latitud','longitud'}``
      - ``list[[lat, lon]]`` / ``[(lat, lon)]``
      - ``str`` con JSON o repr de Python (``json.loads`` / ``ast.literal_eval``)

    Devuelve ``[]`` si no es parseable. (Versión robusta; consolida las copias que
    vivían en ``_lib_features`` ×2 y ``_lib_dashboard``.)
    """
    if raw is None:
        return []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            try:
                raw = ast.literal_eval(raw)
            except Exception:
                return []
    if not isinstance(raw, list):
        return []
    pts: List[Tuple[float, float]] = []
    for item in raw:
        if isinstance(item, dict):
            # is-not-None (no `or`): una coordenada 0.0 es válida, no falsy.
            la = _first_present(item, ("latitude", "lat", "latitud"))
            lo = _first_present(item, ("longitude", "lon", "longitud"))
            if la is not None and lo is not None:
                pts.append((float(la), float(lo)))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            pts.append((float(item[0]), float(item[1])))
    return pts


def _first_present(d: dict, keys):
    """Primer valor no-None entre ``keys`` (a diferencia de ``a or b``, no
    descarta 0.0/0 que son válidos como coordenada)."""
    for k in keys:
        v = d.get(k)
        if v is not None:
            return v
    return None


# Alias públicos (sin guion bajo) por si se prefieren en código nuevo.
haversine_m = _haversine_m
parse_polyline_struct = _parse_polyline_struct
