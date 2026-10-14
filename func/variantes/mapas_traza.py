#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
mapas_traza.py — ¿los buses siguen la traza? Un mapa por ruta, mes a mes, desde el crudo.

Compara los registros AVL crudos de cada ruta (todos los meses de la ventana) con su
polilínea (`data/01_raw/polylines.geojson`) en tramos de 100 m y deja, por ruta, un HTML
con el mapa, un selector de mes y la tira tramo × mes; además un índice de la red, tres
tablas y un .zip con todo. No usa el corte de viajes ni el `df_final`: sólo el crudo.

Dos cosas del crudo mandan en el método:
  - viene recortado a ~50 m de la traza (en 23 de las 27 rutas ningún registro pasa de
    48–68 m; sólo 101, 401, 501 y 601 tienen registros más lejos). Un bus que se sale de la
    traza no aparece lejos: **desaparece**. Por eso, además de medir distancias, se sigue a
    cada bus de un registro al siguiente;
  - de septiembre en adelante se vuelve ralo: el intervalo mediano entre registros de un bus
    pasa de 61 s (enero–agosto) a 2–17 min según la ruta. Los umbrales de tiempo se toman
    del intervalo de cada ruta-mes, y esos meses se marcan como «muestreo ralo» (también
    los mixtos, como septiembre, donde el tramo típico ya tiene ≥ 20 % de saltos).

Por tramo y mes:
  pasos           veces que un bus entra al tramo avanzando por la traza (registros
                  consecutivos a ≤ max(150 s, 2.5 × el intervalo del mes; tope 20 min), en
                  el sentido de la traza)
  huecos          veces que un bus en servicio cruza el tramo SIN que se le vea: entre un
                  registro y el siguiente pasa más que eso (hasta max(30 min, 4 × eso)) y el
                  avance sobre la traza da velocidad de bus (1.5–30 m/s). Concentrados en un
                  tramo: los buses van por otra calle (o la traza va por donde ya no pasan).
                  También es hueco un *rodeo*: un paso (dos registros seguidos a ≤ 150 s, a
                  velocidad de bus sobre la traza) en el que el avance pasa en > RODEO_M y > RODEO_X veces la
                  distancia real: la traza da una vuelta o entra a un ramal y sale por donde
                  entró, y el bus no se vio ahí (sin esto, pasar de largo contaba como paso
                  por toda la vuelta: el ramal de la 601 en el km 28.8–29.45 salía «ok»)
  salto           huecos / (pasos + huecos)
  desplazamiento  mediana de la distancia lateral con signo (+ a la izquierda del sentido de
                  la traza) de los registros en movimiento: la traza dibujada a un lado de
                  por donde pasan (una calle paralela a < 50 m, otro retorno)
  contrasentido   registros cerca de la traza (< 80 m) sin ningún punto de la traza cercano
                  en su sentido: el bus recorre esa calle al revés
  fuera           registros a > 80 m de toda la traza (sólo donde el recorte del crudo se
                  hizo con otra traza)

El rumbo de cada registro (`cape`) decide a qué paso de la traza pertenece cuando la traza
pasa dos veces por la misma calle (ida y vuelta): se empareja con el punto de la traza más
cercano cuyo rumbo difiera ≤ 60°. Si la traza pasa dos veces en el mismo sentido (otra pasada a
≤ 15 m más que la más cercana), decide la continuidad: la pasada que menos avanza desde el
último registro no ambiguo del mismo bus (docstring de `emparejar`).

Estado de cada tramo en cada mes (gana el de más abajo):
  desplazada      |desplazamiento| ≥ 12 m o ≥ 30 % de los registros a > 30 m (con ≥ 10)
  contrasentido   ≥ 30 % de los registros del tramo (con ≥ 10)
  poco recorrida  pasos < 25 % de la mediana de los tramos de la ruta en ese mes
  saltada         salto ≥ 0.5 y ≥ 0.3 sobre la mediana de saltos de los tramos ese mes,
                  con ≥ 5 cruces (pasos + huecos)
Una zona es un tramo seguido con problema en algún mes (≥ 200 m, o en ≥ 3 meses). Un mes
con más de la mitad de la traza con problema no localiza nada y no cuenta para las zonas.

Uso
---
    python mapas_traza.py                              # 27 rutas, todo el crudo (2024)
    python mapas_traza.py --rutas 601 501 --procesos 2
    python mapas_traza.py --desde 2024-01-01 --hasta 2024-08-31
    python mapas_traza.py --geometria "../../data raw/new" --salida reports/mapas_traza_2024-01-01_2024-12-31_hermosillo-tesis-corregido

En el servidor: `bash run_servidor.sh --mapas` (un job; README).

Otra geometría (`--geometria DIR`): la traza y las paradas salen de esa carpeta (con los nombres
de data/01_raw o con los del portal: polylines_decoded.geojson, stops_<fecha>.json,
stops_routes_<fecha>.json); el crudo sigue siendo el mismo. Cada ruta del crudo va con la
polilínea más parecida a la suya de data/01_raw, que es contra la que se recortó el crudo,
porque el número puede cambiar: en la geometría corregida del 5-oct-2026 la traza del crudo
101 es la 102 (y al revés) y la del 1601 es la 1602 (y al revés). `--traza 101=102` fuerza una.
La página dibuja la traza de antes punteada (encendida si se aparta > CAMBIO_M) y el índice
dice qué traza tomó cada ruta. Va a reports/mapas_traza_<desde>_<hasta>_<carpeta>.

Salida: reports/mapas_traza_<desde>_<hasta>/ (index.html, ruta_<n>.html, resumen_mensual.csv,
zonas.csv, tramos_por_mes.csv) y reports/mapas_traza_<desde>_<hasta>.zip con todo eso.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing
import os
import sys
import time
import traceback
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
from scipy.spatial import cKDTree

RAIZ = Path(__file__).resolve().parent
RAW = RAIZ / "data" / "01_raw"

# metros locales (la misma aproximación que analisis/)
LAT0 = 29.08
KX, KY = 111320.0 * math.cos(math.radians(LAT0)), 111320.0

PASO_M = 10.0               # la traza se muestrea cada 10 m
TRAMO_M = 100.0             # la unidad de los mapas y las tablas
VECINOS = 16                # puntos de la traza que se miran por registro
R_MAX_M = 80.0              # más lejos que esto de toda la traza: «fuera»
ANG_MAX = 60.0              # rumbo del registro contra el de la traza
PASADA_M = 300.0            # dos puntos de la traza a más que esto sobre la traza: otra pasada
AMBIGUO_M = 15.0            # otra pasada en el mismo sentido a ≤ esto más que la más cercana: ambiguo
CONTINUIDAD_S = 900         # para elegir la pasada sirve el registro anterior del bus hasta esto antes
LEJOS_M = 30.0
V_MOV_KMH = 3.0             # en movimiento (para el desplazamiento)
DT_PASO_S = 150            # dos registros seguidos: ≤ max(150 s, 2.5 × el intervalo del mes)
DT_PASO_MAX_S = 1200
V_PASO_MAX = 25.0           # m/s
AVANCE_MIN_M = 15.0         # menos que esto entre dos registros: detenido
DT_HUECO_S = 1800           # un hueco: hasta max(30 min, 4 × lo anterior), tope 1 h
HUECO_MIN_M = 200.0
V_HUECO = (1.5, 30.0)       # m/s
RODEO_M, RODEO_X = 250.0, 2.0   # rodeo: el avance sobre la traza pasa en > 250 m y > 2 veces lo que el bus se movió
SALTO_MIN, CRUCES_MIN = 0.5, 5
SALTO_SOBRE_BASE = 0.3      # y 0.3 por encima de la mediana de saltos de los tramos ese mes
POCOS_REGISTROS, POCOS_DIAS = 1000, 5      # menos que esto en un mes: no se evalúa
RALO_S = 120                # «muestreo ralo»: intervalo mediano por encima de esto, o un mes
RALO_SALTO = 0.2            #   mixto (septiembre) donde el tramo típico ya tiene ≥ 20 % de saltos
MES_GENERAL = 0.5           # más de esto de la traza con problema: el mes no cuenta para zonas
POCO_REL = 0.25
CONTRA_MIN, LAT_MIN_M, LEJOS_MIN, N_MIN = 0.3, 12.0, 0.3, 10
LOTE = 400_000              # registros por consulta al árbol (memoria acotada)
MUESTRA_MES = 2500          # registros por mes en el mapa
BUSES_MES = 3               # días de bus por mes para «un bus, paso a paso»…
COBERTURA_BUS = 0.9         # …entre los que recorren ≥ 90 % de la traza en la mayoría de sus
COBERTURA_MIN = 0.5         #   vueltas (o el p90 del mes, pero nunca menos de 50 %)
DIAS_SEM = ["lun", "mar", "mié", "jue", "vie", "sáb", "dom"]
MESES_ES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
ESTADOS = ["sin datos", "ok", "desplazada", "contrasentido", "poco recorrida", "saltada"]
CAMBIO_M = 15.0             # con otra geometría: la traza «cambió» si se aparta más que esto de la de antes


# ═════════════════════════════════════════════════════════════════════════════
#  geometría
# ═════════════════════════════════════════════════════════════════════════════
def archivos_geometria(d: Path) -> dict[str, Path]:
    """La traza y las paradas de una carpeta: con los nombres de data/01_raw o con los del
    portal (polylines_decoded.geojson o, si no está, el codificado polylines_<fecha>.json;
    stops_<fecha>.json, stops_routes_<fecha>.json)."""
    def uno(*patrones):
        for p in patrones:
            c = sorted(d.glob(p))
            if c:
                return c[-1]
        raise FileNotFoundError(f"no veo {patrones[0]} (ni {', '.join(patrones[1:])}) en {d}")
    return {"polylines": uno("polylines.geojson", "polylines_decoded.geojson", "polylines*.geojson", "polylines_2*.json"),
            "stops": uno("stops.json", "stops_2*.json"),
            "stops_routes": uno("stops_routes.json", "stops_routes_2*.json")}


def decodificar(cod: str) -> np.ndarray:
    """Polilínea codificada (formato de Google, 5 decimales) → [[lon, lat], …]."""
    out, lat, lon, i = [], 0, 0, 0
    while i < len(cod):
        dd = []
        for _ in range(2):
            r = sh = 0
            while True:
                b = ord(cod[i]) - 63
                i += 1
                r |= (b & 0x1F) << sh
                sh += 5
                if b < 0x20:
                    break
            dd.append(~(r >> 1) if r & 1 else r >> 1)
        lat, lon = lat + dd[0], lon + dd[1]
        out.append((lon / 1e5, lat / 1e5))
    return np.asarray(out, float)


def fuente_trazas(geom: Path) -> Path:
    """El archivo de donde salen las trazas: el codificado del portal (polylines_<fecha>.json)
    si está, porque es la fuente; si no, el geojson. El 5-oct el polylines_decoded.geojson de
    «Nueva carpeta» seguía siendo el de antes (sin las correcciones de 901, 1001 y 1002)."""
    cod = sorted(geom.glob("polylines_2*.json"))
    return cod[-1] if cod else archivos_geometria(geom)["polylines"]


def leer_polilineas(geom: Path) -> dict[str, np.ndarray]:
    f = fuente_trazas(geom)
    if f.suffix == ".geojson":
        return {str(g["properties"]["id"]): np.asarray(g["geometry"]["coordinates"], float)    # lon, lat
                for g in json.load(open(f, encoding="utf-8"))["features"]}
    return {k: decodificar(v["polyline"]) for grupo in json.load(open(f, encoding="utf-8")).values()
            for k, v in grupo.items()
            if isinstance(v, dict) and isinstance(v.get("polyline"), str) and not v["polyline"].startswith("[")}


def nombre_geom(geom: Path) -> str:
    """Cómo se cita la carpeta de la geometría: relativa al proyecto si se puede."""
    try:
        return geom.resolve().relative_to(RAIZ.parents[1]).as_posix()
    except ValueError:
        return geom.name


def nombres_lineas(geom: Path) -> dict[str, str]:
    """Los nombres de línea del archivo codificado del portal (polylines_<fecha>.json), si está."""
    out = {}
    for f in sorted(geom.glob("polylines_2*.json")):
        for grupo in json.load(open(f, encoding="utf-8")).values():
            out.update({k: v["name"] for k, v in grupo.items() if isinstance(v, dict) and v.get("name")})
    return out


def densa(p: np.ndarray, paso: float = PASO_M) -> np.ndarray:
    xy = np.c_[p[:, 0] * KX, p[:, 1] * KY]
    xy = xy[np.r_[True, np.hypot(*np.diff(xy, axis=0).T) > 0.01]]
    seg = np.hypot(*np.diff(xy, axis=0).T)
    cum = np.r_[0.0, np.cumsum(seg)]
    s = np.arange(0.0, cum[-1], paso)
    return np.c_[np.interp(s, cum, xy[:, 0]), np.interp(s, cum, xy[:, 1])]


def emparejar_trazas(rutas: list[int], geom: Path, forzadas: dict[int, str]) -> dict[int, dict]:
    """Con otra geometría, cada ruta del crudo va con la traza más parecida (distancia media
    en los dos sentidos) a la suya de data/01_raw, que es contra la que se recortó el crudo:
    el número de una línea puede no ser el del crudo (101↔102 y 1601↔1602 en el catálogo
    de 2026). `forzadas` (--traza 101=102) manda sobre eso."""
    ant, nue = leer_polilineas(RAW), leer_polilineas(geom)
    qn = {k: densa(v) for k, v in nue.items()}
    an = {k: cKDTree(v) for k, v in qn.items()}
    out = {}
    for r in rutas:
        if str(r) not in ant:
            raise ValueError(f"la ruta {r} no tiene polilínea en data/01_raw (la referencia del crudo)")
        qa = densa(ant[str(r)])
        aa = cKDTree(qa)
        def costo(k):
            return an[k].query(qa)[0].mean() + aa.query(qn[k])[0].mean()
        k = forzadas.get(r) or min(qn, key=costo)
        if k not in qn:
            raise ValueError(f"--traza {r}={k}: no hay polilínea {k} en {geom}")
        d = an[k].query(qa)[0]
        out[r] = {"id": k, "media_m": round(float(d.mean()), 1), "max_m": round(float(d.max()), 0),
                  "forzada": r in forzadas}
    return out


def leer_traza(ruta: int, geom: Path = RAW, traza: str | None = None) -> dict:
    traza = traza or str(ruta)
    trazas = leer_polilineas(geom)
    if traza not in trazas:
        raise ValueError(f"no hay polilínea {traza} en {fuente_trazas(geom)}")
    p = trazas[traza]
    xy = np.c_[p[:, 0] * KX, p[:, 1] * KY]
    seg = np.hypot(*np.diff(xy, axis=0).T)
    xy = xy[np.r_[True, seg > 0.01]]
    seg = np.hypot(*np.diff(xy, axis=0).T)
    cum = np.r_[0.0, np.cumsum(seg)]
    largo = float(cum[-1])
    s = np.arange(0.0, largo, PASO_M)
    q = np.c_[np.interp(s, cum, xy[:, 0]), np.interp(s, cum, xy[:, 1])]
    j = np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(seg) - 1)
    tan = (xy[j + 1] - xy[j]) / seg[j][:, None]
    nb = int(math.ceil(largo / TRAMO_M))
    return {"q": q, "s": s, "tan": tan, "rumbo": np.degrees(np.arctan2(tan[:, 0], tan[:, 1])) % 360,
            "tramo": np.minimum((s // TRAMO_M).astype(np.int64), nb - 1), "L": largo, "nb": nb,
            "cerrada": float(np.hypot(*(xy[0] - xy[-1]))) < 100, "fin": xy[-1]}


def otra_geometria(ruta: int, geom: Path, traza: str, T: dict) -> dict:
    """Con otra geometría: de dónde sale la traza y la de data/01_raw (la de antes, contra la
    que se recortó el crudo) para dibujarla encima; `cambio_m`, lo más que se aparta de la nueva."""
    if geom.resolve() == RAW.resolve():
        return {}
    qa = densa(leer_polilineas(RAW)[str(ruta)])
    return {"geom": nombre_geom(geom), "traza": traza, "traza_nombre": nombres_lineas(geom).get(traza, ""),
            "cambio_m": round(float(cKDTree(T["q"]).query(qa)[0].max()), 0),
            "traza_ant": [a_latlon(x, y) for x, y in qa[::2]]}


def a_latlon(x, y):
    return [round(float(y) / KY, 5), round(float(x) / KX, 5)]


def dibujo_traza(T: dict) -> tuple[list, list, list]:
    """Por tramo: sus coordenadas (cada 20 m, empalmando con el siguiente) y su tangente
    media; y flechas de sentido cada km."""
    q, tan, nb = T["q"], T["tan"], T["nb"]
    por = int(TRAMO_M // PASO_M)
    coords, tangentes = [], []
    for b in range(nb):
        k0, k1 = b * por, min((b + 1) * por, len(q))
        pts = np.vstack([q[k0:k1:2], q[k1] if k1 < len(q) else T["fin"]])
        coords.append([a_latlon(x, y) for x, y in pts])
        t = tan[k0:k1].mean(0)
        t = t / (np.hypot(*t) or 1.0)
        tangentes.append([round(float(t[0]), 3), round(float(t[1]), 3)])
    flechas = [a_latlon(*q[k]) + [round(float(T["rumbo"][k])), int(round(T["s"][k] / 1000))]
               for k in range(0, len(q), int(1000 // PASO_M))]
    return coords, tangentes, flechas


def paradas(traza: str, geom: Path = RAW) -> list:
    g = archivos_geometria(geom)
    stops = {s["code"]: s for s in json.load(open(g["stops"], encoding="utf-8"))}
    orden = json.load(open(g["stops_routes"], encoding="utf-8")).get(traza, [])
    out = []
    for e in sorted(orden, key=lambda e: e["stop_order"]):
        s = stops.get(e["stop_code"])
        if s:
            out.append([round(float(s["latitude"]), 5), round(float(s["longitude"]), 5),
                        e["stop_order"], s.get("name") or e["stop_code"]])
    return out


# ═════════════════════════════════════════════════════════════════════════════
#  crudo
# ═════════════════════════════════════════════════════════════════════════════
def meses_de(desde: date, hasta: date) -> list[tuple[int, int]]:
    out, (y, m) = [], (desde.year, desde.month)
    while (y, m) <= (hasta.year, hasta.month):
        out.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def leer_crudo(ruta: int, desde: date, hasta: date) -> pl.DataFrame:
    archivos = sorted((RAW / "records" / str(ruta)).glob(f"df_rec_raw_{ruta}_*.parquet"))
    if not archivos:
        raise FileNotFoundError(f"no hay crudo en data/01_raw/records/{ruta}/")
    base = desde.year * 12 + desde.month
    return (pl.scan_parquet([str(a) for a in archivos])
            .select(pl.col("id").alias("veh_cod"),
                    pl.col("latitude").cast(pl.Float64).alias("lat"),
                    pl.col("longitude").cast(pl.Float64).alias("lon"),
                    pl.col("speed").cast(pl.Float64).fill_null(0.0),
                    pl.col("cape").cast(pl.Float64),
                    pl.col("time"))
            .filter((pl.col("time") >= datetime.combine(desde, datetime.min.time()))
                    & (pl.col("time") < datetime.combine(hasta + timedelta(days=1), datetime.min.time())))
            .drop_nulls(["veh_cod", "lat", "lon", "time"])
            .filter(pl.col("lat").is_between(28.0, 30.5) & pl.col("lon").is_between(-112.0, -110.0))
            .unique(subset=["veh_cod", "time"], keep="first")
            .sort(["veh_cod", "time"])
            .with_columns(veh=(pl.col("veh_cod") != pl.col("veh_cod").shift()).fill_null(True)
                          .cast(pl.Int32).cum_sum(),
                          t=pl.col("time").dt.epoch("s"),
                          dia=pl.col("time").dt.date().cast(pl.Int32),
                          mes=(pl.col("time").dt.year().cast(pl.Int32) * 12
                               + pl.col("time").dt.month().cast(pl.Int32) - base))
            .collect())


def emparejar(T: dict, x: np.ndarray, y: np.ndarray, cape: np.ndarray,
              veh: np.ndarray | None = None, t: np.ndarray | None = None):
    """Por registro: el punto de la traza más cercano en su sentido (−1 si no hay a
    < R_MAX_M) y el más cercano sin mirar el rumbo (−1 si no hay).

    Donde la traza pasa dos veces por la misma calle en el mismo sentido (otra pasada, a más
    de PASADA_M sobre la traza, a ≤ AMBIGUO_M más que la más cercana), el más cercano lo decide
    un par de metros: con `veh` y `t` (registros ordenados por bus y hora) se elige la pasada
    por continuidad, la que menos avanza desde el último registro no ambiguo del mismo bus
    (hasta CONTINUIDAD_S antes). Sin esto, en el ramal de la 601 que la traza baja y sube dos veces
    por vuelta (km ~18 y ~43) cada registro se iba a la línea más cercana, no a la pasada en
    que iba el bus, y la del km 43 salía poco recorrida."""
    arbol = cKDTree(T["q"])
    n = len(x)
    s, L, cerrada = T["s"], T["L"], T["cerrada"]
    j = np.full(n, -1, np.int64)
    j_cual = np.full(n, -1, np.int64)
    amb_r, amb_c = [], []
    for a in range(0, n, LOTE):
        z = slice(a, min(a + LOTE, n))
        d, i = arbol.query(np.c_[x[z], y[z]], k=VECINOS, distance_upper_bound=R_MAX_M)
        ok = np.isfinite(d)
        ii = np.where(ok, i, 0)
        c = cape[z][:, None]
        dif = np.abs((c - T["rumbo"][ii] + 180.0) % 360.0 - 180.0)
        mismo = ok & ((dif <= ANG_MAX) | np.isnan(c))
        tiene = mismo.any(1)
        k = mismo.argmax(1)
        filas = np.arange(len(k))
        jb = np.where(tiene, ii[filas, k], -1)
        j[z] = jb
        j_cual[z] = np.where(ok[:, 0], i[:, 0], -1)
        if veh is None:
            continue
        cand = mismo & (d <= np.where(tiene, d[filas, k], np.inf)[:, None] + AMBIGUO_M)
        sep = np.abs(s[ii] - s[np.maximum(jb, 0)][:, None])
        if cerrada:
            sep = np.minimum(sep, L - sep)
        amb = np.flatnonzero(tiene & (cand & (sep > PASADA_M)).any(1))
        if len(amb):
            amb_r.append(amb + a)
            amb_c.append(np.where(cand[amb], ii[amb], -1))
    if not amb_r:
        return j, j_cual
    amb, cs = np.concatenate(amb_r), np.vstack(amb_c)
    # referencia: el último registro anterior del mismo bus en la traza y no ambiguo
    util = j >= 0
    util[amb] = False
    ult = np.where(util, np.arange(n), -1)
    ult = np.maximum.accumulate(np.r_[-1, ult[:-1]])          # el último útil antes de cada registro
    ref = ult[amb]
    vale = (ref >= 0) & (veh[np.maximum(ref, 0)] == veh[amb]) & (t[amb] - t[np.maximum(ref, 0)] <= CONTINUIDAD_S)
    amb, cs, ref = amb[vale], cs[vale], ref[vale]
    av = s[np.maximum(cs, 0)] - s[j[ref]][:, None]
    if cerrada:
        av = np.mod(av + 50.0, L) - 50.0
    av = np.where((cs < 0) | (av < -50.0), np.inf, np.abs(av))   # hasta 50 m hacia atrás: el GPS
    elige = np.isfinite(av).any(1)
    j[amb[elige]] = cs[elige, av[elige].argmin(1)]
    return j, j_cual


# ═════════════════════════════════════════════════════════════════════════════
#  conteos por (mes, tramo); la fila M es todo el periodo
# ═════════════════════════════════════════════════════════════════════════════
def contar(m, b, M, nb):
    c = np.bincount(m * nb + b, minlength=M * nb).reshape(M, nb).astype(np.int64)
    return np.vstack([c, c.sum(0)])


def mediana_por(m, b, v, M, nb):
    out = np.full((M + 1, nb), np.nan)
    if len(v):
        g = pl.DataFrame({"m": m, "b": b, "v": v}).group_by(["m", "b"]).agg(pl.col("v").median())
        out[g["m"].to_numpy(), g["b"].to_numpy()] = g["v"].to_numpy()
        g = pl.DataFrame({"b": b, "v": v}).group_by("b").agg(pl.col("v").median())
        out[M, g["b"].to_numpy()] = g["v"].to_numpy()
    return out


def entradas(m, b1, b2, M, nb):
    """Cuántas veces se entra a cada tramo: los tramos b1+1 … b2 de cada par (dando la
    vuelta por el inicio si la traza es cerrada y b2 < b1)."""
    D = np.zeros(M * (nb + 1), np.int64)

    def rango(mm, a, z):
        ok = a <= z
        mm, a, z = mm[ok], a[ok], z[ok]
        D[:] += np.bincount(mm * (nb + 1) + a, minlength=M * (nb + 1))
        D[:] -= np.bincount(mm * (nb + 1) + z + 1, minlength=M * (nb + 1))

    directo, vuelta = b2 > b1, b2 < b1
    rango(m[directo], b1[directo] + 1, b2[directo])
    rango(m[vuelta], b1[vuelta] + 1, np.full(int(vuelta.sum()), nb - 1))
    rango(m[vuelta], np.zeros(int(vuelta.sum()), np.int64), b2[vuelta])
    c = np.cumsum(D.reshape(M, nb + 1), axis=1)[:, :nb]
    return np.vstack([c, c.sum(0)])


def corridas(mask: np.ndarray, cerrada: bool) -> list[np.ndarray]:
    if not mask.any():
        return []
    if mask.all():
        return [np.arange(len(mask))]
    idx = np.flatnonzero(mask)
    grupos = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    if cerrada and len(grupos) > 1 and grupos[0][0] == 0 and grupos[-1][-1] == len(mask) - 1:
        grupos[0] = np.r_[grupos.pop(), grupos[0]]
    return grupos


def redondear(a, d):
    return [[None if not np.isfinite(v) else (round(float(v), d) if d else int(v)) for v in fila]
            for fila in a]


# ═════════════════════════════════════════════════════════════════════════════
#  una ruta
# ═════════════════════════════════════════════════════════════════════════════
def analizar(ruta: int, desde: date, hasta: date, geom: Path = RAW, traza: str | None = None) -> tuple[dict, dict]:
    traza = traza or str(ruta)
    T = leer_traza(ruta, geom, traza)
    nb, L, cerrada = T["nb"], T["L"], T["cerrada"]
    MS = meses_de(desde, hasta)
    M = len(MS)
    etiquetas = [MESES_ES[m - 1] + (f" {y % 100:02d}" if MS[0][0] != MS[-1][0] else "") for y, m in MS]
    df = leer_crudo(ruta, desde, hasta)
    if df.height == 0:
        raise ValueError("sin registros en la ventana")

    x = df["lon"].to_numpy() * KX
    y = df["lat"].to_numpy() * KY
    cape = df["cape"].to_numpy().astype(float)
    vel = df["speed"].to_numpy()
    mes = df["mes"].to_numpy().astype(np.int64)
    veh = df["veh"].to_numpy()
    t = df["t"].to_numpy().astype(np.int64)
    dia = df["dia"].to_numpy()
    j, j_cual = emparejar(T, x, y, cape, veh, t)
    cat = np.where(j >= 0, 0, np.where(j_cual >= 0, 1, 2))      # traza · contrasentido · fuera
    jj = np.where(j >= 0, j, 0)
    lat_off = np.where(j >= 0, T["tan"][jj, 0] * (y - T["q"][jj, 1]) - T["tan"][jj, 1] * (x - T["q"][jj, 0]), np.nan)

    # ── conteos de registros ──────────────────────────────────────────────────────
    en = cat == 0
    b_en = T["tramo"][j[en]]
    n = contar(mes[en], b_en, M, nb)
    mov = en & (vel >= V_MOV_KMH)
    b_mov = T["tramo"][j[mov]]
    n_mov = contar(mes[mov], b_mov, M, nb)
    lejos = contar(mes[mov & (np.abs(lat_off) > LEJOS_M)],
                   T["tramo"][j[mov & (np.abs(lat_off) > LEJOS_M)]], M, nb)
    lat_med = mediana_por(mes[mov], b_mov, lat_off[mov], M, nb)
    co = cat == 1
    contra = contar(mes[co], T["tramo"][j_cual[co]], M, nb)

    # ── intervalo entre registros de un mismo bus, por mes: de septiembre en adelante
    #    el crudo se vuelve ralo (de 61 s a 2–17 min), y los umbrales de tiempo lo siguen
    dt_todos = np.diff(t).astype(float)
    ok_dt = (veh[1:] == veh[:-1]) & (dt_todos > 0) & (dt_todos <= 3600)
    intervalo = np.full(M + 1, np.nan)
    for m in range(M):
        x_ = dt_todos[ok_dt & (mes[:-1] == m)]
        intervalo[m] = float(np.median(x_)) if len(x_) else np.nan
    intervalo[M] = float(np.median(dt_todos[ok_dt])) if ok_dt.any() else np.nan
    dt_paso = np.clip(np.nan_to_num(2.5 * intervalo[:M], nan=DT_PASO_S), DT_PASO_S, DT_PASO_MAX_S)
    dt_hueco = np.clip(4 * dt_paso, DT_HUECO_S, 2 * DT_HUECO_S)

    # ── pasos y huecos: cada bus, de un registro en la traza al siguiente ─────────
    v_, t_, s_, m_ = veh[en], t[en], T["s"][j[en]], mes[en]
    mismo = v_[1:] == v_[:-1]
    dt = (t_[1:] - t_[:-1]).astype(float)
    ds = s_[1:] - s_[:-1]
    if cerrada:
        ds = np.mod(ds, L)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.where(dt > 0, ds / dt, np.inf)
    dp, dh = dt_paso[m_[:-1]], dt_hueco[m_[:-1]]
    # rodeo: un paso en el que la traza da una vuelta o entra a un ramal y sale cerca de donde
    # entró, y el bus no se vio ahí: el avance sobre la traza es mucho mayor que lo que se movió.
    # No es un paso por esos tramos sino un salto (cuenta como hueco); donde los buses sí dan la
    # vuelta, los que se ven adentro mantienen bajos los saltos. Sólo entre pares que serían
    # paso (con su tope de velocidad): dos registros en una calle que la traza recorre dos
    # veces en el mismo sentido (601: km ~18 y ~43) no son un rodeo de 25 km. Y sólo con
    # registros a ≤ DT_PASO_S: en los meses ralos (8–13 min entre registros) el bus sí alcanza a
    # dar la vuelta sin que se le vea.
    dr = np.hypot(np.diff(x[en]), np.diff(y[en]))
    paso = mismo & (dt > 0) & (dt <= dp) & (ds >= AVANCE_MIN_M) & (ds <= V_PASO_MAX * dt + 100)
    rodeo = paso & (dt <= DT_PASO_S) & (ds - dr > RODEO_M) & (ds > RODEO_X * dr)
    paso &= ~rodeo
    hueco = (mismo & (dt > dp) & (dt <= dh) & (ds >= HUECO_MIN_M)
             & (v >= V_HUECO[0]) & (v <= V_HUECO[1]) & (ds <= 0.5 * L)) | rodeo
    b1, b2, mm = T["tramo"][j[en]][:-1], T["tramo"][j[en]][1:], m_[:-1]
    pasos = entradas(mm[paso], b1[paso], b2[paso], M, nb)
    huecos = entradas(mm[hueco], b1[hueco], b2[hueco], M, nb)

    # ── por mes ───────────────────────────────────────────────────────────────────
    dias = np.zeros(M + 1, np.int64)
    vehs = np.zeros(M + 1, np.int64)
    g = df.group_by("mes").agg(pl.col("dia").n_unique().alias("d"), pl.col("veh").n_unique().alias("v"))
    dias[g["mes"].to_numpy()] = g["d"].to_numpy()
    vehs[g["mes"].to_numpy()] = g["v"].to_numpy()
    dias[M], vehs[M] = df["dia"].n_unique(), df["veh"].n_unique()
    # de la 1001 en adelante no hay crudo después de agosto (0–516 registros al mes) y la 901
    # empieza en abril: un mes con tan poco no se evalúa
    registros = np.r_[np.bincount(mes, minlength=M)[:M], len(mes)]
    insuf = (registros < POCOS_REGISTROS) | (dias < POCOS_DIAS)

    with np.errstate(divide="ignore", invalid="ignore"):
        med = np.array([np.median(f) for f in pasos], float)
        pasos_rel = np.where(med[:, None] > 0, pasos / med[:, None], np.nan)
        cruces = pasos + huecos
        salto = np.where(cruces > 0, huecos / cruces, np.nan)
        lejos_pct = np.where(n_mov > 0, lejos / n_mov, np.nan)
        contra_pct = np.where(contra + n > 0, contra / (contra + n), np.nan)
        lat_med = np.where(n_mov >= 5, lat_med, np.nan)
        est = np.ones((M + 1, nb), np.int8)
        est[(n_mov >= N_MIN) & ((np.abs(lat_med) >= LAT_MIN_M) | (lejos_pct >= LEJOS_MIN))] = 2
        est[((contra + n) >= N_MIN) & (contra_pct >= CONTRA_MIN)] = 3
        est[pasos_rel < POCO_REL] = 4
        # saltada: muchos saltos, y bastantes más que el nivel de saltos del mes (en los
        # meses ralos todos los tramos tienen huecos)
        base = np.array([np.nanmedian(f) if np.isfinite(f).any() else 0.0 for f in salto])
        est[(cruces >= CRUCES_MIN) & (salto >= np.maximum(SALTO_MIN, base + SALTO_SOBRE_BASE)[:, None])] = 5
    est[insuf] = 0
    for a in (pasos_rel, salto, lejos_pct, contra_pct, lat_med):
        a[insuf] = np.nan
    # un mes con más de la mitad de la traza con problema no localiza nada: fuera de las zonas
    general = np.array([(not insuf[m]) and float((est[m] >= 2).mean()) > MES_GENERAL for m in range(M + 1)])

    resumen = []
    for m in range(M + 1):
        r = mes == m if m < M else np.ones(len(mes), bool)
        nr = int(r.sum())
        rm = r & mov
        km = {ESTADOS[e]: round(float((est[m] == e).sum()) * TRAMO_M / 1000, 1) for e in range(6)}
        resumen.append({
            "mes": f"{MS[m][0]}-{MS[m][1]:02d}" if m < M else "todo",
            "etiqueta": etiquetas[m] if m < M else "Todo el periodo",
            "registros": nr, "dias": int(dias[m]), "vehiculos": int(vehs[m]),
            "pct_en_traza": round(float((r & en).sum()) / nr, 4) if nr else None,
            "pct_lejos_30m": round(float((rm & (np.abs(lat_off) > LEJOS_M)).sum()) / max(int(rm.sum()), 1), 4) if nr else None,
            "pct_contrasentido": round(float((r & co).sum()) / nr, 4) if nr else None,
            "pct_fuera_80m": round(float((r & (cat == 2)).sum()) / nr, 4) if nr else None,
            "desv_lateral_mediana_m": round(float(np.median(np.abs(lat_off[rm]))), 1) if rm.any() else None,
            "pasos_dia_mediana": round(float(med[m]) / dias[m], 1) if dias[m] else None,
            "intervalo_mediano_s": round(float(intervalo[m])) if np.isfinite(intervalo[m]) else None,
            "pocos_registros": bool(insuf[m]),
            "muestreo_ralo": bool(not insuf[m] and ((np.isfinite(intervalo[m]) and intervalo[m] > RALO_S)
                                                    or base[m] >= RALO_SALTO)),
            "salto_base": round(float(base[m]), 2),
            "fuera_de_zonas": bool(general[m]),
            **{f"km_{k.replace(' ', '_')}": v for k, v in km.items()},
        })

    # ── zonas ─────────────────────────────────────────────────────────────────────
    P = (est[:M] >= 2) & ~general[:M, None]
    U = P.any(0)
    vecino = (np.roll(U, 1) & np.roll(U, -1)) if cerrada else np.r_[False, U[:-2] & U[2:], False]
    zonas = []
    coords, tangentes, flechas = dibujo_traza(T)
    for grupo in corridas(U | vecino, cerrada):
        Pm = P[:, grupo]
        en_mes = Pm.any(1)
        if len(grupo) < 2 and en_mes.sum() < 3:
            continue
        valores = est[:M, grupo][Pm]
        tipo = int(np.bincount(valores, minlength=6).argmax()) if len(valores) else 1
        peor = int(Pm.sum(1).argmax())
        centro = coords[int(grupo[len(grupo) // 2])]
        with np.errstate(all="ignore"):
            zonas.append({
                "id": len(zonas) + 1, "b": [int(b) for b in grupo],
                "km0": round(float(grupo[0]) * TRAMO_M / 1000, 1),
                "km1": round(min(float(grupo[-1] + 1) * TRAMO_M, L) / 1000, 1),
                "tipo": tipo, "meses": [int(v) for v in en_mes], "peor": peor,
                # a < 500 m del km 0: el cierre de la polilínea, casi siempre una terminal
                "junto_km0": bool(min(grupo) < 5 or max(grupo) >= nb - 5),
                "meses_ok": int((~insuf[:M] & ~general[:M] & ~en_mes).sum()),
                "lat": centro[len(centro) // 2][0], "lon": centro[len(centro) // 2][1],
                "salto_max": round(float(np.nanmax(salto[:M, grupo])), 2) if np.isfinite(salto[:M, grupo]).any() else None,
                "desv_max_m": round(float(np.nanmax(np.abs(lat_med[:M, grupo]))), 1) if np.isfinite(lat_med[:M, grupo]).any() else None,
                "contra_max": round(float(np.nanmax(contra_pct[:M, grupo])), 2) if np.isfinite(contra_pct[:M, grupo]).any() else None,
                "pasos_rel_min": round(float(np.nanmin(pasos_rel[:M, grupo])), 2) if np.isfinite(pasos_rel[:M, grupo]).any() else None,
            })

    # ── lo que se dibuja: muestra de registros, días de bus y los de > 80 m ───────
    rng = np.random.default_rng(ruta)
    dcat = np.where(cat == 0, np.where(np.abs(lat_off) <= 15, 0, np.where(np.abs(lat_off) <= LEJOS_M, 1, 2)),
                    np.where(cat == 1, 3, 4))
    lat_g, lon_g = df["lat"].to_numpy(), df["lon"].to_numpy()
    puntos, buses = [], []
    veh_cod = df["veh_cod"].to_numpy()
    clave = veh.astype(np.int64) * 100_000 + dia.astype(np.int64)
    # km de cada registro: en su sentido; si no hay, el punto de la traza más cercano
    km_reg = np.where(j >= 0, T["s"][jj], np.where(j_cual >= 0, T["s"][np.maximum(j_cual, 0)], np.nan)) / 1000

    # cuántas veces pasa cada bus-día por cada tramo: los que cruza avanzando entre dos
    # registros seguidos (un paso) y el tramo donde se le vuelve a ver tras un corte; lo que
    # cruza en un hueco no cuenta. «Vueltas» del día: lo que pasa por la mitad de los tramos
    # (la mediana, con ceros); un tramo está recorrido si pasa en ≥ la mitad de sus vueltas.
    pp = np.flatnonzero(paso)
    n_ent = np.where(b2[pp] >= b1[pp], b2[pp] - b1[pp], b2[pp] + nb - b1[pp])
    sig = np.arange(int(n_ent.sum())) - np.repeat(np.cumsum(n_ent) - n_ent, n_ent) + 1
    en_idx = np.flatnonzero(en)
    inicio = np.r_[True, ~paso]
    cod, veces = np.unique(np.r_[clave[en_idx[inicio]], np.repeat(clave[en_idx[pp]], n_ent)] * nb
                           + np.r_[b_en[inicio], (np.repeat(b1[pp], n_ent) + sig) % nb], return_counts=True)
    mitad = nb // 2 + 1
    ev = pl.DataFrame({"k": cod // nb, "veces": veces})
    vu = (ev.group_by("k").agg(pl.len().alias("nz"), pl.col("veces").top_k(mitad).min().alias("vk"))
          .with_columns(vueltas=pl.when(pl.col("nz") >= mitad).then(pl.col("vk")).otherwise(0)))
    ev = ev.join(vu.select("k", "vueltas"), on="k").with_columns(
        umbral=pl.max_horizontal(pl.lit(1.0), pl.col("vueltas") * 0.5))
    umbral_ev = ev["umbral"].to_numpy()                     # mismo orden que cod
    cb = (ev.group_by("k").agg((pl.col("veces") >= pl.col("umbral")).sum().alias("n_ok"),
                               pl.col("vueltas").first()).sort("k"))
    k_cob, n_cob, v_cob = cb["k"].to_numpy(), cb["n_ok"].to_numpy(), cb["vueltas"].to_numpy()

    def cobertura(claves):
        pos = np.minimum(np.searchsorted(k_cob, claves), len(k_cob) - 1)
        hay = k_cob[pos] == claves
        return np.where(hay, n_cob[pos], 0) / nb, np.where(hay, v_cob[pos], 0)

    def no_recorrido(kb):
        """Los km de la traza (≥ 200 m seguidos) por los que el bus pasa en menos de la
        mitad de sus vueltas del día."""
        a, z = np.searchsorted(cod, [kb * nb, (kb + 1) * nb])
        hecho = np.zeros(nb, bool)
        hecho[(cod[a:z] - kb * nb)[veces[a:z] >= umbral_ev[a:z]]] = True
        return [[round(float(g[0]) * TRAMO_M / 1000, 1), round(min(float(g[-1] + 1) * TRAMO_M, L) / 1000, 1)]
                for g in corridas(~hecho, cerrada) if len(g) >= 2]

    def dias_de_bus(idx, m):
        """Hasta BUSES_MES días de bus del mes, de vehículos distintos, entre los que recorren
        ≥ COBERTURA_BUS de la traza en la mayoría de sus vueltas (si ninguno llega, el 10 % que
        más recorre: el p90 del mes; nunca menos de COBERTURA_MIN); de ellos, los de más
        registros (no los de más cobertura: eso escogería a los pocos que no toman un atajo
        que toman casi todos). Por registro: hora
        del día, posición, km, categoría y el enlace con el siguiente (1 continuo, 2 hueco: lo
        que cuenta en «saltos», 0 nada)."""
        u, cnt = np.unique(clave[idx], return_counts=True)
        cob, vueltas = cobertura(u)
        cand = np.flatnonzero((cob >= min(COBERTURA_BUS, float(np.quantile(cob, 0.9)))) & (cob >= COBERTURA_MIN))
        orden = cand[np.argsort(-cnt[cand], kind="stable")]
        elegidos, usados = [], set()
        for k in orden:
            if int(u[k] // 100_000) not in usados:
                usados.add(int(u[k] // 100_000))
                elegidos.append((u[k], float(cob[k]), int(vueltas[k])))
            if len(elegidos) == BUSES_MES:
                break
        out = []
        for kb, cb, nv in elegidos:
            r = idx[clave[idx] == kb]
            dt_r = np.diff(t[r]).astype(float)
            ds = (km_reg[r][1:] - km_reg[r][:-1]) * 1000
            if cerrada:
                ds = np.mod(ds, L)
            en_r = cat[r] == 0
            dr_r = np.hypot(np.diff(x[r]), np.diff(y[r]))
            with np.errstate(divide="ignore", invalid="ignore"):
                v_r = np.where(dt_r > 0, ds / dt_r, np.inf)
                hueco_r = (en_r[:-1] & en_r[1:] & (dt_r > dt_paso[m]) & (dt_r <= dt_hueco[m])
                           & (ds >= HUECO_MIN_M) & (v_r >= V_HUECO[0]) & (v_r <= V_HUECO[1]) & (ds <= 0.5 * L))
                rodeo_r = (en_r[:-1] & en_r[1:] & (dt_r > 0) & (dt_r <= min(dt_paso[m], DT_PASO_S)) & (ds >= AVANCE_MIN_M)
                           & (ds <= V_PASO_MAX * dt_r + 100) & (ds - dr_r > RODEO_M) & (ds > RODEO_X * dr_r))
            enlace = np.where(hueco_r | rodeo_r, 2, np.where(dt_r <= dt_paso[m], 1, 0))
            d0 = date(1970, 1, 1) + timedelta(days=int(dia[r[0]]))
            out.append({"veh": str(veh_cod[r[0]]), "dia": d0.isoformat(), "sem": DIAS_SEM[d0.weekday()],
                        "n": int(len(r)), "t": [int(v % 86400) for v in t[r]],
                        "la": [round(float(v), 5) for v in lat_g[r]], "lo": [round(float(v), 5) for v in lon_g[r]],
                        "km": [round(float(v), 2) if np.isfinite(v) else None for v in km_reg[r]],
                        "c": [int(v) for v in dcat[r]], "l": [int(v) for v in enlace],
                        "cob": round(cb, 3), "vueltas": nv, "sin": no_recorrido(kb)})
        return out
    for m in range(M):
        idx = np.flatnonzero(mes == m)
        toma = []
        for c, tope in ((4, 300), (3, 400), (2, 500)):
            k = idx[dcat[idx] == c]
            toma.append(rng.choice(k, tope, replace=False) if len(k) > tope else k)
        resto = max(MUESTRA_MES - sum(len(k) for k in toma), 0)
        k = idx[dcat[idx] <= 1]
        toma.append(rng.choice(k, resto, replace=False) if len(k) > resto else k)
        sel = np.sort(np.concatenate(toma)).astype(np.int64)
        puntos.append([[round(float(lat_g[i]), 5), round(float(lon_g[i]), 5), int(dcat[i])] for i in sel])
        buses.append(dias_de_bus(idx, m) if len(idx) and not insuf[m] else [])
    fu = cat == 2
    fuera = []
    if fu.any():
        cx, cy = np.floor(x[fu] / 50).astype(np.int64), np.floor(y[fu] / 50).astype(np.int64)
        g = (pl.DataFrame({"m": mes[fu], "cx": cx, "cy": cy}).group_by(["m", "cx", "cy"]).len())
        fuera = [a_latlon((r["cx"] + 0.5) * 50, (r["cy"] + 0.5) * 50) + [int(r["m"]), int(r["len"])]
                 for r in g.iter_rows(named=True)]

    datos = {
        "ruta": ruta, "L_km": round(L / 1000, 2), "cerrada": cerrada, "nb": nb,
        "meses": [f"{y}-{m:02d}" for y, m in MS], "etiquetas": etiquetas,
        "dias": [int(v) for v in dias], "insuf": [bool(v) for v in insuf], "KX": KX, "KY": KY,
        "coords": coords, "tan": tangentes, "flechas": flechas, "paradas": paradas(traza, geom),
        **otra_geometria(ruta, geom, traza, T),
        "met": {"estado": est.tolist(), "pasos": redondear(pasos, 0), "huecos": redondear(huecos, 0),
                "pasos_rel": redondear(pasos_rel, 2), "salto": redondear(salto, 2),
                "lat": redondear(lat_med, 1), "lejos": redondear(lejos_pct, 2),
                "contra": redondear(contra_pct, 2), "n": redondear(n, 0)},
        "resumen": resumen, "zonas": zonas, "puntos": puntos, "buses": buses, "fuera": fuera,
        "umbrales": {"tramo_m": TRAMO_M, "lejos_m": LEJOS_M, "lat_min_m": LAT_MIN_M,
                     "salto_min": SALTO_MIN, "poco_rel": POCO_REL, "contra_min": CONTRA_MIN},
    }
    tr = {"traza": traza} if "geom" in datos else {}
    tablas = {
        "resumen": [{"ruta": ruta, **tr, **{k: v for k, v in r.items() if k != "etiqueta"}} for r in resumen],
        "zonas": [{"ruta": ruta, **tr, "zona": z["id"], "km_ini": z["km0"], "km_fin": z["km1"],
                   "largo_km": round(z["km1"] - z["km0"] if z["km1"] >= z["km0"] else z["km1"] + L / 1000 - z["km0"], 1),
                   "tipo": ESTADOS[z["tipo"]],
                   "meses_con_problema": " ".join(datos["meses"][m] for m in range(M) if z["meses"][m]),
                   "n_meses": sum(z["meses"]), "meses_ok": z["meses_ok"], "junto_km0": z["junto_km0"],
                   "lat": z["lat"], "lon": z["lon"],
                   "salto_max": z["salto_max"], "desv_max_m": z["desv_max_m"],
                   "contra_max": z["contra_max"], "pasos_rel_min": z["pasos_rel_min"]} for z in zonas],
        "tramos": [{"ruta": ruta, **tr, "mes": datos["meses"][m], "tramo": b,
                    "km_ini": round(b * TRAMO_M / 1000, 1), "estado": ESTADOS[est[m, b]],
                    "pasos": int(pasos[m, b]), "huecos": int(huecos[m, b]),
                    "salto": datos["met"]["salto"][m][b], "pasos_rel": datos["met"]["pasos_rel"][m][b],
                    "desv_lateral_m": datos["met"]["lat"][m][b], "pct_lejos_30m": datos["met"]["lejos"][m][b],
                    "pct_contrasentido": datos["met"]["contra"][m][b], "registros": int(n[m, b])}
                   for m in range(M) if not insuf[m] for b in range(nb)],
        "indice": {"ruta": ruta, "L_km": datos["L_km"], "cerrada": cerrada, "registros": int(df.height),
                   "meses": etiquetas, "dias": [int(v) for v in dias[:M]],
                   "pct_problema": [None if insuf[m] else round(float((est[m] >= 2).mean()), 4) for m in range(M)],
                   "registros_mes": [int(v) for v in registros[:M]],
                   "pct_en_traza": [r["pct_en_traza"] for r in resumen[:M]],
                   "intervalo": [r["intervalo_mediano_s"] for r in resumen[:M]],
                   "ralo": [r["muestreo_ralo"] for r in resumen[:M]],
                   "zonas": [{"km0": z["km0"], "km1": z["km1"], "tipo": ESTADOS[z["tipo"]],
                              "n_meses": sum(z["meses"]), "meses": z["meses"]} for z in zonas]},
    }
    return datos, tablas


def una_ruta(ruta: int, desde: str, hasta: str, salida: str, geom: str | None = None, traza: str | None = None) -> dict:
    t0 = time.perf_counter()
    out = Path(salida)
    try:
        datos, tablas = analizar(ruta, date.fromisoformat(desde), date.fromisoformat(hasta), Path(geom) if geom else RAW, traza)
        if "geom" in datos:
            tablas["indice"].update({k: datos[k] for k in ("traza", "traza_nombre", "cambio_m")})
        html = (PLANTILLA_RUTA.replace("__RUTA__", str(ruta))
                .replace("__DATOS__", json.dumps(datos, ensure_ascii=False, separators=(",", ":"))
                         .replace("</", "<\\/")))
        (out / f"ruta_{ruta}.html").write_text(html, encoding="utf-8")
        tablas["indice"]["segundos"] = round(time.perf_counter() - t0, 1)
        tablas["indice"]["error"] = None
    except Exception as e:                                          # una ruta no tumba el lote
        tablas = {"resumen": [], "zonas": [], "tramos": [],
                  "indice": {"ruta": ruta, "error": f"{type(e).__name__}: {e}",
                             "detalle": traceback.format_exc(), "segundos": round(time.perf_counter() - t0, 1)}}
    partes = out / "_partes"
    partes.mkdir(exist_ok=True)
    (partes / f"{ruta}.json").write_text(json.dumps(tablas, ensure_ascii=False), encoding="utf-8")
    return tablas["indice"]


# ═════════════════════════════════════════════════════════════════════════════
#  índice, tablas y zip (con todas las rutas que haya en _partes/)
# ═════════════════════════════════════════════════════════════════════════════
def color_pct(p):
    if p is None:
        return "var(--nodata)"
    return ("var(--c0)" if p < 0.02 else "var(--c1)" if p < 0.05 else "var(--c2)" if p < 0.15 else "var(--c3)")


def escribir_indice(out: Path, desde: str, hasta: str, geom: Path = RAW) -> list[dict]:
    partes = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((out / "_partes").glob("*.json"))]
    partes.sort(key=lambda p: p["indice"]["ruta"])
    for nombre, clave in (("resumen_mensual.csv", "resumen"), ("zonas.csv", "zonas"), ("tramos_por_mes.csv", "tramos")):
        filas = [f for p in partes for f in p[clave]]
        with open(out / nombre, "w", newline="", encoding="utf-8-sig") as fh:
            if filas:
                w = csv.DictWriter(fh, fieldnames=list(filas[0]))
                w.writeheader()
                w.writerows(filas)
    buenas = [p["indice"] for p in partes if not p["indice"].get("error")]
    etiquetas = buenas[0]["meses"] if buenas else []
    filas = []
    for p in partes:
        ix = p["indice"]
        r = ix["ruta"]
        if ix.get("error"):
            filas.append(f'<tr><td class="ruta">{r}</td><td colspan="{len(etiquetas) + 4}" class="error">'
                         f'error: {esc(ix["error"])}</td></tr>')
            continue
        celdas = "".join(
            f'<td class="mes{" ralo" if ra else ""}" style="background-color:{color_pct(pp)}" title="{esc(et)}: '
            + (f'{pp * 100:.1f} % de la traza con problema · {d} días · {en * 100:.1f} % de registros en la traza'
               f' · un registro cada {iv} s (mediana){" — muestreo ralo" if ra else ""}'
               if pp is not None else ("sin registros" if not nr else f"sin evaluar: {nr:,} registros en {d} días"))
            + f'">{"" if pp is None else f"{pp * 100:.0f}"}</td>'
            for et, pp, d, en, iv, ra, nr in zip(ix["meses"], ix["pct_problema"], ix["dias"],
                                                 [v if v is not None else 0 for v in ix["pct_en_traza"]],
                                                 ix["intervalo"], ix["ralo"], ix["registros_mes"]))
        zs = sorted(ix["zonas"], key=lambda z: (-z["n_meses"], z["km0"]))
        txt = " · ".join(f'km {z["km0"]}–{z["km1"]} {z["tipo"]} ({z["n_meses"]} m)' for z in zs[:3])
        tr = ""
        if "traza" in ix:
            cambio = f'traza {ix["traza"]}' if ix["traza"] != str(r) else ""
            if ix["cambio_m"] > CAMBIO_M:
                cambio += (" · " if cambio else "") + f'cambió (hasta {ix["cambio_m"]:.0f} m)'
            tr = f'<div class="meta" title="{esc(ix["traza_nombre"])}">{esc(cambio)}</div>' if cambio else ""
        filas.append(f'<tr><td class="ruta"><a href="ruta_{r}.html">{r}</a>{tr}</td>'
                     f'<td class="num">{ix["L_km"]:.1f}</td><td class="num">{ix["registros"]:,}</td>{celdas}'
                     f'<td class="num">{len(ix["zonas"])}</td><td class="zonas">{esc(txt)}</td></tr>')
    fuente = "" if geom.resolve() == RAW.resolve() else (
        f" · traza y paradas de <b>{esc(nombre_geom(geom))}</b> (cada ruta del crudo con la traza más parecida a la de"
        f" data/01_raw; debajo del número, si es otra o si cambió más de {CAMBIO_M:.0f} m)")
    html = (PLANTILLA_INDICE.replace("__DESDE__", desde).replace("__HASTA__", hasta).replace("__GEOM__", fuente)
            .replace("__N__", str(len(buenas))).replace("__FECHA__", datetime.now().strftime("%Y-%m-%d %H:%M"))
            .replace("__CABECERA__", "".join(f"<th>{esc(e)}</th>" for e in etiquetas))
            .replace("__FILAS__", "\n".join(filas)))
    (out / "index.html").write_text(html, encoding="utf-8")
    return [p["indice"] for p in partes]


def esc(s) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def empaquetar(out: Path) -> Path:
    destino = out.with_suffix(".zip")
    tmp = out.with_suffix(".zip.tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f in sorted(out.iterdir()):
            if f.is_file() and f.suffix in (".html", ".csv"):
                z.write(f, f"{out.name}/{f.name}")
    os.replace(tmp, destino)
    return destino


def rutas_con_crudo() -> list[int]:
    rec = RAW / "records"
    return sorted(int(p.name) for p in rec.iterdir()
                  if p.is_dir() and p.name.isdigit() and any(p.glob(f"df_rec_raw_{p.name}_*.parquet")))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rutas", nargs="*", type=int, help="por defecto, todas las que tienen crudo")
    ap.add_argument("--desde", default="2024-01-01")
    ap.add_argument("--hasta", default="2024-12-31")
    ap.add_argument("--procesos", type=int, default=min(4, os.cpu_count() or 1))
    ap.add_argument("--salida", help="por defecto reports/mapas_traza_<desde>_<hasta>[_<geometría>]")
    ap.add_argument("--geometria", help="carpeta con otra traza y paradas (por defecto data/01_raw)")
    ap.add_argument("--traza", nargs="*", default=[], metavar="RUTA=ID",
                    help="con --geometria: la polilínea de esa ruta, en vez de la más parecida")
    a = ap.parse_args()
    date.fromisoformat(a.desde), date.fromisoformat(a.hasta)
    rutas = a.rutas or rutas_con_crudo()
    geom = Path(a.geometria).resolve() if a.geometria else RAW
    otra = geom != RAW.resolve()
    sufijo = ("_" + "-".join("".join(c if c.isalnum() else " " for c in geom.name.lower()).split())) if otra else ""
    out = Path(a.salida) if a.salida else RAIZ / "reports" / f"mapas_traza_{a.desde}_{a.hasta}{sufijo}"
    out.mkdir(parents=True, exist_ok=True)
    trazas = {}
    if otra:
        archivos_geometria(geom)
        forzadas = {int(k): v for k, v in (t.split("=", 1) for t in a.traza)}
        trazas = emparejar_trazas(rutas, geom, forzadas)
        nombres = nombres_lineas(geom)
        print(f"geometría: {geom} (trazas de {fuente_trazas(geom).name})\n  ruta del crudo → traza (distancia a la de data/01_raw: media · máx)", flush=True)
        for r in rutas:
            t = trazas[r]
            nota = (("  ← otro número" if t["id"] != str(r) else "") + ("  ← cambió" if t["max_m"] > CAMBIO_M else "")
                    + ("  (--traza)" if t["forzada"] else ""))
            print(f"  {r:>6} → {t['id']:<6} {nombres.get(t['id'], ''):<34} {t['media_m']:6.1f} · {t['max_m']:4.0f} m{nota}", flush=True)
    elif a.traza:
        ap.error("--traza va con --geometria")
    procesos = max(1, min(a.procesos, len(rutas)))
    # polars ya arrancó sus hilos en este proceso: los hijos van con spawn, y cada uno con
    # su parte de los CPU
    os.environ.setdefault("POLARS_MAX_THREADS", str(max(1, (os.cpu_count() or 1) // procesos)))
    print(f"mapas de la traza · {len(rutas)} rutas · {a.desde} → {a.hasta} · {procesos} a la vez → {out}", flush=True)
    t0 = time.perf_counter()
    errores = []
    with ProcessPoolExecutor(max_workers=procesos, mp_context=multiprocessing.get_context("spawn")) as pool:
        futuros = {pool.submit(una_ruta, r, a.desde, a.hasta, str(out), str(geom) if otra else None,
                               trazas[r]["id"] if otra else None): r for r in rutas}
        for i, fut in enumerate(as_completed(futuros), 1):
            ix = fut.result()
            if ix.get("error"):
                errores.append(ix["ruta"])
                print(f"  [{i}/{len(rutas)}] {ix['ruta']}: ✗ {ix['error']} ({ix['segundos']} s)\n{ix.get('detalle', '')}", flush=True)
            else:
                pp = [p for p in ix["pct_problema"] if p is not None] or [0.0]
                print(f"  [{i}/{len(rutas)}] {ix['ruta']}: {ix['registros']:,} registros · {len(ix['zonas'])} zonas · "
                      f"traza con problema {min(pp) * 100:.0f}–{max(pp) * 100:.0f} % según el mes · {ix['segundos']} s",
                      flush=True)
    escribir_indice(out, a.desde, a.hasta, geom)
    z = empaquetar(out)
    print(f"✓ {z} · {z.stat().st_size / 1e6:.1f} MB · {time.perf_counter() - t0:.0f} s"
          + (f" · con error: {errores}" if errores else ""), flush=True)
    return 1 if errores else 0


# ═════════════════════════════════════════════════════════════════════════════
#  plantillas
# ═════════════════════════════════════════════════════════════════════════════
ESTILO_BASE = """
:root{--bg:#f5f6f8;--panel:#fff;--ink:#1c2230;--muted:#5f6b7d;--line:#e2e5ea;--accent:#2457c5;
 --nodata:#e9ebef;--c0:#d7eed4;--c1:#f4e7a6;--c2:#f6b46b;--c3:#e2674f}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
a{color:var(--accent)}
h1{font-size:20px;margin:0}
h2{font-size:15px;margin:0 0 8px}
.meta{color:var(--muted)}
"""

PLANTILLA_INDICE = """<!doctype html>
<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Traza por mes · red</title>
<style>""" + ESTILO_BASE + """
body{padding:20px 16px 40px;max-width:1500px;margin:0 auto}
header{margin-bottom:14px}
.nota{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px 16px;margin:12px 0 16px;max-width:1100px}
.nota p{margin:6px 0}
.envoltura{overflow-x:auto;background:var(--panel);border:1px solid var(--line);border-radius:8px}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:5px 7px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}
th{font-weight:600;color:var(--muted);font-size:12px;position:sticky;top:0;background:var(--panel)}
td.mes{text-align:center;min-width:38px;font-size:12px}
.ralo{background-image:repeating-linear-gradient(45deg,rgba(28,34,48,.16) 0 2px,transparent 2px 6px)}
td.num{text-align:right}
td.ruta{font-weight:600}
td.zonas{white-space:normal;min-width:280px;font-size:12px;color:var(--muted)}
td.error{color:#b42318}
.ley{display:inline-flex;gap:10px;flex-wrap:wrap;font-size:12px;color:var(--muted)}
.ley span{display:inline-flex;align-items:center;gap:4px}
.ley i{width:14px;height:14px;border-radius:3px;display:inline-block}
</style></head><body>
<header><h1>¿Los buses siguen la traza?</h1>
<div class="meta">Crudo AVL __DESDE__ → __HASTA__ · __N__ rutas · tramos de 100 m · generado __FECHA____GEOM__</div></header>
<section class="nota">
<p><b>Cada celda</b> es el % de la traza de la ruta con algún problema ese mes (tramos desplazados, en
contrasentido, poco recorridos o saltados). Pasa el cursor para ver los días y el % de registros sobre la
traza; entra a la ruta para ver el mapa.</p>
<p><b>El crudo viene recortado a ~50 m de la traza</b>: un bus que se va por otra calle no aparece lejos,
desaparece. Por eso cuentan los <b>huecos</b>: tramos que los buses en servicio cruzan sin que se les vea.</p>
<p><b>Rayado = muestreo ralo</b>: más de 2 min entre registros de un bus (mediana). Pasa de septiembre en
adelante (de 61 s a 2–17 min según la ruta): esos meses sirven para ver por dónde pasan los buses, pero un
desvío corto ya no se distingue de un hueco normal entre registros.</p>
<div class="ley"><span><i style="background:var(--c0)"></i>&lt; 2 %</span><span><i style="background:var(--c1)"></i>2–5 %</span>
<span><i style="background:var(--c2)"></i>5–15 %</span><span><i style="background:var(--c3)"></i>≥ 15 %</span>
<span><i style="background:var(--nodata)"></i>sin registros, o tan pocos que no se evalúa (&lt; 1,000 o &lt; 5 días)</span><span><i class="ralo" style="background-color:var(--c0)"></i>muestreo ralo</span></div>
</section>
<div class="envoltura"><table>
<thead><tr><th>Ruta</th><th>Traza (km)</th><th>Registros</th>__CABECERA__<th>Zonas</th><th>Zonas principales (meses con problema)</th></tr></thead>
<tbody>
__FILAS__
</tbody></table></div>
<p class="meta">Tablas: <a href="resumen_mensual.csv">resumen_mensual.csv</a> (ruta × mes) ·
<a href="zonas.csv">zonas.csv</a> · <a href="tramos_por_mes.csv">tramos_por_mes.csv</a> (ruta × mes × tramo de 100 m).
Método y umbrales: docstring de <code>mapas_traza.py</code>.</p>
</body></html>
"""

PLANTILLA_RUTA = """<!doctype html>
<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ruta __RUTA__ · traza por mes</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>""" + ESTILO_BASE + """
body{padding:14px 16px 40px;max-width:1600px;margin:0 auto}
header{display:flex;align-items:baseline;gap:14px;flex-wrap:wrap;margin-bottom:10px}
.barra{display:flex;flex-wrap:wrap;gap:10px 18px;align-items:center;margin:8px 0}
.grupo{display:flex;flex-wrap:wrap;gap:4px}
button{font:inherit;font-size:13px;border:1px solid var(--line);background:var(--panel);color:var(--ink);
 border-radius:6px;padding:4px 9px;cursor:pointer}
button:hover{border-color:#9aa6b8}
button.on{background:var(--ink);color:#fff;border-color:var(--ink)}
button.vacio{color:#a3abb8}
button.ralo{background-image:repeating-linear-gradient(45deg,rgba(28,34,48,.13) 0 2px,transparent 2px 6px)}
button.ralo.on{background-image:repeating-linear-gradient(45deg,rgba(255,255,255,.22) 0 2px,transparent 2px 6px)}
.aviso{margin-top:6px;padding:6px 8px;border-radius:6px;background:#fff4e0;color:#7a4b00;font-size:12px}
main{display:grid;grid-template-columns:minmax(0,1fr) 340px;gap:12px}
#mapa{height:64vh;min-height:430px;border-radius:8px;border:1px solid var(--line)}
aside{display:flex;flex-direction:column;gap:12px;min-width:0}
.caja{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:10px 12px}
#resumen table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums;font-size:13px}
#resumen td{padding:2px 0}
#resumen td:last-child{text-align:right}
#zonas{list-style:none;margin:0;padding:0;max-height:34vh;overflow:auto}
#zonas li{padding:6px 4px;border-bottom:1px solid var(--line);cursor:pointer;font-size:13px}
#zonas li:hover{background:#f0f3f8}
.chip{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px;vertical-align:-1px}
.tirita{display:inline-flex;gap:1px;margin-left:4px;vertical-align:-1px}
.tirita i{width:6px;height:10px;display:inline-block;border-radius:1px}
#leyenda{display:flex;flex-wrap:wrap;gap:10px;font-size:12px;color:var(--muted);align-items:center}
.grad{width:180px;height:10px;border-radius:3px;display:inline-block}
.tira{margin-top:12px}
.bus{margin-top:12px}
.bus-cab{display:flex;flex-wrap:wrap;gap:8px 14px;align-items:center;margin-bottom:4px}
.bus-cab h2{margin:0}
.bus select{font:inherit;font-size:13px;padding:3px 6px;border:1px solid var(--line);border-radius:6px;background:var(--panel);color:var(--ink);max-width:100%}
.bus label{font-size:13px;color:var(--muted);display:inline-flex;gap:5px;align-items:center}
#bus-t{width:100%;margin:6px 0 2px}
#bus-hora{font-size:13px;min-height:20px;margin-bottom:4px}
#bus-cob{font-size:13px;color:var(--muted);margin-top:2px}
#bus-diag{width:100%;display:block;cursor:ew-resize}
.bus-ley{font-size:12px;margin-top:4px}
#tira{width:100%;display:block;cursor:crosshair}
#tip{position:fixed;pointer-events:none;background:#1c2230;color:#fff;font-size:12px;padding:6px 8px;
 border-radius:6px;display:none;z-index:2000;max-width:300px;line-height:1.4}
.leaflet-tooltip{font-size:12px;line-height:1.4}
.flecha svg{display:block}
.gris{filter:grayscale(1) brightness(1.05) contrast(.92)}
.km{font-size:10px;color:#1c2230;background:rgba(255,255,255,.85);border-radius:3px;padding:0 3px;white-space:nowrap}
.ayuda{font-size:13px;color:var(--muted);max-width:1100px}
.ayuda b{color:var(--ink)}
@media (max-width:900px){main{grid-template-columns:1fr}#mapa{height:56vh}}
</style></head><body>
<header><a href="index.html">← red</a><h1>Ruta __RUTA__</h1><span class="meta" id="meta"></span></header>
<div class="barra"><div class="grupo" id="meses"></div><div class="grupo" id="metricas"></div></div>
<div id="leyenda"></div>
<main>
  <div id="mapa"></div>
  <aside>
    <div class="caja"><h2 id="tit-resumen">Resumen</h2><div id="resumen"></div></div>
    <div class="caja"><h2>Zonas <span class="meta" style="font-weight:400">(clic: ir y ver su peor mes)</span></h2><ol id="zonas"></ol></div>
  </aside>
</main>
<section class="caja bus">
  <div class="bus-cab">
    <h2>Un bus, paso a paso</h2>
    <select id="bus-sel" title="días de bus del mes elegido arriba"></select>
    <button id="bus-play">▶ Reproducir</button>
    <span class="grupo"><button id="bus-hprev" title="ir al hueco anterior">◀ hueco</button><button id="bus-hnext" title="ir al siguiente hueco y acercar el mapa">hueco ▶</button></span>
    <span id="bus-nh" class="meta"></span>
    <label>velocidad <select id="bus-vel"><option value="2">2 min/s</option><option value="5" selected>5 min/s</option><option value="15">15 min/s</option><option value="30">30 min/s</option></select></label>
    <label>estela <select id="bus-ven"><option value="900">15 min</option><option value="1800" selected>30 min</option><option value="3600">1 h</option><option value="86400">todo el día</option></select></label>
    <label><input type="checkbox" id="bus-seguir" checked> seguir al bus</label>
  </div>
  <div id="bus-cob"></div>
  <input type="range" id="bus-t" min="0" max="1" step="30" value="0">
  <div id="bus-hora"></div>
  <canvas id="bus-diag"></canvas>
  <div class="meta bus-ley">Se eligen días de bus que recorren ≥ 90 % de la traza en la mayoría de sus vueltas (si en el mes ninguno
    llega, el 10 % que más recorre; nunca menos de 50 %), de vehículos distintos; de ellos, los de más registros. La barra oscura a la izquierda del diagrama marca los km
    que ese bus se salta en más de la mitad de sus vueltas. El deslizador mueve la hora; en el mapa se ven la posición del bus (amarillo) y su estela. En el diagrama,
    en qué km de la traza va el bus a cada hora: cada vuelta es un diente de sierra.
    <span style="color:#d62728">●</span> hueco: el bus deja de verse y reaparece más adelante, o se salta una vuelta de la traza (lo que cuenta en «saltos»; si los
    puntos rojos caen en fila, siempre en el mismo km, ahí los buses van por otra calle; «hueco ▶» lleva a cada uno en el mapa) · <span style="color:#8a94a6">—</span> continuo ·
    puntos: <span style="color:#d62728">a &gt; 30 m</span>, <span style="color:#8e44ad">contrasentido</span> ·
    franjas de color: las zonas de la ruta · amarillo: la estela que se ve en el mapa. Clic o arrastre en el diagrama para ir a esa hora.</div>
</section>
<section class="caja tira"><h2>Tramo × mes <span class="meta" style="font-weight:400">(cada columna, 100 m de traza desde el km 0; clic: ver ese mes y ese tramo)</span></h2>
<canvas id="tira"></canvas></section>
<div id="tip"></div>
<section class="ayuda">
<p><b>Cómo se lee.</b> La traza se parte en tramos de 100 m (km 0 = inicio de la polilínea; las flechas dan
su sentido). Cada registro AVL se empareja con el punto de la traza más cercano <i>en su mismo sentido</i>
(rumbo ±60°), así que en las calles que la traza recorre de ida y de vuelta cada registro cae en su paso.</p>
<p><b>Pasos</b>: veces que un bus entra al tramo avanzando por la traza (registros seguidos a ≤ 150 s, o a
2.5 × el intervalo del mes si es mayor), entre la mediana de los tramos de la ruta ese mes. <b>Saltos</b>: de los
buses que cruzan el tramo, la parte que lo cruza sin que se le vea (el siguiente registro llega más tarde que eso,
hasta 30 min después, más adelante en la traza y a velocidad de bus), o que se salta una vuelta o un ramal de la
traza: dos registros seguidos entre los que la traza avanza más de 250 m y más del doble de lo que el bus se movió.
Como el crudo viene recortado a ~50 m de la traza, un tramo con muchos saltos quiere decir que los buses van por
otra calle. <b>Desplazamiento</b>: mediana de la distancia lateral de los registros en movimiento (+ a la izquierda
del sentido); la línea punteada negra es ese «camino observado» donde pasa de 6 m. <b>Contrasentido</b>:
registros cerca de la traza sin ningún tramo cercano en su sentido.</p>
<p><b>Estado</b> (gana el de más a la derecha): desplazada (|desv.| ≥ 12 m o ≥ 30 % a > 30 m) · contrasentido
(≥ 30 %) · poco recorrida (pasos &lt; 25 % de la mediana) · saltada (saltos ≥ 50 % y 30 puntos por encima del
tramo típico del mes).</p>
<p><b>Meses con ≈ (botón rayado)</b>: muestreo ralo, más de 2 min entre registros de un bus (de septiembre en
adelante). Los umbrales de tiempo se ajustan al intervalo de cada mes, pero un desvío corto ya no se distingue
de un hueco normal: sirven para ver por dónde pasan los buses, no para fechar cambios chicos.</p>
</section>
<script id="datos" type="application/json">__DATOS__</script>
<script>
(function(){
const D = JSON.parse(document.getElementById('datos').textContent);
const M = D.meses.length, NB = D.nb, TODO = M;
const EST_COL = ['#c9ced7','#2e9e5b','#d9a400','#8e44ad','#ef7d1a','#d62728'];
const EST_NOM = ['sin datos','ok','desplazada','contrasentido','poco recorrida','saltada'];
const ESC = {
  pasos_rel: [[0,'#d62728'],[0.25,'#ef7d1a'],[0.6,'#e9c46a'],[0.9,'#8cc97c'],[1.15,'#2e9e5b'],[2,'#1b6e8f']],
  salto: [[0,'#2e9e5b'],[0.15,'#a7d38c'],[0.3,'#e9c46a'],[0.5,'#ef7d1a'],[1,'#a50f15']],
  lat: [[-30,'#b2182b'],[-12,'#ef8a62'],[-6,'#b5d99c'],[0,'#2e9e5b'],[6,'#b5d99c'],[12,'#67a9cf'],[30,'#2166ac']],
  contra: [[0,'#2e9e5b'],[0.1,'#b5d99c'],[0.3,'#b07cc6'],[1,'#4a1468']]
};
const LEY = {
  pasos_rel: ['× la mediana de la ruta en el mes', ['0','0.25','0.6','1','2']],
  salto: ['parte de los cruces sin registro', ['0 %','15 %','30 %','50 %','100 %']],
  lat: ['m (− derecha · + izquierda del sentido)', ['−30','−12','0','+12','+30']],
  contra: ['de los registros del tramo', ['0 %','10 %','30 %','100 %']]
};
const METS = [['estado','Estado'],['pasos_rel','Pasos'],['salto','Saltos'],['lat','Desplazamiento'],['contra','Contrasentido']];
const PCOL = ['#2f6fdf','#f0a830','#d62728','#8e44ad','#111111'];
const PNOM = ['a ≤ 15 m','15–30 m','a > 30 m','contrasentido','a > 80 m'];
let mes = TODO, met = 'estado';

const vacio = m => D.dias[m] === 0 || D.insuf[m];
const nomMes = m => m === TODO ? 'todo el periodo' : D.etiquetas[m] + ' (' + D.meses[m] + ')';
const fmt = (v, d) => (v === null || v === undefined || !isFinite(v)) ? '–' : Number(v).toLocaleString('es-MX', {minimumFractionDigits:d, maximumFractionDigits:d});
const pct = v => (v === null || v === undefined) ? '–' : fmt(v * 100, v < 0.1 ? 1 : 0) + ' %';
function hex(c){ return [1,3,5].map(i => parseInt(c.slice(i, i + 2), 16)); }
function interp(esc, v){
  if (v <= esc[0][0]) return esc[0][1];
  for (let i = 1; i < esc.length; i++){
    if (v <= esc[i][0]){
      const a = esc[i-1], b = esc[i], t = (v - a[0]) / (b[0] - a[0]), x = hex(a[1]), y = hex(b[1]);
      return 'rgb(' + x.map((u, k) => Math.round(u + (y[k] - u) * t)).join(',') + ')';
    }
  }
  return esc[esc.length-1][1];
}
function color(mt, m, b){
  if (vacio(m)) return EST_COL[0];
  if (mt === 'estado') return EST_COL[D.met.estado[m][b]];
  const v = D.met[mt][m][b];
  return v === null ? EST_COL[0] : interp(ESC[mt], v);
}
function textoTramo(b, m){
  const km0 = fmt(b * D.umbrales.tramo_m / 1000, 1), km1 = fmt(Math.min((b + 1) * D.umbrales.tramo_m / 1000, D.L_km), 1);
  let s = '<b>km ' + km0 + '–' + km1 + '</b> · ' + nomMes(m) + '<br>';
  if (vacio(m)) return s + (D.dias[m] === 0 ? 'sin registros' : 'pocos registros (' + fmt(D.resumen[m].registros, 0) + '): no se evalúa');
  const g = k => D.met[k][m][b], e = g('estado');
  return s + '<span class="chip" style="background:' + EST_COL[e] + '"></span>' + EST_NOM[e] + '<br>' +
    'pasos: ' + fmt(g('pasos'), 0) + ' (' + fmt(g('pasos') / D.dias[m], 1) + '/día, ' + fmt(g('pasos_rel'), 2) + '× la mediana)<br>' +
    'huecos: ' + fmt(g('huecos'), 0) + ' → saltos ' + pct(g('salto')) + '<br>' +
    'desplazamiento: ' + (g('lat') === null ? '–' : fmt(g('lat'), 1) + ' m') + ' · a > 30 m: ' + pct(g('lejos')) + '<br>' +
    'contrasentido: ' + pct(g('contra')) + ' · registros: ' + fmt(g('n'), 0);
}

// ── mapa ─────────────────────────────────────────────────────────────────────
const map = L.map('mapa', {preferCanvas: true});
// Esri primero: OpenStreetMap contesta 403 a los mosaicos pedidos desde un HTML abierto del
// disco (file://, sin Referer); Esri no pide Referer. El gris de Esri llega al zoom 16.
const ESRI = 'https://server.arcgisonline.com/ArcGIS/rest/services/';
const bases = {
  'Calles (gris)': L.tileLayer(ESRI + 'World_Street_Map/MapServer/tile/{z}/{y}/{x}', {maxZoom: 19, className: 'gris', attribution: 'Esri'}),
  'Calles': L.tileLayer(ESRI + 'World_Street_Map/MapServer/tile/{z}/{y}/{x}', {maxZoom: 19, attribution: 'Esri'}),
  'Gris claro': L.tileLayer(ESRI + 'Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}', {maxZoom: 19, maxNativeZoom: 16, attribution: 'Esri'}),
  'Satélite': L.tileLayer(ESRI + 'World_Imagery/MapServer/tile/{z}/{y}/{x}', {maxZoom: 19, attribution: 'Esri'}),
  'OpenStreetMap (sólo servido por http)': L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {maxZoom: 19, attribution: '© OpenStreetMap'})
};
bases['Calles (gris)'].addTo(map);
// con otra geometría: la traza de data/01_raw (la de antes), debajo de la nueva; se ve donde difieren
const gAnt = L.layerGroup();
if (D.traza_ant){
  L.polyline(D.traza_ant, {color: '#1c2230', weight: 3, dashArray: '3 6', opacity: .85, interactive: false}).addTo(gAnt);
  if (D.cambio_m > 15) gAnt.addTo(map);
}
const gTramos = L.featureGroup().addTo(map);
const tramos = D.coords.map((c, b) => {
  const pl = L.polyline(c, {weight: 6, opacity: .92, color: '#999', lineCap: 'butt'});
  pl.bindTooltip(() => textoTramo(b, mes), {sticky: true, direction: 'top'});
  return pl.addTo(gTramos);
});
map.fitBounds(gTramos.getBounds(), {padding: [16, 16]});
const gFlechas = L.layerGroup().addTo(map);
D.flechas.forEach(f => {
  L.marker([f[0], f[1]], {interactive: false, icon: L.divIcon({className: 'flecha', iconSize: [12, 12], iconAnchor: [6, 6],
    html: '<svg viewBox="0 0 10 10" width="11" height="11" style="transform:rotate(' + f[2] + 'deg)"><path d="M5 0L10 10L5 7L0 10Z" fill="#1c2230" fill-opacity=".6" stroke="#fff" stroke-width=".8"/></svg>'})}).addTo(gFlechas);
  if (f[3] % 5 === 0) L.marker([f[0], f[1]], {interactive: false, icon: L.divIcon({className: '', iconAnchor: [-8, 8],
    html: '<span class="km">' + (f[3] === 0 ? '<b>km 0</b>' : 'km ' + f[3]) + '</span>'})}).addTo(gFlechas);
});
const gParadas = L.layerGroup();
D.paradas.forEach(p => L.circleMarker([p[0], p[1]], {radius: 3, color: '#1c2230', weight: 1, fillColor: '#fff', fillOpacity: 1})
  .bindTooltip(p[2] + '. ' + p[3]).addTo(gParadas));
const gObs = L.layerGroup().addTo(map), gPuntos = L.layerGroup(), gFuera = L.layerGroup().addTo(map);
const marca = L.circleMarker([0, 0], {radius: 11, color: '#1c2230', weight: 2, fill: false, interactive: false});

// ── un bus, paso a paso ──────────────────────────────────────────────────────
const gBus = L.layerGroup().addTo(map);
const selBus = document.getElementById('bus-sel'), rngT = document.getElementById('bus-t'),
  btnPlay = document.getElementById('bus-play'), selVel = document.getElementById('bus-vel'),
  selVen = document.getElementById('bus-ven'), chkSeguir = document.getElementById('bus-seguir'),
  lblHora = document.getElementById('bus-hora'), cvB = document.getElementById('bus-diag'),
  btnHp = document.getElementById('bus-hprev'), btnHn = document.getElementById('bus-hnext'), lblNh = document.getElementById('bus-nh');
const CNOM = ['en la traza (≤ 15 m)', 'a 15–30 m de la traza', 'a > 30 m de la traza', 'en contrasentido', 'a > 80 m de la traza'];
const hhmm = s => String(Math.floor(s / 3600)).padStart(2, '0') + ':' + String(Math.floor(s % 3600 / 60)).padStart(2, '0');
const kmTxt = k => k === null ? '–' : 'km ' + fmt(k, 1);
const pctCob = c => c >= 1 ? '100' : fmt(Math.floor(c * 1000) / 10, 1);
let busOpc = [], bus = null, tBus = 0, reloj = null, mesBus = null;

function opcionesBus(){
  busOpc = mes === TODO ? D.buses.map(b => b[0]).filter(Boolean) : D.buses[mes];
  selBus.innerHTML = busOpc.length
    ? busOpc.map((b, i) => '<option value="' + i + '">' + b.veh + ' · ' + b.sem + ' ' + b.dia + ' · recorre ' + pctCob(b.cob) +
        ' % de la traza · ' + b.vueltas + ' vueltas · ' + hhmm(b.t[0]) + '–' + hhmm(b.t[b.t.length - 1]) + ' · ' + b.n + ' registros</option>').join('')
    : '<option>' + (vacio(mes) ? 'sin registros suficientes este mes' : 'ningún bus recorre ≥ 50 % de la traza este mes') + '</option>';
  elegirBus(0);
}
function elegirBus(i){
  parar();
  bus = busOpc[i] || null;
  if (bus){
    rngT.min = bus.t[0]; rngT.max = bus.t[bus.t.length - 1];
    tBus = Math.min(bus.t[0] + 3600, bus.t[bus.t.length - 1]); rngT.value = tBus;
  }
  rngT.disabled = btnPlay.disabled = !bus;
  const nh = bus ? bus.l.filter(x => x === 2).length : 0;
  btnHp.disabled = btnHn.disabled = !nh;
  lblNh.textContent = bus ? nh + (nh === 1 ? ' hueco' : ' huecos') + ' este día' : '';
  document.getElementById('bus-cob').innerHTML = !bus ? '' : 'Recorre el <b>' + pctCob(bus.cob) + ' %</b> de la traza en la mayoría de sus ' +
    bus.vueltas + ' vueltas' + (bus.sin.length ? ' · se salta casi siempre ' + bus.sin.map(r => 'km ' + fmt(r[0], 1) + '–' + fmt(r[1], 1)).join(', ')
      : ' · pasa por toda la traza en casi todas');
  pintarBus();
}
function irHueco(dir){                   // salta al hueco siguiente/anterior y acerca el mapa
  if (!bus) return;
  parar();
  const hs = [];
  for (let i = 0; i < bus.l.length; i++) if (bus.l[i] === 2) hs.push(i);
  if (!hs.length) return;
  let k = dir > 0 ? hs.find(i => bus.t[i + 1] > tBus) : hs.slice().reverse().find(i => bus.t[i + 1] < tBus);
  if (k === undefined) k = dir > 0 ? hs[0] : hs[hs.length - 1];
  tBus = bus.t[k + 1]; rngT.value = tBus;
  if (!map.hasLayer(gBus)) gBus.addTo(map);
  pintarBus();
  map.fitBounds(L.latLngBounds([[bus.la[k], bus.lo[k]], [bus.la[k + 1], bus.lo[k + 1]]]).pad(0.8), {maxZoom: 17});
}
btnHp.addEventListener('click', () => irHueco(-1));
btnHn.addEventListener('click', () => irHueco(1));
function idxEn(t){                      // el último registro con hora ≤ t
  if (!bus || t < bus.t[0]) return -1;
  let a = 0, b = bus.t.length - 1;
  while (a < b){ const k = (a + b + 1) >> 1; if (bus.t[k] <= t) a = k; else b = k - 1; }
  return a;
}
function pintarBus(){
  gBus.clearLayers();
  const i1 = idxEn(tBus);
  if (i1 < 0){ lblHora.textContent = ''; dibujarDiag(); return; }
  let i0 = i1;
  while (i0 > 0 && bus.t[i0 - 1] >= tBus - (+selVen.value)) i0--;
  for (let i = i0; i < i1; i++){
    const a = [bus.la[i], bus.lo[i]], b = [bus.la[i + 1], bus.lo[i + 1]];
    if (bus.l[i] === 1) L.polyline([a, b], {color: '#1c2230', weight: 3, opacity: .7, interactive: false}).addTo(gBus);
    else if (bus.l[i] === 2) L.polyline([a, b], {color: '#d62728', weight: 3.5, dashArray: '7 6'})
      .bindTooltip('hueco: ' + fmt((bus.t[i + 1] - bus.t[i]) / 60, 0) + ' min sin registros (' + hhmm(bus.t[i]) + '–' + hhmm(bus.t[i + 1]) +
        ', ' + kmTxt(bus.km[i]) + ' → ' + kmTxt(bus.km[i + 1]) + ')', {sticky: true}).addTo(gBus);
  }
  for (let i = i0; i <= i1; i++)
    L.circleMarker([bus.la[i], bus.lo[i]], {radius: 3.5, color: '#fff', weight: 1, fillColor: PCOL[bus.c[i]], fillOpacity: 1})
      .bindTooltip(hhmm(bus.t[i]) + ' · ' + kmTxt(bus.km[i]) + ' · ' + CNOM[bus.c[i]]).addTo(gBus);
  const p = [bus.la[i1], bus.lo[i1]];
  L.circleMarker(p, {radius: 9, color: '#1c2230', weight: 2.5, fillColor: '#ffd400', fillOpacity: 1})
    .bindTooltip(bus.veh + ' · ' + hhmm(bus.t[i1]) + ' · ' + kmTxt(bus.km[i1])).addTo(gBus);
  if (chkSeguir.checked && map.hasLayer(gBus) && !map.getBounds().pad(-0.15).contains(p)) map.panTo(p);
  const sin = tBus - bus.t[i1], h = i1 < bus.l.length && bus.l[i1] === 2;
  lblHora.innerHTML = '<b>' + hhmm(tBus) + '</b> · ' + kmTxt(bus.km[i1]) + ' · ' + CNOM[bus.c[i1]] +
    (sin > 150 ? ' · <span style="color:' + (h ? '#d62728' : 'var(--muted)') + '">' + fmt(sin / 60, 0) + ' min sin registros' +
      (h ? ' (hueco: reaparece en ' + kmTxt(bus.km[i1 + 1]) + ')' : '') + '</span>' : '');
  dibujarDiag();
}
function parar(){ if (reloj){ clearInterval(reloj); reloj = null; } btnPlay.textContent = '▶ Reproducir'; }
btnPlay.onclick = () => {
  if (reloj){ parar(); return; }
  if (!bus) return;
  if (!map.hasLayer(gBus)) gBus.addTo(map);
  if (tBus >= bus.t[bus.t.length - 1]) tBus = bus.t[0];
  btnPlay.textContent = '❚❚ Pausa';
  reloj = setInterval(() => {
    tBus = Math.min(tBus + (+selVel.value) * 6, bus.t[bus.t.length - 1]);
    rngT.value = tBus; pintarBus();
    if (tBus >= bus.t[bus.t.length - 1]) parar();
  }, 100);
};
rngT.addEventListener('input', () => { tBus = +rngT.value; pintarBus(); });
selBus.addEventListener('change', () => elegirBus(+selBus.value));
selVen.addEventListener('change', pintarBus);

// diagrama hora × km
const DB = {izq: 48, der: 10, arr: 8, abajo: 22, alto: 220};
function ejesDiag(W){
  const t0 = bus.t[0], t1 = bus.t[bus.t.length - 1], aw = W - DB.izq - DB.der, ah = DB.alto - DB.arr - DB.abajo;
  return {t0, t1, aw, ah, X: t => DB.izq + (t - t0) / Math.max(t1 - t0, 1) * aw, Y: k => DB.arr + ah - k / D.L_km * ah};
}
function dibujarDiag(){
  const dpr = window.devicePixelRatio || 1, W = cvB.clientWidth, H = DB.alto;
  cvB.style.height = H + 'px'; cvB.width = W * dpr; cvB.height = H * dpr;
  const g = cvB.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, W, H);
  g.font = '11px system-ui, sans-serif';
  if (!bus){ g.fillStyle = '#5f6b7d'; g.fillText('sin buses este mes', DB.izq, 20); return; }
  const {t0, t1, aw, ah, X, Y} = ejesDiag(W);
  D.zonas.forEach(z => {                            // franjas: las zonas de la ruta
    g.fillStyle = EST_COL[z.tipo] + '30';
    (z.km1 >= z.km0 ? [[z.km0, z.km1]] : [[z.km0, D.L_km], [0, z.km1]])
      .forEach(([a, b]) => g.fillRect(DB.izq, Y(b), aw, Math.max(Y(a) - Y(b), 2)));
  });
  bus.sin.forEach(([a, b]) => {                     // km que el bus se salta casi siempre: barra a la izquierda
    (b >= a ? [[a, b]] : [[a, D.L_km], [0, b]]).forEach(([p, q]) => {
      g.fillStyle = 'rgba(28,34,48,.07)'; g.fillRect(DB.izq, Y(q), aw, Math.max(Y(p) - Y(q), 2));
      g.fillStyle = '#1c2230'; g.fillRect(DB.izq - 5, Y(q), 4, Math.max(Y(p) - Y(q), 2));
    });
  });
  const tv = Math.max(tBus - (+selVen.value), t0);   // la estela que se ve en el mapa
  g.fillStyle = 'rgba(255,212,0,.22)'; g.fillRect(X(tv), DB.arr, Math.max(X(tBus) - X(tv), 1), ah);
  g.strokeStyle = '#e2e5ea'; g.lineWidth = 1; g.fillStyle = '#5f6b7d';
  g.textAlign = 'right'; g.textBaseline = 'middle';
  const pk = D.L_km > 40 ? 10 : 5;
  for (let k = 0; k <= D.L_km; k += pk){
    g.beginPath(); g.moveTo(DB.izq, Y(k)); g.lineTo(W - DB.der, Y(k)); g.stroke(); g.fillText('km ' + k, DB.izq - 4, Y(k));
  }
  g.textAlign = 'center'; g.textBaseline = 'top';
  const ph = (t1 - t0) > 10 * 3600 ? 2 : 1;
  for (let h = Math.ceil(t0 / 3600); h * 3600 <= t1; h++){
    if (h % ph) continue;
    const x = X(h * 3600);
    g.beginPath(); g.moveTo(x, DB.arr); g.lineTo(x, DB.arr + ah); g.stroke(); g.fillText(h + ':00', x, DB.arr + ah + 5);
  }
  const L_ = D.L_km;
  for (let i = 0; i < bus.t.length - 1; i++){
    const k1 = bus.km[i], k2 = bus.km[i + 1], l = bus.l[i];
    if (k1 === null || k2 === null || l === 0) continue;
    g.setLineDash(l === 2 ? [5, 3] : []);
    g.strokeStyle = l === 2 ? '#d62728' : '#8a94a6'; g.lineWidth = l === 2 ? 2.2 : 1.2;
    const xa = X(bus.t[i]), xb = X(bus.t[i + 1]);
    g.beginPath();
    if (D.cerrada && k2 - k1 < -L_ / 2){             // cruza el km 0 hacia adelante
      const f = (L_ - k1) / (L_ - k1 + k2), xm = xa + (xb - xa) * f;
      g.moveTo(xa, Y(k1)); g.lineTo(xm, Y(L_)); g.moveTo(xm, Y(0)); g.lineTo(xb, Y(k2));
    } else if (Math.abs(k2 - k1) <= L_ / 2){ g.moveTo(xa, Y(k1)); g.lineTo(xb, Y(k2)); }
    g.stroke();
  }
  g.setLineDash([]);
  for (let i = 0; i < bus.l.length; i++){            // cada hueco, un punto rojo: en fila si cae siempre en el mismo km
    if (bus.l[i] !== 2 || bus.km[i] === null || bus.km[i + 1] === null) continue;
    let km = (bus.km[i] + bus.km[i + 1]) / 2;
    if (D.cerrada && bus.km[i + 1] < bus.km[i]) km = ((bus.km[i] + bus.km[i + 1] + D.L_km) / 2) % D.L_km;
    g.beginPath(); g.arc((X(bus.t[i]) + X(bus.t[i + 1])) / 2, Y(km), 4.5, 0, 2 * Math.PI);
    g.fillStyle = '#d62728'; g.fill(); g.strokeStyle = '#fff'; g.lineWidth = 1.2; g.stroke();
  }
  for (let i = 0; i < bus.t.length; i++){
    if (bus.km[i] === null) continue;
    g.fillStyle = PCOL[bus.c[i]];
    const r = bus.c[i] === 0 ? 1.4 : 2.2;
    g.fillRect(X(bus.t[i]) - r, Y(bus.km[i]) - r, 2 * r, 2 * r);
  }
  g.strokeStyle = '#1c2230'; g.lineWidth = 1.5;
  g.beginPath(); g.moveTo(X(tBus), DB.arr); g.lineTo(X(tBus), DB.arr + ah); g.stroke();
}
function tDesdeX(ev){
  const r = cvB.getBoundingClientRect(), {t0, t1} = ejesDiag(r.width);
  const f = (ev.clientX - r.left - DB.izq) / (r.width - DB.izq - DB.der);
  return Math.round(t0 + Math.min(Math.max(f, 0), 1) * (t1 - t0));
}
let arrastre = false;
cvB.addEventListener('mousedown', ev => { if (!bus) return; arrastre = true; parar(); tBus = tDesdeX(ev); rngT.value = tBus; pintarBus(); });
window.addEventListener('mouseup', () => { arrastre = false; });
cvB.addEventListener('mousemove', ev => {
  if (!bus) return;
  const t = tDesdeX(ev);
  if (arrastre){ tBus = t; rngT.value = tBus; pintarBus(); }
  const i = idxEn(t);
  if (i < 0){ tip.style.display = 'none'; return; }
  const h = i < bus.l.length && bus.l[i] === 2;
  tip.innerHTML = '<b>' + hhmm(t) + '</b><br>último registro ' + hhmm(bus.t[i]) + ' · ' + kmTxt(bus.km[i]) + '<br>' + CNOM[bus.c[i]] +
    (h ? '<br><span style="color:#ff8a80">hueco de ' + fmt((bus.t[i + 1] - bus.t[i]) / 60, 0) + ' min: reaparece en ' + kmTxt(bus.km[i + 1]) + '</span>' : '');
  tip.style.display = 'block';
  tip.style.left = Math.min(ev.clientX + 14, window.innerWidth - 310) + 'px';
  tip.style.top = Math.max(4, ev.clientY - tip.offsetHeight - 10) + 'px';
});
cvB.addEventListener('mouseleave', () => { tip.style.display = 'none'; });
window.addEventListener('resize', () => { if (bus) dibujarDiag(); });
L.control.layers(bases, Object.assign({
  'Traza por tramos': gTramos, 'Sentido y km': gFlechas, 'Paradas': gParadas,
  'Camino observado (desv. ≥ 6 m)': gObs, 'Registros del mes (muestra)': gPuntos,
  'Un bus, paso a paso (estela)': gBus, 'Registros a > 80 m': gFuera
}, D.traza_ant ? {'Traza anterior (data/01_raw, punteada)': gAnt} : {}), {collapsed: window.innerWidth < 900}).addTo(map);

function dibujarMapa(){
  const sinDatos = vacio(mes);
  tramos.forEach((pl, b) => pl.setStyle({color: color(met, mes, b), opacity: sinDatos ? .55 : .92}));
  gObs.clearLayers();
  if (!sinDatos) D.coords.forEach((c, b) => {
    const v = D.met.lat[mes][b];
    if (v === null || Math.abs(v) < 6) return;
    const t = D.tan[b], dx = -t[1] * v, dy = t[0] * v;
    L.polyline(c.map(p => [p[0] + dy / D.KY, p[1] + dx / D.KX]), {color: '#111', weight: 2, dashArray: '4 4', interactive: false}).addTo(gObs);
  });
  gPuntos.clearLayers();
  const pts = mes === TODO ? D.puntos.flatMap(p => p.slice(0, Math.ceil(3000 / M))) : D.puntos[mes];
  pts.forEach(p => L.circleMarker([p[0], p[1]], {radius: 2.5, stroke: false, fillColor: PCOL[p[2]], fillOpacity: .8, interactive: false}).addTo(gPuntos));
  gFuera.clearLayers();
  D.fuera.filter(f => mes === TODO || f[2] === mes).forEach(f =>
    L.circleMarker([f[0], f[1]], {radius: Math.min(3 + Math.sqrt(f[3]), 12), color: '#111', weight: 1, fillColor: '#111', fillOpacity: .5})
      .bindTooltip(f[3] + ' registros a > 80 m de la traza · ' + D.etiquetas[f[2]]).addTo(gFuera));
}

// ── controles ────────────────────────────────────────────────────────────────
const cMeses = document.getElementById('meses'), cMets = document.getElementById('metricas');
[TODO].concat([...Array(M).keys()]).forEach(m => {
  const b = document.createElement('button');
  b.textContent = m === TODO ? 'Todo' : D.etiquetas[m];
  const r = D.resumen[m];
  b.title = m === TODO ? 'todo el periodo' : D.meses[m] + ' · ' + D.dias[m] + ' días con registros' +
    (r.intervalo_mediano_s ? ' · un registro cada ' + r.intervalo_mediano_s + ' s' : '') + (r.muestreo_ralo ? ' (muestreo ralo)' : '') +
    (r.pocos_registros && r.registros ? ' · ' + r.registros + ' registros: no se evalúa' : '');
  if (vacio(m)) b.classList.add('vacio');
  if (m !== TODO && r.muestreo_ralo) b.classList.add('ralo');
  b.onclick = () => { mes = m; actualizar(); };
  b.dataset.m = m;
  cMeses.appendChild(b);
});
METS.forEach(([k, n]) => {
  const b = document.createElement('button');
  b.textContent = n; b.dataset.k = k;
  b.onclick = () => { met = k; actualizar(); };
  cMets.appendChild(b);
});
function leyenda(){
  const el = document.getElementById('leyenda');
  let h = '';
  if (met === 'estado') h = EST_NOM.map((n, i) => '<span><span class="chip" style="background:' + EST_COL[i] + '"></span>' + n + '</span>').join('');
  else {
    const e = ESC[met], lo = e[0][0], hi = e[e.length-1][0];
    const g = e.map(s => s[1] + ' ' + ((s[0] - lo) / (hi - lo) * 100).toFixed(0) + '%').join(',');
    h = '<span>' + LEY[met][1][0] + '</span><span class="grad" style="background:linear-gradient(90deg,' + g + ')"></span><span>' +
      LEY[met][1][LEY[met][1].length-1] + '</span><span>' + LEY[met][0] + '</span>';
  }
  h += '<span style="margin-left:12px">registros: ' + PNOM.map((n, i) => '<span class="chip" style="background:' + PCOL[i] + ';border-radius:50%"></span>' + n).join(' ') + '</span>';
  el.innerHTML = h;
}
function resumen(){
  const r = D.resumen[mes];
  document.getElementById('tit-resumen').textContent = 'Resumen · ' + nomMes(mes);
  const fila = (a, b) => '<tr><td>' + a + '</td><td>' + b + '</td></tr>';
  let h = '<table>' + fila('registros', fmt(r.registros, 0)) + fila('días con registros', fmt(r.dias, 0)) + fila('vehículos', fmt(r.vehiculos, 0));
  if (r.registros){
    h += fila('en la traza, en su sentido', pct(r.pct_en_traza)) + fila('en movimiento a > 30 m', pct(r.pct_lejos_30m)) +
      fila('contrasentido', pct(r.pct_contrasentido)) + fila('a > 80 m', pct(r.pct_fuera_80m)) +
      fila('desv. lateral mediana', fmt(r.desv_lateral_mediana_m, 1) + ' m') + fila('pasos por tramo y día (mediana)', fmt(r.pasos_dia_mediana, 1)) +
      fila('un registro cada (mediana)', fmt(r.intervalo_mediano_s, 0) + ' s') + fila('saltos en un tramo típico', pct(r.salto_base));
    h += '<tr><td colspan="2" style="padding-top:6px;color:var(--muted)">km de traza por estado</td></tr>';
    EST_NOM.forEach((n, i) => { const k = 'km_' + n.replace(' ', '_'); if (r[k]) h += fila('<span class="chip" style="background:' + EST_COL[i] + '"></span>' + n, fmt(r[k], 1)); });
  }
  h += '</table>';
  if (mes !== TODO && r.muestreo_ralo) h += '<div class="aviso">Muestreo ralo: un registro cada ' + fmt(r.intervalo_mediano_s, 0) +
    ' s. Los pasos se miden con registros hasta ' + fmt(Math.min(Math.max(2.5 * r.intervalo_mediano_s, 150), 1200) / 60, 0) +
    ' min aparte, así que un desvío corto no se ve.</div>';
  if (mes !== TODO && r.pocos_registros && r.registros) h += '<div class="aviso">Tan pocos registros que el mes no se evalúa (hacen falta ≥ 1,000 en ≥ 5 días).</div>';
  if (r.fuera_de_zonas) h += '<div class="aviso">Más de la mitad de la traza sale con problema este mes: no cuenta para las zonas.</div>';
  document.getElementById('resumen').innerHTML = h;
}
function zonas(){
  const ol = document.getElementById('zonas');
  ol.innerHTML = '';
  if (!D.zonas.length){ ol.innerHTML = '<li style="cursor:default">ninguna</li>'; return; }
  D.zonas.forEach(z => {
    const li = document.createElement('li');
    const tir = z.meses.map((v, m) => '<i style="background:' + (vacio(m) || D.resumen[m].fuera_de_zonas ? '#e3e6ea' : v ? EST_COL[z.tipo] : '#cfe7cf') +
      '" title="' + D.etiquetas[m] + (D.resumen[m].fuera_de_zonas ? ' (no cuenta)' : v ? ': con problema' : ': bien') + '"></i>').join('');
    li.innerHTML = '<span class="chip" style="background:' + EST_COL[z.tipo] + '"></span><b>km ' + fmt(z.km0, 1) + '–' + fmt(z.km1, 1) + '</b> · ' +
      EST_NOM[z.tipo] + '<span class="tirita">' + tir + '</span><br><span class="meta">' + (z.junto_km0 ? 'junto al km 0 (cierre del lazo) · ' : '') +
      [z.salto_max !== null ? 'saltos hasta ' + pct(z.salto_max) : '', z.desv_max_m !== null ? 'desv. hasta ' + fmt(z.desv_max_m, 0) + ' m' : '',
       z.contra_max ? 'contrasentido hasta ' + pct(z.contra_max) : '', z.pasos_rel_min !== null ? 'pasos desde ' + fmt(z.pasos_rel_min, 2) + '×' : ''].filter(Boolean).join(' · ') + '</span>';
    li.onclick = () => {
      mes = z.peor; actualizar();
      map.fitBounds(L.latLngBounds(z.b.flatMap(b => D.coords[b])), {padding: [60, 60], maxZoom: 17});
    };
    ol.appendChild(li);
  });
}

// ── tira tramo × mes ─────────────────────────────────────────────────────────
const cv = document.getElementById('tira'), tip = document.getElementById('tip');
const FILA = 17, SEP = 8, IZQ = 52, DER = 8, ARR = 4, ABAJO = 22;
const filaY = m => m === TODO ? ARR + M * FILA + SEP : ARR + m * FILA;
function dibujarTira(){
  const dpr = window.devicePixelRatio || 1, W = cv.clientWidth, H = ARR + (M + 1) * FILA + SEP + ABAJO;
  cv.style.height = H + 'px'; cv.width = W * dpr; cv.height = H * dpr;
  const g = cv.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, W, H);
  const ab = (W - IZQ - DER) / NB;
  g.font = '11px system-ui, sans-serif'; g.textBaseline = 'middle';
  for (let m = 0; m <= M; m++){
    const y = filaY(m);
    g.fillStyle = m === mes ? '#1c2230' : '#5f6b7d';
    g.fillText(m === TODO ? 'Todo' : D.etiquetas[m] + (D.resumen[m].muestreo_ralo ? ' ≈' : ''), 4, y + FILA / 2);
    for (let b = 0; b < NB; b++){ g.fillStyle = color(met, m, b); g.fillRect(IZQ + b * ab, y, ab + 0.6, FILA - 2); }
  }
  const y = filaY(mes);
  g.strokeStyle = '#1c2230'; g.lineWidth = 1.5; g.strokeRect(IZQ - 1, y - 1, W - IZQ - DER + 2, FILA);
  const paso = D.L_km > 40 ? 5 : D.L_km > 15 ? 2 : 1, yb = ARR + (M + 1) * FILA + SEP + 4;
  g.fillStyle = '#5f6b7d'; g.textBaseline = 'top';
  for (let k = 0; k <= D.L_km; k += paso){
    const x = IZQ + k * 1000 / D.umbrales.tramo_m * ab;
    g.fillRect(x, yb - 3, 1, 4); g.fillText(k + (k === 0 ? ' km' : ''), x + 2, yb);
  }
}
function celda(ev){
  const r = cv.getBoundingClientRect(), x = ev.clientX - r.left, y = ev.clientY - r.top;
  const b = Math.floor((x - IZQ) / ((r.width - IZQ - DER) / NB));
  let m = -1;
  for (let k = 0; k <= M; k++) if (y >= filaY(k) && y < filaY(k) + FILA) m = k;
  return (b >= 0 && b < NB && m >= 0) ? [m, b] : null;
}
const centro = b => D.coords[b][Math.floor(D.coords[b].length / 2)];
cv.addEventListener('mousemove', ev => {
  const c = celda(ev);
  if (!c){ tip.style.display = 'none'; marca.remove(); return; }
  tip.innerHTML = textoTramo(c[1], c[0]); tip.style.display = 'block';
  tip.style.left = Math.min(ev.clientX + 14, window.innerWidth - 310) + 'px';
  tip.style.top = Math.max(4, ev.clientY - tip.offsetHeight - 10) + 'px';
  marca.setLatLng(centro(c[1])).addTo(map);
});
cv.addEventListener('mouseleave', () => { tip.style.display = 'none'; marca.remove(); });
cv.addEventListener('click', ev => {
  const c = celda(ev);
  if (!c) return;
  mes = c[0]; actualizar();
  map.setView(centro(c[1]), Math.max(map.getZoom(), 16));
});
window.addEventListener('resize', dibujarTira);

function actualizar(){
  cMeses.querySelectorAll('button').forEach(b => b.classList.toggle('on', +b.dataset.m === mes));
  cMets.querySelectorAll('button').forEach(b => b.classList.toggle('on', b.dataset.k === met));
  dibujarMapa(); dibujarTira(); resumen(); leyenda();
  if (mesBus !== mes){ mesBus = mes; opcionesBus(); }
}
const tot = D.resumen[TODO];
document.getElementById('meta').textContent = 'traza ' + fmt(D.L_km, 1) + ' km' + (D.cerrada ? ' (cerrada)' : '') + ' · ' +
  fmt(tot.registros, 0) + ' registros · ' + D.etiquetas[0] + '–' + D.etiquetas[M-1] + ' · ' + D.zonas.length + ' zonas' +
  (D.geom ? ' · traza ' + D.traza + (D.traza_nombre ? ' «' + D.traza_nombre + '»' : '') + ' de ' + D.geom +
    (D.cambio_m > 15 ? ' (se aparta hasta ' + fmt(D.cambio_m, 0) + ' m de la anterior, punteada)' : ' (a ≤ 15 m de la anterior)') : '');
zonas(); actualizar();
})();
</script>
</body></html>
"""

if __name__ == "__main__":
    sys.exit(main())
