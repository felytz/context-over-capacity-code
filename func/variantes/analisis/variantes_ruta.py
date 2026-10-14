#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
variantes_ruta.py — en las líneas con variante, ¿cada bus va por la principal o por la alterna?

Cuatro líneas del portal tienen dos trazas que comparten casi todo el recorrido y se separan
en una zona (geometría corregida, data/01_raw; medido en el mismo sentido de circulación):

  familia  variantes (group_id)          compartido   zona distinta (por variante)
  1101     1101 / 1101-ALT (110101)       83 %         km 15.8–23.8 (7.9 km): las mismas calles, recorridas
                                            AL REVÉS (7.3 de los 7.9 km)
  1401     1401 / 1402                    82 %         km 25–34.7 en tres tramos (7.7–7.9 km): 6.3–6.5 km son
                                            las mismas calles al revés (Emiliano Zapata / Calle
                                            Segunda contra Av. Francisco Villa / Lázaro Cárdenas)
  1801     1801 / 1801-ALT (180101)       91–92 %      km 36.3–40.3 (3.4–4.0 km): calles distintas
                                            (la principal por Blvd. Morelos y López Portillo)
  1802     1802 / 1802-ALT (180201)       92–93 %      km 4.1–8.0 (3.2–3.8 km): calles distintas (ídem)

(Las 43 trazas, una contra otra: no hay más pares así. 20000 = 2201 es la misma traza; API01
y 2706 son alimentadoras que van encima de la 401 y la 901. Las -UI del portal son sólo
visuales: dos puntos, sin traza.)

En el AVL las dos variantes vienen con el mismo `group_id` (la -ALT no tiene número propio; la
1401 y la 1402 vienen las dos como 1400, que no es ninguna traza: inventario de la BD, 6-oct) y **un mismo bus cambia de ruta a media
jornada**: en el crudo de las 27 rutas, el 24 % de los bus-día tienen al vehículo en dos o
más rutas, en bloques seguidos (entre rutas hermanas, 401↔402, 301↔302, 201↔202, 1001↔1002,
1901↔1902…, el cambio llega sin pausa: mediana 1.1 min entre el último registro de una y el
primero de la otra). Por eso la unidad no es el bus-día sino el **bloque**: el bus-día se
parte donde la evidencia cambia de variante.

Método
------
1. Zonas discriminantes (sólo geometría). Las dos trazas se muestrean cada 10 m con su rumbo;
   un punto de una traza es *exclusivo* si la otra no pasa a ≤ EXCL_M en el mismo sentido
   (±60°). Así cuenta igual una calle que sólo usa una variante y una calle que las dos
   recorren en sentidos opuestos (la 1101 y la 1401 son casi sólo eso).
2. Evidencia por registro. Cada registro se empareja con las dos trazas con el emparejador de
   mapas_traza.py (el punto más cercano *en su sentido*, con `cape`, y continuidad entre
   pasadas). Es evidencia de la variante V si cae a ≤ EVID_D_M de V, en un punto exclusivo de
   V, y la otra (en su sentido) queda al menos EVID_MARGEN_M más lejos.
3. Saltos (para cuando el crudo no ve una variante). El AVL público viene recortado a ~50 m de
   una traza: si el recorte es el de la principal, un bus de la alterna *desaparece* en la zona
   distinta. Se sigue a cada bus sobre la traza V: si de un registro antes de la zona exclusiva
   de V pasa a uno después, sin registros sobre ella, y con su intervalo de muestreo se
   esperaban ≥ SALTO_MIN_ESPERADOS registros ahí, eso es evidencia de la OTRA variante (con
   ese peso, tope TOPE). Sólo si la zona de V *se ve*: con el crudo recortado a la otra traza,
   los buses de V también la saltan (sus registros se borraron). Se mide por zona:
   evidencia directa / (evidencia + peso de los saltos); con < OBS_MIN sus saltos no cuentan
   (en las pruebas: 0.005–0.025 la zona recortada, 0.68–1.0 las demás).
4. Visitas. Los eventos de evidencia de un bus-día a ≤ VISITA_S uno del otro forman una visita
   a la zona (n_principal, n_alterna), partida en trozos de VISITA_MAX_S.
5. Bloques. Por bus-día, Viterbi de dos estados sobre sus visitas: cada visita suma
   min(n_estado, TOPE) − min(n_otro, TOPE) y cada cambio de variante cuesta CAMBIO. Luego un
   bloque cuyas visitas decisivas (≥ DECISIVA netas) abarcan < LAPSO_MIN_CAMBIO_S se funde con
   su vecino: con una zona de unos km hacen falta dos pasadas en vueltas distintas. Quita el
   bus que va o viene del encierro por calles de la otra variante. La frontera entre dos
   bloques es el hueco más largo (≥ HUECO_FRONTERA_S) entre la última visita de uno y la
   primera del otro; si no hay, el paso por el km 0 de la principal (las cuatro familias salen
   de la misma parada); si no, el punto medio.
6. Etiqueta. Cada registro toma la variante de su bloque (los de antes de la primera visita o
   después de la última, la del bloque más cercano). Un bloque con evidencia neta
   < EVID_MIN_BLOQUE, o un bus-día sin visitas, queda «indeterminado» y no entra al df limpio.
   En las pruebas son casi sólo asomos: bus-días de unos minutos (mediana 5–6 registros), el
   bus que pasa por la ruta al empezar o acabar el día y el resto lo hace en otra.

Números. Las -ALT llevan el número de la principal con «01» al final: 1101-ALT = 110101,
1801-ALT = 180101, 1802-ALT = 180201 (así están en data/01_raw desde el 5-oct-2026,
data raw/preparar_01_raw.py; el pipeline sólo carga geometría con clave entera). La 1402 ya
es un número.

Entrada y salida
----------------
  data/01_raw/records_variantes/<grupo>/df_rec_raw_<grupo>_<rango>.parquet
      el crudo de la BD, con las dos variantes juntas (1101, 1400, 1801, 1802). Va
      aparte a propósito: en data/01_raw/records el lote (correr.py, run_servidor.sh) lo
      procesaría como si fuera una sola ruta
  data/01_raw/records/<n>/df_rec_raw_<n>_<rango>.parquet          (n = 1101, 110101, 1401, …)
      el crudo de cada variante, con el mismo rango, donde el pipeline lo busca: las columnas
      del crudo con `group_id` = el número de la variante, y además `group_id_avl` (el que
      traía el AVL), `variante` ("1101-ALT"…), `bloque` y `bloque_evidencia`
  data/02_interim/variantes/<familia>/indeterminados_<rango>.parquet   los indeterminados
  reports/variantes/<familia>/  zonas.csv · zona.png (variantes_mapa.py) · bloques.csv ·
      indeterminados.csv · resumen.md (cruce con el group_id del AVL, alterna por mes y hora,
      indeterminados por motivo)

Prueba con verdad conocida (`--mezcla A B`): junta el crudo de dos rutas reales que comparten
calles (201/202, 401/402, 1001/1002…), les quita el número y las clasifica con sus trazas; la
verdad es el `group_id` original, y como esos buses sí cambian de ruta a media jornada, mide
también los cambios que trae el AVL; pero el `group_id` del AVL a veces contradice por dónde va el
bus, así que la verdad exacta de un cambio son los `--empalmes` (300 por defecto): el mismo día,
un bus que sólo hizo A hasta τ y otro que sólo hizo B desde τ (10–16 h). Para parecerse a las
familias: `--zona-tipo sentido --zona-max-km 8` (≤ 8 km de las mismas calles al revés, como
1101/1401) o `--zona-tipo calles --zona-max-km 4.5` (un desvío entero por calles distintas, como
1801/1802: 1702/1703 y 1501/1502 lo tienen); `--recorte` tira los registros a > 50 m de la
traza A (crudo recortado a la principal). Leen el crudo de data/01_raw/records (ene–ago por
omisión) y van a reports/variantes/pruebas_<desde>_<hasta>/prueba_*/ (metricas.json,
bloques.csv); `--resumen-pruebas` las junta en pruebas.csv / pruebas.md.

Uso
---
    python analisis/variantes_ruta.py --geometria                     # sólo las zonas (sin crudo)
    python analisis/variantes_ruta.py --familias 1101 1801            # necesita su crudo en records_variantes/
    python analisis/variantes_ruta.py --mezcla 1702 1703 --zona-tipo calles --zona-max-km 4.5 --recorte
    python analisis/variantes_ruta.py --resumen-pruebas
    python analisis/variantes_mapa.py 1101                            # la zona en un mapa (sin crudo)

El crudo de las cuatro familias no está en el proyecto (sólo el de las 27 rutas). Se baja de la
BD como el de las demás (`_lib_data.get_db_rq_from_to(..., int_route_rec=1101)`, tabla
mobility_data) y se deja en data/01_raw/records_variantes/<grupo>/ (arriba). `--geometria`
imprime además la caja (lat/lon) de cada zona para comprobar en la BD qué `group_id` pasan por ahí.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
from scipy.spatial import cKDTree

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
import mapas_traza as mt  # noqa: E402

RAW = RAIZ / "data" / "01_raw"
MEZCLADO = RAW / "records_variantes"      # entrada: el crudo de la BD, con las dos variantes juntas
LIMPIO = RAW / "records"                  # salida: el crudo de cada variante, donde lo busca el pipeline
INTERIM = RAIZ / "data" / "02_interim" / "variantes"     # los indeterminados
REPORTES = RAIZ / "reports" / "variantes"

# familia → grupos del AVL que se leen y las dos trazas (principal, alterna), con su número de
# data/01_raw: las -ALT llevan el de la principal con «01» al final (data raw/preparar_01_raw.py)
FAMILIAS = {
    "1101": {"grupos": [1101], "variantes": ("1101", "110101")},
    "1401": {"grupos": [1400], "variantes": ("1401", "1402")},     # en la BD, las dos como 1400 (inventario, 6-oct)
    "1801": {"grupos": [1801], "variantes": ("1801", "180101")},
    "1802": {"grupos": [1802], "variantes": ("1802", "180201")},
}
NOMBRE = {"110101": "1101-ALT", "180101": "1801-ALT", "180201": "1802-ALT"}   # como las llama el portal

EXCL_M = 50.0               # punto exclusivo: la otra traza, en su sentido, a más que esto
TRAMO_MIN_M = 100.0         # zonas exclusivas más cortas que esto no cuentan (cruces, uniones)
EVID_D_M = 25.0             # evidencia: el registro a ≤ esto de la traza…
EVID_MARGEN_M = 25.0        # …y la otra traza (en su sentido) al menos esto más lejos
SALTO_PRE_M = 1500.0        # salto: el registro de antes, a ≤ esto antes de la zona
SALTO_MAX_S = 2700          #   y el de después a ≤ 45 min
SALTO_MIN_ESPERADOS = 2.0   #   registros que se esperaban sobre la zona con su muestreo
EN_TRAZA_M = 40.0           # para seguir al bus sobre una traza
VISITA_S = 900              # eventos a ≤ 15 min: la misma visita a la zona…
VISITA_MAX_S = 1200         #   partida en trozos de 20 min (si la zona es larga, una visita no se come el día)
TOPE = 6                    # lo más que suma una visita
CAMBIO = 8.0                # costo de cambiar de variante dentro del bus-día
DECISIVA = 2                # visita decisiva: ≥ esto de evidencia neta hacia su variante
LAPSO_MIN_CAMBIO_S = 2700   # un bloque nuevo cuyas visitas decisivas abarcan menos que esto se funde con su vecino
OBS_MIN = 0.2               # los saltos sobre una zona cuentan si ahí se ve al menos esto (evidencia / (evidencia + saltos))
EVID_MIN_BLOQUE = 3         # evidencia neta mínima de un bloque para etiquetarlo
HUECO_FRONTERA_S = 600      # frontera entre bloques: el hueco más largo, si pasa de esto
KM0_M = 3000.0              # paso por el km 0: de > L − esto a < esto
RECORTE_M = 50.0            # --recorte: lo que se deja alrededor de la traza A
BD_MIN = 60                 # prueba: bus-días y lados de un cambio con al menos esto de registros (~1 h)
INDET = "indeterminado"


# ═════════════════════════════════════════════════════════════════════════════
#  geometría
# ═════════════════════════════════════════════════════════════════════════════
def en_sentido(Ta: dict, Tb: dict) -> np.ndarray:
    """Por punto de la traza a: distancia a la traza b en el mismo sentido (inf si > 200 m)."""
    d, i = cKDTree(Tb["q"]).query(Ta["q"], k=24, distance_upper_bound=200.0)
    ok = np.isfinite(d)
    ii = np.where(ok, i, 0)
    dif = np.abs((Ta["rumbo"][:, None] - Tb["rumbo"][ii] + 180.0) % 360.0 - 180.0)
    return np.where(ok & (dif <= mt.ANG_MAX), d, np.inf).min(1)


def geometria(variantes: tuple[str, str], geom: Path, zona_max_km: float | None = None, zona_tipo: str = "todo") -> dict:
    T = {v: mt.leer_traza(0, geom, traza=v) for v in variantes}
    sin_sentido = {}
    for v, o in (variantes, variantes[::-1]):
        d = en_sentido(T[v], T[o])
        sin_sentido[v] = cKDTree(T[o]["q"]).query(T[v]["q"])[0]
        excl = d > EXCL_M
        if zona_tipo == "sentido":      # prueba: sólo calles que la otra recorre al revés (como 1101/1401)
            excl &= sin_sentido[v] <= 30
        tramos = [r for r in mt.corridas(excl, T[v]["cerrada"]) if len(r) * mt.PASO_M >= TRAMO_MIN_M]
        if zona_tipo == "calles":       # prueba: un desvío entero por calles que la otra no usa (como 1801/1802):
            # ≥ 90 % de la zona a > 50 m de la otra traza; en sus dos puntas la otra vuelve a juntarse
            tramos = [r for r in tramos if (sin_sentido[v][r] > EXCL_M).mean() >= 0.9]
            if not tramos:
                raise ValueError(f"{v}: no hay un desvío de calles distintas contra {o}")
            if zona_max_km:
                caben = [r for r in tramos if len(r) * mt.PASO_M <= zona_max_km * 1000]
                tramos = [max(caben, key=len) if caben else min(tramos, key=len)]
        elif zona_max_km:
            r = max(tramos, key=len)
            tramos = [r[: int(zona_max_km * 1000 / mt.PASO_M)]]
        excl = np.zeros(len(T[v]["q"]), bool)
        for r in tramos:
            excl[r] = True
        T[v]["excl"], T[v]["tramos"], T[v]["d_otra"] = excl, tramos, d
    for v in variantes:
        T[v]["contrario_m"] = float(sum((sin_sentido[v][r] <= 30).sum() for r in T[v]["tramos"]) * mt.PASO_M)
    return T


def tabla_zonas(T: dict, variantes) -> list[dict]:
    out = []
    for v in variantes:
        s, q = T[v]["s"], T[v]["q"]
        for r in T[v]["tramos"]:
            lat, lon = q[r, 1] / mt.KY, q[r, 0] / mt.KX
            out.append({"variante": v, "km_ini": round(s[r[0]] / 1000, 2), "km_fin": round(s[r[-1]] / 1000, 2),
                        "largo_m": int(len(r) * mt.PASO_M),
                        "lat_min": round(float(lat.min()), 5), "lat_max": round(float(lat.max()), 5),
                        "lon_min": round(float(lon.min()), 5), "lon_max": round(float(lon.max()), 5)})
    return out


# ═════════════════════════════════════════════════════════════════════════════
#  crudo
# ═════════════════════════════════════════════════════════════════════════════
def archivos_crudo(grupos: list[int], carpeta: Path) -> tuple[list[Path], str]:
    """Los parquet de cada grupo y su rango (df_rec_raw_<grupo>_<rango>.parquet: el mismo en todos)."""
    archivos, faltan = [], []
    for g in grupos:
        hay = sorted((carpeta / str(g)).glob(f"df_rec_raw_{g}_*.parquet"))
        archivos += hay
        if not hay:
            faltan.append(g)
    if faltan:
        raise FileNotFoundError(f"no hay crudo de {faltan} en {carpeta.relative_to(RAIZ)}/<grupo>/ (ver el docstring)")
    rangos = {a.stem.split("_", 4)[4] for a in archivos}       # df_rec_raw_<grupo>_<rango>
    if len(rangos) != 1:
        raise ValueError(f"los crudos de {grupos} tienen rangos distintos: {sorted(rangos)}")
    return archivos, rangos.pop()


def leer_crudo(grupos: list[int], desde: date | None, hasta: date | None, carpeta: Path = LIMPIO) -> pl.DataFrame:
    archivos, _ = archivos_crudo(grupos, carpeta)
    q = pl.scan_parquet([str(a) for a in archivos])
    if desde:
        q = q.filter(pl.col("time") >= datetime.combine(desde, datetime.min.time()))
    if hasta:
        q = q.filter(pl.col("time") < datetime.combine(hasta + timedelta(days=1), datetime.min.time()))
    return (q.filter(pl.col("group_id").cast(pl.Int64).is_in(grupos))
            .drop_nulls(["id", "latitude", "longitude", "time"])
            .filter(pl.col("latitude").cast(pl.Float64).is_between(28.0, 30.5)
                    & pl.col("longitude").cast(pl.Float64).is_between(-112.0, -110.0))
            .unique(subset=["id", "time"], keep="first")
            .sort(["id", "time"])
            .collect())


def arreglos(df: pl.DataFrame) -> dict:
    a = df.select(
        x=pl.col("longitude").cast(pl.Float64) * mt.KX,
        y=pl.col("latitude").cast(pl.Float64) * mt.KY,
        cape=pl.col("cape").cast(pl.Float64),
        t=pl.col("time").dt.epoch("s"),
        bd=((pl.col("id") != pl.col("id").shift()) | (pl.col("time").dt.date() != pl.col("time").dt.date().shift()))
        .fill_null(True).cast(pl.Int32).cum_sum() - 1)
    return {k: a[k].to_numpy() for k in a.columns}


# ═════════════════════════════════════════════════════════════════════════════
#  evidencia
# ═════════════════════════════════════════════════════════════════════════════
def emparejar(T: dict, A: dict, v: str) -> tuple[np.ndarray, np.ndarray]:
    j, _ = mt.emparejar(T[v], A["x"], A["y"], A["cape"], A["bd"], A["t"])
    q = T[v]["q"][np.maximum(j, 0)]
    d = np.where(j >= 0, np.hypot(A["x"] - q[:, 0], A["y"] - q[:, 1]), np.inf)
    return j, d


def intervalo_bus_dia(A: dict) -> np.ndarray:
    """Intervalo mediano entre registros de cada bus-día (por registro)."""
    bd, t = A["bd"], A["t"]
    dt = np.diff(t).astype(float)
    mismo = bd[1:] == bd[:-1]
    nb = int(bd.max()) + 1
    med = np.full(nb, 60.0)
    if mismo.any():
        orden = pl.DataFrame({"bd": bd[1:][mismo], "dt": dt[mismo]}).group_by("bd").agg(pl.col("dt").median())
        med[orden["bd"].to_numpy()] = np.maximum(orden["dt"].to_numpy(), 1.0)
    return med


def saltos(T: dict, v: str, j: np.ndarray, d: np.ndarray, A: dict, med: np.ndarray):
    """Pasos de un registro antes de una zona exclusiva de v a uno después, sin registros sobre
    ella: evidencia de la otra variante. Devuelve (bus-día, t medio, peso)."""
    s, L, cerrada = T[v]["s"], T[v]["L"], T[v]["cerrada"]
    idx = np.flatnonzero((j >= 0) & (d <= EN_TRAZA_M))
    if len(idx) < 2:
        return np.empty(0, np.int64), np.empty(0), np.empty(0)
    p, n = idx[:-1], idx[1:]
    ok = A["bd"][p] == A["bd"][n]
    p, n = p[ok], n[ok]
    sp, sn = s[j[p]], s[j[n]]
    dt = (A["t"][n] - A["t"][p]).astype(float)
    av = np.mod(sn - sp, L) if cerrada else sn - sp
    out_b, out_t, out_w = [], [], []
    for r in T[v]["tramos"]:
        a, largo = s[r[0]], len(r) * mt.PASO_M
        pre = np.mod(a - sp, L) if cerrada else a - sp
        cruza = (pre > 0) & (pre <= SALTO_PRE_M) & (av >= pre + largo) & (av <= pre + largo + SALTO_PRE_M) & (dt <= SALTO_MAX_S)
        k = np.flatnonzero(cruza)
        esperados = dt[k] / med[A["bd"][p[k]]] * largo / np.maximum(av[k], 1.0)
        bien = esperados >= SALTO_MIN_ESPERADOS
        k = k[bien]
        out_b.append(A["bd"][p[k]])
        out_t.append((A["t"][p[k]] + A["t"][n[k]]) / 2.0)
        out_w.append(np.minimum(np.round(esperados[bien]), TOPE))
    return np.concatenate(out_b), np.concatenate(out_t), np.concatenate(out_w)


def evidencia(T: dict, variantes, A: dict) -> dict:
    J, D = {}, {}
    for v in variantes:
        J[v], D[v] = emparejar(T, A, v)
    med = intervalo_bus_dia(A)
    ev = {"positiva": {}, "salto": {}}
    b, tt, ww, vv = [], [], [], []
    for k, (v, o) in enumerate((variantes, variantes[::-1])):
        pos = (J[v] >= 0) & (D[v] <= EVID_D_M) & T[v]["excl"][np.maximum(J[v], 0)] & (D[o] >= D[v] + EVID_MARGEN_M)
        ev["positiva"][v] = pos
        i = np.flatnonzero(pos)
        b.append(A["bd"][i]); tt.append(A["t"][i].astype(float)); ww.append(np.ones(len(i))); vv.append(np.full(len(i), k))
        sb, st, sw = saltos(T, v, J[v], D[v], A, med)
        obs = len(i) / max(1.0, len(i) + float(sw.sum()))
        ev["salto"][v] = (len(sb), float(sw.sum()), round(obs, 3))
        if obs < OBS_MIN:       # la zona no se ve (crudo recortado a la otra traza): sus saltos no dicen nada
            sb, st, sw = sb[:0], st[:0], sw[:0]
        b.append(sb); tt.append(st); ww.append(sw); vv.append(np.full(len(sb), 1 - k))   # un salto sobre v es de la otra
    ev["eventos"] = {"bd": np.concatenate(b).astype(np.int64), "t": np.concatenate(tt),
                     "w": np.concatenate(ww), "v": np.concatenate(vv).astype(np.int8)}
    ev["J"], ev["D"] = J, D
    return ev


# ═════════════════════════════════════════════════════════════════════════════
#  visitas → bloques
# ═════════════════════════════════════════════════════════════════════════════
def visitas(E: dict) -> pl.DataFrame:
    e = pl.DataFrame(E).sort(["bd", "t"])
    return (e.with_columns(nueva=((pl.col("bd") != pl.col("bd").shift()) | (pl.col("t") - pl.col("t").shift() > VISITA_S))
                           .fill_null(True))
            .with_columns(g=pl.col("nueva").cast(pl.Int32).cum_sum())
            .with_columns(trozo=((pl.col("t") - pl.col("t").min().over("g")) // VISITA_MAX_S).cast(pl.Int32))
            .with_columns(visita=pl.struct("g", "trozo").rank("dense"))
            .group_by("visita", maintain_order=True)
            .agg(pl.col("bd").first(), t_ini=pl.col("t").min(), t_fin=pl.col("t").max(),
                 n0=pl.col("w").filter(pl.col("v") == 0).sum(), n1=pl.col("w").filter(pl.col("v") == 1).sum()))


def viterbi(n0: np.ndarray, n1: np.ndarray) -> np.ndarray:
    a0, a1 = np.minimum(n0, TOPE), np.minimum(n1, TOPE)
    r = np.c_[a0 - a1, a1 - a0]
    V = r[0].astype(float).copy()
    atras = np.zeros((len(r), 2), np.int8)
    for k in range(1, len(r)):
        nuevo = np.empty(2)
        for s in (0, 1):
            quedarse, cambiar = V[s], V[1 - s] - CAMBIO
            atras[k, s] = s if quedarse >= cambiar else 1 - s
            nuevo[s] = max(quedarse, cambiar) + r[k, s]
        V = nuevo
    est = np.empty(len(r), np.int8)
    est[-1] = int(np.argmax(V))
    for k in range(len(r) - 1, 0, -1):
        est[k - 1] = atras[k, est[k]]
    return est


def fundir(est: np.ndarray, n0: np.ndarray, n1: np.ndarray, ti: np.ndarray) -> np.ndarray:
    """Un bloque cuyas visitas decisivas (≥ DECISIVA netas hacia su variante) abarcan menos de
    LAPSO_MIN_CAMBIO_S se funde con el vecino, empezando por el de menos lapso, hasta que todos
    lleguen (o quede uno). Con una zona de unos km eso pide dos pasadas en vueltas distintas;
    con una zona larga, ≥ 45 min de evidencia seguida. Quita los bloques de un rato al inicio o
    al final del día: el bus que va o viene del encierro por calles que la otra variante recorre
    (en 1101 y 1401, las mismas calles al revés). Con un umbral en registros, una zona de 2 km
    (≈ 5 registros por pasada) no llegaba nunca."""
    est = est.copy()
    while True:
        cortes = np.flatnonzero(np.diff(est)) + 1
        if not len(cortes):
            return est
        grupos = np.split(np.arange(len(est)), cortes)
        lapsos = []
        for g in grupos:
            neta = (n0[g] - n1[g]) * (1 if est[g[0]] == 0 else -1)
            t = ti[g][neta >= DECISIVA]
            lapsos.append(float(t.max() - t.min()) if len(t) else -1.0)
        k = int(np.argmin(lapsos))
        if lapsos[k] >= LAPSO_MIN_CAMBIO_S:
            return est
        vecino = max((i for i in (k - 1, k + 1) if 0 <= i < len(grupos)), key=lambda i: lapsos[i])
        est[grupos[k]] = est[grupos[vecino][0]]


def frontera(t_bd: np.ndarray, s0_bd: np.ndarray, L: float, ta: float, tb: float) -> float:
    """Entre la última visita de un bloque (ta) y la primera del siguiente (tb): el hueco más
    largo, o el paso por el km 0 de la principal, o el punto medio."""
    m = (t_bd >= ta) & (t_bd <= tb)
    tt = t_bd[m]
    if len(tt) >= 2:
        h = np.diff(tt)
        k = int(np.argmax(h))
        if h[k] >= HUECO_FRONTERA_S:
            return float(tt[k] + h[k] / 2)
        ss = s0_bd[m]
        val = np.isfinite(ss[:-1]) & np.isfinite(ss[1:])
        km0 = np.flatnonzero(val & (ss[:-1] > L - KM0_M) & (ss[1:] < KM0_M))
        if len(km0):
            mitad = (ta + tb) / 2
            c = km0[np.argmin(np.abs(tt[km0] - mitad))]
            return float((tt[c] + tt[c + 1]) / 2)
    return (ta + tb) / 2


def bloques(A: dict, ev: dict, T: dict, variantes) -> tuple[np.ndarray, np.ndarray, np.ndarray, pl.DataFrame]:
    """Por registro: variante (0/1, −1 indeterminado), id de bloque y evidencia neta del bloque."""
    vis = visitas(ev["eventos"])
    n = len(A["t"])
    etiqueta = np.full(n, -1, np.int8)
    bloque = np.full(n, -1, np.int64)
    neta = np.zeros(n, np.int32)
    v0 = variantes[0]
    j0 = ev["J"][v0]
    s0 = np.where((j0 >= 0) & (ev["D"][v0] <= EN_TRAZA_M), T[v0]["s"][np.maximum(j0, 0)], np.nan)
    L0 = T[v0]["L"]
    bd = A["bd"]
    ini = np.searchsorted(bd, np.arange(int(bd.max()) + 2))         # registros de cada bus-día (bd ordenado)
    filas, nb = [], 0
    for (b,), g in vis.group_by(["bd"], maintain_order=True):
        n0, n1 = g["n0"].to_numpy(), g["n1"].to_numpy()
        ti, tf = g["t_ini"].to_numpy(), g["t_fin"].to_numpy()
        est = fundir(viterbi(n0, n1), n0, n1, ti)
        cortes = np.flatnonzero(np.diff(est)) + 1                     # primera visita de cada bloque nuevo
        lo, hi = ini[b], ini[b + 1]
        t_bd = A["t"][lo:hi].astype(float)
        bordes = [-np.inf] + [frontera(t_bd, s0[lo:hi], L0, tf[c - 1], ti[c]) for c in cortes] + [np.inf]
        grupos = np.split(np.arange(len(est)), cortes)
        for k, gv in enumerate(grupos):
            s = int(est[gv[0]])
            a_s, a_o = (n0, n1) if s == 0 else (n1, n0)
            ev_neta = int(round(float((a_s[gv] - a_o[gv]).sum())))
            contrarias = int(((a_o[gv] - a_s[gv]) >= 2).sum())
            m = (t_bd > bordes[k]) & (t_bd <= bordes[k + 1])
            etq = s if ev_neta >= EVID_MIN_BLOQUE else -1
            etiqueta[lo:hi][m] = etq
            bloque[lo:hi][m] = nb
            neta[lo:hi][m] = ev_neta
            filas.append({"bloque": nb, "bd": int(b), "variante": NOMBRE.get(variantes[s], variantes[s]) if etq >= 0 else INDET,
                          "t_ini": float(t_bd[m].min()) if m.any() else np.nan,
                          "t_fin": float(t_bd[m].max()) if m.any() else np.nan,
                          "registros": int(m.sum()), "visitas": len(gv), "n_principal": float(n0[gv].sum()),
                          "n_alterna": float(n1[gv].sum()), "evidencia_neta": ev_neta,
                          "visitas_contrarias": contrarias, "bloques_en_dia": len(grupos)})
            nb += 1
    return etiqueta, bloque, neta, pl.DataFrame(filas)


# ═════════════════════════════════════════════════════════════════════════════
#  una corrida
# ═════════════════════════════════════════════════════════════════════════════
def clasificar(df: pl.DataFrame, T: dict, variantes) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    t0 = time.time()
    A = arreglos(df)
    ev = evidencia(T, variantes, A)
    etq, blq, neta, B = bloques(A, ev, T, variantes)
    nombres = np.array([NOMBRE.get(v, v) for v in variantes] + [INDET], dtype=object)
    numeros = np.array([int(v) for v in variantes] + [-1], dtype=np.int64)
    k = np.where(etq < 0, 2, etq)
    claves = df.select("id", dia=pl.col("time").dt.date()).unique(maintain_order=True)
    B = B.join(claves.with_row_index("bd").with_columns(pl.col("bd").cast(pl.Int64)), on="bd", how="left")
    B = B.with_columns(pl.from_epoch(pl.col("t_ini").fill_nan(None).cast(pl.Int64), "s").alias("hora_ini"),
                       pl.from_epoch(pl.col("t_fin").fill_nan(None).cast(pl.Int64), "s").alias("hora_fin")).drop("t_ini", "t_fin")
    out = df.with_columns(group_id_avl=pl.col("group_id"),
                          group_id=pl.Series(numeros[k]).replace(-1, None),
                          variante=pl.Series(nombres[k].astype(str)),
                          bloque=pl.Series(blq), bloque_evidencia=pl.Series(neta))
    info = {"registros": len(df), "bus_dias": int(A["bd"].max()) + 1,
            "evidencia_positiva": {v: int(ev["positiva"][v].sum()) for v in variantes},
            "saltos": {v: {"n": ev["salto"][v][0], "peso": ev["salto"][v][1], "observable": ev["salto"][v][2],
                           "usados": ev["salto"][v][2] >= OBS_MIN} for v in variantes},
            "segundos": round(time.time() - t0, 1)}
    return out, B, info


def otras_rutas(claves: pl.DataFrame, excluir: set[int]) -> pl.DataFrame:
    """Para cada (id, dia): en qué otras rutas del crudo limpio (data/01_raw/records) anduvo ese
    bus ese día, sin las de la familia (`excluir`: sus grupos y variantes, por si ya hay salida
    de una corrida anterior)."""
    fs = [a for a in sorted(LIMPIO.glob("*/df_rec_raw_*.parquet"))
          if a.parent.name.isdigit() and int(a.parent.name) not in excluir
          and {"id", "time", "group_id"} <= set(pl.read_parquet_schema(a))]
    if not fs or not len(claves):
        return claves.with_columns(otras=pl.lit([], pl.List(pl.Int64)))
    otras = (pl.concat([pl.scan_parquet(a).select("id", dia=pl.col("time").dt.date(), g=pl.col("group_id").cast(pl.Int64))
                        for a in fs])
             .join(claves.lazy(), on=["id", "dia"], how="semi")
             .group_by("id", "dia").agg(otras=pl.col("g").unique().sort()).collect())
    return claves.join(otras, on=["id", "dia"], how="left").with_columns(pl.col("otras").fill_null([]))


def resumen_familia(fam: str, out: pl.DataFrame, B: pl.DataFrame, info: dict, T: dict, variantes, zonas,
                    ind: pl.DataFrame) -> str:
    alterna = NOMBRE.get(variantes[1], variantes[1])
    lin = [f"# Familia {fam}: {variantes[0]} / {alterna} (group_id {variantes[1]})", ""]
    lin += ["## Zonas discriminantes", "", "| variante | km | largo (m) | caja lat | caja lon |", "|---|---|---:|---|---|"]
    lin += [f"| {z['variante']} | {z['km_ini']}–{z['km_fin']} | {z['largo_m']} | {z['lat_min']}–{z['lat_max']} | "
            f"{z['lon_min']}–{z['lon_max']} |" for z in zonas]
    lin += ["", f"De la zona exclusiva, en las mismas calles que la otra variante pero al revés: "
            + ", ".join(f"{v} {T[v]['contrario_m'] / 1000:.1f} km" for v in variantes), ""]
    n = len(out)
    lin += ["## Registros", "", f"{n:,} registros, {info['bus_dias']:,} bus-día.", ""]
    lin += [f"- evidencia directa: " + ", ".join(f"{v} {info['evidencia_positiva'][v]:,}" for v in variantes)]
    lin += [f"- saltos sobre la zona de cada traza (evidencia de la otra): "
            + ", ".join(f"{v} {info['saltos'][v]['n']:,}" for v in variantes)]
    ct = (out.group_by("group_id_avl", "variante", "group_id").agg(pl.len().alias("registros"))
          .sort("group_id_avl", "variante"))
    lin += ["", "| group_id del AVL | variante asignada | group_id nuevo | registros | % |", "|---|---|---|---:|---:|"]
    lin += [f"| {r['group_id_avl']} | {r['variante']} | {r['group_id'] if r['group_id'] is not None else '—'} | "
            f"{r['registros']:,} | {100 * r['registros'] / n:.1f} |" for r in ct.iter_rows(named=True)]
    nd = B.group_by("id", "dia").agg(pl.col("bloques_en_dia").first(),
                                    pl.col("variante").filter(pl.col("variante") != INDET).n_unique().alias("variantes"))
    lin += ["", "## Bus-día", "",
            f"- con un bloque: {(nd['bloques_en_dia'] == 1).sum():,}; con dos o más: {(nd['bloques_en_dia'] >= 2).sum():,}"
            f" (con las dos variantes: {(nd['variantes'] == 2).sum():,})"]
    por_mes = (out.filter(pl.col("variante") != INDET)
               .group_by(pl.col("time").dt.month().alias("mes"))
               .agg((pl.col("variante") == alterna).mean().alias("alterna")).sort("mes"))
    lin += ["", "Fracción de registros en la alterna por mes: "
            + ", ".join(f"{mt.MESES_ES[r['mes'] - 1]} {100 * r['alterna']:.0f} %" for r in por_mes.iter_rows(named=True))]
    por_hora = (out.filter(pl.col("variante") != INDET)
                .group_by(pl.col("time").dt.hour().alias("h"))
                .agg((pl.col("variante") == alterna).mean().alias("alterna")).sort("h"))
    lin += ["", "Por hora: " + ", ".join(f"{r['h']} h {100 * r['alterna']:.0f} %" for r in por_hora.iter_rows(named=True))]
    lin += ["", "## Indeterminados", ""] + texto_indeterminados(ind, len(out))
    return "\n".join(lin) + "\n"


def indeterminados(out: pl.DataFrame, excluir: set[int]) -> pl.DataFrame:
    """Un renglón por bus-día con registros indeterminados: por qué, cuánto, a qué hora y en qué
    rutas del crudo limpio anduvo el bus ese día."""
    bd = (out.with_columns(dia=pl.col("time").dt.date())
          .group_by("id", "dia").agg(
              registros_dia=pl.len(), indeterminados=(pl.col("variante") == INDET).sum(),
              motivo=pl.when((pl.col("bloque") < 0).all()).then(pl.lit("sin pasar por la zona"))
              .otherwise(pl.lit("bloque débil")),
              variante_dia=pl.col("variante").filter(pl.col("variante") != INDET).unique().sort(),
              hora_ini=pl.col("time").filter(pl.col("variante") == INDET).min(),
              hora_fin=pl.col("time").filter(pl.col("variante") == INDET).max())
          .filter(pl.col("indeterminados") > 0).sort("id", "dia"))
    return otras_rutas(bd.select("id", "dia"), excluir).join(bd, on=["id", "dia"])


def texto_indeterminados(ind: pl.DataFrame, n_total: int) -> list[str]:
    n = int(ind["indeterminados"].sum()) if len(ind) else 0
    if not n:
        return ["Ninguno."]
    lin = [f"{n:,} registros ({100 * n / n_total:.2f} %) en {len(ind):,} bus-día. No entran a "
           "`data/01_raw/records`; están en `data/02_interim/variantes/<familia>/`.", "",
           "| motivo | bus-día | registros | mediana de registros del bus-día | bus-día < 60 registros | "
           "con el bus en otra ruta ese día | antes de las 6 o desde las 21 h |", "|---|---:|---:|---:|---:|---:|---:|"]
    for (mot,), x in ind.group_by(["motivo"], maintain_order=True):
        temprano = ((x["hora_fin"].dt.hour() < 6) | (x["hora_ini"].dt.hour() >= 21)).mean()
        lin.append(f"| {mot} | {len(x):,} | {int(x['indeterminados'].sum()):,} | {x['registros_dia'].median():.0f} | "
                   f"{100 * (x['registros_dia'] < 60).mean():.0f} % | {100 * (x['otras'].list.len() > 0).mean():.0f} % | "
                   f"{100 * temprano:.0f} % |")
    largos = ind.filter(pl.col("indeterminados") >= 60)
    lin += ["", f"Bus-día con ≥ 60 registros indeterminados (~1 h de servicio que se pierde): {len(largos):,}"
            f" ({int(largos['indeterminados'].sum()):,} registros)."]
    return lin


def empalmar(df: pl.DataFrame, va: str, vb: str, n: int, semilla: int = 0) -> pl.DataFrame:
    """Prueba: bus-días sintéticos con un cambio de ruta a una hora conocida. El mismo día se
    toma un bus que sólo hizo A y otro que sólo hizo B (≥ 300 registros, de antes de las 9 a
    después de las 18 h): los registros del primero hasta τ y los del segundo desde τ, con τ al
    azar entre las 10 y las 16 h, pasan a ser un bus «EMPALME-k» (los dos originales salen del
    crudo). El orden A→B / B→A se alterna. El `group_id` del AVL cambia a media jornada sin
    que el bus cambie de calle (ver el resumen), así que sólo aquí la hora del cambio es exacta."""
    rng = np.random.default_rng(semilla)
    d = df.with_columns(dia=pl.col("time").dt.date())
    k = (d.group_by("id", "dia").agg(g=pl.col("group_id").unique(), n=pl.len(), t0=pl.col("time").min(), t1=pl.col("time").max())
         .filter((pl.col("g").list.len() == 1) & (pl.col("n") >= 300) & (pl.col("t0").dt.hour() < 9) & (pl.col("t1").dt.hour() >= 18))
         .with_columns(g=pl.col("g").list.first().cast(pl.Utf8)).sort("dia", "id"))
    pares = []
    for (dia,), sub in k.group_by(["dia"], maintain_order=True):
        a, b = sub.filter(pl.col("g") == va)["id"].to_list(), sub.filter(pl.col("g") == vb)["id"].to_list()
        rng.shuffle(a), rng.shuffle(b)
        pares += [(dia, x, y) for x, y in zip(a, b)]
    orden = rng.permutation(len(pares))[:n]
    filas = []
    for j, i in enumerate(orden):
        dia, x, y = pares[i]
        tau = datetime.combine(dia, datetime.min.time()) + timedelta(minutes=int(rng.uniform(600, 960)))
        primero, segundo = (x, y) if j % 2 == 0 else (y, x)
        filas += [{"id": primero, "dia": dia, "nuevo": f"EMPALME-{j}", "lado": 0, "tau": tau},
                  {"id": segundo, "dia": dia, "nuevo": f"EMPALME-{j}", "lado": 1, "tau": tau}]
    if not filas:
        return df
    plan = pl.DataFrame(filas).with_columns(pl.col("tau").cast(d["time"].dtype))
    emp = (d.join(plan, on=["id", "dia"])
           .filter(((pl.col("lado") == 0) & (pl.col("time") < pl.col("tau"))) | ((pl.col("lado") == 1) & (pl.col("time") >= pl.col("tau"))))
           .with_columns(id=pl.col("nuevo")).drop("nuevo", "lado", "tau"))
    resto = d.join(plan.select("id", "dia"), on=["id", "dia"], how="anti")
    return pl.concat([resto, emp]).drop("dia").sort(["id", "time"])


def metricas_prueba(out: pl.DataFrame, ev_reg: int) -> dict:
    """La verdad es el group_id del AVL de cada ruta mezclada (en los empalmes, exacta)."""
    o = out.with_columns(verdad=pl.col("group_id_avl").cast(pl.Utf8), emp=pl.col("id").str.starts_with("EMPALME"))
    det = pl.col("variante") != INDET
    m = {}
    for nombre, sub in (("reales", o.filter(~pl.col("emp"))), ("empalmes", o.filter(pl.col("emp")))):
        if len(sub):
            cl = sub.filter(det)
            m[f"{nombre}_registros"] = len(sub)
            m[f"{nombre}_clasificados"] = round(len(cl) / len(sub), 4)
            m[f"{nombre}_acierto_registros"] = round(float((cl["variante"] == cl["verdad"]).mean()), 4)
    bd = (o.group_by("id", pl.col("time").dt.date().alias("dia"))
          .agg(pl.col("emp").first(), n=pl.len(), n_menor=pl.col("verdad").value_counts().struct.field("count").min(),
               verdad=pl.col("verdad").unique().sort(), pred=pl.col("variante").filter(det).unique().sort(),
               verdad_ini=pl.col("verdad").first(), pred_ini=pl.col("variante").filter(det).first(),
               acierto=(pl.col("variante") == pl.col("verdad")).filter(det).mean(),
               t_cambio=pl.col("time").filter(pl.col("verdad") != pl.col("verdad").first()).first(),
               p_cambio=pl.col("time").filter(det & (pl.col("variante") != pl.col("variante").filter(det).first())).first()))
    # bus-días con ≥ BD_MIN registros (~1 h); un cambio «sustancial» deja ≥ BD_MIN de cada ruta. Los
    # fragmentos de 3–5 registros (el bus asoma por la otra ruta al arrancar) no se pueden ver.
    real = bd.filter(~pl.col("emp"))
    m["bus_dias_total"] = len(real)
    m["registros_en_bus_dias_cortos"] = round(float(real.filter(pl.col("n") < BD_MIN)["n"].sum() / max(1, real["n"].sum())), 4)
    real = real.filter(pl.col("n") >= BD_MIN)
    una = real.filter(pl.col("verdad").list.len() == 1)
    m["bus_dias_una_ruta"] = len(una)
    m["una_ruta_bien"] = round(float(((una["pred"].list.len() == 1) & (una["pred"].list.first() == una["verdad"].list.first())).mean()), 4)
    m["una_ruta_cambio_falso"] = round(float((una["pred"].list.len() == 2).mean()), 4)
    m["una_ruta_indeterminado"] = round(float((una["pred"].list.len() == 0).mean()), 4)
    m["bus_dias_cambio_avl_fragmento"] = len(real.filter((pl.col("verdad").list.len() == 2) & (pl.col("n_menor") < BD_MIN)))
    for nombre, dos in (("cambio_avl", real.filter((pl.col("verdad").list.len() == 2) & (pl.col("n_menor") >= BD_MIN))),
                        ("empalme", bd.filter(pl.col("emp")))):
        m[f"{nombre}_bus_dias"] = len(dos)
        if not len(dos):
            continue
        ambos = dos.filter(pl.col("pred").list.len() == 2)
        m[f"{nombre}_detectado"] = round(len(ambos) / len(dos), 4)
        m[f"{nombre}_detectado_en_orden"] = round(float((ambos["pred_ini"] == ambos["verdad_ini"]).sum() / len(dos)), 4)
        m[f"{nombre}_acierto_registros"] = round(float(dos["acierto"].mean()), 4)
        err = ambos.select(((pl.col("p_cambio") - pl.col("t_cambio")).dt.total_seconds().abs() / 60).alias("e"))["e"].drop_nulls()
        if len(err):
            for q in (0.5, 0.75, 0.9):
                m[f"{nombre}_error_frontera_min_p{int(q * 100)}"] = round(float(err.quantile(q)), 1)
    m["registros_con_evidencia"] = ev_reg
    return m


def resumen_pruebas(rep: Path) -> None:
    """Junta los metricas.json de reports/variantes_<desde>_<hasta>/prueba_*/ en pruebas.csv y pruebas.md."""
    filas = []
    for f in sorted(rep.glob("prueba_*/metricas.json")):
        m = json.load(open(f, encoding="utf-8"))
        largo = {}
        for z in m["zonas"]:
            largo[z["variante"]] = largo.get(z["variante"], 0) + z["largo_m"]
        filas.append({"prueba": f.parent.name, "zona": m.get("zona_tipo", "todo"),
                      "zona_km": "/".join(f"{v / 1000:.1f}" for v in largo.values()), "recorte": m["recorte"],
                      "acierto_reg": m["reales_acierto_registros"], "clasif_reg": m["reales_clasificados"],
                      "una_bien": m["una_ruta_bien"], "una_cambio_falso": m["una_ruta_cambio_falso"],
                      "una_indet": m["una_ruta_indeterminado"],
                      "emp_detectado": m.get("empalme_detectado"), "emp_en_orden": m.get("empalme_detectado_en_orden"),
                      "emp_frontera_p50": m.get("empalme_error_frontera_min_p50"),
                      "emp_frontera_p90": m.get("empalme_error_frontera_min_p90"),
                      "emp_acierto_reg": m.get("empalme_acierto_registros"),
                      "avl_cambio_detectado": m.get("cambio_avl_detectado"), "avl_cambio_n": m.get("cambio_avl_bus_dias")})
    if not filas:
        print("no hay pruebas en", rep)
        return
    pl.DataFrame(filas).write_csv(rep / "pruebas.csv")
    cols = list(filas[0])
    pc = lambda v: "—" if v is None else (f"{100 * v:.1f}" if isinstance(v, float) and v <= 1 and not isinstance(v, bool) else str(v))
    lin = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lin += ["| " + " | ".join(pc(r[c]) for c in cols) + " |" for r in filas]
    (rep / "pruebas.md").write_text("Proporciones en %; fronteras en minutos.\n\n" + "\n".join(lin) + "\n", encoding="utf-8")
    print("\n".join(lin))


# ═════════════════════════════════════════════════════════════════════════════
def main() -> int:
    global CAMBIO, LAPSO_MIN_CAMBIO_S
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--familias", nargs="+", default=list(FAMILIAS), choices=list(FAMILIAS))
    ap.add_argument("--geometria", action="store_true", help="sólo las zonas discriminantes (no lee crudo)")
    ap.add_argument("--mezcla", nargs=2, metavar=("A", "B"), help="prueba: mezcla el crudo de dos rutas reales")
    ap.add_argument("--zona-max-km", type=float, help="prueba: una sola zona de X km por traza")
    ap.add_argument("--zona-tipo", default="todo", choices=["todo", "calles", "sentido"],
                    help="prueba: zona de calles distintas (1801/1802) o de las mismas calles al revés (1101/1401)")
    ap.add_argument("--recorte", action="store_true", help="prueba: sólo registros a ≤ 50 m de la traza A")
    ap.add_argument("--empalmes", type=int, default=300, help="prueba: bus-días sintéticos con un cambio a hora conocida")
    ap.add_argument("--desde", default="2024-01-01")
    ap.add_argument("--hasta", default="2024-08-31")
    ap.add_argument("--geom", default=str(RAW), help="carpeta de la geometría (data/01_raw)")
    ap.add_argument("--resumen-pruebas", action="store_true", help="junta las pruebas hechas en pruebas.csv/.md")
    ap.add_argument("--cambio", type=float, default=CAMBIO, help="costo de un cambio de variante (Viterbi)")
    ap.add_argument("--lapso-cambio", type=float, default=LAPSO_MIN_CAMBIO_S, help="s que debe abarcar la evidencia de un bloque nuevo")
    a = ap.parse_args()
    CAMBIO, LAPSO_MIN_CAMBIO_S = a.cambio, a.lapso_cambio
    desde, hasta = date.fromisoformat(a.desde), date.fromisoformat(a.hasta)
    geom = Path(a.geom)
    rep = REPORTES / f"pruebas_{a.desde}_{a.hasta}"
    if a.resumen_pruebas:
        resumen_pruebas(rep)
        return 0

    if a.mezcla:
        va, vb = a.mezcla
        T = geometria((va, vb), geom, a.zona_max_km, a.zona_tipo)
        df = leer_crudo([int(va), int(vb)], desde, hasta)
        if a.recorte:
            d = cKDTree(T[va]["q"]).query(np.c_[df["longitude"].cast(pl.Float64).to_numpy() * mt.KX,
                                                df["latitude"].cast(pl.Float64).to_numpy() * mt.KY])[0]
            df = df.filter(pl.Series(d <= RECORTE_M))
        if a.empalmes:
            df = empalmar(df, va, vb, a.empalmes)
        out, B, info = clasificar(df, T, (va, vb))
        m = metricas_prueba(out, sum(info["evidencia_positiva"].values()))
        m.update({"zonas": tabla_zonas(T, (va, vb)), "info": info, "zona_max_km": a.zona_max_km, "recorte": a.recorte,
                  "zona_tipo": a.zona_tipo, "cambio": CAMBIO, "lapso_cambio": LAPSO_MIN_CAMBIO_S})
        nombre = (f"prueba_{va}_{vb}" + (f"_{a.zona_tipo}" if a.zona_tipo != "todo" else "")
                  + (f"_zona{a.zona_max_km:g}km" if a.zona_max_km else "") + ("_recorte" if a.recorte else "")
                  + (f"_c{CAMBIO:g}_l{LAPSO_MIN_CAMBIO_S:g}" if a.cambio != ap.get_default("cambio")
                     or a.lapso_cambio != ap.get_default("lapso_cambio") else ""))
        d = rep / nombre
        d.mkdir(parents=True, exist_ok=True)
        json.dump(m, open(d / "metricas.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
        B.write_csv(d / "bloques.csv")
        print(json.dumps({k: v for k, v in m.items() if k not in ("zonas", "info")}, ensure_ascii=False, indent=1))
        print(f"  saltos {info['saltos']}  evidencia {info['evidencia_positiva']}  {info['segundos']} s  → {d}")
        return 0

    for fam in a.familias:
        variantes = FAMILIAS[fam]["variantes"]
        T = geometria(variantes, geom)
        zonas = tabla_zonas(T, variantes)
        print(f"\n{fam}: {variantes[0]} / {variantes[1]}")
        for z in zonas:
            print(f"  {z['variante']:>9} km {z['km_ini']:6.2f}–{z['km_fin']:6.2f} ({z['largo_m']:5d} m)  "
                  f"lat {z['lat_min']}–{z['lat_max']}  lon {z['lon_min']}–{z['lon_max']}")
        print("  al revés sobre las mismas calles: " + ", ".join(f"{v} {T[v]['contrario_m'] / 1000:.1f} km" for v in variantes))
        d = REPORTES / fam
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "zonas.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(zonas[0]))
            w.writeheader()
            w.writerows(zonas)
        if a.geometria:
            continue
        grupos = FAMILIAS[fam]["grupos"]
        try:
            _, rango = archivos_crudo(grupos, MEZCLADO)
        except (FileNotFoundError, ValueError) as e:
            print(f"  ⚠ {e}")
            continue
        df = leer_crudo(grupos, None, None, MEZCLADO)          # todo el archivo: la salida lleva su mismo rango
        out, B, info = clasificar(df, T, variantes)
        ind = indeterminados(out, {*grupos, *map(int, variantes)})
        B.write_csv(d / "bloques.csv")
        ind.with_columns(pl.col("otras", "variante_dia").list.eval(pl.element().cast(pl.Utf8)).list.join(" ")) \
           .write_csv(d / "indeterminados.csv")
        (d / "resumen.md").write_text(resumen_familia(fam, out, B, info, T, variantes, zonas, ind), encoding="utf-8")
        for v in variantes:
            destino = LIMPIO / v / f"df_rec_raw_{v}_{rango}.parquet"
            if destino.exists() and "group_id_avl" not in pl.read_parquet_schema(destino):
                print(f"  ⚠ {destino.relative_to(RAIZ)} existe y no salió de aquí (no trae group_id_avl): no lo piso")
                continue
            destino.parent.mkdir(parents=True, exist_ok=True)
            sub = out.filter(pl.col("group_id") == int(v))
            sub.write_parquet(destino)
            print(f"  {NOMBRE.get(v, v)} (group_id {v}): {len(sub):,} registros → {destino.relative_to(RAIZ)}")
        di = INTERIM / fam
        di.mkdir(parents=True, exist_ok=True)
        out.filter(pl.col("variante") == INDET).write_parquet(di / f"indeterminados_{rango}.parquet")
        print(f"  indeterminado: {int(ind['indeterminados'].sum()) if len(ind) else 0:,} registros "
              f"({info['segundos']} s)  → {di.relative_to(RAIZ)}; resumen en {d.relative_to(RAIZ)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
