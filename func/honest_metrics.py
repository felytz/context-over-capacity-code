"""
honest_metrics
==============
Fase 5 del plan de re-corrida (`docs/PLAN_REPRESENTATIVIDAD.md`): separar
**calibración** de **precisión** en el reporte de on-time.

El problema
-----------
La banda operacional APTA es ``dev = t_true − t_pred ∈ [−60, +300]`` s:
**asimétrica**, 60 s del lado temprano y 300 s del tardío. Un modelo que
subestima sistemáticamente el tiempo restante (``bias_s < 0`` ⇒ ``dev > 0``)
cae del lado ANCHO y gana on-time sin ser más preciso. Medido en la corrida
parcial del 2026-07-24: ``xgb_mae_b`` marca 61.4% de on-time contra 50.9% del
LSTM, pero con bias −44.5 s frente a −8.7 s. ``delay_hour`` es el único que gana
sin ese asterisco (bias 0.0).

Las dos métricas que lo separan
-------------------------------
1. **Banda simétrica ±180 s** — la MISMA ANCHURA que APTA (360 s), centrada en
   0. Comparar APTA contra ±180 aísla el efecto de la asimetría: la anchura no
   cambia, solo el centro.
2. **On-time calibrado** — se resta un desplazamiento constante ``c`` a la
   predicción (``t_pred' = t_pred + c`` ⇒ ``dev' = dev − c``) con ``c`` = mediana
   de ``dev``, y se aplica la banda a ``dev'``. Responde: *si el modelo
   estuviera perfectamente centrado, ¿cuánto on-time tendría?* Lo que sobreviva
   es precisión; lo que se pierda era calibración.

   Dos variantes, ambas se reportan:

   ``loo``   ``c_k`` = mediana de ``dev`` sobre los folds ≠ k, aplicada al fold
             k. **Fuera de muestra** respecto del fold evaluado ⇒ es la cifra
             honesta, la que va al paper.
   ``self``  ``c`` = mediana de ``dev`` sobre TODOS los folds, aplicada a todos.
             Optimista (usa el fold evaluado para centrarse); es la **cota
             superior** de lo que la calibración puede dar.

   El plan pedía "el bias mediano del fold de TRAIN". Eso obligaría a evaluar
   cada modelo también sobre sus filas de entrenamiento (×4 el coste de
   evaluación) y además el bias de train está optimísticamente sesgado. La
   variante ``loo`` da lo mismo que se buscaba —un centro estimado sin mirar los
   datos que se van a puntuar— gratis, porque los folds ya están calculados.

Cómo se lee (la descomposición)
-------------------------------
Las tres cifras se leen en cadena, y cada salto aísla un efecto:

    pct_ontime_apta          on-time tal cual se reporta hoy
      │  ← el salto mide el premio por CAER DEL LADO ANCHO (misma anchura,
      │     otro centro)
    pct_ontime_sym180        on-time sin ventaja por asimetría
      │  ← el salto mide cuánto costaba el sesgo residual
    pct_ontime_sym180_cal    on-time del modelo ya centrado  =  PRECISIÓN pura

``pct_ontime_apta_cal`` es la lectura operacional del mismo centrado: qué
marcaría en la banda de la agencia un modelo sin sesgo. Ojo con su signo: como
la banda APTA está centrada en +120 s, centrar en 0 a un modelo que ya caía del
lado ancho lo puede **bajar**. Eso no es un fallo del cálculo, es exactamente la
distorsión que la fase 5 documenta.

Por qué histogramas y no residuos
---------------------------------
El harness de `xgb_prediction` es streaming a propósito: una leg de la ruta 101
son ~300M pares (viaje × corte × horizonte) por algoritmo, que no caben en disco.
Un histograma de ``dev`` con bins de 1 s pesa nada y permite derivar
**cualquier** banda y cualquier calibración a posteriori, sin volver a correr.
Con MAE del orden de 200 s, un bin de 1 s es ruido despreciable (el error de
borde al contar una banda es a lo sumo medio bin).

Los bins extremos (±``DEV_MAX_ABS_S``) **saturan**: acumulan toda la cola. Es
inofensivo para contar bandas —cualquier cosa a 2 h del objetivo está fuera de
todas ellas— y `honest_from_hist` reporta ``sat_frac`` para que se vea si alguna
vez deja de serlo.

Esquema del artefacto ``*_devhist.parquet``
-------------------------------------------
    route · leg · algorithm · fold · dev_s (Int32, BORDE IZQUIERDO) · count (Int64)

Lo escriben las dos familias con la misma convención:
`xgb_prediction/run_experiment.py` (baselines + XGB) y
`lstm_prediction_tf/run_frame.py` (LSTM).
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import numpy as np

# ──────────────────────────────────────────────────────────────────
# Convención del histograma
# ──────────────────────────────────────────────────────────────────

DEV_BIN_S     = 1.0      # 1 s: ruido despreciable frente a MAE ~200 s
DEV_MAX_ABS_S = 7200.0   # ±2 h; los bins de los extremos SATURAN

# Bandas reportadas. Ambas miden 360 s de ancho: la única diferencia es el
# centro (+120 s en APTA, 0 en la simétrica), que es justo lo que se quiere
# aislar. NO invertir el signo de APTA (ver docs/historial_paper/11_*).
BAND_APTA   = (-60.0, 300.0)
BAND_SYM180 = (-180.0, 180.0)
BANDS: Dict[str, Tuple[float, float]] = {"apta": BAND_APTA, "sym180": BAND_SYM180}


def n_bins(bin_s: float = DEV_BIN_S, max_abs_s: float = DEV_MAX_ABS_S) -> int:
    return int(2.0 * max_abs_s / bin_s) + 1


# ──────────────────────────────────────────────────────────────────
# Horizonte de predicción (§4.5 del paper)
# ──────────────────────────────────────────────────────────────────
# El horizonte es el TIEMPO RESTANTE REAL hasta el objetivo, no el índice de
# celda: es lo que un operador entiende por "¿a cuánto está el camión?". Los
# cortes, en minutos: 0-2, 2-5, 5-10, 10-20, 20-40, 40-60, 60-90, >90.
#
# Los cinco primeros son los del manuscrito de agosto. 60 y 90 los añade la
# enmienda E4 (docs/PREREGISTRO_VUELTA.md §6, 8-oct-2026): con la vuelta (59-179
# min) «> 40» juntaba la mitad de los pares, y en agosto, con legs de 20-30 min,
# era una cola. Los cortes viejos son un subconjunto de los nuevos, así que el
# «> 40» de antes se recupera EXACTO uniendo los tres últimos cubos, ponderados
# por `n_pairs`: MAE, RMSE, sesgo y on-time (la mediana y el P90, no).
#
# ⚠ Se binea por `dt_true`, NUNCA por `dt_pred`: bineando por la predicción, el
# estrato dependería del modelo y los algoritmos dejarían de compararse sobre
# las mismas filas — exactamente lo que el frame único existe para evitar.
HORIZON_EDGES_S: Tuple[float, ...] = (0.0, 120.0, 300.0, 600.0, 1200.0, 2400.0,
                                      3600.0, 5400.0)
HORIZON_LABELS: Tuple[str, ...] = ("0-2 min", "2-5 min", "5-10 min",
                                   "10-20 min", "20-40 min", "40-60 min",
                                   "60-90 min", "> 90 min")


def horizon_bin(dt_true: np.ndarray) -> np.ndarray:
    """Tiempo restante real (s) → índice de bin en [0, len(HORIZON_LABELS)).

    Intervalos CERRADOS POR LA IZQUIERDA — [0,120), [120,300), … — que es lo que
    dicen las etiquetas. La rama polars del LSTM debe pasar `left_closed=True`
    explícitamente: su default es el contrario y un valor justo en el borde
    caería en bins distintos en cada familia.
    """
    return np.searchsorted(np.asarray(HORIZON_EDGES_S[1:]),
                           np.asarray(dt_true, dtype=np.float64),
                           side="right")


def bin_index(dev: np.ndarray, bin_s: float = DEV_BIN_S,
              max_abs_s: float = DEV_MAX_ABS_S) -> np.ndarray:
    """``dev`` (s) → índice de bin, saturando en los extremos.

    El bin se etiqueta por su **borde izquierdo** (``floor``), no por su centro,
    y eso hace que contar una banda sea EXACTO en vez de aproximado: el bin ``b``
    cubre ``[b, b+1)``, así que ``lo ≤ b < hi`` reproduce ``dev ∈ [lo, hi)``
    sin error mientras ``lo`` y ``hi`` sean enteros — y lo son (−60/+300,
    ±180, y los desplazamientos de calibración, que salen de un cuantil del
    propio histograma). Con redondeo al centro la banda se ensanchaba medio bin
    por cada borde: +0.15 pp de on-time medidos en un ensayo de 1M de pares,
    siempre en la misma dirección.
    """
    off = int(max_abs_s / bin_s)
    idx = np.floor(np.asarray(dev, dtype=np.float64) / bin_s).astype(np.int64)
    return np.clip(idx + off, 0, n_bins(bin_s, max_abs_s) - 1)


def bin_left(idx: np.ndarray, bin_s: float = DEV_BIN_S,
             max_abs_s: float = DEV_MAX_ABS_S) -> np.ndarray:
    """Índice de bin → borde IZQUIERDO en segundos (la etiqueta ``dev_s``)."""
    off = int(max_abs_s / bin_s)
    return (np.asarray(idx, dtype=np.int64) - off) * bin_s


# ──────────────────────────────────────────────────────────────────
# Derivaciones sobre un histograma disperso (dev_s, count)
# ──────────────────────────────────────────────────────────────────

def pct_in_band(dev_s: np.ndarray, count: np.ndarray,
                lo: float, hi: float, offset: float = 0.0) -> float:
    """% de pares con ``dev − offset ∈ [lo, hi)``.

    ``offset`` es el desplazamiento de calibración ``c``: predecir
    ``t_pred + c`` mueve ``dev`` a ``dev − c``, así que equivale a contar la
    banda desplazada ``[lo + c, hi + c)`` sobre el ``dev`` original.

    ``dev_s`` son bordes izquierdos, así que el conteo es exacto para límites
    enteros (ver `bin_index`). El intervalo es semiabierto por la derecha; con
    residuos en coma flotante el punto ``dev = hi`` tiene medida nula.
    """
    n = float(count.sum())
    if n <= 0:
        return float("nan")
    m = (dev_s >= lo + offset) & (dev_s < hi + offset)
    return 100.0 * float(count[m].sum()) / n


def quantile_from_hist(dev_s: np.ndarray, count: np.ndarray, q: float) -> float:
    """Cuantil de ``dev``, redondeado al borde izquierdo de su bin (≤ 1 s de
    sesgo a la baja). Los bins ya vienen ordenados por `_sorted`."""
    n = int(count.sum())
    if n <= 0:
        return float("nan")
    cum = np.cumsum(count)
    i = int(np.searchsorted(cum, q * n, side="left"))
    return float(dev_s[min(i, len(dev_s) - 1)])


def _sorted(dev_s: np.ndarray, count: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    o = np.argsort(dev_s, kind="stable")
    return np.asarray(dev_s, dtype=np.float64)[o], np.asarray(count, dtype=np.int64)[o]


def _merge(parts: Iterable[Tuple[np.ndarray, np.ndarray]]
           ) -> Tuple[np.ndarray, np.ndarray]:
    """Suma histogramas dispersos con soportes distintos."""
    ds = np.concatenate([p[0] for p in parts]) if parts else np.zeros(0)
    cs = np.concatenate([p[1] for p in parts]) if parts else np.zeros(0, dtype=np.int64)
    if ds.size == 0:
        return ds, cs.astype(np.int64)
    uniq, inv = np.unique(ds, return_inverse=True)
    return uniq, np.bincount(inv, weights=cs).astype(np.int64)


def honest_from_hist(hist_by_fold: Dict[int, Tuple[np.ndarray, np.ndarray]],
                     bands: Optional[Dict[str, Tuple[float, float]]] = None,
                     max_abs_s: float = DEV_MAX_ABS_S,
                     causal: bool = False) -> Dict[str, float]:
    """Métricas honestas de UN algoritmo en UNA leg, desde sus histogramas por fold.

    Devuelve, por cada banda ``b`` de `bands`:
      ``pct_ontime_<b>``       tal cual (lo que hoy reportan los agregados)
      ``pct_ontime_<b>_cal``   calibrado leave-one-fold-out  ← la cifra honesta
      ``pct_ontime_<b>_selfcal`` calibrado con todos los folds (cota superior)

    más ``n_pairs``, ``dev_median_s`` (= −bias mediano; el desplazamiento que
    aplicaría la calibración), ``cal_offset_s`` (media de los ``c_k`` del LOO,
    solo informativa) y ``sat_frac``.
    """
    bands = bands or BANDS
    folds = sorted(hist_by_fold)
    parts = [_sorted(*hist_by_fold[f]) for f in folds]
    all_ds, all_cs = _merge(parts)
    n_tot = int(all_cs.sum())

    out: Dict[str, float] = {
        "n_pairs": n_tot,
        "n_folds_hist": len(folds),
        "dev_median_s": quantile_from_hist(all_ds, all_cs, 0.50),
        "dev_p10_s": quantile_from_hist(all_ds, all_cs, 0.10),
        "dev_p90_s": quantile_from_hist(all_ds, all_cs, 0.90),
    }
    edge = np.isclose(np.abs(all_ds), max_abs_s)
    out["sat_frac"] = (float(all_cs[edge].sum()) / n_tot) if n_tot else float("nan")

    # Desplazamiento por fold: mediana de dev sobre los OTROS folds.
    #
    # ⚠ Con `causal=True` (ventana expansiva) solo entran los folds ANTERIORES.
    # El leave-one-out de siempre mira también los posteriores, y en una
    # escalera temporal eso es exactamente el look-ahead que la escalera existe
    # para eliminar: en despliegue, al centrar la predicción del mes k solo se
    # tienen los residuos de los meses ya vividos. El primer escalón evaluado no
    # tiene ninguno detrás, así que no se calibra (offset 0), que es lo honesto
    # y no lo cómodo.
    offs: Dict[int, float] = {}
    for i, f in enumerate(folds):
        rest = ([p for j, p in enumerate(parts) if j < i] if causal
                else [p for j, p in enumerate(parts) if j != i])
        if not rest:
            offs[f] = 0.0
            continue
        rds, rcs = _merge(rest)
        offs[f] = quantile_from_hist(rds, rcs, 0.50)
    out["cal_offset_s"] = float(np.mean(list(offs.values()))) if offs else float("nan")

    c_self = out["dev_median_s"]
    for name, (lo, hi) in bands.items():
        out[f"pct_ontime_{name}"] = pct_in_band(all_ds, all_cs, lo, hi)
        out[f"pct_ontime_{name}_selfcal"] = pct_in_band(all_ds, all_cs, lo, hi, c_self)
        # LOO: cada fold con SU desplazamiento, sumando aciertos, no promediando
        # porcentajes (los folds no tienen el mismo nº de pares).
        hit = 0.0
        for i, f in enumerate(folds):
            ds, cs = parts[i]
            # Semiabierto por la derecha, IGUAL que `pct_in_band`. Con `<=` la
            # banda se ensancha un bin y el LOO sale ~0.13 pp por encima del
            # auto-calibrado, que es imposible: lo detectó `verify_honest_metrics`.
            m = (ds >= lo + offs[f]) & (ds < hi + offs[f])
            hit += float(cs[m].sum())
        out[f"pct_ontime_{name}_cal"] = (100.0 * hit / n_tot) if n_tot else float("nan")
    return out


# ──────────────────────────────────────────────────────────────────
# Lectura de los artefactos
# ──────────────────────────────────────────────────────────────────

def honest_table(df_hist, bands: Optional[Dict[str, Tuple[float, float]]] = None,
                 causal: bool = False):
    """`*_devhist.parquet` (una o varias legs) → una fila por (route, leg, algorithm).

    Import de polars diferido: `xgb_prediction.cross_validation` importa este
    módulo solo por las constantes y no debe arrastrar polars.
    """
    import polars as pl

    need = {"route", "leg", "algorithm", "fold", "dev_s", "count"}
    missing = need - set(df_hist.columns)
    if missing:
        raise ValueError(f"al devhist le faltan columnas: {sorted(missing)}")

    rows = []
    for (route, leg, algo), g in df_hist.group_by(["route", "leg", "algorithm"],
                                                  maintain_order=True):
        hbf: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        for fold, gf in g.group_by("fold"):
            f = int(fold[0] if isinstance(fold, tuple) else fold)
            hbf[f] = (gf["dev_s"].to_numpy().astype(np.float64),
                      gf["count"].to_numpy().astype(np.int64))
        rows.append({"route": str(route), "leg": str(leg), "algorithm": str(algo),
                     **honest_from_hist(hbf, bands, causal=causal)})
    return pl.DataFrame(rows)
