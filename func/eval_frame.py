"""
eval_frame
==========
Marco de evaluación ÚNICO para las tres familias de modelos (baselines,
XGBoost, LSTM).

El problema que resuelve
-----------------------
Hasta ahora cada familia definía su propia muestra y sus propios folds:

===============  ============================================  =====================
familia          partición                                     población
===============  ============================================  =====================
baselines / XGB  ``rng.permutation(M)`` sobre FILAS de ``T``   ``build_trajectory_matrix``
                 (``arrival_prediction/cross_validation.py``)  (filtro de cobertura)
LSTM             ``rng.permutation`` sobre trip_ids ORDENADOS   todos los viajes con
                 (``lstm_prediction_tf/cross_validation.py``)   ``n_cells >= 5``
===============  ============================================  =====================

Con ``n_route_quantile=0.95`` eso significaba que el LSTM entrenaba con ~7,700
viajes por leg y los baselines con ~491 (15×), con poblaciones y folds distintos.
La saga del matched (v6/v7, ``idx_off`` / ``dt_rem``) fue el parche *a posteriori*
de esa asimetría.

Este módulo la elimina en el origen: un solo objeto define **qué viajes** entran,
**en qué fold** cae cada uno y **hasta dónde llega la grid**. Las tres familias lo
leen. La comparación queda pareada por construcción y el certificado matched se
reemplaza por un ``assert`` de igualdad de frames.

Alineación de índices entre familias
------------------------------------
``build_trajectory_matrix`` produce ``l_grid = [0, step, ..., n_route_m]``, así que
la columna ``l_idx`` de ``T`` está a distancia ``l_idx * step`` del inicio del
viaje. El builder del LSTM numera las celdas ``1..n_cells`` con ``n_cells =
d_max // step``, y el tiempo acumulado al final de la celda ``c`` es el tiempo a
distancia ``c * step``. Es decir: **celda c del LSTM ≡ l_idx = c de T**, siempre
que ambas se corten en el mismo ``n_route_m``. Por eso el frame guarda
``n_route_m``: el runner del LSTM trunca ahí (``max_cells``) y los índices dejan
de necesitar ``idx_off``.

Clave de viaje
--------------
``build_trajectory_matrix`` usa ``f"{bus_code}|{trip_id}"``; el builder del LSTM
usa ``trip_id`` pelado. Verificado en las 26 rutas del estudio (2026-07-24):
``trip_id`` ya incluye el código de bus (``SIT-0263_M2024-01-01_0_L0``) y **no
colisiona** ni entre ``bus_code`` ni entre ``leg``, así que la correspondencia es
1:1. El frame guarda las tres columnas y ``load_frame`` valida la biyección.

Uso
---
    from eval_frame import build_frame, save_frame, load_frame, frame_path

    fr = build_frame(df_full, leg="T0->T1", n_route_quantile=0.05,
                     max_trips=2500)
    save_frame(fr, route="101")
    ...
    fr = load_frame(route="101", leg="T0->T1", q=0.05)   # otra familia, mismo frame
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import polars as pl

FRAME_SCHEMA_VERSION = 1


# ──────────────────────────────────────────────────────────────────
# Particiones — copia literal de arrival_prediction, sin importarlo
# ──────────────────────────────────────────────────────────────────

def make_l_indices(n: int) -> List[int]:
    """Posiciones de corte. Copia literal de `arrival_prediction:47`."""
    return np.linspace(2, n - 2, min(30, n - 4), dtype=int).tolist()


def make_folds(M: int, n_folds: int = 5, seed: int = 42) -> List[np.ndarray]:
    """Particiones de viajes. Copia literal de `arrival_prediction:50-52`."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(M)
    return np.array_split(perm, n_folds)


#: Los dos regímenes de partición. NO son el mismo experimento remedido:
#: `random` responde "¿cuánta señal hay en la trayectoria?" y es válido para
#: modelos autorregresivos con errores no correlacionados (Bergmeir, Hyndman &
#: Koo 2018); `expanding` responde "¿sirve en despliegue?" — entrena con los
#: meses anteriores y predice el siguiente, que es la evaluación de origen
#: móvil estándar en pronóstico. Se reportan los dos, no uno en lugar del otro.
SPLIT_RANDOM    = "random"
SPLIT_EXPANDING = "expanding"
SPLIT_TAGS = {SPLIT_RANDOM: "", SPLIT_EXPANDING: "tw"}

#: Fracción de viajes de entrenamiento que cada régimen reserva para el early
#: stopping POR DEFECTO. `random` hereda el 0.0 histórico —el early stopping
#: miraba el fold evaluado (§7.11 del handoff)— y `expanding` exige 0.15, que es
#: lo que usa XGBoost, porque parar mirando el mes que se va a predecir es el
#: look-ahead que la escalera elimina.
#:
#: ⚠ Sirve para DOS cosas y las dos importan: resolver el default del runner, y
#: decidir si `es_frac` entra en el nombre del artefacto (ver `es_tag`).
DEFAULT_ES_FRAC = {SPLIT_RANDOM: 0.0, SPLIT_EXPANDING: 0.15}


def make_folds_expanding(periods: Sequence[str]) -> tuple:
    """Folds = periodo natural del viaje, en orden cronológico.

    El fold deja de ser un trozo intercambiable y pasa a ser una posición en el
    tiempo: el escalón *k* entrena con los folds ``< k`` y evalúa sobre el *k*.
    El fold 0 NUNCA es conjunto de prueba — no tiene historia detrás —, así que
    con *P* periodos salen *P−1* escalones.

    Devuelve ``(folds, period_labels)``; `folds[i]` son las filas del periodo
    *i*, que es lo que la columna `fold` del frame guarda.
    """
    labels = sorted(set(str(p) for p in periods))
    if len(labels) < 2:
        raise ValueError(
            f"la ventana expansiva necesita al menos 2 periodos y hay "
            f"{len(labels)}: sin historia previa no hay escalón que evaluar."
        )
    idx = {p: i for i, p in enumerate(labels)}
    fold_by_row = np.array([idx[str(p)] for p in periods], dtype=np.int32)
    folds = [np.flatnonzero(fold_by_row == i) for i in range(len(labels))]
    return folds, labels


def describe_rungs(folds: Sequence[np.ndarray],
                   period_labels: Sequence[str]) -> List[Dict[str, Any]]:
    """Tamaño de cada escalón de la escalera, para grabarlo en el meta.

    Es lo que decide qué legs entran en el panel temporal: el criterio
    pre-registrado pide 100 viajes, y en los primeros escalones no todas los
    tienen. Grabarlo por leg evita que cada análisis lo re-derive.
    """
    return [{
        "rung":          k,
        "train_periods": list(period_labels[:k]),
        "test_period":   period_labels[k],
        "n_train":       int(sum(len(folds[j]) for j in range(k))),
        "n_test":        int(len(folds[k])),
    } for k in range(1, len(period_labels))]


def trip_periods(df_full: "pl.DataFrame", trip_keys: Sequence[str],
                 group_col: str, leg: Any, freq: str = "month") -> List[str]:
    """Periodo natural de cada viaje del frame, por su PRIMER ping.

    Se lee de `df_full` y no del texto del `trip_id` — que también lleva la
    fecha — porque una convención de nombre no es un dato: si cambiara, el
    parseo seguiría devolviendo algo y la partición saldría mal sin avisar.
    """
    fmt = {"month": "%Y-%m", "week": "%G-W%V", "day": "%Y-%m-%d"}[freq]
    d = (df_full
         .filter(pl.col(group_col) == leg, pl.col("trip_valid"))
         .group_by(["bus_code", "trip_id"])
         .agg(pl.col("time").min().alias("t0")))
    start = {f"{r['bus_code']}|{r['trip_id']}": r["t0"]
             for r in d.iter_rows(named=True)}

    out, missing = [], []
    for k in trip_keys:
        t0 = start.get(str(k))
        if t0 is None:
            missing.append(str(k))
            continue
        out.append(t0.strftime(fmt))
    if missing:
        raise ValueError(
            f"leg {leg!r}: {len(missing)} viaje(s) del frame sin hora de inicio "
            f"en df_full (p. ej. {missing[:3]}). Sin fecha no hay partición "
            f"temporal posible; abortar es preferible a inventar un periodo."
        )
    return out


def subsample_idx(M: int, max_trips: Optional[int], seed: int = 42) -> Optional[np.ndarray]:
    """Índices a conservar si hay tope de viajes. Copia literal de
    `xgb_prediction/run_experiment.py:156-158` — MISMO rng y MISMO orden, para
    que el frame reproduzca exactamente la muestra de la corrida XGB."""
    if max_trips is None or M <= max_trips:
        return None
    return np.sort(np.random.default_rng(seed).choice(M, size=max_trips,
                                                      replace=False))


# ──────────────────────────────────────────────────────────────────
# Etiquetas y rutas de disco
# ──────────────────────────────────────────────────────────────────

def safe_label(group_value: Any) -> str:
    """Etiqueta de archivo para una leg. Copia de `_safe_label` del paso 10/12."""
    if isinstance(group_value, (int, np.integer)) or (
            isinstance(group_value, str) and str(group_value).isdigit()):
        return f"dir{group_value}"
    s = str(group_value).replace("->", "_").replace(" ", "")
    s = "".join(c if (c.isalnum() or c in "_-") else "_" for c in s)
    return f"leg{s}"


#: Cobertura histórica. Las corridas anteriores a 2026-07-24 la usaban y sus
#: archivos NO llevan sufijo de cobertura; se preserva para no invalidarlos.
_LEGACY_COVERAGE = 0.99


def es_tag(es_frac: Optional[float], split: str = SPLIT_RANDOM) -> str:
    """Sufijo de early stopping: ``""`` si es el default del régimen, ``es15``
    si no. Es el TERCER eje de nombre, por la misma razón que los otros dos.

    ⚠ Solo etiqueta la DESVIACIÓN, y esto no es cosmético. Las corridas
    publicadas son `random` con 0.0 y `expanding` con 0.15: las dos son el
    default de su régimen, así que las dos siguen sin sufijo y **ni un archivo
    existente cambia de nombre**. Etiquetar por valor absoluto renombraría los
    artefactos de la escalera del 19-ago y sus 15 GPU-h volverían a parecer
    pendientes.
    """
    if es_frac is None:
        return ""
    default = DEFAULT_ES_FRAC.get(split, 0.0)
    if abs(float(es_frac) - default) < 1e-12:
        return ""
    return f"es{int(round(float(es_frac) * 100)):02d}"


def q_tag(n_route_quantile: float, min_coverage: float = _LEGACY_COVERAGE,
          split: str = SPLIT_RANDOM, es_frac: Optional[float] = None,
          variant: Optional[str] = None) -> str:
    """
    Etiqueta de régimen para nombres de archivo: ``q05`` (cobertura histórica),
    ``q05c100`` (cualquier otra), ``q05c100_tw`` (partición temporal),
    ``q05c100_es15`` (early stopping distinto al del régimen) o
    ``q05c100_np`` (variante de features).

    ⚠ La cobertura TIENE que entrar en el nombre. Sin ella, una corrida con
    ``min_coverage=1.0`` encuentra los agregados de una corrida previa con 0.99,
    los da por buenos y se salta el trabajo entero como ``cached`` — pasó el
    2026-07-24: las 26 rutas se saltaron contra los archivos del 23-jul y no se
    escribió ni un frame. `q` y `cov` definen tareas DISTINTAS, no la misma
    tarea remedida.

    ⚠ Y `es_frac` tampoco es una remedición: entró como tercer eje el 2026-08-23
    al preparar la corrida honesta que pide el revisor. Una CV aleatoria con
    ``es_frac=0.15`` escribe sobre la publicada si no se distingue en el nombre,
    y las dos salidas del fallo son malas — sin ``--force`` todo sale ``cached``
    en segundos y parece éxito; con ``--force`` sobrescribe los artefactos que
    el manuscrito reporta. Se quiere lo contrario: que **coexistan**, porque la
    comparación entre ambas es la que mide la asimetría que §7.11 solo declara.
    """
    q = f"q{int(round(n_route_quantile * 100)):02d}"
    if abs(float(min_coverage) - _LEGACY_COVERAGE) >= 1e-12:
        q = f"{q}c{int(round(float(min_coverage) * 100)):03d}"
    # La partición entra en el nombre por la MISMA razón que la cobertura: un
    # frame temporal y uno aleatorio son tareas distintas sobre la misma leg, y
    # sin sufijo el segundo se saltaría como `cached` contra el primero.
    if split not in SPLIT_TAGS:
        raise ValueError(f"split desconocido: {split!r} "
                         f"(esperaba {sorted(SPLIT_TAGS)})")
    # El orden de los sufijos es parte del contrato: régimen → partición →
    # early stopping → variante de features. Cambiarlo renombra artefactos.
    for suf in (SPLIT_TAGS[split], es_tag(es_frac, split), variant or ""):
        if suf:
            q = f"{q}_{suf}"
    return q


def frame_dir(route: str | int, analysis_range: Optional[str] = None,
              root: Optional[Path] = None) -> Path:
    rng_ = analysis_range or os.environ.get("BUS_ANALYSIS_RANGE", "2024_01-08_tmin")
    base = Path(root) if root is not None else _repo_root()
    return base / "data" / "02_interim" / "routes" / str(route) / "frames" / rng_


def frame_path(route: str | int, leg: Any, n_route_quantile: float,
               analysis_range: Optional[str] = None,
               root: Optional[Path] = None,
               min_coverage: float = _LEGACY_COVERAGE,
               split: str = SPLIT_RANDOM) -> Path:
    d = frame_dir(route, analysis_range, root)
    tag = q_tag(n_route_quantile, min_coverage, split)
    return d / f"{safe_label(leg)}_{tag}.parquet"


def _repo_root() -> Path:
    env = os.environ.get("BUSONORA_ROOT")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parent.parent


# ──────────────────────────────────────────────────────────────────
# Hash de identidad
# ──────────────────────────────────────────────────────────────────

def compute_hash(trip_ids: Sequence[str], folds_by_row: Sequence[int],
                 n_route_m: int, grid_step_m: int,
                 l_indices: Sequence[int]) -> str:
    """
    Huella del frame: define la IDENTIDAD de la tarea de evaluación.

    Se ordena por trip_id para que el hash NO dependa del orden de las filas
    (dos familias pueden recorrer los viajes en orden distinto y seguir siendo
    el mismo frame). Incluye la geometría (`n_route_m`, `grid_step_m`) y los
    cortes: un cambio ahí cambia la tarea, no solo la muestra.
    """
    pairs = sorted((str(t), int(f)) for t, f in zip(trip_ids, folds_by_row))
    h = hashlib.sha256()
    h.update(f"v{FRAME_SCHEMA_VERSION}|{int(n_route_m)}|{int(grid_step_m)}|".encode())
    h.update(",".join(str(int(l)) for l in l_indices).encode())
    h.update(b"|")
    for t, f in pairs:
        h.update(f"{t}:{f};".encode())
    return h.hexdigest()[:16]


# ──────────────────────────────────────────────────────────────────
# Construcción
# ──────────────────────────────────────────────────────────────────

def build_frame(
    df_full:          pl.DataFrame,
    leg:              Any,
    n_route_quantile: float = 0.05,
    grid_step_m:      int   = 100,
    min_coverage:     float = 0.99,
    n_folds:          int   = 5,
    seed:             int   = 42,
    max_trips:        Optional[int] = None,
    group_col:        str   = "leg",
    route:            Optional[str] = None,
    analysis_range:   Optional[str] = None,
    split:            str   = SPLIT_RANDOM,
    period:           str   = "month",
    verbose:          bool  = False,
) -> Dict[str, Any]:
    """
    Construye el frame de una leg: muestra + folds + geometría de la grid.

    Reproduce EXACTAMENTE el preámbulo de `xgb_prediction.run_one_leg`
    (build_trajectory_matrix → submuestreo opcional → make_folds), así que el
    frame que se persiste es el que esa corrida usó, no una reconstrucción
    aproximada.

    Returns
    -------
    dict con 'T', 'l_grid', 'trip_ids', 'folds', 'fold_by_row', 'table', 'meta'.
    """
    from arrival_prediction import build_trajectory_matrix  # import perezoso

    n_trips_raw = (
        df_full.filter(pl.col(group_col) == leg, pl.col("trip_valid"))
        ["trip_id"].n_unique()
    )

    ds = build_trajectory_matrix(
        df_full, group_col=group_col, group_value=leg,
        grid_step_m=grid_step_m, n_route_quantile=n_route_quantile,
        min_coverage=min_coverage, verbose=False,
    )
    T        = ds["T"]
    l_grid   = ds["l_grid"]
    trip_keys = list(ds["trip_ids"])
    M_full   = T.shape[0]

    keep = subsample_idx(M_full, max_trips, seed)
    if keep is not None:
        T = T[keep]
        trip_keys = [trip_keys[i] for i in keep]
    M = T.shape[0]

    if M < n_folds:
        raise ValueError(
            f"leg {leg!r}: M={M} < n_folds={n_folds}; no hay frame posible."
        )

    n = T.shape[1] - 1
    l_indices = [int(l) for l in make_l_indices(n) if l < n]

    periods: Optional[List[str]] = None
    if split == SPLIT_EXPANDING:
        # El número de folds lo fijan los datos, no el argumento: hay tantos
        # como periodos con viajes en esta leg.
        periods = trip_periods(df_full, trip_keys, group_col, leg, period)
        folds, period_labels = make_folds_expanding(periods)
        n_folds = len(period_labels)
    else:
        period_labels = []
        folds = make_folds(M, n_folds, seed)

    fold_by_row = np.empty(M, dtype=np.int32)
    for fid, idx in enumerate(folds):
        fold_by_row[idx] = fid

    # `bus_code|trip_id` → columnas separadas. El split es por el PRIMER '|'
    # porque trip_id puede contener guiones/underscores pero no '|'.
    bus_codes, trip_ids = [], []
    for k in trip_keys:
        bus, _, trip = str(k).partition("|")
        bus_codes.append(bus)
        trip_ids.append(trip if trip else bus)

    if len(set(trip_ids)) != M:
        dup = M - len(set(trip_ids))
        raise ValueError(
            f"leg {leg!r}: {dup} trip_id duplicado(s) tras separar la clave "
            f"'bus_code|trip_id'. El frame exige biyección trip_id ↔ fila porque "
            f"el builder del LSTM indexa por trip_id pelado."
        )

    table = pl.DataFrame({
        "row_idx":  np.arange(M, dtype=np.int32),
        "trip_key": trip_keys,
        "bus_code": bus_codes,
        "trip_id":  trip_ids,
        "fold":     fold_by_row,
    })

    fhash = compute_hash(trip_ids, fold_by_row, int(ds["n_route"]),
                         grid_step_m, l_indices)

    # Tamaño de cada escalón, grabado por leg. Es lo que decide qué legs entran
    # en el panel temporal —el criterio pre-registrado pide 100 viajes de
    # entrenamiento y en el primer escalón no lo cumplen todas— y grabarlo aquí
    # evita que cada análisis lo re-derive por su cuenta.
    rungs = (describe_rungs(folds, period_labels)
             if split == SPLIT_EXPANDING else [])

    meta = {
        "frame_schema_version": FRAME_SCHEMA_VERSION,
        "route":            str(route) if route is not None else None,
        "analysis_range":   analysis_range,
        "group_col":        group_col,
        "leg":              str(leg),
        "label":            safe_label(leg),
        "n_route_quantile": n_route_quantile,
        "grid_step_m":      grid_step_m,
        "min_coverage":     min_coverage,
        "n_route_m":        int(ds["n_route"]),
        "n_grid_pts":       int(len(l_grid)),
        "n_cells":          int(len(l_grid) - 1),
        "l_indices":        l_indices,
        "n_folds":          n_folds,
        "split":            split,
        "period":           period if split == SPLIT_EXPANDING else None,
        "periods":          period_labels,
        "rungs":            rungs,
        "seed":             seed,
        "n_trips_raw":      int(n_trips_raw),
        "M_retained":       int(M_full),
        "retention_pct":    round(100.0 * M_full / max(n_trips_raw, 1), 2),
        "max_trips":        max_trips,
        "subsampled":       keep is not None,
        "M":                int(M),
        "fold_sizes":       [int(len(f)) for f in folds],
        "frame_hash":       fhash,
        "created_utc":      datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    if verbose:
        print(f"  frame {meta['label']} "
              f"{q_tag(n_route_quantile, min_coverage, split)}: "
              f"M={M:,} (de {n_trips_raw:,} crudos, ret={meta['retention_pct']}%) "
              f"· n_route={meta['n_route_m']:,} m ({meta['n_cells']} celdas) "
              f"· folds={meta['fold_sizes']} · hash={fhash}")
        if split == SPLIT_EXPANDING:
            print(f"    escalera de {len(rungs)} escalones sobre "
                  f"{period_labels[0]}…{period_labels[-1]}: " +
                  " · ".join(f"→{r['test_period'][-2:]} "
                             f"({r['n_train']}/{r['n_test']})" for r in rungs))

    return {"T": T, "l_grid": l_grid, "trip_ids": trip_ids,
            "trip_keys": trip_keys, "folds": folds,
            "fold_by_row": fold_by_row, "table": table, "meta": meta}


# ──────────────────────────────────────────────────────────────────
# Persistencia
# ──────────────────────────────────────────────────────────────────

def save_frame(frame: Dict[str, Any], route: Optional[str] = None,
               analysis_range: Optional[str] = None,
               root: Optional[Path] = None, verbose: bool = False) -> Path:
    """Escribe `<leg>_q<NN>.parquet` + `<leg>_q<NN>_meta.json`."""
    meta = dict(frame["meta"])
    route = str(route if route is not None else meta.get("route"))
    if route in (None, "None", ""):
        raise ValueError("save_frame necesita `route` (o meta['route']).")
    rng_ = analysis_range or meta.get("analysis_range") \
        or os.environ.get("BUS_ANALYSIS_RANGE", "2024_01-08_tmin")
    meta["route"] = route
    meta["analysis_range"] = rng_

    d = frame_dir(route, rng_, root)
    d.mkdir(parents=True, exist_ok=True)
    stem = (f"{meta['label']}_"
            f"{q_tag(meta['n_route_quantile'], meta.get('min_coverage', _LEGACY_COVERAGE), meta.get('split', SPLIT_RANDOM))}")
    p_tab = d / f"{stem}.parquet"
    p_met = d / f"{stem}_meta.json"

    # El frame NUNCA debe quedar vacío en disco (guard del proyecto).
    if frame["table"].height == 0:
        raise ValueError(f"frame vacío para {stem}: no se escribe.")

    frame["table"].write_parquet(p_tab)
    p_met.write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                     encoding="utf-8")
    if verbose:
        print(f"  ✓ frame → {p_tab.name} (M={frame['table'].height:,}, "
              f"hash={meta['frame_hash']})")
    return p_tab


def load_frame(route: str | int, leg: Any, n_route_quantile: float,
               analysis_range: Optional[str] = None,
               root: Optional[Path] = None,
               min_coverage: float = _LEGACY_COVERAGE,
               split: str = SPLIT_RANDOM) -> Dict[str, Any]:
    """
    Lee un frame de disco. Devuelve dict con 'table', 'meta', 'trip_ids',
    'fold_of' (trip_id → fold) y 'folds' (lista de arrays de row_idx).

    Valida la huella: si el parquet y el meta no concuerdan, aborta. Un frame
    corrupto o de otra corrida silenciaría toda la comparación pareada.
    """
    d = frame_dir(route, analysis_range, root)
    stem = f"{safe_label(leg)}_{q_tag(n_route_quantile, min_coverage, split)}"
    p_tab, p_met = d / f"{stem}.parquet", d / f"{stem}_meta.json"
    if not p_tab.exists() or not p_met.exists():
        raise FileNotFoundError(
            f"no hay frame para ruta {route} leg {leg} "
            f"{q_tag(n_route_quantile, min_coverage, split)}: "
            f"falta {p_tab if not p_tab.exists() else p_met}. Genéralo con "
            f"`python func/xgb_prediction/run_experiment.py --write-frames` o "
            f"`python func/eval_frame.py --route {route}`."
        )

    tab  = pl.read_parquet(p_tab)
    meta = json.loads(p_met.read_text(encoding="utf-8"))

    recomputed = compute_hash(tab["trip_id"].to_list(), tab["fold"].to_list(),
                              meta["n_route_m"], meta["grid_step_m"],
                              meta["l_indices"])
    if recomputed != meta["frame_hash"]:
        raise ValueError(
            f"frame {stem} CORRUPTO: hash del parquet {recomputed} ≠ "
            f"{meta['frame_hash']} del meta. No se puede usar para una "
            f"comparación pareada."
        )

    folds: List[np.ndarray] = [
        tab.filter(pl.col("fold") == f)["row_idx"].to_numpy()
        for f in range(meta["n_folds"])
    ]
    return {
        "table":    tab,
        "meta":     meta,
        "trip_ids": tab["trip_id"].to_list(),
        "fold_of":  dict(zip(tab["trip_id"].to_list(), tab["fold"].to_list())),
        "folds":    folds,
        "path":     p_tab,
    }


def frame_from_arrays(trip_ids: Sequence[str], fold_by_row: Sequence[int],
                      n_route_m: int, grid_step_m: int,
                      l_indices: Sequence[int], **meta_extra) -> Dict[str, Any]:
    """
    Empaqueta un frame desde arrays ya calculados por un runner (sin re-construir
    T). Lo usa `xgb_prediction.run_one_leg`, que ya tiene la muestra y los folds
    en memoria: así lo que se persiste es lo que la corrida usó de verdad.
    """
    trip_ids = [str(t) for t in trip_ids]
    fold_by_row = np.asarray(fold_by_row, dtype=np.int32)
    M = len(trip_ids)
    bus_codes, tids = [], []
    for k in trip_ids:
        bus, _, trip = k.partition("|")
        bus_codes.append(bus)
        tids.append(trip if trip else bus)
    if len(set(tids)) != M:
        raise ValueError("trip_id duplicado: el frame exige biyección.")

    table = pl.DataFrame({
        "row_idx":  np.arange(M, dtype=np.int32),
        "trip_key": trip_ids,
        "bus_code": bus_codes,
        "trip_id":  tids,
        "fold":     fold_by_row,
    })
    l_indices = [int(l) for l in l_indices]
    meta = {
        "frame_schema_version": FRAME_SCHEMA_VERSION,
        "n_route_m":  int(n_route_m),
        "grid_step_m": int(grid_step_m),
        "l_indices":  l_indices,
        "M":          M,
        "frame_hash": compute_hash(tids, fold_by_row, int(n_route_m),
                                   int(grid_step_m), l_indices),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    meta.update(meta_extra)
    return {"table": table, "meta": meta, "trip_ids": tids,
            "fold_by_row": fold_by_row}


# ──────────────────────────────────────────────────────────────────
# CLI: generar los frames de una ruta sin correr ningún modelo
# ──────────────────────────────────────────────────────────────────

def main() -> int:
    import argparse
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import PATHS, INT_ROUTE, ANALYSIS_RANGE   # noqa: E402

    ap = argparse.ArgumentParser(
        description="Genera los frames de evaluación de una ruta (sin modelos)")
    ap.add_argument("--legs", default=None, help="lista separada por comas")
    ap.add_argument("--n-route-quantile", type=float, default=0.05)
    ap.add_argument("--grid-step-m", type=int, default=100)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-trips", type=int, default=None)
    ap.add_argument("--min-coverage", type=float, default=1.0)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    df_full = pl.read_parquet(PATHS.df_full)
    if a.legs:
        legs = [s.strip() for s in a.legs.split(",") if s.strip()]
    else:
        legs = (
            df_full.filter(pl.col("trip_valid"))
            .group_by("leg").agg(pl.col("trip_id").n_unique().alias("m"))
            .sort("m", descending=True)["leg"].to_list()
        )

    print(f"  frames · ruta {INT_ROUTE} · {ANALYSIS_RANGE} · "
          f"{q_tag(a.n_route_quantile, a.min_coverage)} · legs={legs}")
    n_ok = 0
    for leg in legs:
        p = frame_path(INT_ROUTE, leg, a.n_route_quantile, ANALYSIS_RANGE,
                       min_coverage=a.min_coverage)
        if p.exists() and not a.force:
            print(f"  · {safe_label(leg)}: ya existe (--force para rehacer)")
            n_ok += 1
            continue
        try:
            fr = build_frame(
                df_full, leg, n_route_quantile=a.n_route_quantile,
                grid_step_m=a.grid_step_m, n_folds=a.n_folds, seed=a.seed,
                max_trips=a.max_trips, min_coverage=a.min_coverage,
                route=str(INT_ROUTE),
                analysis_range=ANALYSIS_RANGE, verbose=True,
            )
        except ValueError as e:
            print(f"  ⚠ {safe_label(leg)} SALTADA: {e}")
            continue
        save_frame(fr, route=str(INT_ROUTE), analysis_range=ANALYSIS_RANGE,
                   verbose=True)
        n_ok += 1

    print(f"\n  ✅ {n_ok}/{len(legs)} frames en "
          f"{frame_dir(INT_ROUTE, ANALYSIS_RANGE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
