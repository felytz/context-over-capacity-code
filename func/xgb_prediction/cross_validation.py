"""
xgb_prediction.cross_validation
=================================
Etapa C: validación cruzada MATCHED en streaming.

Por qué un harness nuevo y no `arrival_prediction.run_cross_validation`
-----------------------------------------------------------------------
Aquel materializa un `df_errors` con una fila por (algoritmo, fold, viaje,
l_idx, h). Para la ruta 101 eso son ~297M filas por algoritmo (medido en
`101_fullgrid.parquet`: `n_pairs = 297,349,450`). Con 5 algoritmos no cabe en
disco ni en RAM.

Este harness evalúa TODOS los algoritmos sobre EXACTAMENTE las mismas filas, en
bloques de (fold, l_idx), y acumula los estadísticos al vuelo. Nunca materializa
la tabla de errores.

Paridad matched por construcción
--------------------------------
Se replican literalmente las particiones de `arrival_prediction`:

    l_indices = np.linspace(2, n - 2, min(30, n - 4), dtype=int)
    rng   = np.random.default_rng(seed)      # seed=42
    perm  = rng.permutation(M)
    folds = np.array_split(perm, n_folds)    # n_folds=5

Como baselines y XGBoost comparten `T`, `l_grid`, `trip_ids`, folds y filas de
evaluación, la comparación es pareada sin necesidad del protocolo matched
offline (idx_off / dt_rem) que hizo falta para cruzar corridas separadas.

Espacio de trabajo: tiempo RESTANTE
-----------------------------------
Todo se evalúa sobre ``dt_rem``. Es equivalente al espacio acumulado porque

    |t_pred − t_true| = |(t_obs_l + dt_pred) − (t_obs_l + dt_true)| = |dt_pred − dt_true|
    dev = t_true − t_pred = dt_true − dt_pred

así que MAE, RMSE, MedAE, P90AE y % on-time son idénticos en ambos espacios.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

# Cuántos hilos ve el proceso. XGBoost corre con `n_jobs=-1` (model.py) y los
# `.slrm` exportan `OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK` (16), así que su
# reloj de PARED no es tiempo de CPU y compararlo a pelo contra un baseline de
# numpy lo favorece hasta 16×. Se graba en el artefacto en vez de asumirlo en el
# análisis, que es donde rompería en silencio si el `.slrm` cambiara de
# `cpus-per-task`.
_OMP_THREADS = int(os.environ.get("OMP_NUM_THREADS") or 0)

from .feature_builder import (
    FEATURE_NAMES,
    build_block,
    compute_prior,
    feature_names,
)

# `honest_metrics` vive en la raíz de func/, como `config` y `eval_frame`. Los
# entrypoints ya la insertan en sys.path, pero este módulo también se importa
# como paquete (`from xgb_prediction import ...`) desde otros directorios.
_FUNC_ROOT = str(Path(__file__).resolve().parent.parent)
if _FUNC_ROOT not in sys.path:
    sys.path.insert(0, _FUNC_ROOT)

from honest_metrics import (                                    # noqa: E402
    DEV_BIN_S,
    DEV_MAX_ABS_S,
    HORIZON_LABELS,
    bin_index,
    bin_left,
    horizon_bin,
    n_bins as _dev_n_bins,
)

# Banda operacional APTA — MISMA convención que el paso 10 y el paso 12:
# dev = t_true − t_pred; + = el bus llega DESPUÉS de lo predicho. NO invertir
# (ver docs/historial_paper/11_pre_resultados_auditoria).
APTA_EARLY_S = -60.0
APTA_LATE_S  = 300.0

#: Viajes de entrenamiento mínimos para que un escalón pueda correr XGBoost.
#: El holdout de early stopping (`es_frac`, Tabla S9) reserva viajes ENTEROS, así
#: que con uno solo el ajuste se queda VACÍO. No es un umbral de calidad —de eso
#: se encarga el criterio pre-registrado de 100 viajes al fijar el panel— sino el
#: mínimo estructural para que fit y holdout coexistan.
MIN_TRAIN_TRIPS_ES = 2


class InsufficientDataError(ValueError):
    """M < n_folds: algún fold de entrenamiento quedaría vacío."""


# ──────────────────────────────────────────────────────────────────
# Acumulador de errores en streaming
# ──────────────────────────────────────────────────────────────────

class ErrorAccumulator:
    """
    Acumula estadísticos de error sin guardar los residuos.

    Sumas exactas para MAE y RMSE; histograma de 1 s para los cuantiles
    (MedAE, P90AE). Con MAE del orden de 250 s, un bin de 1 s es ruido
    despreciable.

    Con ``track_dev`` (fase 5) guarda además el histograma **con signo** de
    ``dev = t_true − t_pred``. Es lo que permite derivar a posteriori CUALQUIER
    banda de on-time (p. ej. la simétrica ±180 s) y el on-time calibrado por
    sesgo, sin volver a correr: ver `honest_metrics`. Pesa ~115 KB por
    acumulador, así que solo se activa donde hace falta (overall y por fold,
    no en los 30 acumuladores por posición).
    """

    def __init__(self, max_abs_s: float = 20000.0, bin_s: float = 1.0,
                 track_dev: bool = False,
                 dev_max_abs_s: float = DEV_MAX_ABS_S) -> None:
        self.bin_s   = float(bin_s)
        self.n_bins  = int(max_abs_s / bin_s) + 1
        self.hist    = np.zeros(self.n_bins + 1, dtype=np.int64)  # +1 = overflow
        self.n          = 0
        self.sum_abs    = 0.0
        self.sum_sq     = 0.0
        self.sum_signed = 0.0
        self.n_ontime   = 0

        self.dev_max_abs_s = float(dev_max_abs_s)
        self.dev_hist = (np.zeros(_dev_n_bins(self.bin_s, self.dev_max_abs_s),
                                  dtype=np.int64) if track_dev else None)

    def update(self, dt_pred: np.ndarray, dt_true: np.ndarray) -> None:
        err = (dt_pred - dt_true).astype(np.float64)
        abs_err = np.abs(err)

        self.n          += err.size
        self.sum_abs    += float(abs_err.sum())
        self.sum_sq     += float((err ** 2).sum())
        self.sum_signed += float(err.sum())

        # dev = t_true − t_pred = −err
        dev = -err
        self.n_ontime += int(np.count_nonzero(
            (dev >= APTA_EARLY_S) & (dev <= APTA_LATE_S)
        ))

        idx = np.minimum((abs_err / self.bin_s).astype(np.int64), self.n_bins)
        self.hist += np.bincount(idx, minlength=self.n_bins + 1)

        if self.dev_hist is not None:
            d_idx = bin_index(dev, self.bin_s, self.dev_max_abs_s)
            self.dev_hist += np.bincount(d_idx, minlength=self.dev_hist.size)

    def dev_bins(self) -> tuple:
        """Histograma disperso de ``dev``: (bordes izquierdos en s, cuentas).

        Solo los bins no vacíos — de 14,401 posibles suelen estar poblados unos
        pocos miles, así que el parquet resultante pesa KB.
        """
        if self.dev_hist is None:
            return np.zeros(0), np.zeros(0, dtype=np.int64)
        nz = np.nonzero(self.dev_hist)[0]
        return (bin_left(nz, self.bin_s, self.dev_max_abs_s),
                self.dev_hist[nz])

    def _quantile(self, q: float) -> float:
        if self.n == 0:
            return float("nan")
        target = q * self.n
        cum = np.cumsum(self.hist)
        i = int(np.searchsorted(cum, target, side="left"))
        return float(i * self.bin_s)

    def result(self) -> Dict[str, float]:
        if self.n == 0:
            return {"n_pairs": 0, "MAE_s": float("nan"), "RMSE_s": float("nan"),
                    "MedAE_s": float("nan"), "P90AE_s": float("nan"),
                    "bias_s": float("nan"), "pct_ontime_dev": float("nan")}
        return {
            "n_pairs":        int(self.n),
            "MAE_s":          self.sum_abs / self.n,
            "RMSE_s":         float(np.sqrt(self.sum_sq / self.n)),
            "MedAE_s":        self._quantile(0.50),
            "P90AE_s":        self._quantile(0.90),
            # bias = media de (pred − true). Negativo = subestima el tiempo
            # restante ⇒ dev = true − pred positivo ⇒ cae del lado TARDE de la
            # banda, que es el lado ancho ([−60, +300]). Necesario para saber si
            # una ganancia de on-time viene de precisión o de sesgo.
            "bias_s":         self.sum_signed / self.n,
            "pct_ontime_dev": 100.0 * self.n_ontime / self.n,
        }


# ──────────────────────────────────────────────────────────────────
# Particiones (idénticas a arrival_prediction)
# ──────────────────────────────────────────────────────────────────

def make_l_indices(n: int) -> List[int]:
    """Posiciones de corte — copia literal de `arrival_prediction`."""
    return np.linspace(2, n - 2, min(30, n - 4), dtype=int).tolist()


def make_folds(M: int, n_folds: int, seed: int = 42) -> List[np.ndarray]:
    """Particiones de viajes — copia literal de `arrival_prediction`."""
    rng  = np.random.default_rng(seed)
    perm = rng.permutation(M)
    return np.array_split(perm, n_folds)


# ──────────────────────────────────────────────────────────────────
# CV principal
# ──────────────────────────────────────────────────────────────────

#: Tiers de la ablación de INFORMACIÓN, por debajo de Tier A. Son sentinelas y
#: no tuplas de columnas exógenas a propósito: así `isinstance(tier, tuple)`
#: sigue distinguiendo "subconjunto de exógenos" de "Tier A", y estos casos no
#: se confunden con ninguno de los dos.
#:
#:   NO_PRIOR  nivel 1 — Tier A sin `PRIOR_FEATURES`
#:   TRIP_ONLY nivel 0 — además sin `delay_vs_mu`, el gancho histórico que vive
#:             escondido en el bloque de viaje. Es el único verdaderamente
#:             *trajectory-only*.
NO_PRIOR  = "no_prior"
TRIP_ONLY = "trip_only"

#: (with_prior, with_delay_vs_mu) por sentinela. UNA sola tabla: mientras las
#: banderas estuvieron esparcidas por el módulo, añadir un nivel obligaba a
#: tocar cinco sitios y olvidarse de uno solo —el de la ruta de EVALUACIÓN—
#: producía matrices de distinto ancho entre ajuste y predicción.
_TIER_FLAGS = {
    NO_PRIOR:  (False, True),
    TRIP_ONLY: (False, False),
}


def _tier_cols(tier: Any) -> Optional[List[int]]:
    """Columnas exógenas de un tier. `None` para Tier A y para los sentinelas."""
    return list(tier) if isinstance(tier, tuple) else None


def _tier_flags(tier: Any) -> tuple:
    """(with_prior, with_delay_vs_mu) — Tier A y los tiers exógenos llevan todo."""
    return _TIER_FLAGS.get(tier, (True, True))


def _tier_label(tier: Any) -> str:
    """Etiqueta del tier para el reporte de timings y para los logs.

    ⚠ Una sola definición para los dos consumidores. Cuando el sentinela era
    solo una rama más, `f"+{len(tier)}exo"` lo etiquetaba como `+8exo` —cuenta
    las letras de la cadena— y la Tabla 12 habría atribuido el coste del nivel 1
    a un tier exógeno inexistente.
    """
    if tier is None:
        return "A"
    if tier is NO_PRIOR:
        return "A-prior"
    if tier is TRIP_ONLY:
        return "A-hist"
    return f"+{len(tier)}exo"


def _stack_blocks(
    T_src:     np.ndarray,
    l_grid:    np.ndarray,
    l_indices: Sequence[int],
    mu:        np.ndarray,
    priors:    Dict[int, Dict[str, np.ndarray]],
    n_h_train: int,
    rng:       np.random.Generator,
    exo_src:   Optional[np.ndarray] = None,
    with_prior: bool = True,
    with_delay_vs_mu: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Apila bloques con horizontes submuestreados (para entrenar / early stop)."""
    n = T_src.shape[1] - 1
    Xs, ys = [], []
    for li, l in enumerate(l_indices):
        n_h_full = n - l
        k = min(n_h_train, n_h_full)
        h_sub = np.sort(rng.choice(n_h_full, size=k, replace=False))
        Xb, yb, _ = build_block(
            T_src, l_grid, l, mu, priors[l], h_subset=h_sub,
            exo=(exo_src[:, li, :] if exo_src is not None else None),
            with_prior=with_prior, with_delay_vs_mu=with_delay_vs_mu,
        )
        Xs.append(Xb)
        ys.append(yb)
    return np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0)


def run_matched_cv(
    T:              np.ndarray,
    l_grid:         np.ndarray,
    baselines:      Optional[Dict[str, Any]] = None,
    xgb_models:     Optional[Dict[str, Any]] = None,
    exo:            Optional[np.ndarray] = None,
    strata:         Optional[np.ndarray] = None,
    n_folds:        int   = 5,
    seed:           int   = 42,
    folds:          Optional[Sequence[np.ndarray]] = None,
    split:          str   = "random",
    l_indices:      Optional[Sequence[int]] = None,
    n_h_train:      int   = 20,
    max_train_rows: int   = 4_000_000,
    es_frac:        float = 0.15,
    verbose:        bool  = False,
) -> Dict[str, Any]:
    """
    Corre la CV pareada sobre todos los algoritmos.

    Parameters
    ----------
    T : (M, n+1) matriz de trayectorias de `build_trajectory_matrix`.
    l_grid : (n+1,) distancias en metros.
    baselines : dict[str, BasePredictor] | None
        Predictores de `arrival_prediction` (delay, linreg, knn, kernel). Se
        evalúan en las MISMAS filas que XGBoost.
    xgb_models : dict[str, callable] | None
        Nombre → factory sin argumentos que devuelve un `XGBRemainingTime`
        fresco por fold. Permite reportar varias variantes (p.ej. `xgb_mse` y
        `xgb_mae`) sobre exactamente las mismas filas. None = solo baselines
        (modo del test de fidelidad).

        Cada entrada puede ser la factory sin más (Tier A) o la tupla
        ``(factory, "B")`` para que esa variante consuma también los exógenos.
        Así Tier A y Tier B se entrenan y evalúan en el MISMO loop, sobre las
        MISMAS filas: la comparación A-vs-B queda pareada por construcción.
    exo : np.ndarray (M, n_l, k) | None
        Tensor de Tier B (`exo_features.build_exo_matrix`), alineado fila a fila
        con `T` y columna a columna con `l_indices`.
    strata : np.ndarray (M, n_l) | None
        Estrato por (viaje, corte) — hora del día. Solo lo consumen los
        predictores que declaran `needs_strata` (p.ej. `HourStratifiedDelay`);
        el resto ignora este argumento.
    n_h_train : int
        Horizontes muestreados por (viaje, l_idx) en ENTRENAMIENTO. La
        evaluación siempre usa el horizonte completo: submuestrear en eval
        rompería la paridad con los baselines.
    max_train_rows : int
        Tope de filas de entrenamiento por fold (submuestreo aleatorio).
    es_frac : float
        Fracción de los VIAJES de train reservada para early stopping. El
        holdout es por viaje, no por fila: las filas de un mismo viaje están
        fuertemente correlacionadas (comparten toda la trayectoria observada) y
        un holdout por fila filtraría información entre fit y early stopping.

    Returns
    -------
    dict con:
      'overall'      : dict[algorithm] -> métricas sobre todas las filas
      'by_fold'      : dict[algorithm] -> list[dict] por fold
      'by_position'  : dict[algorithm] -> dict[l_idx] -> métricas
      'by_horizon'   : dict[algorithm] -> dict[bin] -> métricas, con el bin
                       definido por el TIEMPO RESTANTE REAL (§4.5). Eje
                       ortogonal a `by_position`, que agrega sobre horizontes.
      'dev_hist'     : dict[algorithm] -> list[(fold, centros_s, cuentas)]
                       histograma con signo de dev, bins de 1 s. De ahí salen
                       las métricas honestas de la fase 5 (banda simétrica,
                       on-time calibrado) sin re-correr — ver `honest_metrics`.
      'importance'   : dict[algorithm] -> dict[feature] -> ganancia normalizada
      'metadata'     : dict

    Notes
    -----
    Para XGBoost, μ y el prior se calculan sobre el SUBCONJUNTO DE AJUSTE del
    fold (train menos el holdout de early stopping), no sobre train completo.
    Así las filas de early stopping son fuera de muestra también respecto del
    prior. Es ligeramente conservador para XGBoost (usa ~85% de los viajes de
    train para el prior, frente al 100% de los baselines) y mantiene la
    comparación honesta en su contra, no a su favor.
    """
    M, n_plus_1 = T.shape
    n = n_plus_1 - 1

    if M < n_folds:
        raise InsufficientDataError(
            f"Solo {M} viaje(s) interpolado(s) para n_folds={n_folds}."
        )

    if l_indices is None:
        l_indices = make_l_indices(n)
    l_indices = [int(l) for l in l_indices if l < n]

    # Las particiones pueden venir dadas (frame temporal: el fold es el mes del
    # viaje, no un trozo intercambiable) o derivarse aquí como siempre. Cuando
    # vienen dadas mandan ellas, y `n_folds` pasa a ser cuántas hay.
    expanding = (split == "expanding")
    if folds is None:
        if expanding:
            raise ValueError(
                "split='expanding' exige `folds` explícitos: el periodo de cada "
                "viaje no se puede derivar de (M, seed)."
            )
        folds = make_folds(M, n_folds, seed)
    else:
        folds = [np.asarray(f, dtype=np.int64) for f in folds]
        n_folds = len(folds)
        if any(f.size == 0 for f in folds):
            raise InsufficientDataError(
                f"hay {sum(1 for f in folds if f.size == 0)} fold(s) vacío(s) "
                f"de {n_folds}: la partición no es evaluable."
            )
    baselines  = baselines or {}
    xgb_models = xgb_models or {}

    # Normalizar a {nombre: (factory, tier)} y validar la forma de exo.
    specs: Dict[str, tuple] = {}
    for name, v in xgb_models.items():
        fac, cols = v if isinstance(v, tuple) else (v, None)
        # ⚠ `isinstance(cols, str)` primero: `cols` suele ser una LISTA de
        # índices exógenos, y `lista in dict` intenta hashearla y revienta.
        if isinstance(cols, str) and cols in _TIER_FLAGS:
            # No pide exógenos: pide MENOS features que Tier A.
            specs[name] = (fac, cols)
            continue
        if cols is not None and exo is None:
            raise ValueError(
                f"la variante {name!r} pide exógenos pero no se pasó `exo`."
            )
        # La clave del tier es la tupla de columnas: variantes que comparten
        # subconjunto reutilizan la MISMA matriz de entrenamiento.
        specs[name] = (fac, tuple(cols) if cols is not None else None)

    if strata is not None and (strata.shape[0] != M
                               or strata.shape[1] != len(l_indices)):
        raise ValueError(
            f"strata tiene shape {strata.shape}; se esperaba "
            f"({M}, {len(l_indices)}) — debe alinear con T y l_indices."
        )

    if exo is not None:
        if exo.shape[0] != M or exo.shape[1] != len(l_indices):
            raise ValueError(
                f"exo tiene shape {exo.shape}; se esperaba "
                f"({M}, {len(l_indices)}, k) — debe alinear con T y l_indices."
            )

    algos: List[str] = list(baselines.keys()) + list(specs.keys())

    acc_overall  = {a: ErrorAccumulator() for a in algos}
    acc_position = {a: {l: ErrorAccumulator() for l in l_indices} for a in algos}
    by_fold: Dict[str, List[Dict[str, float]]] = {a: [] for a in algos}
    # Histograma de `dev` POR FOLD (fase 5). Se guarda solo aquí: el overall es
    # la suma de los folds, y la calibración leave-one-fold-out necesita
    # justamente el desglose. Los acumuladores por posición no lo llevan.
    dev_hist: Dict[str, List[tuple]] = {a: [] for a in algos}
    # Rendimiento por HORIZONTE (§4.5). El eje es ortogonal al de posición:
    # `by_position` agrega sobre todos los horizontes en cada corte, así que de
    # él no se puede recuperar esta lectura. Son 6 acumuladores por algoritmo,
    # sin histograma: cuesta nada.
    acc_horizon = {a: [ErrorAccumulator() for _ in HORIZON_LABELS] for a in algos}
    importances: Dict[str, List[Dict[str, float]]] = {a: [] for a in specs}
    best_iters:  Dict[str, List[int]] = {a: [] for a in specs}
    tier_keys = {t for _, t in specs.values()}
    names_by_tier = {}
    for t in tier_keys:
        wp, wd = _tier_flags(t)
        names_by_tier[t] = feature_names(_tier_cols(t), with_prior=wp,
                                         with_delay_vs_mu=wd)

    # ── Coste computacional por (fold, algoritmo) ────────────────────
    # §3.9 del manuscrito promete reportar el coste de cada familia junto a su
    # precisión, y hasta ahora el único medido era el del LSTM (`elapsed_s` por
    # leg, en `lstm_prediction_tf/run_frame.py`): del lado CPU solo existía el
    # total por ruta del job, que mezcla los nueve algoritmos con la
    # construcción del frame. Sin desglose no se puede colocar `delay_hour` —
    # que no ajusta un solo parámetro — en un eje de coste.
    #
    # Los relojes son `perf_counter` pegados a la llamada y nada más: no
    # cambian el orden de ejecución ni tocan un solo número de las métricas.
    # La featurización se mide POR TIER, no por algoritmo, porque dos variantes
    # del mismo tier reutilizan la misma matriz (ver `specs` arriba); se
    # atribuye entera a cada una de ellas —que es lo que costaría desplegar esa
    # sola— y `n_share_tier` deja amortizarla si se prefiere la otra lectura.
    timing: List[Dict[str, Any]] = []

    for fold_id, val_idx in enumerate(folds):
        if expanding and fold_id == 0:
            # El primer periodo no tiene historia detrás: es entrenamiento
            # siempre y conjunto de prueba nunca. P periodos ⇒ P−1 escalones.
            continue
        src = (range(fold_id) if expanding
               else [j for j in range(n_folds) if j != fold_id])
        train_idx = np.concatenate([folds[j] for j in src])
        # ⚠ El holdout de early stopping reserva viajes enteros, así que un
        # escalón con UN solo viaje de entrenamiento deja `T_fit` vacío:
        # `mu_ref` sale NaN y `compute_prior` revienta pidiendo un cuantil sobre
        # cero filas (IndexError, 1602/T0->T0 el 2026-08-17). Con folds
        # sorteados es imposible —cada fold entrena con (n−1)/n de los viajes—;
        # la escalera SÍ produce primeros escalones de un viaje. Se salta el
        # escalón, igual que ya se saltan los meses sin viajes, en vez de tumbar
        # la leg y con ella el resto de las legs de la ruta.
        if specs and train_idx.size < MIN_TRAIN_TRIPS_ES:
            print(f"    escalón {fold_id}: {train_idx.size} viaje(s) de "
                  f"entrenamiento (<{MIN_TRAIN_TRIPS_ES}) — se salta: no caben "
                  f"ajuste y holdout de early stopping", flush=True)
            continue
        T_train = T[train_idx]
        T_val   = T[val_idx]

        acc_fold = {a: ErrorAccumulator(track_dev=True) for a in algos}
        rng = np.random.default_rng(seed + 1000 * fold_id)

        sec_fit:     Dict[str, float] = {a: 0.0 for a in algos}
        sec_pred:    Dict[str, float] = {a: 0.0 for a in algos}
        sec_feat_tr: Dict[Any, float] = {t: 0.0 for t in tier_keys}
        sec_feat_ev: Dict[Any, float] = {t: 0.0 for t in tier_keys}
        rows_tier:   Dict[Any, int]   = {t: 0 for t in tier_keys}
        sec_prior = 0.0

        # ── 1) Entrenar las variantes de XGBoost ─────────────────
        fitted: Dict[str, tuple] = {}
        if specs:
            # Holdout de early stopping POR VIAJE.
            perm  = rng.permutation(T_train.shape[0])
            n_es  = max(1, int(round(es_frac * T_train.shape[0])))
            # El holdout no puede vaciar el ajuste: con pocos viajes el
            # redondeo se lo llevaba entero (ver MIN_TRAIN_TRIPS_ES).
            n_es  = min(n_es, T_train.shape[0] - 1)
            idx_es, idx_fit = perm[:n_es], perm[n_es:]
            T_es, T_fit = T_train[idx_es], T_train[idx_fit]

            _t = perf_counter()
            mu_ref     = T_fit.mean(axis=0)
            priors_ref = {l: compute_prior(T_fit, l) for l in l_indices}
            sec_prior = perf_counter() - _t

            # exo debe seguir EXACTAMENTE las mismas particiones que T.
            exo_tr = exo[train_idx] if exo is not None else None
            exo_fit = exo_tr[idx_fit] if exo_tr is not None else None
            exo_es  = exo_tr[idx_es]  if exo_tr is not None else None

            data: Dict[Any, tuple] = {}
            for tier in tier_keys:
                _t = perf_counter()
                cols = _tier_cols(tier)
                wp, wd = _tier_flags(tier)
                e_fit = exo_fit[:, :, cols] if cols is not None else None
                e_es  = exo_es[:, :, cols] if cols is not None else None
                X_tr, y_tr = _stack_blocks(T_fit, l_grid, l_indices, mu_ref,
                                           priors_ref, n_h_train,
                                           np.random.default_rng(seed + fold_id),
                                           exo_src=e_fit, with_prior=wp,
                                           with_delay_vs_mu=wd)
                if X_tr.shape[0] > max_train_rows:
                    sel = np.random.default_rng(seed).choice(
                        X_tr.shape[0], size=max_train_rows, replace=False)
                    X_tr, y_tr = X_tr[sel], y_tr[sel]
                X_es, y_es = _stack_blocks(T_es, l_grid, l_indices, mu_ref,
                                           priors_ref, n_h_train,
                                           np.random.default_rng(seed + fold_id),
                                           exo_src=e_es, with_prior=wp,
                                           with_delay_vs_mu=wd)
                data[tier] = (X_tr, y_tr, X_es, y_es)
                sec_feat_tr[tier] = perf_counter() - _t
                rows_tier[tier] = int(X_tr.shape[0])
                if verbose:
                    lbl = _tier_label(tier)
                    print(f"    fold {fold_id} [{lbl}]: "
                          f"fit={X_tr.shape[0]:,} ({T_fit.shape[0]} viajes) · "
                          f"es={X_es.shape[0]:,} × {X_tr.shape[1]} features",
                          flush=True)

            for name, (factory, tier) in specs.items():
                X_tr, y_tr, X_es, y_es = data[tier]
                mdl = factory()
                _t = perf_counter()
                mdl.fit(X_tr, y_tr, eval_set=(X_es, y_es))
                sec_fit[name] = perf_counter() - _t
                fitted[name] = (mdl, tier)
                importances[name].append(
                    mdl.feature_importance(names_by_tier[tier]))
                bi = mdl.best_iteration
                if bi is not None:
                    best_iters[name].append(int(bi))
            del data
        else:
            _t = perf_counter()
            mu_ref     = T_train.mean(axis=0)
            priors_ref = {l: compute_prior(T_train, l) for l in l_indices}
            sec_prior = perf_counter() - _t

        # ── 2) Ajustar baselines del fold (sobre train COMPLETO) ─
        strata_tr = strata[train_idx] if strata is not None else None
        strata_va = strata[val_idx] if strata is not None else None
        for name, pred in baselines.items():
            _t = perf_counter()
            if getattr(pred, "needs_strata", False):
                pred.fit(T_train, l_grid, strata_tr)
            else:
                pred.fit(T_train, l_grid)
            sec_fit[name] = perf_counter() - _t

        # ── 3) Evaluar en streaming por posición ─────────────────
        exo_val = exo[val_idx] if exo is not None else None

        for li, l in enumerate(l_indices):
            n_h = n - l

            # Verdad y features de evaluación (horizonte COMPLETO).
            # y_ev y t_obs_l solo dependen de T_val: los baselines no se ven
            # afectados por qué μ/prior se usó para featurizar.
            X_ev: Dict[Any, np.ndarray] = {}
            y_ev = t_obs_l = None
            for tier in {t for _, t in fitted.values()}:
                _t = perf_counter()
                cols = _tier_cols(tier)
                Xb, y_ev, t_obs_l = build_block(
                    T_val, l_grid, l, mu_ref, priors_ref[l], h_subset=None,
                    exo=(exo_val[:, li, :][:, cols]
                         if (cols is not None and exo_val is not None) else None),
                    with_prior=_tier_flags(tier)[0],
                    with_delay_vs_mu=_tier_flags(tier)[1],
                )
                X_ev[tier] = Xb
                sec_feat_ev[tier] += perf_counter() - _t
            if y_ev is None:
                _, y_ev, t_obs_l = build_block(
                    T_val, l_grid, l, mu_ref, priors_ref[l], h_subset=None,
                )

            # Máscaras de horizonte: dependen SOLO de la verdad, así que se
            # calculan una vez por (fold, corte) y valen para los 9 algoritmos.
            h_bin = horizon_bin(y_ev)
            h_masks = [(h_bin == b) for b in range(len(HORIZON_LABELS))]

            def _acc_h(name: str, dt_pred: np.ndarray) -> None:
                for b, m in enumerate(h_masks):
                    if m.any():
                        acc_horizon[name][b].update(dt_pred[m], y_ev[m])

            for name, (mdl, tier) in fitted.items():
                _t = perf_counter()
                dt_pred = mdl.predict(X_ev[tier])
                sec_pred[name] += perf_counter() - _t
                acc_overall[name].update(dt_pred, y_ev)
                acc_position[name][l].update(dt_pred, y_ev)
                acc_fold[name].update(dt_pred, y_ev)
                _acc_h(name, dt_pred)
            del X_ev

            if baselines:
                T_obs = T_val[:, :l + 1]
                for name, pred in baselines.items():
                    # Los baselines devuelven tiempo ACUMULADO → a restante.
                    _t = perf_counter()
                    if getattr(pred, "needs_strata", False):
                        t_pred = pred.predict_batch(T_obs, l, li, strata_va)[:, :n_h]
                    else:
                        t_pred = pred.predict_batch(T_obs, l)[:, :n_h]
                    sec_pred[name] += perf_counter() - _t
                    dt_b = (t_pred - t_obs_l[:, None]).reshape(-1).astype(np.float32)
                    acc_overall[name].update(dt_b, y_ev)
                    acc_position[name][l].update(dt_b, y_ev)
                    acc_fold[name].update(dt_b, y_ev)
                    _acc_h(name, dt_b)

        for a in algos:
            r = acc_fold[a].result()
            r["fold"] = fold_id
            by_fold[a].append(r)
            centers, counts = acc_fold[a].dev_bins()
            dev_hist[a].append((fold_id, centers, counts))

        for a in algos:
            tier   = specs[a][1] if a in specs else None
            is_xgb = a in specs
            timing.append({
                "fold":              fold_id,
                "algorithm":         a,
                "family":            "xgb" if is_xgb else "baseline",
                "tier":              _tier_label(tier) if is_xgb else "",
                # Coste propio del modelo: lo que se paga por ESTE algoritmo.
                "fit_s":             round(sec_fit[a], 4),
                "predict_s":         round(sec_pred[a], 4),
                # Coste compartido, atribuido entero (ver nota arriba).
                "prior_s":           round(sec_prior if is_xgb else 0.0, 4),
                "featurize_train_s": round(sec_feat_tr.get(tier, 0.0)
                                           if is_xgb else 0.0, 4),
                "featurize_eval_s":  round(sec_feat_ev.get(tier, 0.0)
                                           if is_xgb else 0.0, 4),
                "n_share_tier":      (sum(1 for _, t in specs.values() if t == tier)
                                      if is_xgb else 1),
                "n_train_trips":     int(T_train.shape[0]),
                "n_val_trips":       int(T_val.shape[0]),
                "n_train_rows":      int(rows_tier.get(tier, 0)) if is_xgb else 0,
                "n_cuts":            len(l_indices),
                "omp_threads":       _OMP_THREADS,
            })

        if verbose:
            msg = " · ".join(
                f"{a}={acc_fold[a].result()['MAE_s']:.1f}" for a in algos
            )
            print(f"    fold {fold_id} MAE: {msg}", flush=True)

    # ── Importancias promediadas entre folds ─────────────────────
    imp_out: Dict[str, Dict[str, float]] = {}
    for a, lst in importances.items():
        if not lst:
            continue
        names = names_by_tier[specs[a][1]]
        d = {f: float(np.mean([x.get(f, 0.0) for x in lst])) for f in names}
        imp_out[a] = dict(sorted(d.items(), key=lambda kv: -kv[1]))

    return {
        "overall":     {a: acc_overall[a].result() for a in algos},
        "by_fold":     by_fold,
        "timing":      timing,
        "by_position": {
            a: {l: acc_position[a][l].result() for l in l_indices} for a in algos
        },
        "dev_hist":    dev_hist,
        "by_horizon":  {a: {HORIZON_LABELS[b]: acc_horizon[a][b].result()
                            for b in range(len(HORIZON_LABELS))} for a in algos},
        "importance":  imp_out,
        "metadata": {
            "dev_hist_bin_s": DEV_BIN_S, "dev_hist_max_abs_s": DEV_MAX_ABS_S,
            "M": int(M), "n": int(n), "n_folds": n_folds, "seed": seed,
            "split": split,
            "evaluated_folds": ([] if not expanding else list(range(1, n_folds))),
            "l_indices": list(l_indices), "n_h_train": n_h_train,
            "es_frac": es_frac, "algorithms": algos,
            "tiers": {a: ("A" if t is None else list(t))
                      for a, (_, t) in specs.items()},
            "best_iterations": {a: v for a, v in best_iters.items() if v},
        },
    }
