"""
xgb_prediction.feature_builder
================================
Etapa A: construcción de features tabulares para XGBoost.

Formulación
-----------
Una fila = una tupla (viaje, l_idx, h). El target es el tiempo RESTANTE

    dt_rem = t_{l+h} − t_l

y la predicción se re-expresa en la escala de los baselines / LSTM como

    t_pred = t_obs_l + dt_rem_hat

Predecir el restante (y no el acumulado) es lo que corrigió el falso negativo
del certificado v6: `t_true` acumulado y restante difieren por `t_obs_l`, y
entrenar sobre el acumulado dejaría que el modelo aprenda `t_obs_l` como atajo.

Tier A — solo información de trayectoria
----------------------------------------
Todo sale de la matriz `T` (viajes × celdas de grid, tiempo acumulado) y del
fold de entrenamiento. NINGÚN feature viene de `df_full`. Es la condición de
paridad de información con los baselines y con el LSTM en modo grid.

El prior histórico incluye `prior_mean[h] = μ_{l+h} − μ_l`, que **es exactamente
la predicción del DelayPredictor**. Eso hace de XGB-A un superconjunto estricto
de ese baseline: si no lo gana, es señal y no artefacto de ajuste.

Anti-leakage
------------
  * μ, p50, p90 y σ del prior se calculan SIEMPRE sobre `T_train` del fold.
  * NO se usan las columnas `hist_*` de `df_full`: se construyeron con su propio
    split de train en el paso 07, que no coincide con estos folds.
  * Los features de estado del viaje solo miran celdas ≤ l_idx.
"""
from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple

# Ventanas (en celdas de grid) para la dinámica reciente.
K_SHORT = 5
K_LONG  = 20

_EPS = 1e-6


# ──────────────────────────────────────────────────────────────────
# Nombres de features (orden canónico — el modelo depende de él)
# ──────────────────────────────────────────────────────────────────

#: Lo que se observa del viaje EN CURSO y de nada más. Ni una de estas
#: cantidades mira a otro viaje: se calculan sobre la traza del propio bus hasta
#: el corte.
CURRENT_TRIP_FEATURES: List[str] = [
    "t_obs_l",            # tiempo transcurrido hasta l (s)
    "v_avg_sofar",        # velocidad media del viaje hasta l (m/s)
    "v_recent_short",     # velocidad en las últimas K_SHORT celdas (m/s)
    "v_recent_long",      # velocidad en las últimas K_LONG celdas (m/s)
    "accel",              # v_recent_short − v_recent_long (proxy acel/desacel)
    "celltime_std",       # σ de tiempos por celda en las últimas K_LONG
    "celltime_mean",      # media de tiempos por celda en las últimas K_LONG
]

#: ⚠ El gancho histórico ESCONDIDO en el bloque de viaje. `t_l − μ_l` parece una
#: cantidad del viaje en curso —y se lee como «el retraso actual»— pero μ es la
#: media por posición del fold de ENTRENAMIENTO. Quitar solo `PRIOR_FEATURES` y
#: dejar esto deja al nivel 1 con condicionamiento histórico dentro, y la
#: atribución sale sesgada a la baja: en el panel de 42 legs el prior parecía
#: aportar un 5.5% de la ganancia con este hueco abierto.
HIST_TRIP_FEATURES: List[str] = [
    "delay_vs_mu",        # t_l − μ_l  (el retraso actual; señal del baseline delay)
]

TRIP_FEATURES: List[str] = CURRENT_TRIP_FEATURES + HIST_TRIP_FEATURES

#: Geometría del par (l, h). No condiciona en nada histórico: dónde está el
#: corte y a qué distancia está el horizonte.
GEOM_FEATURES: List[str] = [
    "l_idx",
    "l_meters",
    "frac_route",         # l_meters / n_route
    "h_idx",
    "h_meters",
    "h_frac",             # h_meters / (n_route − l_meters)
]

#: El bloque de CONDICIONAMIENTO HISTÓRICO: la distribución del tiempo restante
#: en (l, h) estimada sobre el fold de entrenamiento. Es lo que separa el nivel
#: 1 del nivel 2 de la ablación de información, así que va aparte y no mezclado
#: en `PAIR_FEATURES` — el revisor de agosto-2026 señaló, con razón, que llamar
#: *trajectory-only* a un Tier A que lleva esto dentro es un nombre equivocado.
PRIOR_FEATURES: List[str] = [
    "prior_mean",         # μ_{l+h} − μ_l   ← predicción del baseline delay
    "prior_p50",
    "prior_p90",
    "prior_std",
]

PAIR_FEATURES: List[str] = GEOM_FEATURES + PRIOR_FEATURES

FEATURE_NAMES: List[str] = TRIP_FEATURES + PAIR_FEATURES


def feature_names(exo_cols=None, with_prior: bool = True,
                  with_delay_vs_mu: bool = True) -> List[str]:
    """
    Orden canónico de columnas de X. Debe coincidir con `build_block`.

    exo_cols : None = Tier A · lista de índices sobre EXO_FEATURES = ese
        subconjunto (se usa para la ablación por grupos).
    with_prior : False deja fuera `PRIOR_FEATURES` — el nivel 1 de la ablación
        de información.
    with_delay_vs_mu : False deja fuera además `delay_vs_mu` — el nivel 0, que
        es el único verdaderamente *trajectory-only*: sin él no queda ninguna
        cantidad que mire al conjunto de entrenamiento.
    """
    trip = TRIP_FEATURES if with_delay_vs_mu else CURRENT_TRIP_FEATURES
    pair = PAIR_FEATURES if with_prior else GEOM_FEATURES
    if exo_cols is None:
        return trip + pair
    from .exo_features import EXO_FEATURES
    sel = [EXO_FEATURES[i] for i in exo_cols]
    return trip + sel + pair


# ──────────────────────────────────────────────────────────────────
# Prior histórico (SOLO train fold)
# ──────────────────────────────────────────────────────────────────

def compute_prior(T_train: np.ndarray, l_idx: int) -> Dict[str, np.ndarray]:
    """
    Estadísticos del tiempo restante en el fold de entrenamiento.

    Para cada horizonte h = 1..n−l calcula la distribución de
    ``dt_rem = T_train[:, l+h] − T_train[:, l]`` sobre los viajes de train.

    Returns
    -------
    dict con arrays (n_h,): 'mean', 'p50', 'p90', 'std'.

    Notes
    -----
    ``mean`` coincide algebraicamente con ``μ_{l+h} − μ_l`` (linealidad de la
    media), es decir, con la predicción del DelayPredictor.
    """
    if T_train.ndim != 2:
        raise ValueError(f"T_train debe ser 2-D, recibí shape={T_train.shape}")

    # (M_tr, n_h) — restante de cada viaje de train para cada horizonte
    rem = T_train[:, l_idx + 1:] - T_train[:, l_idx][:, None]

    return {
        "mean": rem.mean(axis=0),
        "p50":  np.median(rem, axis=0),
        "p90":  np.quantile(rem, 0.90, axis=0),
        "std":  rem.std(axis=0, ddof=0),
    }


# ──────────────────────────────────────────────────────────────────
# Features de estado del viaje (por viaje, dado l_idx)
# ──────────────────────────────────────────────────────────────────

def build_trip_features(
    T:          np.ndarray,
    l_grid:     np.ndarray,
    l_idx:      int,
    mu_train:   np.ndarray,
    with_delay_vs_mu: bool = True,
) -> np.ndarray:
    """
    Features del viaje observado hasta `l_idx`, para cada fila de `T`.

    Solo mira columnas ≤ l_idx (nunca el futuro del viaje).

    Parameters
    ----------
    T : (M, n+1) tiempos acumulados interpolados en la grid.
    l_grid : (n+1,) distancias en metros.
    l_idx : índice de la posición de corte.
    mu_train : (n+1,) media por posición del fold de TRAIN (no de T).

    Returns
    -------
    np.ndarray (M, len(TRIP_FEATURES))
    """
    M = T.shape[0]
    step = float(l_grid[1] - l_grid[0])

    t_l = T[:, l_idx]

    # Ventanas recortadas al inicio del viaje (l_idx puede ser < K).
    k_s = max(1, min(K_SHORT, l_idx))
    k_l = max(1, min(K_LONG,  l_idx))

    dt_s = np.maximum(t_l - T[:, l_idx - k_s], _EPS)
    dt_l = np.maximum(t_l - T[:, l_idx - k_l], _EPS)

    v_short = (k_s * step) / dt_s
    v_long  = (k_l * step) / dt_l

    # Tiempos por celda en la ventana larga: (M, k_l)
    win = np.diff(T[:, l_idx - k_l: l_idx + 1], axis=1)
    if win.shape[1] == 0:                       # l_idx == 0 (no debería ocurrir)
        celltime_std  = np.zeros(M)
        celltime_mean = np.zeros(M)
    else:
        celltime_std  = win.std(axis=1, ddof=0)
        celltime_mean = win.mean(axis=1)

    v_avg = l_grid[l_idx] / np.maximum(t_l, _EPS)

    names = TRIP_FEATURES if with_delay_vs_mu else CURRENT_TRIP_FEATURES
    out = np.empty((M, len(names)), dtype=np.float32)
    out[:, 0] = t_l
    out[:, 1] = v_avg
    out[:, 2] = v_short
    out[:, 3] = v_long
    out[:, 4] = v_short - v_long
    out[:, 5] = celltime_std
    out[:, 6] = celltime_mean
    if with_delay_vs_mu:
        out[:, 7] = t_l - mu_train[l_idx]
    return out


# ──────────────────────────────────────────────────────────────────
# Ensamblado de un bloque (todas las filas de un l_idx)
# ──────────────────────────────────────────────────────────────────

def build_block(
    T:          np.ndarray,
    l_grid:     np.ndarray,
    l_idx:      int,
    mu_train:   np.ndarray,
    prior:      Dict[str, np.ndarray],
    h_subset:   np.ndarray | None = None,
    exo:        np.ndarray | None = None,
    with_prior: bool = True,
    with_delay_vs_mu: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construye la matriz de features para TODAS las filas (viaje × horizonte)
    de una posición de corte `l_idx`.

    El bloque tiene estructura de producto externo: los features de viaje se
    repiten sobre horizontes y los de par (l,h) se repiten sobre viajes. Se
    ensambla con `repeat`/`tile` en vez de bucles.

    Parameters
    ----------
    T : (M, n+1) — trayectorias del conjunto a featurizar (train o val).
    prior : salida de `compute_prior` sobre el T_train del fold.
    h_subset : índices (0-based sobre el eje de horizontes) a incluir. None =
        todos. Se usa para submuestrear horizontes en ENTRENAMIENTO; en
        evaluación debe ser None o la comparación deja de ser matched.
    exo : (M, k_exo) features de Tier B para ESTE l_idx, en el mismo orden de
        filas que `T`. None = solo Tier A. Se insertan entre los features de
        viaje y los de par, igual que en `feature_names(with_exo=True)`.
    with_prior : False omite las cuatro columnas de `PRIOR_FEATURES`. El
        `prior` se sigue exigiendo y se sigue calculando sobre el fold de
        entrenamiento; lo único que cambia es que el modelo no lo ve. La
        alternativa —pasar `prior=None`— dejaría que un fallo al calcularlo se
        colara como una ablación silenciosa.

    Returns
    -------
    X : (M·n_h, n_features) float32
    y : (M·n_h,) float32 — dt_rem verdadero
    t_obs_l : (M,) float32 — tiempo observado en l (para reconstruir t_pred)
    """
    M      = T.shape[0]
    n      = T.shape[1] - 1
    n_full = n - l_idx
    if n_full <= 0:
        raise ValueError(f"l_idx={l_idx} no deja horizontes (n={n})")

    h_all = np.arange(n_full) if h_subset is None else np.asarray(h_subset)
    n_h   = len(h_all)

    n_route  = float(l_grid[-1])
    l_meters = float(l_grid[l_idx])

    # ── features de par (l, h): (n_h, len(PAIR_FEATURES)) ────────────
    h_idx    = (h_all + 1).astype(np.float32)
    h_meters = (l_grid[l_idx + 1: l_idx + 1 + n_full] - l_meters)[h_all]
    remaining = max(n_route - l_meters, _EPS)

    pair_names = PAIR_FEATURES if with_prior else GEOM_FEATURES
    pair = np.empty((n_h, len(pair_names)), dtype=np.float32)
    pair[:, 0] = l_idx
    pair[:, 1] = l_meters
    pair[:, 2] = l_meters / max(n_route, _EPS)
    pair[:, 3] = h_idx
    pair[:, 4] = h_meters
    pair[:, 5] = h_meters / remaining
    if with_prior:
        pair[:, 6] = prior["mean"][h_all]
        pair[:, 7] = prior["p50"][h_all]
        pair[:, 8] = prior["p90"][h_all]
        pair[:, 9] = prior["std"][h_all]

    # ── features de viaje: (M, len(TRIP_FEATURES)) ───────────────────
    trip = build_trip_features(T, l_grid, l_idx, mu_train,
                               with_delay_vs_mu=with_delay_vs_mu)
    if exo is not None:
        if exo.shape[0] != M:
            raise ValueError(
                f"exo tiene {exo.shape[0]} filas y T tiene {M}: el orden de "
                f"viajes debe coincidir (¿submuestreaste T sin submuestrear exo?)"
            )
        trip = np.concatenate([trip, exo.astype(np.float32)], axis=1)

    # ── producto externo → (M·n_h, n_features) ───────────────────────
    # Orden de filas: viaje-mayor (viaje 0 con todos sus h, viaje 1, …), que
    # es el mismo que recorre run_cross_validation (for vi: for hi:).
    n_trip = trip.shape[1]
    X = np.empty((M * n_h, n_trip + len(pair_names)), dtype=np.float32)
    X[:, :n_trip] = np.repeat(trip, n_h, axis=0)
    X[:, n_trip:] = np.tile(pair, (M, 1))

    # ── target: dt_rem = t_{l+h} − t_l ───────────────────────────────
    t_obs_l = T[:, l_idx]
    rem     = T[:, l_idx + 1:][:, h_all] - t_obs_l[:, None]   # (M, n_h)
    y       = rem.reshape(-1).astype(np.float32)

    return X, y, t_obs_l.astype(np.float32)
