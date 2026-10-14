"""
lstm_prediction_tf.data_loader
================================
Conversión de las secuencias por viaje del dataset_builder en muestras
(X, y) consumibles por TensorFlow, con padding, máscaras y loss MSE
enmascarada.

Responsabilidades
-----------------
1. `SequenceDataset`: clase Python pura que genera muestras al vuelo.
2. `make_tf_dataset`: factory que envuelve SequenceDataset en `tf.data.Dataset`
   con padding nativo via `padded_batch`.
3. `masked_mse_loss`: MSE solo sobre pasos del horizonte válidos.

Diferencias clave respecto a la versión PyTorch
-----------------------------------------------
- En PyTorch necesitábamos `collate_sequences` para padding manual.
  En TF, `tf.data.Dataset.padded_batch()` hace el padding nativamente.
- En PyTorch teníamos `Batch` como dataclass propio. En TF las muestras
  fluyen como tuplas (stops, times, stops_horizon, obs_mask,
  horizon_mask, target).
- El `filter_by_trip_ids` para CV folds se hace en la clase pura, no en
  el `tf.data.Dataset`.

Paridad con el paso 10
----------------------
Para comparar contra los baselines del paso 10, el LSTM debe predecir EN
LAS MISMAS POSICIONES l. Por defecto, si no se pasa l_indices explícito,
generamos una muestra por cada posición intermedia del viaje; el
cross_validation luego puede filtrarlas.
"""
from __future__ import annotations

import atexit
import os
import shutil
import tempfile
import uuid
import weakref
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import tensorflow as tf

from .vocab import StopVocab, PAD_IDX


# ──────────────────────────────────────────────────────────────────────
# Cache a disco para tf.data — evita el OOM de `.cache()` en RAM
# ──────────────────────────────────────────────────────────────────────
# El dataset expandido (una muestra por corte l) puede pesar decenas de GB.
# `.cache()` sin argumento lo guarda TODO en RAM → causaba SIGKILL (exit -9)
# en rutas grandes. Cachear a un archivo temporal mantiene la RAM acotada;
# los epochs siguientes leen del disco. Cada archivo se borra solo cuando su
# Dataset es recolectado (fin del fold) y, como red de seguridad, al salir.

# Cap del shuffle buffer: buffer=n cargaba TODAS las muestras en RAM. 50k
# muestras dan una mezcla más que suficiente sin reventar memoria.
_SHUFFLE_BUFFER_CAP = int(os.environ.get("LSTM_SHUFFLE_BUFFER_CAP", "50000"))

# Directorio raíz para los caches en disco (configurable; por defecto el
# scratch del sistema). En SLURM conviene apuntarlo a $TMPDIR / scratch local.
_CACHE_ROOT = os.environ.get("LSTM_TF_CACHE_DIR") or os.path.join(
    tempfile.gettempdir(), "lstm_tf_cache"
)
_cache_files: set[str] = set()


def _new_cache_prefix() -> str:
    """Ruta única para un cache de tf.data (TF crea <prefix>.index/.data-*)."""
    os.makedirs(_CACHE_ROOT, exist_ok=True)
    prefix = os.path.join(_CACHE_ROOT, f"ds_{uuid.uuid4().hex}")
    _cache_files.add(prefix)
    return prefix


def _cleanup_cache(prefix: str) -> None:
    """Borra los ficheros <prefix>.* que TF crea para un cache en disco."""
    folder = os.path.dirname(prefix)
    base = os.path.basename(prefix)
    try:
        for fn in os.listdir(folder):
            if fn.startswith(base):
                try:
                    os.remove(os.path.join(folder, fn))
                except OSError:
                    pass
    except OSError:
        pass
    _cache_files.discard(prefix)


@atexit.register
def _cleanup_all_caches() -> None:
    """Red de seguridad: limpia cualquier cache que sobreviva al proceso."""
    for prefix in list(_cache_files):
        _cleanup_cache(prefix)
    try:
        if os.path.isdir(_CACHE_ROOT) and not os.listdir(_CACHE_ROOT):
            shutil.rmtree(_CACHE_ROOT, ignore_errors=True)
    except OSError:
        pass


# ──────────────────────────────────────────────────────────────────────
# Estructura de una muestra individual
# ──────────────────────────────────────────────────────────────────────

@dataclass
class TrainingSample:
    """
    Una muestra para entrenar/evaluar el modelo.

    Attributes
    ----------
    trip_id : str | int
    l_idx : int
    stops_obs : np.ndarray (L,) int64
    times_obs : np.ndarray (L,) float32
    stops_horizon : np.ndarray (H,) int64
    times_target : np.ndarray (H,) float32
    """
    trip_id:       Any
    l_idx:         int
    stops_obs:     np.ndarray
    times_obs:     np.ndarray
    stops_horizon: np.ndarray
    times_target:  np.ndarray
    context:       np.ndarray      # (CTX_DIM,) float32 en el corte l_idx (o vacío)


# ──────────────────────────────────────────────────────────────────────
# Dataset Python puro (sin dependencias de TF)
# ──────────────────────────────────────────────────────────────────────

class SequenceDataset:
    """
    Dataset sobre secuencias del builder.

    Genera muestras (X, y) al vuelo desde la lista de secuencias por viaje.
    Cada secuencia de longitud n produce muestras correspondientes a
    posiciones de corte l_idx.

    Esta clase NO depende de TensorFlow — es Python puro. Se envuelve en
    `tf.data.Dataset` mediante `make_tf_dataset()`.

    Parameters
    ----------
    sequences : list[dict]
    vocab : StopVocab
    min_l : int, default=2
    max_l_offset : int, default=2
    max_context_length : int | None, default=None
    l_strategy : str, default="all"
    l_indices_per_trip : dict[trip_id → list[int]] | None
    """

    def __init__(
        self,
        sequences:          List[Dict[str, Any]],
        vocab:              StopVocab,
        min_l:              int = 2,
        max_l_offset:       int = 2,
        max_context_length: Optional[int] = None,
        l_strategy:         str = "all",
        l_indices_per_trip: Optional[Dict[Any, List[int]]] = None,
    ) -> None:
        if l_strategy not in ("all", "uniform", "explicit"):
            raise ValueError(
                f"l_strategy debe ser 'all', 'uniform' o 'explicit'; "
                f"recibido: {l_strategy!r}"
            )
        if l_strategy == "explicit" and l_indices_per_trip is None:
            raise ValueError("l_strategy='explicit' requiere l_indices_per_trip")

        self.vocab              = vocab
        self.min_l              = int(min_l)
        self.max_l_offset       = int(max_l_offset)
        self.max_context_length = max_context_length
        self.l_strategy         = l_strategy

        # Pre-encodificar cada secuencia
        self._encoded_seqs: List[Dict[str, np.ndarray]] = []
        for s in sequences:
            stops_ids = np.asarray(
                vocab.encode_sequence(s["stops"]), dtype=np.int64
            )
            times = np.asarray(s["times"], dtype=np.float32)
            ctx = s.get("context")
            self._encoded_seqs.append({
                "trip_id": s["trip_id"],
                "stops":   stops_ids,
                "times":   times,
                "context": (np.asarray(ctx, dtype=np.float32)
                            if ctx is not None else None),
            })
        # Dimensión del contexto (0 = sin calendario). Se infiere de la 1ª seq.
        self.context_dim = 0
        for e in self._encoded_seqs:
            if e["context"] is not None:
                self.context_dim = int(e["context"].shape[1])
                break

        # Construir el índice plano (sample_idx → (trip_idx, l_idx))
        self._index: List[tuple] = []
        for trip_idx, eseq in enumerate(self._encoded_seqs):
            n = len(eseq["stops"])
            max_l = n - 1 - max_l_offset
            if max_l < min_l:
                continue

            if l_strategy == "all":
                l_values = range(min_l, max_l + 1)
            elif l_strategy == "uniform":
                n_uniform = min(30, max_l - min_l + 1)
                l_values = np.linspace(
                    min_l, max_l, n_uniform, dtype=int
                ).tolist()
                l_values = sorted(set(l_values))
            else:
                trip_id = eseq["trip_id"]
                if trip_id not in l_indices_per_trip:
                    continue
                l_values = [
                    l for l in l_indices_per_trip[trip_id]
                    if min_l <= l <= max_l
                ]

            for l in l_values:
                self._index.append((trip_idx, l))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> TrainingSample:
        trip_idx, l_idx = self._index[idx]
        eseq = self._encoded_seqs[trip_idx]

        stops = eseq["stops"]
        times = eseq["times"]

        if self.max_context_length is None:
            obs_start = 0
        else:
            obs_start = max(0, l_idx + 1 - self.max_context_length)

        stops_obs     = stops[obs_start: l_idx + 1].copy()
        times_obs     = times[obs_start: l_idx + 1].copy()
        stops_horizon = stops[l_idx + 1:].copy()
        times_target  = times[l_idx + 1:].copy()

        # Contexto del corte l_idx: lo QUE SE CONOCE al predecir (nunca el
        # futuro). times/stops del token l se indexan igual, así que ctx[l_idx].
        cd = getattr(self, "context_dim", 0)
        if eseq.get("context") is not None:
            context = eseq["context"][l_idx].copy()
        else:
            context = np.zeros((cd,), dtype=np.float32)

        return TrainingSample(
            trip_id       = eseq["trip_id"],
            l_idx         = l_idx,
            stops_obs     = stops_obs,
            times_obs     = times_obs,
            stops_horizon = stops_horizon,
            times_target  = times_target,
            context       = context,
        )

    def n_trips(self) -> int:
        return len({trip_idx for trip_idx, _ in self._index})

    def get_trip_ids(self) -> List[Any]:
        seen = set()
        ids = []
        for trip_idx, _ in self._index:
            tid = self._encoded_seqs[trip_idx]["trip_id"]
            if tid not in seen:
                seen.add(tid)
                ids.append(tid)
        return ids

    def filter_by_trip_ids(
        self, allowed_trip_ids: Sequence[Any],
    ) -> "SequenceDataset":
        """Devuelve un nuevo dataset que solo incluye los trips indicados."""
        allowed_set = set(allowed_trip_ids)
        new = SequenceDataset.__new__(SequenceDataset)
        new.vocab              = self.vocab
        new.min_l              = self.min_l
        new.max_l_offset       = self.max_l_offset
        new.max_context_length = self.max_context_length
        new.l_strategy         = self.l_strategy
        new.context_dim        = getattr(self, "context_dim", 0)
        new._encoded_seqs      = self._encoded_seqs
        new._index = [
            (trip_idx, l_idx)
            for trip_idx, l_idx in self._index
            if self._encoded_seqs[trip_idx]["trip_id"] in allowed_set
        ]
        return new

    def iter_samples(self):
        """Generator que yieldea TrainingSample uno a uno (para tf.data)."""
        for i in range(len(self)):
            yield self[i]


# ──────────────────────────────────────────────────────────────────────
# Conversión a tf.data.Dataset con padding nativo
# ──────────────────────────────────────────────────────────────────────

def make_tf_dataset(
    dataset:    SequenceDataset,
    batch_size: int  = 64,
    shuffle:    bool = True,
    shuffle_buffer: Optional[int] = None,
    drop_remainder: bool = False,
    cache: Union[bool, str] = "disk",
) -> tf.data.Dataset:
    """
    Envuelve un SequenceDataset en un tf.data.Dataset con padding nativo.

    Las muestras fluyen como tuplas de 6 elementos:
        (stops_obs, times_obs, stops_horizon,
         obs_mask, horizon_mask, times_target)

    El padding se hace via `padded_batch`. Los valores de padding:
        - stops_*: PAD_IDX (=0)
        - times_*: 0.0
        - masks  : False

    El obs_mask y horizon_mask se calculan dinámicamente durante el
    padding (los valores no-padded son True, los padded son False).

    Parameters
    ----------
    dataset : SequenceDataset
    batch_size : int, default=64
    shuffle : bool, default=True
    shuffle_buffer : int | None
        Tamaño del buffer de shuffle. None = min(len(dataset), cap), donde el
        cap (default 50k, env LSTM_SHUFFLE_BUFFER_CAP) evita cargar TODO el
        dataset en RAM. Antes usaba n → causaba OOM en rutas grandes.
    cache : bool | str, default="disk"
        "disk"  → cachea el dataset expandido a un archivo temporal (RAM
                  acotada; el archivo se borra solo al terminar el fold).
        True    → cache en RAM (rápido pero puede reventar memoria; el
                  comportamiento antiguo, causaba exit -9).
        False   → sin cache (el generator Python corre cada epoch; mínima
                  RAM, algo más lento).
    drop_remainder : bool, default=False
        Si True, descarta el último batch incompleto. Útil para tf.function.

    Returns
    -------
    tf.data.Dataset que produce batches.
    """
    n = len(dataset)
    if n == 0:
        # Crear un dataset vacío con la signature correcta
        empty_signature = _get_output_signature()
        return tf.data.Dataset.from_generator(
            lambda: iter([]),
            output_signature=empty_signature,
        ).padded_batch(batch_size, drop_remainder=drop_remainder)

    # Generator que produce tuplas con máscara incluida
    def gen():
        for s in dataset.iter_samples():
            L = len(s.stops_obs)
            H = len(s.stops_horizon)
            obs_mask     = np.ones((L,), dtype=np.bool_)
            horizon_mask = np.ones((H,), dtype=np.bool_)
            yield (
                s.stops_obs,        # (L,) int64
                s.times_obs,        # (L,) float32
                s.stops_horizon,    # (H,) int64
                obs_mask,           # (L,) bool
                horizon_mask,       # (H,) bool
                s.times_target,     # (H,) float32
                # Metadata para evaluación (no entra al modelo)
                np.int64(s.l_idx),
                str(s.trip_id),
                s.context,          # (CTX_DIM,) float32 — 0-dim si sin calendario
            )

    output_signature = (
        tf.TensorSpec(shape=(None,), dtype=tf.int64,   name="stops_obs"),
        tf.TensorSpec(shape=(None,), dtype=tf.float32, name="times_obs"),
        tf.TensorSpec(shape=(None,), dtype=tf.int64,   name="stops_horizon"),
        tf.TensorSpec(shape=(None,), dtype=tf.bool,    name="obs_mask"),
        tf.TensorSpec(shape=(None,), dtype=tf.bool,    name="horizon_mask"),
        tf.TensorSpec(shape=(None,), dtype=tf.float32, name="times_target"),
        tf.TensorSpec(shape=(),      dtype=tf.int64,   name="l_idx"),
        tf.TensorSpec(shape=(),      dtype=tf.string,  name="trip_id"),
        tf.TensorSpec(shape=(None,), dtype=tf.float32, name="context"),
    )

    ds = tf.data.Dataset.from_generator(
        gen, output_signature=output_signature,
    )

    # Cache para que el generator Python solo se ejecute una vez.
    # Por defecto a DISCO (RAM acotada); el archivo se borra cuando el
    # Dataset es recolectado (fin del fold) y, como red, al salir del proceso.
    cache_prefix: Optional[str] = None
    if cache == "disk":
        cache_prefix = _new_cache_prefix()
        ds = ds.cache(cache_prefix)
    elif cache:          # True → cache en RAM (comportamiento antiguo)
        ds = ds.cache()
    # cache=False → sin cache (el generator corre cada epoch)

    if shuffle:
        # Cap del buffer: nunca cargar más de _SHUFFLE_BUFFER_CAP muestras.
        buf = min(n, shuffle_buffer or _SHUFFLE_BUFFER_CAP)
        ds = ds.shuffle(buffer_size=buf, seed=42)

    # padded_batch con valores de relleno por tensor
    padded_shapes = (
        [None],   # stops_obs
        [None],   # times_obs
        [None],   # stops_horizon
        [None],   # obs_mask
        [None],   # horizon_mask
        [None],   # times_target
        [],       # l_idx (scalar)
        [],       # trip_id (scalar)
        [None],   # context (fijo por dataset; no se rellena en la práctica)
    )
    padding_values = (
        tf.constant(PAD_IDX, dtype=tf.int64),    # stops_obs
        tf.constant(0.0,     dtype=tf.float32),  # times_obs
        tf.constant(PAD_IDX, dtype=tf.int64),    # stops_horizon
        tf.constant(False,   dtype=tf.bool),     # obs_mask
        tf.constant(False,   dtype=tf.bool),     # horizon_mask
        tf.constant(0.0,     dtype=tf.float32),  # times_target
        tf.constant(-1,      dtype=tf.int64),    # l_idx (sentinel)
        tf.constant("",      dtype=tf.string),   # trip_id
        tf.constant(0.0,     dtype=tf.float32),  # context
    )

    ds = ds.padded_batch(
        batch_size       = batch_size,
        padded_shapes    = padded_shapes,
        padding_values   = padding_values,
        drop_remainder   = drop_remainder,
    )

    out = ds.prefetch(tf.data.AUTOTUNE)

    # Ata la limpieza del cache en disco al ciclo de vida del Dataset: cuando
    # el caller suelta `out` (p.ej. al pasar al siguiente fold), el finalizer
    # borra los ficheros del cache y libera el scratch.
    if cache_prefix is not None:
        weakref.finalize(out, _cleanup_cache, cache_prefix)

    return out


def _get_output_signature():
    """Signature común para crear datasets vacíos."""
    return (
        tf.TensorSpec(shape=(None,), dtype=tf.int64),
        tf.TensorSpec(shape=(None,), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.int64),
        tf.TensorSpec(shape=(None,), dtype=tf.bool),
        tf.TensorSpec(shape=(None,), dtype=tf.bool),
        tf.TensorSpec(shape=(None,), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int64),
        tf.TensorSpec(shape=(), dtype=tf.string),
        tf.TensorSpec(shape=(None,), dtype=tf.float32),
    )


# ──────────────────────────────────────────────────────────────────────
# Loss MSE enmascarada
# ──────────────────────────────────────────────────────────────────────

def masked_mse_loss(
    target: tf.Tensor,  # (B, H_max)
    pred:   tf.Tensor,  # (B, H_max)
    mask:   tf.Tensor,  # (B, H_max) bool
    reduction: str = "mean",
) -> tf.Tensor:
    """
    MSE solo sobre los pasos del horizonte que son válidos según mask.

    Convención de argumentos: (target, pred, mask) — compatible con
    `keras.losses.Loss.call(y_true, y_pred)` si se quiere wrap.

    Parameters
    ----------
    target, pred : (B, H_max) float
    mask : (B, H_max) bool
    reduction : "mean" | "sum" | "none"

    Returns
    -------
    Tensor escalar (mean/sum) o (B, H_max) (none).
    """
    if reduction not in ("mean", "sum", "none"):
        raise ValueError(f"reduction inválida: {reduction!r}")

    sq_err = tf.square(pred - target)                  # (B, H_max)
    mask_f = tf.cast(mask, sq_err.dtype)
    sq_err = sq_err * mask_f                           # ceros en padding

    if reduction == "none":
        return sq_err

    total = tf.reduce_sum(sq_err)
    if reduction == "sum":
        return total

    n_valid = tf.maximum(tf.reduce_sum(mask_f), 1.0)
    return total / n_valid


def masked_rmse(
    target: tf.Tensor, pred: tf.Tensor, mask: tf.Tensor,
) -> tf.Tensor:
    """RMSE enmascarada."""
    return tf.sqrt(masked_mse_loss(target, pred, mask, reduction="mean"))


def masked_mae(
    target: tf.Tensor, pred: tf.Tensor, mask: tf.Tensor,
) -> tf.Tensor:
    """MAE enmascarada."""
    abs_err = tf.abs(pred - target) * tf.cast(mask, pred.dtype)
    n_valid = tf.maximum(tf.reduce_sum(tf.cast(mask, pred.dtype)), 1.0)
    return tf.reduce_sum(abs_err) / n_valid
