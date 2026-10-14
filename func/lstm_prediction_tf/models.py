"""
lstm_prediction_tf.models
===========================
Etapa C: arquitecturas neuronales (TensorFlow/Keras 3).

Adaptación del Sequential LSTM de Sun & Kim (2021) a predicción de
tiempos de llegada en ruta fija.

Diferencias respecto al paper original
--------------------------------------
El paper resuelve la tarea conjunta de predecir siguiente ubicación
(clasificación) y tiempo (regresión). En ruta fija con orden de paradas
determinístico, la siguiente ubicación es trivial (stop_order[idx+1]).
Por ello eliminamos la cabeza de clasificación y conservamos solo la
estructura LSTM compartida + self-attention para la regresión de tiempo.

Adicionalmente, en lugar de predecir solo el siguiente paso (autoregresivo),
predecimos el horizonte completo h = 1..n-l usando query-attention con la
parada destino como query. Esto:
  - elimina error compounding
  - acepta horizonte variable por muestra
  - hace la comparación directa con los 4 baselines del paso 10
    (delay, linreg, KNN, kernel)

Estructura del modelo SequentialLSTM
------------------------------------
    token_t   = [embed(ℓ_t) ; proj(τ_t)]              (B, L, 2·d_emb)
    h_1..L    = LSTM(token_1..L)                       (B, L, d_hid)
    para cada destino ℓ_{L+h} en el horizonte:
        q_h       = embed(ℓ_{L+h})                    (B, d_emb)
        ctx_h     = AdditiveAttention(q_h, h_1..L)    (B, d_hid)
        τ̂_{L+h}  = MLP([ctx_h ; q_h])                 (B,)

Notas sobre Keras 3
-------------------
Este módulo usa la API de subclassing de Keras 3 (compatible con TF >= 2.16).
Los modelos heredan de `keras.Model`, las capas de `keras.layers.Layer`.
El método principal es `call()`, no `forward()`. La compilación
(`model.compile`) es opcional cuando se usa un loop custom, pero la API
funcional `model.fit` también está soportada (ver trainer.py).

Referencia
----------
Sun, J. & Kim, J. (2021). Transportation Research Part C, 128.
Bahdanau, D., Cho, K. & Bengio, Y. (2015). Neural Machine Translation
by Jointly Learning to Align and Translate.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import tensorflow as tf
from tensorflow import keras
from keras import layers


# ──────────────────────────────────────────────────────────────────────
# Atención por trozos (enmienda E3 del pre-registro)
# ──────────────────────────────────────────────────────────────────────

# Los kernels elementales de TF en GPU indexan con int32: un tensor de más de
# 2^31 − 1 elementos da «Memory access fault by GPU» y mata el proceso. Medido
# el 7-oct-2026 en la MI210 (TF 2.18.1, ROCm 6.4.3; PREREGISTRO_VUELTA.md, E3):
# 64·511²·128 = 2,139,103,232 pasa y 64·513²·128 = 2,155,880,448 cae.
GPU_MAX_ELEMS = 2**31 - 1
# Cuando hay que trocear, cada trozo se queda en la mitad del tope, por margen.
ATTN_CHUNK_ELEMS = 2**30


def attention_chunks(batch_size: int, max_len: int, attention_hidden_dim: int) -> int:
    """Trozos del horizonte que necesita la atención aditiva (enmienda E3).

    El tensor más grande del modelo es (B, H, L, d_attn) en
    `SequentialLSTM._attention_scores`. Con el padding por batch, H y L llegan
    a `max_len − 1` (L + H = la secuencia entera). Si eso cabe en int32 → 1, el
    camino original; si no, los trozos para que cada uno quede en
    `ATTN_CHUNK_ELEMS`. Con la vuelta y batch 64 eso pasa en las rutas de
    más de 512 celdas.
    """
    n = max(int(max_len) - 1, 1)
    elems = int(batch_size) * n * n * int(attention_hidden_dim)
    if elems <= GPU_MAX_ELEMS:
        return 1
    return math.ceil(elems / ATTN_CHUNK_ELEMS)


# ──────────────────────────────────────────────────────────────────────
# Interfaz común
# ──────────────────────────────────────────────────────────────────────

class BaseLSTMModel(keras.Model):
    """
    Interfaz común para todos los modelos del paquete.

    Subclases deben implementar `call()` y devolver un tensor (B, H)
    con las predicciones de link travel time para cada paso del horizonte.
    """

    name: str = "base"

    def call(
        self,
        inputs,
        training: bool = False,
        return_attention: bool = False,
    ):
        """
        Parameters
        ----------
        inputs : tuple or dict
            (stops, times, stops_horizon, obs_mask, horizon_mask)
            o dict con esas claves.
            - stops          : LongTensor (B, L) - IDs de paradas observadas
            - times          : FloatTensor (B, L) - link_travel_time observados
            - stops_horizon  : LongTensor (B, H) - IDs paradas futuras
            - obs_mask       : BoolTensor (B, L) o None - máscara de padding
            - horizon_mask   : BoolTensor (B, H) o None
        training : bool
            Activa/desactiva dropout y otros módulos sensibles a fase.
        return_attention : bool
            Si True, retorna también los pesos de atención.

        Returns
        -------
        time_pred : FloatTensor (B, H)
        attention_weights : opcional, FloatTensor (B, H, L)
        """
        raise NotImplementedError

    def num_parameters(self) -> int:
        """Cuenta los parámetros entrenables."""
        return sum(
            int(tf.size(v).numpy()) for v in self.trainable_variables
        )


# ──────────────────────────────────────────────────────────────────────
# Componentes internos
# ──────────────────────────────────────────────────────────────────────

class AdditiveAttention(layers.Layer):
    """
    Atención aditiva (Bahdanau et al. 2015):

        e_i = v^T · tanh(W_q · q + W_k · k_i)
        α_i = softmax(e_i)
        ctx = Σ α_i · v_i

    El paper de Sun & Kim no especifica el tipo de attention; usamos
    aditiva por ser la formulación original referenciada en su §3.3
    y por ser estable con secuencias cortas (~100 tokens).
    """

    def __init__(
        self,
        query_dim:   int,
        key_dim:     int,
        hidden_dim:  int,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.supports_masking = True
        self.query_dim  = query_dim
        self.key_dim    = key_dim
        self.hidden_dim = hidden_dim
        self.W_q = layers.Dense(hidden_dim, use_bias=False, name="W_q")
        self.W_k = layers.Dense(hidden_dim, use_bias=False, name="W_k")
        self.v   = layers.Dense(1,          use_bias=False, name="v")

    def call(
        self,
        query:    tf.Tensor,           # (B, query_dim)
        keys:     tf.Tensor,           # (B, L, key_dim)
        values:   tf.Tensor,           # (B, L, value_dim)
        mask:     Optional[tf.Tensor] = None,   # (B, L) bool
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Returns
        -------
        ctx     : (B, value_dim)
        weights : (B, L) pesos de atención post-softmax
        """
        q_proj = tf.expand_dims(self.W_q(query), axis=1)   # (B, 1, hidden_dim)
        k_proj = self.W_k(keys)                            # (B, L, hidden_dim)
        scores = tf.squeeze(self.v(tf.tanh(q_proj + k_proj)), axis=-1)  # (B, L)

        if mask is not None:
            # -inf donde mask=False → softmax da 0
            mask_f = tf.cast(mask, scores.dtype)
            scores = scores + (1.0 - mask_f) * -1e9

        weights = tf.nn.softmax(scores, axis=-1)           # (B, L)

        # (B, 1, L) @ (B, L, value_dim) → (B, value_dim)
        ctx = tf.squeeze(
            tf.matmul(tf.expand_dims(weights, 1), values),
            axis=1,
        )
        return ctx, weights


# ──────────────────────────────────────────────────────────────────────
# SequentialLSTM (modelo principal)
# ──────────────────────────────────────────────────────────────────────

class SequentialLSTM(BaseLSTMModel):
    """
    Adaptación del Sequential LSTM de Sun & Kim 2021 §3.3 para
    predicción de link travel time en ruta fija con horizonte completo.

    Parameters
    ----------
    n_stops : int
    embed_dim : int, default=64
    time_proj_dim : int, default=64
    hidden_dim : int, default=128
    n_lstm_layers : int, default=1
    dropout : float, default=0.1
    attention_hidden_dim : int, default=128
    pad_idx : int, default=0
        ID del token PAD. En Keras se maneja vía `mask_zero=True`
        en el Embedding; pad_idx debe ser 0 para que funcione.
    attention_chunks : int, default=1
        Trozos del horizonte en que se calcula la atención (enmienda E3; ver
        `attention_chunks()`). 1 = el camino original. Con más, la misma
        función: sólo cambia cuántas filas ve cada kernel.

    Notes
    -----
    Keras 3 acepta `mask_zero=True` en `Embedding` y propaga la máscara
    automáticamente a través de capas que la soporten (incluido LSTM).
    Sin embargo, la atención custom NO recibe la máscara por propagación
    automática; debemos pasarla explícitamente. Por eso `call()` acepta
    `obs_mask` y `horizon_mask` como entradas.
    """

    name = "sequential_lstm"

    def __init__(
        self,
        n_stops:              int,
        embed_dim:            int   = 64,
        time_proj_dim:        int   = 64,
        hidden_dim:           int   = 128,
        n_lstm_layers:        int   = 1,
        dropout:              float = 0.1,
        attention_hidden_dim: int   = 128,
        pad_idx:              int   = 0,
        context_dim:          int   = 0,
        attention_chunks:     int   = 1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        if embed_dim <= 0 or hidden_dim <= 0:
            raise ValueError("embed_dim y hidden_dim deben ser positivos")
        if pad_idx != 0:
            raise ValueError(
                "En Keras 3, mask_zero requiere pad_idx=0. "
                "Reordena tu vocabulario para que PAD ocupe el índice 0."
            )

        self.n_stops       = n_stops
        self.embed_dim     = embed_dim
        self.time_proj_dim = time_proj_dim
        self.hidden_dim    = hidden_dim
        self.dropout_rate  = float(dropout)
        # context_dim > 0 → fase 4: el contexto de calendario (por corte) se
        # concatena a [ctx_attn ; query] antes del head. =0 reproduce EXACTAMENTE
        # el modelo certificado (el contexto se ignora aunque venga en la tupla).
        self.context_dim   = int(context_dim)
        self.attention_chunks = max(int(attention_chunks), 1)

        # 1) Embedding de paradas con PAD enmascarado automáticamente
        self.stop_emb = layers.Embedding(
            input_dim    = n_stops,
            output_dim   = embed_dim,
            mask_zero    = True,
            name         = "stop_emb",
        )

        # 2) Proyección del escalar de tiempo a vector
        self.time_proj = keras.Sequential([
            layers.Dense(time_proj_dim, activation="relu"),
            layers.Dense(time_proj_dim),
        ], name="time_proj")

        # 3) LSTM compartida sobre tokens conjuntos (stacked si n_lstm_layers > 1)
        lstm_dropout = dropout if n_lstm_layers > 1 else 0.0
        if n_lstm_layers == 1:
            self.lstm = layers.LSTM(
                hidden_dim, return_sequences=True, name="lstm",
            )
        else:
            self.lstm = keras.Sequential([
                layers.LSTM(
                    hidden_dim, return_sequences=True,
                    dropout=0.0, recurrent_dropout=0.0,
                )
                for _ in range(n_lstm_layers)
            ], name="lstm_stack")

        # 4) Attention: query = embed(stop_horizon), keys/values = h_1..L
        self.attention = AdditiveAttention(
            query_dim  = embed_dim,
            key_dim    = hidden_dim,
            hidden_dim = attention_hidden_dim,
            name       = "attention",
        )

        # 5) Head de regresión sobre [ctx ; query]
        self.head = keras.Sequential([
            layers.Dense(hidden_dim, activation="relu"),
            layers.Dropout(dropout),
            layers.Dense(1),
        ], name="head")

        self.dropout_in = layers.Dropout(dropout, name="dropout_in")

    # ────────────────────────────────────────────────────────────────
    # Forward
    # ────────────────────────────────────────────────────────────────

    def call(
        self,
        inputs,
        training: bool = False,
        return_attention: bool = False,
    ):
        # Aceptar tupla o dict
        context = None
        if isinstance(inputs, dict):
            stops         = inputs["stops"]
            times         = inputs["times"]
            stops_horizon = inputs["stops_horizon"]
            obs_mask      = inputs.get("obs_mask")
            horizon_mask  = inputs.get("horizon_mask")
            context       = inputs.get("context")
        else:
            # tupla (stops, times, stops_horizon[, obs_mask, horizon_mask[, ctx]])
            if len(inputs) == 3:
                stops, times, stops_horizon = inputs
                obs_mask = horizon_mask = None
            elif len(inputs) == 5:
                stops, times, stops_horizon, obs_mask, horizon_mask = inputs
            elif len(inputs) == 6:
                (stops, times, stops_horizon, obs_mask, horizon_mask,
                 context) = inputs
            else:
                raise ValueError(
                    f"inputs debe ser tuple de 3, 5 o 6 elementos, "
                    f"recibí {len(inputs)}"
                )

        # ── 1) Encoder: tokens conjuntos → LSTM ────────────────────
        e_loc  = self.stop_emb(stops)                              # (B, L, d_emb)

        # times: (B, L) → (B, L, 1) → (B, L, d_time)
        e_time = self.time_proj(tf.expand_dims(times, axis=-1), training=training)

        x = tf.concat([e_loc, e_time], axis=-1)                    # (B, L, d_emb+d_time)
        x = self.dropout_in(x, training=training)

        h_states = self.lstm(x, training=training)                 # (B, L, d_hid)

        # ── 2) Decoder por horizonte (vectorizado) ────────────────
        q = self.stop_emb(stops_horizon)                           # (B, H, d_emb)
        scores = self._attention_scores(q, h_states)               # (B, H, L)

        # Aplicar obs_mask sobre keys (broadcast a H dimensiones)
        if obs_mask is not None:
            mask_f = tf.cast(obs_mask, scores.dtype)               # (B, L)
            mask_f = tf.expand_dims(mask_f, axis=1)                # (B, 1, L)
            scores = scores + (1.0 - mask_f) * -1e9

        attn_w = tf.nn.softmax(scores, axis=-1)                    # (B, H, L)

        # ctx = einsum("bhl,bld->bhd", attn_w, h_states)
        ctx = tf.einsum("bhl,bld->bhd", attn_w, h_states)          # (B, H, d_hid)

        # ── 3) Head: regresión sobre [ctx_attn ; query (; calendario)] ──
        head_parts = [ctx, q]                                      # (B, H, d_hid+d_emb)
        if self.context_dim > 0 and context is not None:
            # context: (B, k) al corte l → broadcast a los H horizontes.
            H = tf.shape(q)[1]
            cal = tf.repeat(tf.expand_dims(context, axis=1), H, axis=1)  # (B, H, k)
            head_parts.append(cal)
        head_in = tf.concat(head_parts, axis=-1)
        time_pred = tf.squeeze(self.head(head_in, training=training), axis=-1)

        if return_attention:
            return time_pred, attn_w
        return time_pred

    def _attention_scores(self, q: tf.Tensor, h_states: tf.Tensor) -> tf.Tensor:
        """scores[b, h, l] = v · tanh(W_q q_h + W_k h_l), de forma (B, H, L).

        Vectorizado: H queries contra L keys en un tensor (B, H, L, attn_hid).
        Con `attention_chunks > 1` (enmienda E3) se calcula por trozos del
        horizonte y se concatena. Cada (b, h, l) es la misma cuenta: en la GPU
        la salida sale bit a bit igual y los gradientes, a 3e-7 relativo (por
        el orden de las sumas).
        """
        # q_lin: (B, H, attn_hid)
        q_lin = self.attention.W_q(q)
        # k_proj: (B, 1, L, attn_hid)
        k_proj = tf.expand_dims(self.attention.W_k(h_states), axis=1)
        if self.attention_chunks <= 1:
            # q_proj: (B, H, 1, attn_hid)
            q_proj = tf.expand_dims(q_lin, axis=2)
            return tf.squeeze(self.attention.v(tf.tanh(q_proj + k_proj)), axis=-1)

        # Trozos de ceil(H / k) filas; el último puede salir corto o vacío.
        size = (tf.shape(q_lin)[1] + self.attention_chunks - 1) // self.attention_chunks
        parts = []
        for i in range(self.attention_chunks):
            q_c = tf.expand_dims(q_lin[:, i * size:(i + 1) * size, :], axis=2)
            parts.append(tf.squeeze(self.attention.v(tf.tanh(q_c + k_proj)), axis=-1))
        return tf.concat(parts, axis=1)


# ──────────────────────────────────────────────────────────────────────
# HybridLSTM (stub — no es default del pipeline)
# ──────────────────────────────────────────────────────────────────────

class HybridLSTM(BaseLSTMModel):
    """
    Adaptación del Hybrid LSTM de Sun & Kim 2021 §3.2 para predicción
    de tiempo en ruta fija (variante para ablation, no es default).

    Estructura
    ----------
        h^L = LSTM_loc(embed(stops))       # torre de ubicación
        h^T = LSTM_time(proj(times))       # torre de tiempo
        h   = concat(h^L, h^T)             # fusión tardía
        ctx_h = AdditiveAttention(embed(ℓ_{L+h}), h)
        τ̂_{L+h} = Head([ctx_h ; embed(ℓ_{L+h})])
    """

    name = "hybrid_lstm"

    def __init__(
        self,
        n_stops:              int,
        embed_dim:            int   = 64,
        time_proj_dim:        int   = 64,
        hidden_dim:           int   = 128,
        n_lstm_layers:        int   = 1,
        dropout:              float = 0.1,
        attention_hidden_dim: int   = 128,
        pad_idx:              int   = 0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        if pad_idx != 0:
            raise ValueError("Keras 3 requiere pad_idx=0 (mask_zero)")

        self.n_stops    = n_stops
        self.embed_dim  = embed_dim
        self.hidden_dim = hidden_dim

        # Torre de ubicación
        self.stop_emb = layers.Embedding(
            n_stops, embed_dim, mask_zero=True, name="stop_emb",
        )
        self.lstm_loc = layers.LSTM(
            hidden_dim, return_sequences=True, name="lstm_loc",
        )

        # Torre de tiempo
        self.time_proj = keras.Sequential([
            layers.Dense(time_proj_dim, activation="relu"),
            layers.Dense(time_proj_dim),
        ], name="time_proj")
        self.lstm_time = layers.LSTM(
            hidden_dim, return_sequences=True, name="lstm_time",
        )

        # Attention sobre concatenación
        self.attention = AdditiveAttention(
            query_dim  = embed_dim,
            key_dim    = 2 * hidden_dim,
            hidden_dim = attention_hidden_dim,
            name       = "attention",
        )

        self.head = keras.Sequential([
            layers.Dense(hidden_dim, activation="relu"),
            layers.Dropout(dropout),
            layers.Dense(1),
        ], name="head")

        self.dropout_in = layers.Dropout(dropout, name="dropout_in")

    def call(
        self,
        inputs,
        training: bool = False,
        return_attention: bool = False,
    ):
        if isinstance(inputs, dict):
            stops         = inputs["stops"]
            times         = inputs["times"]
            stops_horizon = inputs["stops_horizon"]
            obs_mask      = inputs.get("obs_mask")
        else:
            if len(inputs) == 3:
                stops, times, stops_horizon = inputs
                obs_mask = None
            elif len(inputs) == 5:
                stops, times, stops_horizon, obs_mask, _ = inputs
            else:
                raise ValueError(f"len(inputs)={len(inputs)}")

        # Torre de ubicación
        e_loc = self.dropout_in(self.stop_emb(stops), training=training)
        h_loc = self.lstm_loc(e_loc, training=training)

        # Torre de tiempo
        e_time = self.dropout_in(
            self.time_proj(tf.expand_dims(times, -1), training=training),
            training=training,
        )
        h_time = self.lstm_time(e_time, training=training)

        # Fusión tardía
        h_states = tf.concat([h_loc, h_time], axis=-1)              # (B, L, 2·d_hid)

        # Queries por horizonte
        q = self.stop_emb(stops_horizon)                            # (B, H, d_emb)
        q_proj = tf.expand_dims(self.attention.W_q(q), axis=2)
        k_proj = tf.expand_dims(self.attention.W_k(h_states), axis=1)
        scores = tf.squeeze(
            self.attention.v(tf.tanh(q_proj + k_proj)), axis=-1,
        )

        if obs_mask is not None:
            mask_f = tf.cast(obs_mask, scores.dtype)
            scores = scores + (1.0 - tf.expand_dims(mask_f, 1)) * -1e9

        attn_w = tf.nn.softmax(scores, axis=-1)
        ctx = tf.einsum("bhl,bld->bhd", attn_w, h_states)

        head_in = tf.concat([ctx, q], axis=-1)
        time_pred = tf.squeeze(self.head(head_in, training=training), axis=-1)

        if return_attention:
            return time_pred, attn_w
        return time_pred


# ──────────────────────────────────────────────────────────────────────
# Inicialización (Keras 3 ya usa Glorot por defecto, esta función es no-op)
# ──────────────────────────────────────────────────────────────────────

def init_weights(model: keras.Model) -> None:
    """
    Reinicialización de pesos.

    Keras 3 inicializa por defecto con `glorot_uniform` (equivalente a
    `xavier_uniform_` de PyTorch) y biases en cero. Esta función está
    aquí por paridad de API con la versión PyTorch pero NO necesita hacer
    nada — los pesos ya están bien inicializados al construir las capas.

    Si en el futuro quieres reinicializar (e.g. para reentrenar desde
    scratch un modelo cargado), llama explícitamente a:

        for layer in model.layers:
            if hasattr(layer, 'kernel_initializer'):
                layer.build(None)   # fuerza reinicialización
    """
    pass
