"""
lstm_prediction.vocab
=======================
Etapa B: vocabulario de paradas (stop_code ↔ entero) para el embedding
del SequentialLSTM.

Por qué un archivo dedicado
---------------------------
El modelo neuronal opera sobre IDs enteros (entrada de nn.Embedding),
no sobre strings. El vocabulario es el mapeo `stop_code → int` y su
inverso. Tres propiedades NO negociables:

  1. Determinismo. La misma lista de paradas debe producir el mismo
     mapeo siempre. Si entrenas un modelo, guardas los pesos, y luego
     reconstruyes el vocabulario con paradas en otro orden, el modelo
     queda inutilizable (los IDs ya no corresponden a las paradas).

  2. Persistencia. Al serializar un modelo entrenado, hay que serializar
     también el vocabulario. `save()`/`load()` están aquí.

  3. Tokens especiales (PAD, UNK). El padding es necesario para procesar
     batches de secuencias de longitudes distintas; UNK protege ante
     paradas no vistas en train (e.g. una ruta que añade una parada
     nueva).

Convención de IDs
-----------------
  0  → PAD       (reservado para padding, ignorado en la loss)
  1  → UNK       (parada desconocida; activable opcionalmente)
  2+ → paradas ordenadas alfabéticamente

El orden alfabético es la elección más simple que cumple determinismo.
Alternativas (orden por frecuencia, orden geográfico, orden secuencial
en la ruta) son válidas pero acoplan el vocabulario a metadata externa.

Uso típico
----------
    from lstm_prediction.vocab import StopVocab

    # Construir desde el dataset
    vocab = StopVocab.from_stop_counts(dataset["stop_counts"])

    # Encodificar
    ids = vocab.encode_sequence(["101-S001", "101-S002", "DESCONOCIDA"])
    # → [2, 3, 1]   (1 = UNK)

    # Persistir
    vocab.save("vocab.json")
    vocab2 = StopVocab.load("vocab.json")
    assert vocab == vocab2
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Union


# ──────────────────────────────────────────────────────────────────────
# Tokens especiales
# ──────────────────────────────────────────────────────────────────────

PAD_TOKEN: str = "<PAD>"
UNK_TOKEN: str = "<UNK>"

PAD_IDX:   int = 0
UNK_IDX:   int = 1


class StopVocab:
    """
    Mapeo determinístico entre códigos de parada y enteros.

    Parameters
    ----------
    stops : Iterable[str]
        Códigos de parada (e.g. ["101-FQAE", "101-S001", ...]).
        Pueden venir en cualquier orden; se ordenan alfabéticamente
        internamente para garantizar determinismo.
    add_unk : bool, default=True
        Si True, reserva el ID 1 para paradas desconocidas. Si False,
        toda parada fuera del vocabulario lanza KeyError al encodificar.

    Attributes
    ----------
    stop_to_id : dict[str, int]
    id_to_stop : list[str]   (índice = ID)
    n_stops    : int         (total incluyendo PAD y UNK)

    Notes
    -----
    PAD siempre ocupa el índice 0. UNK ocupa el índice 1 si add_unk=True.
    Las paradas reales empiezan en el índice 2 (o 1 si add_unk=False).
    """

    def __init__(
        self,
        stops:   Iterable[str],
        add_unk: bool = True,
    ) -> None:
        # Ordenar alfabéticamente y deduplicar manteniendo orden
        # (sorted ya deduplica + ordena con set; usamos sorted(set(...)) explícito)
        unique_stops: List[str] = sorted({str(s) for s in stops if s is not None})

        self.add_unk: bool = bool(add_unk)

        # Construir mapeos
        self.id_to_stop: List[str] = [PAD_TOKEN]
        if self.add_unk:
            self.id_to_stop.append(UNK_TOKEN)
        self.id_to_stop.extend(unique_stops)

        self.stop_to_id: Dict[str, int] = {
            s: i for i, s in enumerate(self.id_to_stop)
        }

        # Sanidad: PAD siempre en 0
        assert self.stop_to_id[PAD_TOKEN] == PAD_IDX
        if self.add_unk:
            assert self.stop_to_id[UNK_TOKEN] == UNK_IDX

    # ────────────────────────────────────────────────────────────────
    # Construcción desde fuentes habituales
    # ────────────────────────────────────────────────────────────────

    @classmethod
    def from_stop_counts(
        cls,
        stop_counts: Dict[str, int],
        add_unk: bool = True,
        min_count: int = 1,
    ) -> "StopVocab":
        """
        Construye el vocabulario desde el dict de frecuencias del
        dataset_builder.

        Parameters
        ----------
        stop_counts : dict[str, int]
            Salida de `build_sequence_dataset()["stop_counts"]`.
        add_unk : bool
        min_count : int, default=1
            Frecuencia mínima para incluir la parada. Paradas con menos
            apariciones se mapean a UNK (si add_unk=True) o se omiten.

        Returns
        -------
        StopVocab
        """
        stops = [s for s, c in stop_counts.items() if c >= min_count]
        return cls(stops=stops, add_unk=add_unk)

    @classmethod
    def from_sequences(
        cls,
        sequences: List[dict],
        add_unk:   bool = True,
    ) -> "StopVocab":
        """
        Construye el vocabulario directamente desde el output del
        dataset_builder (`sequences` key).
        """
        stops = set()
        for s in sequences:
            stops.update(s["stops"].tolist())
        return cls(stops=sorted(stops), add_unk=add_unk)

    # ────────────────────────────────────────────────────────────────
    # Encoding / decoding
    # ────────────────────────────────────────────────────────────────

    def encode(self, stop: str) -> int:
        """Mapea un código de parada a su entero. UNK si no existe."""
        if stop in self.stop_to_id:
            return self.stop_to_id[stop]
        if self.add_unk:
            return UNK_IDX
        raise KeyError(
            f"Parada {stop!r} no está en el vocabulario y add_unk=False. "
            f"Reconstruye el vocabulario incluyendo esta parada o "
            f"habilita add_unk=True."
        )

    def decode(self, idx: int) -> str:
        """Mapea un entero a su código de parada."""
        if not (0 <= idx < len(self.id_to_stop)):
            raise IndexError(
                f"ID {idx} fuera de rango [0, {len(self.id_to_stop) - 1}]"
            )
        return self.id_to_stop[idx]

    def encode_sequence(self, stops: Iterable[str]) -> List[int]:
        """Encodifica una lista/array de códigos de parada."""
        return [self.encode(str(s)) for s in stops]

    def decode_sequence(self, ids: Iterable[int]) -> List[str]:
        """Decodifica una lista de IDs a códigos de parada."""
        return [self.decode(int(i)) for i in ids]

    # ────────────────────────────────────────────────────────────────
    # Propiedades de conveniencia
    # ────────────────────────────────────────────────────────────────

    @property
    def n_stops(self) -> int:
        """Tamaño total del vocabulario (incluye PAD y UNK)."""
        return len(self.id_to_stop)

    @property
    def n_real_stops(self) -> int:
        """Número de paradas reales (sin contar PAD ni UNK)."""
        return len(self.id_to_stop) - (2 if self.add_unk else 1)

    @property
    def pad_idx(self) -> int:
        return PAD_IDX

    @property
    def unk_idx(self) -> int:
        if not self.add_unk:
            raise AttributeError("Este vocabulario no tiene UNK (add_unk=False)")
        return UNK_IDX

    # ────────────────────────────────────────────────────────────────
    # Persistencia
    # ────────────────────────────────────────────────────────────────

    def save(self, path: Union[str, Path]) -> None:
        """
        Persiste el vocabulario en JSON.

        Formato:
            {
                "version":   1,
                "add_unk":   bool,
                "id_to_stop": ["<PAD>", "<UNK>", "101-S000", ...],
            }
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version":    1,
            "add_unk":    self.add_unk,
            "id_to_stop": self.id_to_stop,
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "StopVocab":
        """Reconstruye un vocabulario desde un JSON previamente guardado."""
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        version = payload.get("version")
        if version != 1:
            raise ValueError(
                f"Versión de vocabulario desconocida: {version}. "
                f"Este código solo entiende version=1."
            )

        # Reconstrucción: extraemos las paradas reales saltando PAD y UNK
        # según add_unk
        add_unk = payload["add_unk"]
        id_to_stop = payload["id_to_stop"]

        skip = 2 if add_unk else 1
        real_stops = id_to_stop[skip:]

        vocab = cls(stops=real_stops, add_unk=add_unk)

        # Sanidad: el roundtrip debe ser idéntico
        if vocab.id_to_stop != id_to_stop:
            raise RuntimeError(
                "El vocabulario reconstruido no coincide con el guardado. "
                "Esto indica corrupción o un bug en save/load."
            )
        return vocab

    # ────────────────────────────────────────────────────────────────
    # Dunder methods
    # ────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return self.n_stops

    def __contains__(self, stop: str) -> bool:
        return stop in self.stop_to_id

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, StopVocab):
            return NotImplemented
        return (
            self.add_unk    == other.add_unk and
            self.id_to_stop == other.id_to_stop
        )

    def __repr__(self) -> str:
        return (
            f"StopVocab(n_stops={self.n_stops}, "
            f"n_real_stops={self.n_real_stops}, "
            f"add_unk={self.add_unk})"
        )
