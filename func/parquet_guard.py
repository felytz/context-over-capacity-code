"""
func/parquet_guard.py — ¿este parquet se puede LEER, no solo existe?
=====================================================================
Por qué existe
--------------
En la corrida del 2026-07-25 el packer certificó `FRESCURA: OK` y
`leg_inclusion.csv` marcó `has_devhist_lstm: True` sobre once ficheros que
terminaban en `\\0\\0\\0\\0` en vez de `PAR1`: escrituras del LSTM que Lustre
nunca vació (proceso muerto a media escritura o nodo desalojado). Todos los
chequeos del pipeline preguntaban `is_file()`, así que el daño pasó tres
auditorías sin una sola señal, y el reintento del array los daba por hechos y
los saltaba: un fichero corrupto era MÁS pegajoso que uno ausente.

`is_file()` no es una comprobación de integridad. Este módulo es la que sí.

Qué comprueba
-------------
La envoltura del formato, que es donde se nota una escritura truncada:

    PAR1 ... <footer> <u32 LE: longitud del footer> PAR1
    ^^^^                                            ^^^^

Un fichero cortado pierde el `PAR1` final; uno con la cola sin vaciar lo
sustituye por ceros; uno con la longitud del footer basura apunta fuera del
fichero. Los tres casos se ven leyendo 12 bytes.

NO valida el esquema ni descomprime los grupos de filas: es O(1) y sirve para
barrer miles de artefactos, no para certificar su contenido. Con `strict=True`
se puede pedir además que el footer se parsee de verdad (necesita polars o
pyarrow), a costa de leerlo entero.

Sólo stdlib a propósito: lo importan tanto los jobs de `func/` como los scripts
de auditoría de `docs/paper/analysis/`, y no debe arrastrar dependencias.
"""
from __future__ import annotations

import struct
from pathlib import Path
from typing import Optional, Union

MAGIC = b"PAR1"
#: cabecera + longitud del footer + cola. Nada más pequeño puede ser parquet.
MIN_SIZE = len(MAGIC) * 2 + 4


def parquet_problem(path: Union[str, Path], *,
                    strict: bool = False) -> Optional[str]:
    """Devuelve None si `path` es un parquet íntegro; si no, el motivo.

    El motivo es una frase corta pensada para imprimirse en una auditoría.
    """
    p = Path(path)
    try:
        size = p.stat().st_size
    except OSError:
        return "no existe"
    if not p.is_file():
        return "no es un fichero"
    if size == 0:
        return "vacío (0 bytes)"
    if size < MIN_SIZE:
        return f"demasiado pequeño para ser parquet ({size} B)"

    try:
        with p.open("rb") as fh:
            head = fh.read(len(MAGIC))
            fh.seek(-(len(MAGIC) + 4), 2)
            foot_len = struct.unpack("<I", fh.read(4))[0]
            tail = fh.read(len(MAGIC))
    except OSError as exc:
        return f"ilegible ({exc.__class__.__name__})"

    if head != MAGIC:
        return "no empieza por PAR1 (no es parquet)"
    if tail != MAGIC:
        # El caso de Lustre: la cola quedó sin vaciar.
        extra = " — cola de ceros: escritura sin vaciar" if tail == b"\0" * 4 else ""
        return f"TRUNCADO: no termina en PAR1{extra}"
    if not 0 < foot_len <= size - MIN_SIZE:
        return (f"footer inconsistente ({foot_len} B declarados en un fichero "
                f"de {size} B)")

    if strict:
        try:
            import polars as pl
            pl.read_parquet_schema(p)
        except ImportError:
            try:
                import pyarrow.parquet as pq
                pq.read_metadata(p)
            except ImportError:
                return None          # sin motor: la envoltura ya está validada
            except Exception as exc:
                return f"footer no parseable ({exc.__class__.__name__})"
        except Exception as exc:
            return f"footer no parseable ({exc.__class__.__name__})"

    return None


def parquet_ok(path: Union[str, Path], *, strict: bool = False) -> bool:
    """`True` sólo si `path` existe Y es un parquet con la envoltura intacta.

    Reemplazo directo de `Path(...).is_file()` en cualquier sitio donde la
    respuesta decida si un artefacto se da por calculado.
    """
    return parquet_problem(path, strict=strict) is None


def scan(root: Union[str, Path], pattern: str = "*.parquet", *,
         strict: bool = False):
    """Recorre `root` y devuelve [(ruta, motivo)] de lo que esté roto."""
    return [(p, why) for p in sorted(Path(root).rglob(pattern))
            if (why := parquet_problem(p, strict=strict)) is not None]


if __name__ == "__main__":
    import argparse
    import sys

    ap = argparse.ArgumentParser(
        description="Busca parquets corruptos bajo un directorio.")
    ap.add_argument("root", nargs="?", default=".")
    ap.add_argument("--pattern", default="*.parquet")
    ap.add_argument("--strict", action="store_true",
                    help="además, parsea el footer (necesita polars/pyarrow)")
    a = ap.parse_args()

    bad = scan(a.root, a.pattern, strict=a.strict)
    total = sum(1 for _ in Path(a.root).rglob(a.pattern))
    for p, why in bad:
        print(f"  ✗ {p}: {why}")
    print(f"{total - len(bad)}/{total} parquet íntegros bajo {a.root}")
    sys.exit(1 if bad else 0)
