# El entorno de Python

Esta carpeta usa un **`.venv` propio**, dentro de ella. Lo crea un script:

```bash
cd ~/thesis_final
bash setup_venv.sh
```

`config.env` ya apunta a `./.venv/bin/python` por defecto, así que si el script
termina bien **no hay que editar nada**.

---

## La única restricción que importa: la versión del Python base

`python -m venv` **no elige** la versión de Python: hereda la del intérprete
sobre el que lo montas. Y TensorFlow para AMD viene en una rueda con la versión
grabada en el nombre:

```
tensorflow_rocm-2.18.1-cp310-cp310-manylinux_2_28_x86_64.whl
                       ^^^^^ esta, solo Python 3.10
```

Así que el venv sirve perfectamente **si el python base tiene rueda**. Es lo
único que hay que acertar, y `setup_venv.sh` lo comprueba **antes** de crear
nada: consulta el índice de AMD y aborta con instrucciones si tu versión no
tiene rueda, en vez de dejarte un venv a medias que revienta en la fase 2 seis
horas después.

**No estás atado a 3.10.** Comprobado contra `rocm-rel-6.4.2` el 2026-08, AMD
publica TF-ROCm 2.18.1 para:

| Python | ¿rueda? |
|---|---|
| 3.9 (`cp39`) | sí |
| **3.10** (`cp310`) | sí ← la del paper |
| 3.11 (`cp311`) | **no** |
| 3.12 (`cp312`) | sí |

El script prefiere 3.10 (por reproducibilidad), luego 3.12, luego 3.9 — y si
solo encuentra 3.11, para y te enseña la lista real del índice. Esa lista se
consulta en vivo: si AMD publica más ruedas, las verá.

Dos cosas más sobre esa rueda:

- **No sale de PyPI.** La aloja AMD en `repo.radeon.com`, y su rama tiene que
  casar con el ROCm del sistema — los `.slrm` cargan `module load rocm/6.4.3` y
  la del paper es de `rocm-rel-6.4.2`.
- **`pip install tensorflow` a secas instala el build de CUDA**, que en una
  MI210 no ve la GPU. Y no da error: solo lentitud. Por eso
  `requirements-server.txt` no lleva TensorFlow y el script lo instala aparte.

## Qué hace `setup_venv.sh`

1. Busca un intérprete base y **prefiere 3.10 → 3.12 → 3.9**, que son las que
   tienen rueda. Mira `--python` si se lo pasas, los `python3.X` del PATH, los
   pythons de conda que haya en el sistema, y `python3`.
2. Consulta `https://repo.radeon.com/rocm/manylinux/rocm-rel-<ver>/` y busca la
   rueda de `tensorflow_rocm` para esa versión. Si no hay, **para ahí**.
3. Crea `.venv`, actualiza pip, instala `requirements-server.txt`.
4. Instala la rueda de ROCm (~600 MB).
5. Comprueba los imports y te dice si algo falta.

```bash
bash setup_venv.sh                                   # autodetecta
bash setup_venv.sh --python /ruta/al/python3.10      # base explícito
bash setup_venv.sh --rocm 6.3.4                      # otra rama de ROCm
bash setup_venv.sh --no-tf                           # solo CPU (fases 0, 1, 3)
bash setup_venv.sh --recreate                        # borrar y empezar de cero
```

## Si no hay ningún Python con rueda

Por orden de preferencia:

```bash
# 1 · ¿hay un 3.10, 3.12 o 3.9 por módulos?
module avail python
bash setup_venv.sh --python /ruta/al/python3.10

# 2 · ¿hay rueda para tu versión en otra rama de ROCm?
#     mira https://repo.radeon.com/rocm/manylinux/ y prueba
bash setup_venv.sh --rocm 6.3.4

# 3 · último recurso: venv solo para CPU, y la fase 2 con otro intérprete
bash setup_venv.sh --no-tf
```

Con la opción 3, las fases 0, 1 y 3 van con el venv y solo el LSTM necesita otro
python:

```bash
PY=/ruta/al/python/con/tf-rocm sbatch -p gpu --array=0-21%4 \
    jobs/job_lstm_frame.slrm --require-frame
```

## ⚠ Nunca hagas `source .venv/bin/activate` antes de un `sbatch`

Y no es una manía: los `.slrm` hacen

```bash
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV
```

**Eso NO rompe el venv.** Un venv se reconoce por el `pyvenv.cfg` que hay junto
al ejecutable, no por la variable `VIRTUAL_ENV`; llamar a `.venv/bin/python` por
ruta absoluta funciona sin activar nada. Lo que ese `unset` evita es lo
contrario: que un entorno activo **en la shell desde la que envías el job** se
cuele por `--export=ALL` y le pise el `sys.path` al intérprete del job, mezclando
site-packages y stdlib de dos instalaciones distintas.

Del mismo modo, `config.env` pide una **ruta absoluta** y nunca `python` a secas:
dentro de un job, activar un entorno falla en silencio y `python` cae al
intérprete equivocado. El síntoma es carísimo — el LSTM entrena en CPU a ~29 min
por época con la GPU asignada ociosa, **y el job termina en verde**.

## `libhsa-runtime64.so.1: cannot open shared object file`

**No es un error del entorno.** Es la runtime de ROCm, que solo existe en los
nodos con GPU; el env que produjo el paper da exactamente el mismo mensaje en el
login node.

Comprueba las dependencias de CPU y TensorFlow **por separado**, porque un
import conjunto aborta en TF y parece que falta todo:

```bash
PY=./.venv/bin/python
$PY -c "import polars, numpy, scipy, pyarrow, h3, matplotlib, xgboost, sklearn; print('deps CPU ok')"
$PY -m pip list 2>/dev/null | grep -iE "tensorflow|keras"
```

Para verlo importar en el propio login node:

```bash
module load rocm/6.4.3
$PY -c "import tensorflow as tf; print(tf.__version__, tf.config.list_physical_devices('GPU'))"
```

Debe imprimir `2.18.1 []`. La lista vacía es correcta: el login node no tiene
GPU. Lo que se comprueba es que **importa**.

Que **vea la GPU** se comprueba en el nodo de cómputo: cada job del LSTM
(`jobs/job_lstm_frame.slrm`) importa TensorFlow, cuenta las GPUs y aborta en
rojo si no ve ninguna, antes de entrenar nada. Busca `TF chk  : rc=0 · gpus=1`
al principio de su log.

## Qué es obligatorio y qué no

| Paquete | Fase | ¿Obligatorio? |
|---|---|---|
| `polars` `numpy` `scipy` `pyarrow` | 0, 1 | **sí** |
| `h3` `matplotlib` | 0 | **sí** (cadena de imports de `_lib_turn_hex`) |
| `pandas` `folium` `plotly` | 0 | sí para mapas/dashboards; esta corrida va con `--no-maps --no-dashboards` |
| `xgboost` `scikit-learn` | 1 | **sí** |
| `tensorflow_rocm` | 2 | **sí**, y con soporte ROCm de verdad |
| `sqlalchemy` `pg8000` | — | solo para descargar de la BD; aquí se parte de `data/01_raw` |
| `psutil` | 0 | **no** — todos sus usos están en `try/except`. Sin él, `perf_*.json` sale sin `peak_rss_gb` ni `--mem sugerido`, nada más. El env que produjo el paper **no lo tenía** |

## Versiones de la corrida publicada

Las registra `RUNINFO_q05c100.txt` del bundle (`artefacts/run_bundle_q05c100_20261009_171428`)
y son las que fija `requirements-server.txt`:

| | |
|---|---|
| Python | 3.10.19 |
| polars | 1.41.2 |
| numpy | 1.26.4 |
| scipy | 1.15.3 · pyarrow 21.0.0 |
| xgboost | 3.2.0 · scikit-learn 1.7.2 |
| tensorflow-rocm | 2.18.1 (ROCm 6.4, fase GPU) |

⚠ Reproducir el entorno **no** reproduce las cifras al dígito: el LSTM no es
bit-reproducible (el no-determinismo es de la GPU). Rehacer rutas con el frame,
los folds y las semillas fijos mueve su MAE (Tabla S10 del artículo). Fijar versiones
sirve para que las diferencias sean atribuibles a esa fuente y no a otra, no
para eliminarlas.
