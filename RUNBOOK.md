# Manual de operación — dos comandos

| | Dónde | Comando | Qué deja |
|---|---|---|---|
| **1** | servidor (SLURM + GPU AMD) | `bash run_pipeline.sh` | `data/03_processed/run_bundle_q05c100_<fecha>.tgz` |
| **2** | tu máquina (sin GPU, minutos) | `python reproduce_tables.py [--run <bundle>]` | todas las tablas y figuras, comparadas con las publicadas |

Sin `--run`, el comando 2 parte de los artefactos publicados (`artefacts/`) y no
necesita el comando 1. Con `--run` procesa una corrida nueva del comando 1.

Régimen fijo del artículo: **q = 0.05 · min_coverage = 1.0 · max_trips = 2500**,
ventana **enero–agosto 2024**. Todo lo que produce lleva la etiqueta `q05c100` sobre el
rango `2024_01-08_tmin`.

**La corrida publicada** usa la vuelta como unidad de viaje, en 35 rutas:
- el corte por cruce de km, con los puntos de `cortes/cortes_vuelta.json`;
- la malla sobre la traza (`GRID_AXIS=traza`);
- sólo los LSTM `es15` y `tw`.

Lo fija `docs/PREREGISTRO_VUELTA.md`, y `config.env` §7 lo trae por defecto. El análisis
anterior (agosto de 2026: terminales por descubrimiento automático, legs A/B) no es parte
del artículo ni se reproduce con este repositorio.

---

## 1 · Servidor: `bash run_pipeline.sh`

### Una sola vez: preparar la carpeta

Una carpeta **nueva y vacía**. En una que ya tenga resultados, la reanudación
da por hechos los que encuentre y los salta como `cached`.

```bash
# en tu máquina: el código tal como está en git (con LF, que sbatch exige)
git archive --format=tar.gz -o repro.tgz HEAD
scp repro.tgz usuario@servidor:~/

# en el servidor
mkdir ~/repro && tar -xzf ~/repro.tgz -C ~/repro && cd ~/repro
rsync -av --checksum --partial /ruta/de/los/crudos/data/01_raw/ data/01_raw/
```

`--checksum --partial` y nunca un `cp` a pelo: una copia interrumpida ya dejó
una vez parquet truncados sin que nada se quejara (ver `data/01_raw/README.md`).

**Intérprete.** Lo más fiel es reusar el venv con el que se hicieron los bundles
de agosto. Comprueba que siga vivo; debe imprimir `3.10.19 1.41.2 1.26.4 3.2.0 1.7.2`:

```bash
/lustre/home/usuario/db_latest/.venv/bin/python -c "import sys,polars,numpy,xgboost,sklearn;print(sys.version.split()[0],polars.__version__,numpy.__version__,xgboost.__version__,sklearn.__version__)"
```

Si lo hace, pon esa ruta en la línea `PY=` de `config.env`. Si no, crea uno con
`bash setup_venv.sh`, en el login node (descarga de internet); la restricción de
versión de Python que impone TensorFlow-ROCm está en `ENVIRONMENT.md`.

**`config.env`** es el único archivo que hay que editar: intérprete (`PY`),
proyecto de SLURM (`ACCOUNT`, obligatorio desde el 7-sep-2026; `misproyectos`
lista los tuyos), particiones y régimen. Si cambias el proyecto, cámbialo
también en las líneas `#SBATCH --account` de `jobs/*.slrm`.

### Lanzar

```bash
bash run_pipeline.sh --dry-run     # comprueba todo y muestra los sbatch
bash run_pipeline.sh               # lo envía todo; te puedes ir
bash run_pipeline.sh --solo-datos  # sólo la fase de datos + digest, para revisar un
                                   # cambio del método antes de gastar GPU
```

**El corte de viajes** (`docs/HANDOFF_METODO_V2.md` §2 y §9): la proyección coherente, la
cobertura mínima según la leg, un orden cíclico que no tira los viajes válidos y el corte
por cruce de km de la vuelta (`TRIP_CUT=cruce`) van por defecto. Los interruptores están en `config.env` §7, con los valores que
reproducen la corrida publicada.

En una terminal SSH del login node, **nunca desde una celda `%%bash` de
Jupyter**: un Jupyter de OnDemand es un job de Slurm y hereda variables
`SLURM_*` que pisan las directivas `#SBATCH`.

Encola de una vez, encadenado por dependencias de SLURM:

| job | recurso | tareas | qué hace | → para |
|---|---|---|---|---|
| datos | CPU | 35 | pings crudos → `df_rec_full` por ruta | todo lo demás |
| xgb | CPU | 35 | frame de evaluación + baselines + XGBoost | Tablas 2–4, 7, S11–S13, S16, S17 |
| ablacion | CPU | 35 | XGBoost sin prior / solo trayectoria | Tabla 5 |
| escalera | CPU | 35 | ventana expansiva por mes | Tablas 6, S15 · Fig. 3 |
| lstm_es15 | GPU | 35 | LSTM, early stopping honesto (la leg `T0->T0` de cada ruta) | Tablas 2–5, 7, S11, S12, S16, S17 · Figs. S2, S3 |
| lstm_tw | GPU | 35 | LSTM sobre la escalera | Tablas 6, S15 · Fig. 3 |
| digest | CPU | 1 | funnel por ruta | Tablas 1, S2 |
| flota | CPU | 1 | concentración de flota | Tabla S2 |
| apendices | CPU | 1 | representatividad de la malla (con la vuelta; ε y reversión, sólo con `APPENDICES=A,C,D`) | Tabla S3 |
| paquete | CPU | 1 | todo lo anterior → un `.tgz` | comando 2 |


El encadenado es **por ruta** (`aftercorr`): la ruta 101 entra a la GPU en cuanto
tiene frames, sin esperar a las demás. Si una ruta falla, sus tareas
posteriores se cancelan en vez de salir en verde sin haber entrenado nada.

Coste: el de la corrida publicada está en la Tabla 7 del artículo
(`docs/paper/tables/table_12_cost_accuracy.md`; el LSTM `es15`, 62.65 h-GPU) y en los
logs del bundle. La fase de datos es la más larga en tiempo de reloj.

```bash
squeue -u "$USER" -o '%.10i %.8a %.14j %.8T %.10M %.20E'    # E = dependencia pendiente
```

### Si algo falla

Se vuelve a lanzar **el mismo comando** (`--skip-data` si los datos ya salieron):
todo es reanudable y lo hecho se salta. La excepción es un parquet **corrupto**:
la reanudación lo da por hecho para siempre. El job `paquete` los lista y sale
con rc=5; **bórralos** antes de relanzar. Una ruta que murió por OOM en la fase
de datos: sube `MEM_DATA` en `config.env`. Si el `RUNINFO` marca `FUNNEL ⚠`
(rc=4), esa ruta murió entre el `df_final` (paso 4) y el funnel (paso 7): relanza
solo su job de datos, `sbatch --array=<idx> jobs/job_data_pipeline.slrm` (reusa
las cachés); al terminar, rehaz el digest,
`sbatch jobs/job_report_pack.slrm --mode digest`, y empaqueta de nuevo,
`sbatch jobs/job_pack.slrm`.

### Al terminar

```bash
sacct -j <id del paquete> --format=JobID,JobName,State,ExitCode
tail -30 logs/pack_<id>.out
```

Ese log dice dónde quedó el `.tgz` y trae los comandos para bajarlo. Su
`RUNINFO_q05c100.txt` (dentro del `.tgz`) cuenta los artefactos y registra la
integridad, la frescura, el intérprete y el `config.env` de la corrida.

---

## 2 · Local: `python reproduce_tables.py`

Python 3.11 con `pip install -r requirements-analysis.txt`.

```bash
# desde una corrida del comando 1
scp servidor:~/repro/data/03_processed/run_bundle_q05c100_<fecha>.tgz .
mkdir -p artefacts/run_bundle_q05c100_<fecha>
tar -xzf run_bundle_q05c100_<fecha>.tgz -C artefacts/run_bundle_q05c100_<fecha>
python reproduce_tables.py --run artefacts/run_bundle_q05c100_<fecha>

# un piloto (sólo algunas rutas): la tabla del funnel lo dice al pie en vez de parar
python reproduce_tables.py --run artefacts/run_bundle_q05c100_<fecha> --partial

# desde la corrida publicada (PUBLISHED en reproduce_tables.py)
python reproduce_tables.py
```

Reescribe las tablas y figuras de `docs/paper/` (las copias versionadas, que están
en git) y las compara con git: Markdown idéntico byte a byte, CSV con las mismas
filas, PNG idéntico píxel a píxel.

**Todo sale de UN paquete.** La referencia es la corrida completa del 8–9 oct 2026:
`PUBLISHED` =
`artefacts/run_bundle_q05c100_20261009_171428`, verificada con `artefacts/MANIFEST.sha256`.
Sin `--run`, el comando la regenera y todo debe salir idéntico. ⚠ Con `--run` reescribe
`docs/paper/` en git: commitea antes los cambios de los generadores, o corre en una copia
del repo.

Con `--run` y otra corrida, las tablas y figuras que llevan el LSTM **cambian en sus
últimas cifras**: no es bit-reproducible en GPU (con la vuelta, hasta ±1 % de MAE por
ruta; la 601 `es15`, 241.0 → 243.4 s entre el re-piloto y la corrida completa). Para
separar eso de una diferencia real, el comando compara después la corrida ruta a ruta
con la publicada, una vez por variante (`q05c100`, `es15`, ablación, escalera). Los
nueve modelos deterministas deben salir bit-idénticos (en agosto, una re-corrida
reprodujo 375 de 378 pares; con la vuelta, los cuatro pilotos salieron idénticos). Un Δ
apreciable fuera del LSTM no es ruido: es régimen o datos distintos.

Para volver a las salidas versionadas: `git checkout -- docs/paper`.

---

## Lo que hay que saber antes de tocar nada

Cada punto costó al menos una corrida.

- **El intérprete, siempre por ruta absoluta.** Nunca `python` a secas ni
  `source .venv/bin/activate` antes de un `sbatch`: en los nodos el entorno cae
  al intérprete equivocado, el LSTM entrena en CPU (~29 min/época) y el job
  termina en verde. Los `.slrm` llaman a `$PY` y hacen `unset VIRTUAL_ENV`.
- **El LSTM no es bit-reproducible.** No apoyes ninguna afirmación en el puesto
  del ranking; lo robusto son los tests pareados y los ratios.
- **Un parquet corrupto es peor que uno ausente.** Pasa por presente y la
  reanudación lo salta. Por eso `func/parquet_guard.py` y el chequeo del job
  `paquete`.
- **La etiqueta lleva el régimen completo** (`q05c100`, no `q05`). Si cambias `Q`
  o `COV` en `config.env` y dejas el `TAG` viejo, la reanudación da por hechos
  artefactos de otro régimen.
- **El caché no se invalida por cambios de código**, solo por parámetros. Si
  tocas el código, corre en una carpeta nueva: `FORCE=1` solo rehace la fase de
  datos, y los modelos seguirían saltando sus resultados viejos como `cached`.
- **CRLF.** `sbatch` rechaza scripts con `\r\n`. `.gitattributes` fuerza LF y
  `git archive` lo respeta; si la carpeta viaja por un zip hecho en Windows,
  `run_pipeline.sh` lo detecta y te da el comando que lo arregla.

---

## Estructura

```
config.env              ★ el único archivo a editar
run_pipeline.sh         ★ comando 1 (servidor)
reproduce_tables.py     ★ comando 2 (local)
setup_venv.sh             crea ./.venv en el servidor (deps + TensorFlow-ROCm)
ENVIRONMENT.md            la restricción de versión de Python, y qué hacer
requirements-server.txt   dependencias del servidor (sin TensorFlow)
requirements-analysis.txt dependencias del comando 2
jobs/                     un .slrm por job de la tabla de arriba
func/                     el código del pipeline
docs/paper/analysis/      generadores de tablas, análisis del servidor y sus CSV
docs/paper/figures/       generador de la Fig. 1 y las figuras publicadas
docs/paper/tables/        las tablas publicadas
artefacts/                paquetes derivados (se descargan aparte; no en git)
data/01_raw/              los pings crudos (se descargan aparte; no en git)
```

`docs/paper/analysis/` conserva esa ruta a propósito: sus scripts resuelven la
raíz del proyecto contando directorios hacia arriba (`parents[3]`).
