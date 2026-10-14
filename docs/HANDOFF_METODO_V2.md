# HANDOFF — método v2 del corte de viajes · 2026-10-02 (rev. 2026-10-06)

> **Note for readers of the deposit (English).** These are the authors' working notes,
> kept as written because the pre-registration, `config.env` and the code cite them.
> The trip cut of the published run is described in §2 (the data-stage fixes) and §9
> (the round trip, cut by km crossing). §3–§8 record tests, server runs and
> comparisons made before the cut was fixed; nothing in them is needed to reproduce
> the article, and the table numbers they mention are those of earlier drafts.

> **Estado:** rama `metodo-v2` (el repo no tiene remoto; v2 commiteado el 6-oct). Tres
> arreglos a la fase de datos (proyección coherente, cobertura según la leg, orden cíclico)
> y una opción para fijar las terminales por ruta, cada uno con su interruptor en
> `config.env` §7. Probado en local con marzo de 2024 (§3) y, **el 2–3 oct, en la red
> completa en el servidor** (27 rutas, enero–agosto, sólo la fase de datos, con el test
> `../test/hermosillo-bus-test`, mismo código de `func/`): con las terminales del explorador
> (`TERMINAL_RULE=file`) los viajes direccionales pasan de 299,176 a 537,597 y el veredicto
> de `calidad_corte` (ok / revisar / mal) de 2 / 6 / 19 a 5 / 14 / 8 (§7). Queda una cola de
> viajes que no se cortan en la terminal, tres rutas con una sola terminal y la 601 (§7.3).
> **6-oct: la vuelta** (§9): el corte por cruce de km con un punto por ruta (`TRIP_CUT=cruce`),
> más una proyección que cobra el retroceso. Corrido en el servidor con el test (35 rutas,
> enero–agosto): 32 `ok` / 3 `revisar` / 0 `mal`.
> **Siguiente:** decidir si la vuelta entra al artículo (re-correr modelos; §9).
> Ningún número publicado cambia mientras no se vuelva a correr el artículo.

---

## 1. En una frase

La corrida `tmin` publicada cortaba bien los viajes, pero después los filtraba con un
orden cíclico que no sirve en trazas cerradas (tiraba del 29 al 99.9 % de los viajes
válidos), y escogía las terminales por los sitios donde los buses descansan, que en 8
rutas son dos rincones del mismo complejo y en otras un relevo a media ruta. v2 corrige
el orden cíclico, trae de `thesis` la proyección coherente, ajusta la cobertura mínima a
la leg y deja fijar las terminales por ruta con las que escoge el explorador.

## 2. Qué cambia

| # | Pieza | Interruptor (v2 · publicado) | Dónde |
|---|---|---|---|
| 1 | Proyección coherente en el tiempo (de `thesis`, commit `1007faa`) | `COHERENT_PROJECTION` 1 · 0 | `_lib_terminal_discovery._project_many_coherent`, `_lib_new_trips._project_seq` |
| 2 | Terminales fijadas por ruta (opcional) | `TERMINAL_RULE` file · dwell | `load_terminal_coords`, `build_trips(terminal_coords=)` |
| 3 | Cobertura mínima según la leg | `COVERAGE_BY_LEG` 1 · 0 | `build_trips` |
| 4 | Orden cíclico por secuencia de terminales | `CYCLIC_ORDER` v2 · v1 | `mark_cyclic_order` |
| 5 | Corte por cruce de km; con un punto por ruta, la vuelta (6-oct, §9) | `TRIP_CUT` cruce · vacío | `_build_trips_cruce` (desde `build_trips`), `mark_cyclic_order`, `assign_terminal_legs`, `run_new_pipeline` |

Además: el funnel (`perf/funnel_*.json`) gana `trips_valid_cyclic` (el paso que no tenía
número en la Tabla S1), `method` (el marcador `nt.method_tag()`, que también va en la
clave de la caché de viajes), `cyclic_order` y `terminal_fix` (qué terminales y de dónde
salieron); `run_pipeline.sh --solo-datos` encola la fase de datos y el digest, nada más.

### 2.1 Proyección coherente

Ya estaba en `thesis/func/lib` desde el 7-sep y nunca llegó a este repo: la corrida `tmin`
proyectaba cada ping al punto más cercano de la traza, y en una traza de ida y vuelta las
dos ramas van por la misma calle. Medido en la 601 (thesis/docs/03 §4.3): 84 % de los
viajes con al menos un salto de rama, distancia declarada 1.79× la real, cobertura mediana
0.42 cuando era 0.25. Se porta tal cual (`_lib_terminal_discovery.py` queda idéntico al de
`thesis`).

### 2.2 Terminales: las publicadas, o fijadas por ruta

La corrida `tmin` toma como terminales las 1–2 zonas con más descansos largos (dwell ≥ 10
min, AUTO_min) y corta en cada visita a ellas. Dos problemas medidos sobre sus `stats.json`:

- **8 rutas con las dos terminales a 0.8–7 km** (0.02–0.12 del lazo: 201, 202, 302, 401,
  402, 1501, 1502, 1902). Son dos sitios de descanso del mismo complejo; sus «legs
  direccionales» miden 0.83–0.92 del lazo (Tabla S1, mediana de viaje), o sea, vueltas
  completas, el mismo caso que el estrato N=1 que el artículo excluye.
- **Un descanso a media ruta no es un extremo.** En marzo, la 201 toma como terminal un
  descanso del km 39.5; su vuelta en U está en el 47.9. Como el bus pasa dos veces por ese
  descanso (ida y vuelta por la misma calle), una terminal ahí tiene dos posiciones en el
  lazo, y las legs que la usan quedan mal definidas.

Se probaron, sobre las 27 rutas, cuatro reglas para escoger las dos terminales de servicio
(en el explorador, `hermosillo-bus-trips-explorer`, docs/HANDOFF.md §9): el par de
reversión que menos viajes atraviesan, el más parejo sobre el lazo, el par entre el que más
alternan los buses y el de más descansos. **Ninguna acierta en todas**: la 201 recorre tres
zonas de vuelta en U por ciclo, la 1901 tiene una vuelta que sólo dan algunos buses, la 101
cinco zonas. La que sí funciona es **medir**: el explorador corta los viajes de su muestra
de 41 días con cada par candidato y se queda con el que más ciclos completos de ida y
vuelta produce (el menor de los dos conteos). Hacerlo aquí costaría horas por ruta (ocho
meses por par), así que el pipeline puede **leer** ese resultado:

- `TERMINAL_RULE=dwell` (por defecto): las terminales de la corrida publicada.
- `TERMINAL_RULE=file TERMINALS_FILE=<…>/terminales_servicio.json`: las de ese JSON
  (`{ruta: [[lat, lon], …]}`), que escribe el lote del explorador (`2.5`, una entrada por
  ruta). El corte, la validación y las legs usan esas coordenadas; una ruta que no está en
  el archivo usa `dwell`. Las legs quedan iguales a las direcciones A y B del dashboard.

El funnel lleva `terminal_fix = {rule, lin_pos, gap_frac}`: con `gap_frac` < ~0.15 las dos
terminales son del mismo complejo.

### 2.3 Cobertura mínima según la leg

En modo dwell un viaje es válido si cubre ≥ 0.375 del lazo, que supone legs de medio lazo.
Con dos terminales a un hueco g < ½ la leg corta mide g y no llega nunca: con el mínimo
global no se notaba (los saltos de rama inflaban la cobertura), con la proyección
coherente se rechazan todas. v2 usa 0.75·g, la misma fracción (0.375 = 0.75·½).

### 2.4 Orden cíclico

`mark_cyclic_order` (v1) marca `cyclic_ok` el viaje que (a) cubre ≥ 0.4 del lazo en
**bins únicos** y (b) **alterna** `direction` con la leg anterior del bus en el día;
`direction` es el signo del avance sobre la polilínea. Medido sobre los cachés `tmin` de
las 27 rutas (`thesis/data/02_interim/routes/*/dataframes/trips_cache_2024_01-08_tmin`):

- **(b) no alterna:** en una traza cerrada las dos direcciones avanzan en el mismo sentido.
  Viajes seguidos del mismo bus con la misma «dirección»: 44–96 %, salvo 101, 102 y 601
  (16–27 %), donde alterna porque la proyección cae en la rama equivocada.
- **(a) penaliza las rutas cortas:** con un ping por minuto los bins únicos no se llenan
  (la razón por la que `build_trips` pasó a medir la cobertura por span). Viajes con
  cobertura ≥ 0.4: 902 0.1 % en bins únicos contra 91 % por span; 1602 2.5 % contra 98 %;
  1601 5.1 % contra 99 %.

| ruta | válidos (build_trips) | tras el orden cíclico | queda |
|---|---|---|---|
| 902 | 20,300 | 18 | 0.1 % |
| 1602 | 17,148 | 408 | 2.4 % |
| 1601 | 18,198 | 829 | 4.6 % |
| 501 | 18,002 | 1,145 | 6.4 % |
| 20000 | 4,965 | 338 | 6.8 % |
| 801 | 33,294 | 3,210 | 9.6 % |
| 201 | 7,535 | 2,830 | 37.6 % |
| 101 | 21,804 | 15,461 | 70.9 % |
| **27 rutas** | **458,309** | **147,665** | **32 %** (mediana 34.5 %) |

Esto es la mayor parte de la caída de «Valid trips» a «Directional» de la Tabla S1, y no
es lo que dicen sus notas (902 fuera, self-loops, legs con menos viajes que folds). La 902
no se pierde por no hacer layover —su funnel tiene 0 % de self-loops: el corte la partió
bien— sino aquí.

v2: una leg es **completa** si es válida para `build_trips` (cuya cobertura ya se mide
sobre la leg, §2.3, y con la proyección coherente) y sigue el **ciclo** si arranca en la
terminal donde terminó la leg válida anterior del mismo bus y día. No usa la posición de
las terminales sobre el lazo (un descanso sobre una calle de ida y vuelta tiene dos).
`CYCLIC_ORDER=v2-sin-ciclo` deja sólo la condición de completa, para medir cuánto quita
el ciclo.

## 3. Pruebas locales (marzo de 2024)

`func/run_new_pipeline.py --mode custom --start 2024-03-01 --end 2024-04-01 --terminals
AUTO_min --no-maps --no-dashboards --no-baselines --no-lstm`, con `BUSONORA_ROOT` en una
carpeta aparte por versión; «publicado» es `git archive main`. Un mes va en modo directo
(sin consenso mensual): sirve para ver el mecanismo, no da los números de la corrida.

Viajes en el `df_final` (válidos de `build_trips` → tras el orden cíclico; entre paréntesis
los direccionales y el % de self-loops):

| ruta | publicado (`main`) | v2, terminales publicadas | v2, terminales del explorador |
|---|---|---|---|
| 902 | 2,732 → **3** (3, 0 %) | 2,746 → **2,315** (2,306, 0.4 %) | — |
| 1601 | 2,478 → **100** (95, 5 %) | 2,481 → **2,424** (2,404, 0.8 %) | — |
| 201 | 1,432 → **173** (128, 26 %) | 2,213 → 1,897 (1,309, **31 %**) | 1,800 → **1,746** (1,707, 2.2 %) |
| 101 | 2,640 → 1,887 (1,866, 1.1 %) | 2,607 → **2,579** (2,522, 2.2 %) | — |

- Los tres arreglos solos recuperan casi todo lo que tiraba el orden cíclico (902, 1601).
- En la 201 de marzo las terminales de dwell son el km 19.5 y un **descanso a media ruta**
  (lin_pos 0.685): 580 viajes salen `T1->T1`. Con las del explorador (lin_pos 0.334 y
  0.831, sus dos vueltas en U extremas) quedan 2.2 % de self-loops.
- Un mes de 902 o 1601 tarda ~4 min; 101 y 1701 ~20–30 min en local (el *headway* del
  paso 7 domina).
- No se probó en local la 1701 (una sola terminal de dwell) ni ninguna ruta con
  `TERMINAL_RULE=file` salvo la 201: eso lo dirá la corrida de la red (→ §7).

## 4. Cómo correrlo en el servidor

> **Sólo la fase de datos** (lo que se hizo el 2–3 oct, §7): el paquete del test,
> `../validacion_corte.tgz`, que se descomprime en una carpeta nueva y corre todo con
> `bash run_servidor.sh` (venv, prueba, terminales del explorador, 27 rutas, dashboards,
> calidad del corte y un `.tgz` de resultados; README del test). Usa el mismo código de
> `func/` que este repo y no corre el enrich ni los modelos. Lo de abajo sigue siendo el
> camino para la **corrida completa** del artículo (datos + modelos).

Dos variantes de la fase de datos, cada una en su carpeta **nueva** (el manifest de la
caché de viajes invalida por método, pero los modelos no saben del cambio y saltarían sus
resultados viejos):

| carpeta | terminales | para qué |
|---|---|---|
| `~/repro_v2` | `dwell`, las publicadas | el efecto de los tres arreglos solos |
| `~/repro_v2_file` | `file`, las del explorador | además, las terminales medidas |

```bash
# en tu máquina: el código de la rama (git archive respeta el LF de .gitattributes)
git -C hermosillo-bus-arrival-paper archive --format=tar.gz --prefix=repro_v2/ -o repro_v2.tgz metodo-v2
scp repro_v2.tgz usuario@servidor:~/

# en el servidor
tar -xzf repro_v2.tgz && cp -r repro_v2 repro_v2_file
for d in repro_v2 repro_v2_file; do ln -s ~/repro/data/01_raw ~/$d/data/01_raw; done   # ~/repro: la carpeta de la corrida publicada
```

En las dos, `PY` de `config.env` tiene que apuntar al mismo intérprete de la corrida
publicada (la línea `PY=`). En `repro_v2_file/config.env`, §7:

```bash
export TERMINAL_RULE="${TERMINAL_RULE:-file}"
export TERMINALS_FILE="${TERMINALS_FILE:-/lustre/home/usuario/dashb/reports/lote_2024-01-01_2024-08-31/terminales_servicio.json}"
```

(ese archivo lo deja el lote del explorador con su versión v2: hay que correrlo antes,
`bash run_pipeline.sh --solo propios` en `/lustre/home/usuario/dashb`). **Ya existe uno de
la red completa** (3-oct, el que usó §7):
`/lustre/home/usuario/validacion_corte/reports/terminales_2024-01-01_2024-08-31/terminales_servicio.json`,
con copia local en `../test/hermosillo-bus-test/reports/resultados_test_2024-01-01_2024-08-31_v2-explorador_completo/reports/terminales_2024-01-01_2024-08-31/`;
sirve igual en `TERMINALS_FILE`. Luego, en cada carpeta:

```bash
bash run_pipeline.sh --dry-run
bash run_pipeline.sh --solo-datos                 # fase de datos (27 rutas) + digest
```

Al terminar, el comando imprime cómo empaquetar los funnels (`funnels_v2.tgz`): con ellos
se compara contra la Tabla S1. Con la variante que convenza:

```bash
bash run_pipeline.sh --skip-data                  # modelos, auxiliares y paquete
```

Antes de eso, revisar §5: los estratos y los arrays de rutas están escritos a mano.

## 5. Lo que hay que rehacer después de la corrida

- **Estratos** (`validated` / `group B` / `N=1`): están escritos a mano en
  `docs/paper/analysis/audit_run.py` y `grid_representativity.py`, y el orden de rutas en
  los `ROUTES=(…)` de `jobs/*.slrm`. Con v2, rutas N=1 (301, 901, 1201, 1701) pueden pasar a
  dos terminales: entonces tienen legs direccionales y `GPU_ARRAY` debería ser `0-25%4`.
  Medido en la red (§7): con las terminales del explorador 301, 1201 y 1701 pasan a dos y
  quedan con una 501, 901 y 1002; con las de dwell, las mismas cuatro de siempre.
- **Números de ruta** (§8): siete rutas tienen otro número en el catálogo actual de UNE
  (101↔102, 1001↔1002, 1601↔1602, 20000 = 2201). Decidir con cuál se nombran.
- **Apéndices C y D**: `terminal_eps_sensitivity.py` y `reversal_params_sensitivity.py`
  reproducen la lógica vieja de `_terminal_like_cut_events` (conteo por dwell); con v2 el
  conteo sale de la reversión y el apéndice C deja de medir lo que decide las legs.
- **Tabla S1 y texto**: la nota que atribuye la caída valid→modeled, la razón de excluir la
  902 y la definición de terminal (§3.4 del artículo).
- **README §6 «Known limitations»**: la 902.

## 6. Reproducir la corrida publicada

```bash
COHERENT_PROJECTION=0 CYCLIC_ORDER=v1 COVERAGE_BY_LEG=0 TERMINAL_RULE=dwell bash run_pipeline.sh …
```

Comprobado en local con la 902 de marzo: con esos valores el código de `metodo-v2` da el
mismo `df_final` (156 filas) y el mismo caché de viajes (94,899 filas) que `git archive
main`, idénticos (`polars.DataFrame.equals`).

## 7. La fase de datos v2 en la red (servidor, 2–3 oct)

**Cómo.** Con el test (`../test/hermosillo-bus-test`, su README): los pasos 1, 3 y 4 de
`run_new_pipeline.py` por ruta, con el mismo código de `func/` que este repo (el `df_final`
de la 902 de marzo sale idéntico al de `run_new_pipeline.py`), en la ventana del artículo
(`2024_01-08_tmin`, `MODE=full`), un job por ruta (8 CPU, 160 GB); después los dashboards del
explorador en modo `--df-final` y `calidad_corte.py`. No corre el enrich ni los modelos.
Cada corrida en su carpeta:

| corrida | terminales | carpeta en el servidor | resultados aquí |
|---|---|---|---|
| v2-dwell (2-oct) | `TERMINAL_RULE=dwell`, las publicadas | `/lustre/home/usuario/db_test_df` | `../test/hermosillo-bus-test/reports/resultados_test_2024-01-01_2024-08-31_v2-dwell_completo/` |
| v2-explorador (2–3 oct) | `TERMINAL_RULE=file`, las que mide el lote de viajes propios del explorador en un job previo (136 s) | `/lustre/home/usuario/validacion_corte` | `…_v2-explorador_completo/` |

En las dos: 27/27 rutas en la fase de datos (1–24 min cada una) y 27/27 dashboards;
`terminal_fix.rule` es `dwell` en las 27 de la primera y `file` en las 27 de la segunda. Cada
carpeta de resultados trae los funnels (`data/02_interim/routes/<ruta>/perf/funnel_2024_01-08_tmin.json`)
y los `stats.json` de la caché de viajes, para comparar con la Tabla S1.

### 7.1 Totales

| | v2 + dwell | v2 + explorador |
|---|---|---|
| válidos (`build_trips`) | 486,781 | 640,179 |
| tras el orden cíclico v2 | 392,538 | 600,735 |
| direccionales | 299,176 | **537,597** |
| self-loops | 93,362 | 63,138 |
| rutas con dos terminales (hueco entre ellas, fracción del lazo) | 23 (0.02–0.49) | 24 (0.37–0.50) |
| `calidad_corte`: ok / revisar / mal | 2 / 6 / 19 | **5 / 14 / 8** |

`calidad_corte.py` (README del test) mide si A y B salen como arcos complementarios del
lazo: km que recorren las dos direcciones en el dashboard, % de viajes que pasan la salida de
la otra dirección por > 0.5 km, > 2 km o > media vuelta, % que se quedan a > 2 km, y los
self-loops del funnel. `mal`: > 10 % rebasa por > 2 km, > 20 % de self-loops, más de una
vuelta con > 5 % de los viajes pasando media vuelta, o una sola dirección con dos
terminales. `revisar`: > 1 km compartido o sin cubrir, > 15 % rebasa por 0.5 km o se queda
corto, o direcciones disparejas (< 0.8).

### 7.2 Por ruta

| ruta | hueco entre terminales (dwell → explorador) | veredicto | direccionales | self-loops % | km compartidos | qué queda con el explorador |
|---|---|---|---|---|---|---|
| 101 | 0.48 → 0.48 | revisar → **revisar** | 20,735 → 20,314 | 2.25 → 2.35 | 3.41 → 3.41 | 3.41 km compartidos |
| 102 | 0.49 → 0.50 | revisar → **revisar** | 28,082 → 27,991 | 0.98 → 0.99 | 16.47 → 16.75 | B da más de una vuelta (2.1 % de sus viajes pasa media vuelta); 16.75 km compartidos |
| 201 | 0.09 → 0.50 | mal → **mal** | 3,083 → 13,111 | 7.31 → 3.45 | 51.79 → 5.53 | B: 17.9 % rebasa la otra terminal por > 2 km; 5.53 km compartidos |
| 202 | 0.12 → 0.50 | mal → **revisar** | 2,720 → 16,748 | 20.05 → 3.13 | 49.64 → 25.86 | 25.86 km compartidos |
| 301 | una → 0.38 | mal → **revisar** | 0 → 23,251 | 100 → 3.95 | 0 → 7.15 | 7.15 km compartidos |
| 302 | 0.02 → 0.39 | mal → **mal** | 4,593 → 22,056 | 33.63 → 4.15 | 46.07 → 32.63 | A: 10.8 % rebasa por > 2 km; 32.63 km compartidos |
| 401 | 0.06 → 0.41 | mal → **revisar** | 6,545 → 34,018 | 18.7 → 4.91 | 46.02 → 1.1 | 1.1 km compartidos |
| 402 | 0.05 → 0.49 | mal → **ok** | 3,588 → 34,703 | 38.65 → 2.84 | 46.07 → 0.11 | — |
| 501 | 0.34 → una | mal → **mal** | 12,665 → 0 | 28.94 → 100 | 27.36 → 0 | una sola terminal: 100 % self-loops |
| 601 | 0.46 → 0.47 | revisar → **revisar** | 13,125 → 13,798 | 1.53 → 1.52 | 0 → 0.55 | B: 84.7 % se queda a > 2 km (§7.3) |
| 801 | 0.22 → 0.46 | mal → **ok** | 6,855 → 18,710 | 50.17 → 1.64 | 28.29 → 0.32 | — |
| 901 | una → una | mal → **mal** | 0 → 0 | 100 → 100 | 0 → 0 | una sola terminal: 100 % self-loops |
| 902 | 0.46 → 0.49 | revisar → **ok** | 17,260 → 22,168 | 0.23 → 0.47 | 1.57 → 0.55 | — |
| 1001 | 0.21 → 0.44 | mal → **revisar** | 4,765 → 16,585 | 5.16 → 4.95 | 43.16 → 18.02 | 18.02 km compartidos |
| 1002 | 0.19 → una | mal → **mal** | 3,129 → 0 | 25.54 → 100 | 35.05 → 0 | una sola terminal: 100 % self-loops |
| 1201 | una → 0.37 | mal → **mal** | 0 → 8,360 | 100 → 2.84 | 0 → 13.11 | A: 27.9 % y B: 14.1 % rebasan por > 2 km; 13.11 km compartidos |
| 1301 | 0.28 → 0.47 | mal → **revisar** | 14,863 → 13,913 | 29.5 → 1.59 | 26.48 → 2.51 | 2.51 km compartidos |
| 1501 | 0.04 → 0.39 | mal → **revisar** | 3,287 → 23,616 | 29.49 → 5.02 | 44.05 → 27.8 | A da más de una vuelta (2.8 % pasa media vuelta); 27.8 km compartidos |
| 1502 | 0.03 → 0.48 | mal → **revisar** | 8,064 → 32,486 | 49.69 → 0.4 | 24.75 → 4.14 | 4.14 km compartidos |
| 1601 | 0.48 → 0.47 | ok → **ok** | 17,656 → 17,806 | 0.75 → 0.75 | 0.47 → 0.47 | — |
| 1602 | 0.48 → 0.48 | ok → **ok** | 16,554 → 16,539 | 0.8 → 0.8 | 0 → 0 | — |
| 1701 | una → 0.49 | mal → **mal** | 0 → 28,643 | 100 → 1.93 | 0 → 20.2 | A: 12.0 % rebasa por > 2 km; 20.2 km compartidos |
| 1702 | 0.35 → 0.41 | revisar → **revisar** | 32,894 → 32,965 | 1.89 → 2.15 | 6.5 → 4.4 | 4.4 km compartidos |
| 1703 | 0.31 → 0.46 | revisar → **revisar** | 49,705 → 51,776 | 2.46 → 2.05 | 12.23 → 4.03 | 4.03 km compartidos |
| 1901 | 0.41 → 0.44 | mal → **revisar** | 16,194 → 22,344 | 4.31 → 3.47 | 9.9 → 1.8 | 1.8 km compartidos |
| 1902 | 0.03 → 0.45 | mal → **mal** | 5,167 → 16,893 | 35.59 → 3.74 | 48.11 → 7.54 | B: 10.3 % rebasa por > 2 km; 7.54 km compartidos |
| 20000 | 0.34 → 0.50 | mal → **revisar** | 7,647 → 8,803 | 0.96 → 1.11 | 6.87 → 2.24 | 2.24 km compartidos |

Lectura: con dwell, las 8 rutas de terminales pegadas (0.02–0.12) dan vueltas completas; en
otras 8 una terminal queda lejos del extremo (0.19–0.35; en 501, 801, 1002 y 1301 con 26–50 %
de self-loops); 301, 901, 1201 y 1701 tienen una sola. Con las del explorador, 402, 801 y 902
pasan a `ok`, las de terminales pegadas multiplican sus direccionales ×3–10 y 301, 1201 y
1701 ganan dos direcciones; 501 y 1002 quedan con una.

### 7.3 Lo que queda

1. **Viajes que no se cortan en la terminal.** `_terminal_like_cut_events` corta en cada
   visita, y una visita es un tramo de registros a < `visit_max` = 2 × `TERMINAL_NEAR_M` =
   600 m de una terminal: si en esa pasada no hay registro a < 600 m (un hueco del GPS), el
   viaje sigue de largo hasta la siguiente. No hay cap de longitud (el explorador sí:
   `cap_longitud` parte los viajes de más de L·(1+margen) en su acercamiento más cercano a
   una terminal, su §4.21). Son 2–10 % de los viajes en casi todas las `revisar` y 10–28 % en
   201, 302, 1201, 1701 y 1902. El dashboard llega al percentil 99, así que 1–3 % alcanza
   para que una dirección se meta kilómetros en la otra (202: 26 km compartidos con 2.7 % de
   viajes; 102: 2.1 % pasa media vuelta).
2. **Visitas que no son la terminal.** El mismo radio alcanza otros pasos de la traza
   (`../test/hermosillo-bus-test/reports/terminales_dwell_vs_explorador.csv`, columnas
   `pasos_traza_600m` y `km_de_los_pasos_600m`). La T1 de la 601 está en el km 33.9 y la traza
   pasa a < 600 m también en los km 28.0–30.3: el 85 % de la B termina > 2 km antes de la
   salida de la A, igual con dwell, y lo más probable es que el corte dispare en ese paso.
   Pasos extra también en 1201 (T0: 4, T1: 2), 1301 y 1501.
3. **Tres rutas con una sola terminal del explorador** (501, 901, 1002; el explorador deja
   una si una dirección no llega a ¼ de la otra), y la traza pasa por ella dos o tres veces
   por vuelta (501: km 10–11 y 24–26; 901: 13–14 y 31–33; 1002: 0–1, 8–9 y 21–22): sus
   self-loops podrían ser las dos mitades de una ruta de ida y vuelta. Con dwell, 501 y 1002
   tenían dos terminales (malas: 29 % y 26 % de self-loops).
4. Los registros caen sobre la traza en las 27 (distancia perpendicular mediana 2.2–3.1 m):
   ninguna ruta con la geometría equivocada.

Candidatos (nada implementado; esperan la revisión de §8): partir los viajes más largos que
su leg aunque no haya visita (como `cap_longitud`); radio de visita menor, o exigir la llegada
a la terminal sobre la traza; terminales a mano para 501, 901 y 1002 si son de ida y vuelta
(`TERMINALS_FILE`); en el dashboard, percentil 95 en vez de 99 (sólo visual). Cada uno con su
interruptor, probado en local (202, 201, 1201, 601) y después `validacion_corte.tgz` otra vez,
en una carpeta nueva, para comparar `calidad_corte.csv` contra las dos corridas.

## 8. La geometría contra el catálogo actual de UNE (3-oct)

La geometría de `data/01_raw/` (`polylines.geojson`, `segments*`, `stops*`; fechada
13-jul-2026, byte a byte la misma en `thesis`, este repo, el explorador y el test) contra el
catálogo del portal de UNE del 11-sep-2026
(`../archive/tesis-referencia/thesis/data/01_raw_public/UNE_lineas.json`: 49 líneas con su
polilínea y su recorrido; `UNE_paradas.json`; ningún pipeline los usa). Por ruta, en
`../test/hermosillo-bus-test/reports/trazas_vs_lineas_une.csv`
(`../test/hermosillo-bus-test/analisis/trazas_vs_lineas_une.py`):

- 17 trazas idénticas (0.0 m) a la línea de UNE con el mismo `group_id`, con las mismas
  paradas (todas a < 15 m).
- 301, 302 y 601 cambiaron un poco: 5–11 m de distancia media, la de UNE 0.2–0.6 km más
  corta, 3–5 paradas distintas.
- Siete con otro número en el catálogo, idénticas en traza y paradas: nuestra 101 es la 102
  de UNE («Línea 1 Manga») y nuestra 102 la 101 («Línea 1 Sahuaro»); nuestra 1001 es la 1002
  («Línea 10 Caturegli») y nuestra 1002 la 1001 («Línea 10 VH»); nuestra 1601 es la 1602
  («Línea 16 Monteverde») y nuestra 1602 la 1601 («Línea 16 Reforma»); la 20000 es la 2201
  («Línea 20 Montalban», con 2 paradas más en UNE).
- Los crudos traen `group_id` igual a nuestro número (el archivo de la 101, `group_id` 101) y
  caen sobre nuestra traza a 2–3 m: el análisis es consistente consigo mismo; lo que difiere
  es el número contra el catálogo actual. Importa al nombrar las rutas en el artículo.

## 9. La vuelta: corte por cruce de km (6-oct, `da4a02d`)

Lo decidió el autor tras el comparativo de métodos (`../Comparativo_metodos_viajes.md` §8 y
§10, en la carpeta del proyecto): **la vuelta es la unidad de viaje en todas las rutas**, y el
corte es por cruce de km sobre el progreso continuo, no en el registro más cercano a la
terminal.

- **Interruptor**: `TRIP_CUT=cruce`, con `TERMINAL_RULE=file`. `TERMINALS_FILE` trae los
  puntos de corte de cada ruta como `[lat, lon, km]`:
  - con un punto, la leg es `T0->T0` y es la vuelta;
  - con dos, las legs A/B por cruce.

  Los puntos de la vuelta están en `../test/hermosillo-bus-test/cortes_vuelta.json` (35 rutas;
  `analisis/cortes_vuelta.py`). Sin `TRIP_CUT`, nada cambia: la 902 de marzo da el mismo
  `df_final` que antes.
- **`_build_trips_cruce`**: el viaje empieza en el primer registro después de que M(t) pase
  por el km de un corte.
  - **Tramos de servicio**: por bus, se parten al cambiar de día o con > `GAP_SPLIT_MIN` sin
    registros. Cada uno lleva su proyección coherente y su máximo acumulado M(t).
  - **Saltos**: un avance > 25 m/s + 300 m parte el tramo, salvo un **atajo**: hacia
    adelante, de ≤ 8 km, y los 3 registros siguientes avanzan normal. El atajo no parte el
    tramo y queda marcado (columna `atajo`; `atajo_m` en el `frag`).
  - **Válido**: completo (entre dos cruces del mismo tramo), densidad, sin cola a depósito, y
    un recorrido real de ≤ 1.5 veces su arco. Si recorrió más, la proyección se saltó un
    cruce: es una vuelta doble (`cruce_perdido`).
  - **Columnas nuevas del `df_final`**: `prog_m`, `atajo`, `cut_from` y `cut_to`.
  - **El funnel lleva `cruce`**: cortes, tramos y por qué se partieron, atajos, cobertura,
    vueltas, cruces inferidos, cruces perdidos, duración y vueltas raras.
- **`mark_cyclic_order`** no juzga el ciclo (lo da la construcción), y
  **`assign_terminal_legs`** toma la leg de los cortes cruzados.
- **`run_new_pipeline`**, con `TRIP_CUT=cruce`, no omite `T0->T0`: la vuelta es la unidad y no
  hay residuo. Probado en local con la 902 de marzo, sin modelos (con la `.venv` del test,
  porque la de este repo no trae `pandas`): enrich completo, `df_full` de 81,449 filas y la
  leg `T0->T0`. El paso 8b falló porque esa `.venv` no trae `h3`.
- **En el test**: `TERMINALES=vuelta` es el valor por defecto. Ahí se corrió marzo con las 35
  rutas (`../test/hermosillo-bus-test/README.md`, «La vuelta»).
- **En el servidor** (6-oct, enero–agosto, 35 rutas, con el test): 32 `ok`, 3 `revisar`,
  0 `mal`; 313,580 vueltas. Re-corrido con los arreglos de abajo: 31 / 4 / 0 (abajo).
- **Arreglado esa noche**: la proyección coherente tiene `direccional=w`
  (`_lib_terminal_discovery._project_many_coherent`). Retroceder d sobre la traza cuesta además
  w·d; el corte por cruce usa w = 1 (`CRUCE_DIRECCIONAL`).
  - Sin eso, en las rutas de ida y vuelta (20000, 902) el bus caía media vuelta en la otra
    pasada, al revés, y se perdía un cruce.
  - Marzo: vueltas dobles de 96 a 12.
  - Por defecto w = 0: lo demás no cambia.
- **Re-corrido en el servidor con los arreglos** (6-oct, 17:30; el `validacion_corte.tgz` de
  las 15:06, con el test; detalle en `../test/hermosillo-bus-test/README.md`, «Resultados (6-oct
  17:30…)», y en el handoff del proyecto §4.14):
  - 31 `ok`, 4 `revisar`, 0 `mal`. Las cuatro son por atajos: 101, 302 y 601, como antes, y
    la 901 por 0.05 puntos (10.05 %);
  - 317,968 vueltas; cobertura mediana 75.2 %, y ninguna ruta pierde;
  - vueltas dobles de 742 a 119 (la 20000, de 275 a 4);
  - la duración, a −0.8 % de la de UNE (mediana).

  Falta el `sacct` de memoria.
- **Decidido (6-oct, noche): entra al artículo**, que está sin enviar. Se trabaja en la rama
  `vuelta`, que sale de esta.
  - **El plan**: `../Plan_nueva_corrida_articulo.md`, con las decisiones P1–P8: 35 rutas, los
    puntos de `cortes_vuelta.json`, la malla sobre la traza, sólo `es15` y `tw`.
  - **El pre-registro**: `docs/PREREGISTRO_VUELTA.md`, con datos, método, criterio de inclusión,
    paneles e inferencia, fijados antes de correr modelos.
  - Con la vuelta, los grupos de predicción pasan de las legs a uno por ruta.
