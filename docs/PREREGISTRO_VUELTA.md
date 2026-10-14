# Pre-registro — la corrida del artículo con la vuelta (6-oct-2026)

Lo que sigue se fija **antes de correr ningún modelo con la vuelta**: datos, método, criterio de
inclusión, paneles e inferencia. Sustituye, para esta corrida, al criterio de agosto
(`docs/paper/analysis/leg_inclusion.py`, cuyo C1 «direccional» ya no aplica: con la vuelta
todas las legs son `T0->T0`).

**Qué cuenta como prueba de que se fijó antes.** La fecha del commit de este archivo en la rama
`vuelta`.

**Cómo se cambia.** Después de ver métricas de modelos, nada de §1–§5 se cambia por lo que
digan los resultados.
- **Lo que sí se puede**: los arreglos técnicos (un error del código, un job que muere, la
  memoria o el tiempo pedidos). Van en §6 con fecha, motivo y qué tocan.
- **Lo que se excluye no se borra**: va a una tabla con su motivo, como en agosto.

**De dónde sale.**
- Las decisiones P1–P8 están en el plan de la carpeta del proyecto
  (`../Plan_nueva_corrida_articulo.md` §2, fuera de este repo).
- La validación de la vuelta, en el handoff del proyecto (§4.14) y en el §9 de
  `HANDOFF_METODO_V2.md`.
- La evidencia de P2 y P3, en `../test/hermosillo-bus-test/reports/plan_articulo_2026-10-06/`.

## 1. Datos y método

**Ventana**: del 1 de enero al 31 de agosto de 2024 (`MODE=full`, `RANGE=2024_01-08_tmin`) (P7).

**Rutas: 35** (P1):
- **las 26 del estudio y la 902.** Su crudo es el público, byte a byte el de `data/01_raw/records/`
  de este repo al 6-oct;
- **las 8 de las familias con variante**: 1101 y 110101 (1101-ALT); 1401 y 1402 (AVL 1400); 1801
  y 180101 (1801-ALT); 1802 y 180201 (1802-ALT). Su crudo es el de la BD de 2024, separado bus por
  bus con `variantes_ruta.py` (`../Variantes_rutas_alternas.md`). Los archivos de entrada:

  | ruta | `df_rec_raw_<ruta>_2024_01-12.parquet`, sha256 |
  |---|---|
  | 1101 | `afaf8e18e96b77f9f0787aca1bfbf114193caf96caec150eba6d54de314463b5` |
  | 110101 | `3475e2d4d04b97c1c29378360d01b8767f76673e8fa1787b180ce7450a74a7b4` |
  | 1401 | `1cebc08144c7372cb63e310a8ed4e2e6165a8200af7a0db71f3929b589d7a45b` |
  | 1402 | `637f5ff985b6abf227274bab6064c952973174ace1c47dfad4f88f84d08473f4` |
  | 1801 | `0757412811160e6498cc7cfae7bd8e77430bbd2cce46b8f986388c3acce38ce6` |
  | 180101 | `db77a0502730b00033009ee42e103fa1bdcd5fd666b7113b1ebed8453f892a0f` |
  | 1802 | `8ed0a14f08948d9adc8998503a858e4dba957bf44a94a9862c301d0a8a64e44d` |
  | 180201 | `d03ece3887ad74b2a04b431a67d37d67ea7b9c5db3fba3a00f979cd95e0e6c58` |

**Geometría**: la corregida, `../data raw/new-01_raw/`, con la numeración
del AVL:

| archivo | sha256 |
|---|---|
| `polylines.geojson` | `b0eae7c62429b17cffe72cb154eda8895d666e1ac06d7d5854c39832e8986629` |
| `segments.geojson` | `2c57ba2341920ee36417e168f2ba0ebe7424c7b6e9b28f23a2c500c9d7732b56` |
| `segments_route.json` | `578440b7ad99d2ea9a0240c67a81d130938b846d84dad1610942763ed0702899` |
| `stops.json` | `af8517b47f860a9840fdc9fadaef95e2f503513b313187a11e35d05fbb936c03` |
| `stops_routes.json` | `166d98457bbca7377029d61cc45cc7ecd867b099e51428447dd6987f860b19f1` |

**El corte** (P2):
- **interruptores**: `TRIP_CUT=cruce`, `TERMINAL_RULE=file`, `COHERENT_PROJECTION=1`,
  `CYCLIC_ORDER=v2`, `COVERAGE_BY_LEG=1`, `CRUCE_DIRECCIONAL=1`;
- **los parámetros de `func/lib/_lib_new_trips.py` al 6-oct**: `_cutcruce_v25+300_atajo8000x3_rec1.5_dir1`
  en `method_tag()`, más `GAP_SPLIT_MIN=20`, `MIN_TRIP_PTS=5`, `MIN_TRIP_DENSITY=0.15` y
  `CRUCE_INFERIDO_S=300`;
- **los puntos de corte**: un punto por ruta, `cortes_vuelta.json` (sha256
  `d5ca2c92c6e37559df7470dc37b7cb31a0ec3f3d58f9365a927c90432bf835db`; el de
  `../test/hermosillo-bus-test/`, validado dos veces en el servidor). No se cambian;
- **la duración que declara UNE**: `vueltas_une.csv` (sha256
  `f44ae1abe0e4481268d89c46d68290d131f2a280a67c94362852be1dd40d1f8f`).

## 2. Unidad, malla y régimen

- **La unidad es la vuelta**: la leg `T0->T0` de cada ruta, de un cruce del punto de corte al
  siguiente del mismo tramo de servicio.
- **La malla va sobre el progreso en la traza** (P3): la distancia de cada registro es su avance
  sobre la traza desde el corte (`prog_m` menos el del primer registro de la vuelta), no la
  suma de líneas rectas entre registros.
- **El régimen de agosto, sin cambios** (`q05c100`):
  - paso de 100 m, malla hasta el P05 de la longitud (`n_route_quantile = 0.05`);
  - `min_coverage = 1.0`;
  - `max_trips = 2500`, submuestreados una vez con semilla 42 e iguales para todos los
    algoritmos;
  - 5 folds con semilla 42;
  - las posiciones de corte de `eval_frame.make_l_indices` (hasta 30);
  - la escalera mensual de `make_folds_expanding`.

## 3. Criterio de inclusión (P4)

**Por ruta.** Una ruta entra si cumple las tres:
- **C1 · el corte no sale mal**. Se mide en la fase de datos de esta corrida, con su funnel
  (`cruce`). Sale mal si cumple cualquiera de éstas (los umbrales de `mal` de `calidad_corte.py`
  del test):
  - cobertura < 50 %;
  - > 10 % de cruces inferidos;
  - > 5 % de vueltas raras (duración fuera de 0.6–1.6 veces la mediana).
- **C2 · M ≥ 100** vueltas en el frame.
- **C3 · computable**: M ≥ 5 (los folds).

Las rutas que en la calidad del corte saldrían `revisar` entran, y van marcadas en la tabla.

**Por vuelta.** Entra al frame si cumple las tres:
- **es válida**: completa entre dos cruces, ≥ 5 registros, densidad ≥ 0.15, sin cola a depósito
  y sin cruce perdido (recorrido ≤ 1.5 veces la traza);
- **no tiene atajo**: ningún registro marcado `atajo`, es decir, un salto hacia adelante de
  ≤ 8 km que no parte el tramo. Sus celdas saltadas no tienen tiempo observado. Estas vueltas
  se quedan en las tablas descriptivas;
- **cubre toda la malla** (`min_coverage = 1.0`).

**No hay filtro por duración.** Las vueltas raras se quedan: filtrar por duración sería filtrar
por lo que se predice.

**Lo esperado.** En la validación del 6-oct (mismo código y crudo) ninguna ruta sale `mal`, y la
que menos vueltas tiene suma 4,406, así que se espera que **entren las 35**. Si la fase de datos
de esta corrida difiere de la validación, se reporta la diferencia.

## 4. Paneles

- **Principal**: todas las rutas incluidas.
- **La escalera temporal, panel constante**: las rutas con ≥ 100 vueltas de entrenamiento en
  **todos** los escalones (`min_n_train ≥ 100`, como en agosto). El de sensibilidad: todas las
  incluidas.
- **Las variantes, la 902 y las rutas `revisar`** van en el panel principal, sin estrato aparte.

## 5. Modelos e inferencia

- **Los modelos**: los diez predictores de agosto, con los mismos ajustes (Tabla S5) y las mismas
  métricas: MAE, RMSE y on-time en las tres bandas.
- **El LSTM** (P5):
  - `es15` (early stopping sobre el 15 % del entrenamiento) para el titular y para todos los
    desgloses: por posición, por horizonte y correlatos;
  - `tw` para la escalera;
  - `es0` no se corre.
- **Las pruebas pareadas por ruta**, como en agosto.
- **El bootstrap agrupado**: el grupo es la ruta, salvo en las familias con variante, donde el
  grupo es la familia (1101 + 110101; 1401 + 1402; 1801 + 180101; 1802 + 180201). Son 31 grupos
  para 35 rutas. Las dos variantes comparten el 82–93 % de la traza y la flota.
- **La espera en la terminal intermedia** (P8) queda dentro del tiempo que se predice. No se
  quita ni se modela aparte.

## 6. Enmiendas

Cada una, con fecha, motivo y qué toca.

**E1 · 7-oct-2026 · `segment_count_ahead`, un error del código** (decisión del autor).
- **Qué estaba mal**: `compute_segment_count_ahead` (`func/lib/_lib_features/extended.py`)
  ordenaba las paradas por la media de `m_next_stop`, que son los metros que faltan para el fin
  del tramo actual.
  - Eso ordena por el largo del tramo, no por la posición de la parada en el viaje.
  - La feature, que dice ser «las paradas que faltan», era un código de la parada siguiente, sin
    orden.
  - Viene de agosto: es una de las doce de Tier B, y la mitad del grupo «route» de la ablación
    por grupos (Tabla S9, «Route position»).
- **Cómo se vio**: en el piloto local de la fase 2 (`../test/hermosillo-bus-test/reports/
  piloto_articulo_2026-10-07/`), mirando la feature misma, no las métricas de los modelos.
  - Su correlación de rangos con el avance sobre la traza, vuelta por vuelta, iba de −0.13 a
    +0.04 (mediana por ruta). Si fueran las paradas que faltan, sería −1.
  - En una vuelta de la 902 valía 9, 26, 11, 37, 42, 13, 8…
- **El arreglo**: cada parada va en la posición de la mediana de `cum_dist_m` (el avance del
  enrich en el viaje) de los registros que la tienen por siguiente parada, de menor a mayor.
  - Desempata el código, para que el orden no dependa de `group_by`.
  - Las paradas sin código ya no cuentan.
  - La primera parada del viaje vale N y la última, 1. En el piloto, la correlación con el
    avance queda en −0.98 a −1.00 (mediana por ruta).
- **Qué toca**: sólo la columna `segment_count_ahead` del `df_full`, y con ella los modelos que
  la leen.
  - **Los que cambian**: los de Tier B, `xgb_mse_b` y `xgb_mae_b`, en la corrida principal, la
    ablación y la escalera. También su descripción en el texto (la tabla de features del ESM y
    la Tabla S9).
  - **Los que no**: el corte, el frame, Tier A, los baselines (incluido `delay_hour`) y el LSTM.
- **Cómo se corre**: en una carpeta nueva (RUNBOOK). El caché del enrich no se invalida por
  cambios de código.

**E2 · 7-oct-2026 · `compute_headways`, el tiempo de la fase de datos** (un arreglo técnico, el
tiempo pedido; decisión del autor).
- **Qué pasaba**: `compute_headways` (`func/lib/_lib_features/extended.py`) recorría, por cada
  registro, los registros de todos los demás buses de la semana. Es O(n²) en Python puro.
  - En el piloto local era casi toda la fase de datos: 55 de los 65 min de la 601 en marzo.
  - Con la vuelta, todas las vueltas de una ruta caen en la misma `direction`, y los pares se
    duplican. Proyectado a enero–agosto en el servidor, eran hasta 6.6–12.5 h por ruta.
- **El arreglo**: el mismo cálculo, por bloques con numpy.
  - Los mismos pares, las mismas condiciones y las mismas operaciones en double. El cruce se
    redondea a microsegundos como `timedelta`.
  - El bucle queda como referencia (`compute_headways(..., bucle=True)`).
- **Qué toca**: nada de lo que sale. `headway_ahead` y `headway_behind` salen idénticos, bit a bit:
  - contra el bucle, en 3,000 tablas sintéticas, con los empates y los NaN, y en trozos reales
    de las cuatro rutas del piloto. También con numpy 1.26.4, la del servidor;
  - de punta a punta: la fase de datos de marzo de las cuatro rutas, con el `df_full` idéntico
    columna a columna y el mismo funnel.
  - Evidencia: `../test/hermosillo-bus-test/reports/piloto_articulo_2026-10-07/` (`LEEME.md` §8).
- **Lo que cambia es el tiempo**: la fase de datos de la 601 en marzo baja de 65 a 5 min. En
  enero–agosto, el enrich de la ruta más lenta queda en ~1 h.

**E3 · 7-oct-2026 · la atención del LSTM por trozos en las rutas largas** (un arreglo técnico, que
el LSTM corra; decisión del autor).
- **Qué pasaba**: en el piloto del servidor (fase 3), el LSTM de la 601 (640 celdas) murió en el
  primer paso de entrenamiento, `es15` y `tw`, con «Memory access fault by GPU». Las otras tres
  rutas (219–428 celdas) terminaron.
  - La atención aditiva arma un tensor (B, H, L, d_attn). Con el padding por batch, H y L
    llegan casi siempre a n_celdas − 2: en la 601 son 64 × 638 × 638 × 128 = 3.3e9 elementos.
  - Los kernels elementales de TF en GPU indexan con int32, así que un tensor de más de 2^31
    elementos escribe fuera de su memoria.
  - Con las legs A/B nunca pasó: en agosto la secuencia más larga tenía 446 celdas. Con la vuelta
    pasa en las rutas de más de 512 celdas: la 601, la 201 y la 202.
- **Cómo se vio**: un diagnóstico de una GPU con el modelo del repo y batches sintéticos
  (MI210, TF 2.18.1, ROCm 6.4.3). Evidencia en `../test/hermosillo-bus-test/reports/
  piloto_articulo_2026-10-07/` (`LEEME.md` §9, `diag_atencion.csv`, `e3_verificacion.txt`).
  - El tope es exacto: 64·511²·128 = 2,139,103,232 pasa y 64·513²·128 = 2,155,880,448 cae.
  - El LSTM solo, sin la atención, aguanta 640 pasos.
- **El arreglo**: `scores` se calcula por trozos del horizonte y se concatena
  (`SequentialLSTM._attention_scores`).
  - Hay trozos sólo si 64 · (n_celdas − 1)² · 128 pasa de 2^31 − 1 (`attention_chunks()` en
    `models.py`). Cada trozo queda en 2^30 elementos, por margen: 4 en la 601 y 3 en la 201 y
    la 202.
  - En las demás rutas el modelo no cambia: el código es el mismo y `model_kwargs`, también.
  - El meta de cada LSTM lo dice en `attention_chunks`.
- **Qué toca**: nada del modelo ni del entrenamiento. Los hiperparámetros, el batch, los cortes y
  los folds son los mismos.
  - **La misma función**: cada (b, h, l) es la misma cuenta; sólo cambia cuántas filas ve cada
    kernel.
    - En la GPU, la salida sale bit a bit igual, y los gradientes, a 3e-7 relativo (por el orden
      de las sumas).
    - En CPU, contra el `models.py` de `cd1c8b5` y con batches con padding, pasa lo mismo: la
      predicción, la pérdida y los pesos de atención salen idénticos, y los gradientes, a ≤ 2.7e-7.
    - Con un solo trozo sale idéntico en todo, gradientes incluidos.
  - **El coste**: el mismo. Con la 601 en la GPU, 0.2 s por paso y un pico de 27 GB de 64.
  - La otra salida era batch 32 en esas tres rutas. Se descartó porque cambia el entrenamiento.

**E4 · 8-oct-2026 · los cubos de horizonte: 40-60, 60-90 y > 90 min** (la resolución de un
desglose; decisión del autor). Se fija antes de correr los modelos de la corrida completa (fase 4).
- **Qué pasaba**: el desglose por horizonte (§5) usa los cubos de agosto, fijos en el harness
  (`func/honest_metrics.py`): 0-2, 2-5, 5-10, 10-20, 20-40 y > 40 min, por el tiempo restante real.
  - Con las legs de agosto (20–30 min), «> 40» era una cola. La vuelta dura de 59 a 179 min
    (mediana 138, funnels de la fase de datos), y «> 40» junta la mitad de los pares.
  - En el piloto del servidor, contado: 58 % de los pares de la 601, 49 % de la 180101, 29 % de
    la 1601 y 6 % de la 902. En las 35 rutas, proyectado: 50 %.
  - El desglose perdía la resolución justo donde están la mitad de los datos.
- **Cómo se eligió**: sólo con el reparto de los pares por horizonte, que depende de cuánto dura
  cada vuelta, no de los errores de ningún modelo. Del piloto se habían visto las cifras por
  horizonte de cuatro rutas; no entran en la elección.
  - Evidencia: `artefacts/cubos_horizonte_2026-10-08.{py,txt}` (fuera de git). Trae el reparto
    exacto del piloto y un modelo para las 35 rutas (los cortes del harness, el tiempo lineal en
    la distancia y la duración de cada vuelta de su funnel), que se aparta de lo exacto 4 puntos
    como mucho.
  - Con 60 y 90, los tres cubos nuevos juntan el 17.9 %, el 18.5 % y el 13.5 % de los pares.
    En «> 90» sólo cinco rutas, las de vuelta de 95 min o menos (902, 20000, 1601, 1602 y
    1502), tienen menos del 1 % de sus pares.
  - Se descartaron un corte en 120 («> 120» tendría el 3.7 % de los pares, y en 12 rutas menos
    del 1 %) y cortes cada 20 min.
- **El arreglo**: `HORIZON_EDGES_S` y `HORIZON_LABELS` ganan 3600 y 5400 s. Quedan 0-2, 2-5,
  5-10, 10-20, 20-40, 40-60, 60-90 y > 90 min.
  - Los cortes de agosto son un subconjunto de los nuevos. El «> 40» de antes se recupera exacto
    uniendo los tres últimos cubos, ponderados por `n_pairs`: MAE, RMSE, sesgo y on-time. La
    mediana y el P90 del error, no.
- **Qué toca**: sólo los `*_by_horizon` de XGBoost y del LSTM, que pasan de 6 a 8 filas por
  algoritmo, y con ellos la Fig. S3 (`make_fig6_horizon_frame.py`) y su texto en el manuscrito.
  - **Lo que no**: el binado va después de predecir, así que los modelos, el titular, el
    desglose por posición, los histogramas de `dev` y los frames no cambian.
  - `make_fig6_horizon_frame.py` se detiene si un bundle trae otros cubos, como los del piloto
    del 7-oct, en vez de saltarse «> 40 min» en silencio.
- **Verificado** (`artefacts/verificacion_e4_2026-10-08.{py,txt}`, fuera de git):
  - **XGBoost**: el job `xgb` del piloto local (601 y 180101 de marzo, 12 hilos), con los cubos
    de antes y con los nuevos, sobre los mismos datos.
    - El frame sale idéntico (mismo hash), y también el titular, los folds, las posiciones, los
      histogramas y las importancias. En el meta sólo cambia el tiempo de reloj.
    - Los cinco primeros cubos salen idénticos.
    - Los tres nuevos, unidos, dan el «> 40» de antes: los mismos pares en los 9 algoritmos, el
      MAE exacto, y el RMSE, el sesgo y el on-time a ≤ 4e-16 relativo.
  - **El LSTM bina igual que XGBoost** (`pl.cut` cerrado por la izquierda contra
    `np.searchsorted`): el mismo cubo en 200,024 tiempos, incluidos los siete bordes y ±1e-9 s.
  - **La Fig. S3** corre con los ocho cubos (un bundle sintético del piloto) y se detiene con el
    del piloto del 7-oct.
- **Cómo se corre**: la fase 4, en una carpeta nueva. Los `by_horizon` del piloto que quedan en
  `~/repro_vuelta` tienen los cubos de antes y saldrían como `cached`.

**E5 · 9-oct-2026 · la escalera temporal, alineada por mes** (un error del código, que aplica §4
tal como está escrito; decisión del autor). Se fija después de correr los modelos: no cambia
ninguno, sólo cómo la tabla junta los que ya están.
- **Qué estaba mal**: `make_table_13_temporal.py` (Tabla 6, Tabla S12 y Fig. 3) agrupaba por el
  número de fold, y tomaba el mes de cada fila del meta de la primera ruta.
  - Pero `make_folds_expanding` numera los periodos **de cada ruta**. La 901 tiene AVL desde
    abril: sus escalones 1–4 evalúan mayo–agosto y salían en las filas de febrero–mayo.
  - El panel constante sólo miraba el escalón más flaco (`min_n_train ≥ 100`), no cuántos
    escalones tiene la ruta. La 901 (455 vueltas en su primer escalón) entraba con cuatro de
    siete: 34 rutas en los escalones 1–4 y 33 en los 5–7. §4 pide ≥ 100 vueltas en **todos**.
  - Sólo la 901 tiene menos de siete escalones (los metas `frames/…_tw_meta.json`, `rungs`).
- **Cómo se vio**: al leer las tablas de `2a45f23` para el manuscrito
  (`../hermosillo-bus-arrival-manuscript/PLAN_DE_ACCION.md` §3, punto 0.1).
- **El arreglo**: cada fold se traduce a su `test_period` con el meta de su ruta, y el escalón es
  la posición de ese mes en la escalera (la unión de los meses evaluados: febrero–agosto). Un
  escalón que la ruta no tiene cuenta como cero vueltas de entrenamiento, así que el panel
  constante exige los siete. Si falta el meta de una ruta incluida, o trae otros folds, el
  generador se detiene. La nota de la tabla dice qué rutas quedan fuera y por qué.
  - De paso, la leyenda de la Fig. 3 sale debajo de los ejes: tapaba un punto.
- **Qué toca**: sólo esas tres salidas. Los modelos, los frames y las demás 55 salidas de
  `reproduce_tables.py` salen idénticos.
  - **Tabla 6, panel constante**: 33 rutas en los siete escalones (fuera la 101, con 88 vueltas
    en febrero, y la 901). XGBoost con contexto le gana al LSTM en 229 de 231 comparaciones
    (antes 233 de 235), por 5.2–7.8 % (igual); la media por hora, 2.8–5.8 % (antes 2.8–5.6 %).
    Cambian los escalones 1–4; los 5–7 ya eran estas 33 rutas.
  - **Tabla S12, todas las rutas**: 34 rutas en febrero–abril y 35 en mayo–agosto (antes 35 y
    34). XGBoost con contexto gana 240 de 242, como antes, por 5.2–7.7 % (antes 5.1–7.8 %).
- **Verificado**: `reproduce_tables.py` con el bundle publicado; sólo difieren esas seis salidas
  (dos tablas en `.md` y `.csv`, y la figura en `.png` y `.pdf`), y dos corridas seguidas las dan
  idénticas byte a byte.

**E6 · 9-oct-2026 · la ablación por grupos en el frame de la vuelta, y la sensibilidad al tope
de árboles** (dos corridas nuevas de XGBoost, sólo CPU; decisión del autor, V1 del plan del
manuscrito). Se fijó **antes** de correrlas: este texto y el código que corre (el paquete
`e6_servidor.tgz`, sha256 `35d94be4…`, armado de este mismo árbol) son anteriores al
lanzamiento. El autor lanzó `run_e6.sh` la noche del 9-oct y el commit llegó después, sin
cambiar nada de lo que aquí se fija salvo esta nota. Lo que dice no se cambia al ver los
resultados.
- **Qué pasaba**:
  - **La Tabla S9** (qué grupo del bloque de contexto compra qué) era un artefacto congelado de
    agosto: 43 legs de 11 rutas, tope de 1,200 viajes, fuera del frame común, la geometría de
    antes y `segment_count_ahead` con el error de E1. Sostiene un párrafo de §3.4 y otro de §4.
  - **El tope de 600 árboles**: la S5 medida (`make_appendix_f_model_fits.py`, sobre el bundle
    de la fase 4) muestra que el early stopping no paró a muchos ajustes con error absoluto antes
    del tope. Llegaron a 600 árboles 59 de 175 en XGBoost (MAE, + contexto), 55 en el nivel 1 y
    71 en el nivel 0 de la Tabla 5, contra 10 en el nivel 2. El 9k decía «muy por debajo del
    tope». Si los niveles quedan recortados de forma desigual, el reparto de la Tabla 5 y la
    paridad del nivel 0 con el LSTM podrían ser en parte del tope.
- **Corrida A, la S9** (la principal; reemplaza a la de agosto):
  `run_routes.py --n-route-quantile 0.05 --min-coverage 1.0 --max-trips 2500 --exo-ablate
  --no-baselines --no-frames`, las 35 rutas.
  - El régimen publicado, sin cambios: el mismo frame (se reconstruye igual y no se escribe),
    la misma semilla, los mismos folds y 600 árboles.
  - Ajusta Tier A + **un** grupo a la vez (calendario, headway, dwell, velocidad reciente,
    posición en la ruta: `EXO_GROUPS`), sólo con error absoluto, como en agosto; y además
    `xgb_mse`, `xgb_mae`, `xgb_mse_b` y `xgb_mae_b`, que tienen que reproducir la corrida
    publicada con |ΔMAE| = 0 en las 35 rutas. Si no, la tabla no se escribe.
  - Se lee como la de agosto (`make_table_11_ablation.py`): mediana por ruta del cambio relativo
    contra Tier A, rutas que mejoran, Wilcoxon y Holm sobre los cinco grupos, y la fracción de
    la ganancia de Tier B completo.
  - De paso guarda los descriptivos del bloque de contexto en los cortes (`*_exodesc.parquet`):
    cuantiles, fracción de ceros y, en el headway, la fracción en el tope de 3,600 s. Sustituyen
    a los de agosto en §4 («dwell median 0 s», «headway median 15.7 min»).
- **Corrida B, el tope** (sensibilidad; no sustituye a nada):
  lo mismo que A más `--no-prior --trip-only --n-estimators 2000`. Las once variantes de
  XGBoost (las cuatro de siempre, los niveles 0 y 1 y los cinco grupos) con el tope en 2,000,
  todo lo demás igual (incluido el early stopping de 30 rondas sobre el 15 % de los viajes de
  entrenamiento y la tasa de aprendizaje de 0.05). Escribe con sufijo `_n2000`
  (`run_experiment.py --n-estimators`, nuevo).
  - 2,000 es más de tres veces el tope que se alcanzó; la mediana de árboles con error absoluto
    está entre 380 y 550.
  - Se lee con `make_table_17_tree_cap.py`: cuántos ajustes llegan al tope en cada versión,
    cuánto cambia el MAE de cada variante, y lo que el tope podría mover: los cuatro niveles de
    la Tabla 5 y su reparto, el nivel 0 contra el LSTM, XGBoost (MAE, + contexto) contra el LSTM
    y contra la media por hora, y el Δ de cada grupo de la S9.
- **Regla fijada ahora**: las tablas publicadas se quedan con el tope pre-especificado de 600
  (§5: «los mismos ajustes»). La corrida B se reporta como sensibilidad, en la S5 y en §5 del
  manuscrito, con su tabla, **diga lo que diga**. Si con 2,000 árboles alguna afirmación cambia
  de signo o deja de ser significativa, el manuscrito lo dice junto a la afirmación; no se
  cambia de tope para el titular.
- **Qué no toca**: la corrida publicada (el paquete de resultados sólo añade archivos con sufijo
  `_ablate`), el LSTM y los baselines, que no tienen árboles.
- **El número de hilos importa**: XGBoost suma histogramas en paralelo, y con otro número de
  hilos los árboles cambian en el último dígito. Las dos corridas van con 16, como la publicada
  (`omp_threads` en sus `*_timing`).
- **Probado** antes de lanzar, en local, con los datos de marzo del piloto:
  - la 601 con la corrida A y los mismos 12 hilos que su piloto: los cuatro modelos compartidos
    dan |ΔMAE| = 0 y los mismos árboles en los cinco folds. La 902 con 12 hilos contra los 8 de
    su piloto difiere hasta 0.14 s: es el número de hilos;
  - la 902 con un tope de 300 viajes: las dos corridas terminan, escriben sus archivos con su
    sufijo, y los modelos compartidos dan el mismo MAE en las dos;
  - los generadores de A y B corren sobre un bundle sintético de 35 rutas. Con el tope de 600
    reproducen las cifras de la Tabla 5 y de su nota, y el de A se detiene si los modelos
    compartidos no casan.
- **Resultado** (10-oct; añadido después de correr, sin tocar lo anterior):
  - **Las dos corridas terminaron**: 70 de 70 tareas `COMPLETED` (jobs 2078729, la A, y
    2078730, la B; 16–35 min por ruta), 18 archivos por ruta en las 35. El paquete
    (`e6_resultados_20261009_2148.tgz`, sha256 `f5e26e6f…`) se descomprimió dentro del bundle
    publicado: 842 archivos nuevos, ninguno reemplaza a otro; los 2,701 renglones previos de
    `MANIFEST.sha256` no cambian. El código que corrió casa con `252cb83` (`E6_SHA256SUMS`).
  - **La comprobación pasa**: los cuatro modelos compartidos reproducen la corrida publicada con
    |ΔMAE| = 0 en las 35 rutas.
  - **La S9** (corrida A): calendario −5.31 % contra Tier A (35/35 rutas, 88 % de la ganancia
    de Tier B, −6.04 %); headway −0.61 % (34/35, 10 %); posición en la ruta −0.38 % (34/35,
    6 %); velocidad reciente −0.24 % (32/35, 4 %); dwell −0.05 % (23/35, 1 %). Los cinco,
    p (Holm) ≤ 0.003. En agosto, headway y dwell empeoraban (+0.30 %, +0.17 %) y sólo
    calendario y posición eran significativos.
  - **El tope** (corrida B): con 2,000 árboles ningún ajuste llega al tope (0/175 en las once
    variantes). El MAE mediano de cada variante se mueve ≤ 0.1 s; los niveles de la Tabla 5
    (219.3 / 219.1 / 211.4 / 197.2 s), su reparto (36 % / 64 %), el nivel 0 contra el LSTM
    (mejor en 16/35, p 0.24 → 0.23) y XGBoost (MAE, + contexto) contra el LSTM (35/35) y contra
    la media por hora (35/35, −2.2 %) quedan iguales; los Δ de la S9 se mueven ≤ 0.02 puntos.
    Ninguna afirmación cambia de signo ni de significancia: el tope de 600 no recorta nada que
    importe.
  - **Verificado**: `reproduce_tables.py` con el bundle: sólo difieren las salidas de E6 (la S9,
    la Tabla 17 y `context_descriptives.csv`); dos corridas seguidas las dan idénticas byte a
    byte. Al generador de la S9 se le corrigió el título (Table 15, como la de agosto; «Table
    11» chocaba con la tabla por posición) y la nota, que ahora dice «exactly» cuando
    |ΔMAE| = 0.
