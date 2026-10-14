> **English summary.** This folder receives the raw inputs of the pipeline: the
> AVL records of the 35 routes (`records/<route>/df_rec_raw_<route>_*.parquet`)
> and five network-wide geometry and stop files. They were collected from the
> public web portal of UNE, Hermosillo's integrated transport system
> (https://unesonora.com), and are published **without the licence-plate column**
> (`plate_number`), which the analysis does not use; the loader works the same
> with or without it. Download the dataset *Hermosillo bus AVL records, route
> geometry and round trips, January–August 2024* from Zenodo
> (https://doi.org/10.5281/zenodo.23384907): extract `avl_records.zip` and `geometry.zip`
> into one folder and run, from the repository root,
> `python prepare_raw_from_deposit.py <that folder>`: it copies `avl_records/` and
> `geometry/corrected/` here with the names below. They are not needed to
> regenerate the tables and figures (`reproduce_tables.py` reads only
> `artefacts/`), only to re-run the pipeline. The rest of this file, in Spanish,
> describes the layout.

# `data/01_raw/` — lo ÚNICO que hay que traer

Todo lo demás (`02_interim/`, `03_processed/`) lo genera el pipeline. Esta
carpeta es la entrada y hay que copiarla a mano.

## Qué va aquí

```
data/01_raw/
├── records/
│   ├── 101/   df_rec_raw_101_2024_01-12.parquet
│   ├── 102/   df_rec_raw_102_2024_01-12.parquet
│   └── …      una carpeta por ruta, 35 en total
├── polylines.geojson       geometría de cada ruta (anidado por operador/ruta)
├── segments.geojson        tramos entre paradas
├── segments_route.json     qué segmentos pertenecen a cada ruta, y en qué orden
├── stops.json              catálogo global de paradas
└── stops_routes.json       qué paradas pertenecen a cada ruta, y en qué orden
```

Los cinco ficheros sueltos son de **toda la red**, no por ruta: se copian una
vez. Pesan ~6 MB entre todos.

## La geometría (desde el 6-oct-2026, rama `vuelta`)

Es la **corregida por el autor**, `data raw/new-01_raw/` de la carpeta del proyecto:
- la traza y las paradas del portal, corregidas a mano ruta por ruta y contrastadas con
  el crudo, en tramos de 100 m, mes a mes;
- con la numeración del AVL (101↔102 y 1601↔1602 renumeradas contra el portal);
- con los códigos de parada de UNE de 6 caracteres;
- las variantes -ALT, con número entero (110101, 180101, 180201).

Sus sha256 están en `docs/PREREGISTRO_VUELTA.md` §1. Los kilómetros de
`cortes/cortes_vuelta.json` están medidos sobre ella.

La de agosto (la del artículo hasta el 6-oct, del 13-jul) es `data raw/article-old/`. Para
reproducir aquella corrida hay que volver a copiar esos cinco archivos aquí.

## Las 35 rutas

```
101 102 201 202 301 302 401 402 501 601 801 901 902 1001 1002 1201 1301
1501 1502 1601 1602 1701 1702 1703 1901 1902 20000
1101 110101 1401 1402 1801 180101 1802 180201
```

- **Las 27 de la primera línea** son el crudo público: el mismo que en agosto, byte a byte.
- **Las 8 de la segunda** son las cuatro familias cuyo `group_id` del AVL mezcla dos
  variantes: 1101 y 1101-ALT, la 1400 (1401 y 1402), 1801 y 1801-ALT, 1802 y 1802-ALT. Su crudo
  de 2024 se bajó de la BD el 6-oct (el de las otras 27 es idéntico al público). Se separó bus
  por bus con `variantes_ruta.py` (`Variantes_rutas_alternas.md` en la carpeta del proyecto).
  Cada archivo lleva además `group_id_avl`, `variante`, `bloque` y `bloque_evidencia`, que el
  pipeline no lee. Sus sha256 también están en el pre-registro.
- **La 902**, excluida en agosto (sin descansos en terminal no se partían sus viajes de ida y
  vuelta), entra con la vuelta.

## El nombre del archivo importa

```
df_rec_raw_<ruta>_<etiqueta>.parquet
```

La `<etiqueta>` es libre, pero el pipeline elige **el extracto más grande** de
cada ruta cuando hay varios (y prefiere `2024_01-12` si está). Para forzar uno
concreto: `RAW_DATE_RANGE=2024_01-12` en `config.env`.

⚠ La ventana de análisis **no** sale del nombre del extracto: la fija
`MODE=full`, que recorta a **2024-01-01 … 2024-09-01** y etiqueta los artefactos
como `2024_01-08`. Eso es **enero–agosto de 2024** (~242 días), no la semana del
1 al 8 de enero.

## Cómo copiarlo

```bash
rsync -av --checksum --partial /ruta/vieja/data/01_raw/ ./data/01_raw/
```

`--checksum --partial`, **nunca `cp` a pelo**. Una copia interrumpida ya dejó
una vez el árbol con 21 parquet de miles y un `df_rec_full` cortado a 16 MB de
23, y nada se quejó. Después de copiar, comprueba que se leen de verdad:

```bash
python func/parquet_guard.py data/01_raw --pattern 'df_rec_raw_*.parquet'
```

Un parquet truncado **no** es un parquet ausente: es peor. El ausente se nota;
el truncado pasa las auditorías que preguntan `is_file()` y la reanudación lo da
por hecho.

## Si el extracto no está en disco

El paso 01 sabe descargarlo de la base de datos (`--prefer-full-range`), pero
esta carpeta está pensada para correr **sin BD**: si falta un extracto, la tarea
de esa ruta aborta con `rc=2` y un mensaje que dice exactamente qué fichero
esperaba.
