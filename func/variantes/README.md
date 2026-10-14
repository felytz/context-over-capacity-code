> **English summary.** Four route codes in the AVL records (1101, 1400, 1801 and 1802)
> carry two services each, which share most of their path and split in one zone. These
> scripts assign every record of those codes to one of the two services, vehicle by
> vehicle and day by day, from the route geometry and the vehicle's heading. Their output
> is the raw input of the 8 variant routes (`data/01_raw/records/{1101,110101,1401,1402,
> 1801,180101,1802,180201}/`), whose sha256 are fixed in `docs/PREREGISTRO_VUELTA.md` §1.
> The article only reads that output; these scripts document how it was produced.

# `func/variantes/` — separar la principal de la alterna, bus por bus

Cuatro códigos de ruta del AVL mezclan dos servicios: 1101 / 1101-ALT, 1400 (1401 y 1402),
1801 / 1801-ALT y 1802 / 1802-ALT. `variantes_ruta.py` decide, para cada bus y cada día, por
cuál iba, con la traza y el rumbo; después reasigna el `group_id`. Así salieron las 8 rutas
con variante del artículo (decisión P1 del plan; método y resultado en
`../../../Variantes_rutas_alternas.md`, en la carpeta del proyecto).

## De dónde salen estos archivos

Se copiaron del test (`test/hermosillo-bus-test/`) el 7-oct-2026, sin cambios de contenido:
son los que produjeron el crudo separado el 6-oct. Allá tienen fines de línea CRLF; aquí, LF
(`.gitattributes` normaliza los `.py`), así que el sha256 cambia y el contenido no.

| archivo | sha256 en el test (CRLF) | sha256 aquí (LF) |
|---|---|---|
| `mapas_traza.py` | `b50d2e395a468f3a78b768160fb2fac20fd30734e62743bd8a780ff996b352d9` | `c500e1febf416406ac940f270c3a882eaf268db103dc99ec00ad5681faae798e` |
| `analisis/variantes_ruta.py` | `85009a2f34adcaad1dc566ad75b4284e9aa7f41f552f2d40b5b13174c830da55` | `0449b860e3cc95a14fdfa83b52d212cba3e9aa8dcfcc017681123ebd0d25e3a2` |
| `analisis/variantes_mapa.py` | `af8745b2b4a8936aad0bfa40635f6cb6e6738cf053c12b3513071d061c1b1cb1` | `6eaa1bd2c7d6a984c1386a293ddb31f9a1ffba84bdccce960c4768bf28f432b6` |

- `mapas_traza.py` es el emparejador de registros con la traza del test (el punto más
  cercano en el sentido del bus, con continuidad entre pasadas). `variantes_ruta.py` lo
  importa; del resto del archivo (los mapas) no usa nada.
- Se conserva la estructura de carpetas del test (`mapas_traza.py` arriba y el script en
  `analisis/`) porque los dos resuelven sus rutas relativas a sí mismos.

## Cómo se corre

Lee y escribe con la estructura del test, **dentro de esta carpeta** (`data/` y `reports/`
de aquí están en `.gitignore`):

```
func/variantes/data/01_raw/
├── polylines.geojson …                 la geometría (o `--geom data/01_raw`, la del artículo)
├── records_variantes/<grupo>/          ENTRADA: el crudo de la BD con las dos variantes
│   └── df_rec_raw_<grupo>_2024_01-12.parquet      (grupo = 1101, 1400, 1801, 1802)
└── records/<n>/                        SALIDA: el crudo de cada variante
    └── df_rec_raw_<n>_2024_01-12.parquet          (n = 1101, 110101, 1401, …)
```

```bash
python func/variantes/analisis/variantes_ruta.py --geometria --geom data/01_raw   # sólo las zonas (segundos)
python func/variantes/analisis/variantes_ruta.py --familias 1101 1401 1801 1802 --geom data/01_raw
```

La salida se copia a `data/01_raw/records/` del artículo. Con el crudo de la BD del 6-oct
tiene que dar los ocho sha256 de `docs/PREREGISTRO_VUELTA.md` §1.

- **El crudo mezclado no viaja con el artículo**: viene de la BD (`mobility_data`, grupos
  1101, 1400, 1801 y 1802; `Variantes_rutas_alternas.md` §5). Lo que se publica es la
  salida, el crudo separado, que es la entrada del artículo.
- **Lo que queda fuera** («indeterminado»: 0.4–1.4 % de los registros, sobre todo bus-días
  de minutos) se escribe en `data/02_interim/variantes/<familia>/` y no entra al crudo
  separado. Sus cuentas están en `docs/paper/analysis/variants_unassigned.csv`.
- **La validación** con verdad conocida (`--mezcla A B`, rutas reales mezcladas) está en
  `docs/paper/analysis/variants_validation.csv` (copia de `pruebas.csv` del test).

La tabla del artículo (Tabla S1) la arma `docs/paper/analysis/variants_separation.py`, que
lee el crudo separado y estos dos CSV, y recalcula las zonas con `geometria()` de aquí.
