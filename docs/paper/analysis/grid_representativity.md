# Representatividad de la grid — retenidos vs descartados

Generado por `docs/paper/analysis/grid_representativity.py`. Un viaje se
retiene si tiene ≥5 pings y cubre `min_coverage · n_route_m`, donde
`n_route_m` es el cuantil `q` de las longitudes de viaje de la leg.

Todas las comparaciones son **muestra analizada (retenidos) vs población
completa de la leg**, no retenidos vs descartados: la pregunta es si lo
que se analiza representa a la ruta.

SMD = (media_retenidos − media_población) / sd_población.
Convención: |SMD| < 0.1 equilibrado · 0.1–0.25 leve · **> 0.25
desplazamiento material**. TVD = distancia de variación total; KS =
estadístico de Kolmogórov-Smirnov contra la población.

## 1. Retención por régimen

| q | min_cov | legs | retención mediana | p25 | p75 | legs < 20% |
|---|---|---|---|---|---|---|
| 0.95 | 1 | 35 | **0.1%** | 0.0% | 1.6% | 35 |
| 0.95 | 0.99 | 35 | **77.0%** | 59.4% | 85.5% | 0 |
| 0.50 | 1 | 35 | **37.5%** | 31.3% | 42.0% | 1 |
| 0.50 | 0.99 | 35 | **90.3%** | 87.0% | 93.2% | 0 |
| 0.10 | 1 | 35 | **87.7%** | 86.1% | 89.2% | 0 |
| 0.10 | 0.99 | 35 | **96.8%** | 95.7% | 97.5% | 0 |
| 0.05 | 1 | 35 | **94.1%** | 93.7% | 94.5% | 0 |
| 0.05 | 0.99 | 35 | **97.7%** | 97.0% | 98.5% | 0 |

## 2. Desplazamiento de covariables (min_coverage = 1)

Mediana sobre legs del |SMD| / TVD de la muestra analizada respecto de
la población de la leg.

| q | legs | |SMD| trip_len_m | |SMD| duration_s | |SMD| hour_dep | TVD day_of_week | TVD month | legs con \|SMD\| longitud > 0.25 |
|---|---|---|---|---|---|---|---|
| 0.95 | 18 | 0.779 | 0.238 | 0.141 | 0.114 | 0.144 | 18/18 |
| 0.50 | 35 | 0.529 | 0.118 | 0.051 | 0.030 | 0.032 | 35/35 |
| 0.10 | 35 | 0.218 | 0.038 | 0.016 | 0.013 | 0.014 | 6/35 |
| 0.05 | 35 | 0.153 | 0.021 | 0.011 | 0.008 | 0.010 | 0/35 |

## 3. Sesgo de longitud (el mecanismo)

`len_ratio_keep_all` = longitud mediana de los retenidos / longitud
mediana de TODOS los viajes de la leg. 1.00 = sin sesgo.

| q | legs | ratio mediano | p25 | p75 | máx |
|---|---|---|---|---|---|
| 0.95 | 18 | **1.008** | 1.006 | 1.009 | 1.017 |
| 0.50 | 35 | **1.004** | 1.003 | 1.005 | 1.008 |
| 0.10 | 35 | **1.001** | 1.000 | 1.001 | 1.002 |
| 0.05 | 35 | **1.000** | 1.000 | 1.000 | 1.001 |

## 4. Flota

`bus_kept_frac` = buses presentes entre los retenidos / buses de la leg.
Si baja mucho, el filtro además selecciona vehículos.

| q | legs | fracción mediana de buses conservados | p25 |
|---|---|---|---|
| 0.95 | 18 | **0.163** | 0.082 |
| 0.50 | 35 | **0.769** | 0.714 |
| 0.10 | 35 | **0.981** | 0.974 |
| 0.05 | 35 | **0.992** | 0.984 |

