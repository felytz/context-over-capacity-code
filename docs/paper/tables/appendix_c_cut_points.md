**Table S4. Where each route's round trip is cut, and the round trip measured against the operator's declared duration.**

Each route is cut at one point of its polyline, fixed before any model was run (pre-registration, `cortes/cortes_vuelta.json`); a round trip runs from one crossing of that point to the next. On one-terminal routes the point is the location chosen by the rule shown; on two-terminal routes, of the two terminals, the one that leaves more of the service in complete round trips; on the variant routes, a single-cut rule run on their own records. *km* is the position of the point along the polyline and the loop length; *stop* is the nearest stop and its distance. *Median* is the median duration of valid round trips in this run (10th–90th percentile), against the round-trip duration the operator's portal declares (*UNE*). *Inferred* is the share of crossings whose time is interpolated across a gap of more than five minutes between records; *rare*, the share of round trips outside 0.6–1.6 times the median.

| Route | Type | Cut point (stop, distance m) | km / loop | Located by | Median (min) | UNE (min) | Δ vs UNE (%) | Inferred (%) | Rare (%) | Cut quality |
|---|---|---|---|---|---|---|---|---|---|---|
| 101 | two terminals | Calle del Jet y Garcia Morales (134) | 24.26 / 50.1 | lowest occupancy (two) | 149.5 (138–161) | 150 | -0.3 | 0.4 | 0.18 | review |
| 102 | one terminal | Blvd. Enrique Mazon Lopez Y Pilares (0) | 0.00 / 32.9 | km 0 of the portal polyline | 115.5 (107–123) | 117 | -1.3 | 0.5 | 0.13 | ok |
| 201 | one terminal | Solildaridad y Emancipacion (85) | 19.49 / 57.6 | main layover point | 179.0 (164–193) | 180 | -0.6 | 1.4 | 0.07 | ok |
| 202 | one terminal | Angel Arreola y Mario MORUA Jhonson (26) | 32.83 / 57.2 | main layover point | 178.1 (165–191) | 180 | -1.1 | 1.3 | 0.09 | ok |
| 301 | two terminals | Blvd. Luz Valencia y Jose Maria Escriva (52) | 22.35 / 45.7 | terminals with most complete cycles | 134.9 (124–144) | 136 | -0.8 | 1.4 | 0.11 | ok |
| 302 | one terminal | Av. del Jet y Concord (31) | 26.88 / 47.6 | main layover point | 138.2 (126–151) | 140 | -1.3 | 3.0 | 0.13 | review |
| 401 | one terminal | Calle Margarita Maza de Juarez y Placeritas (2) | 3.43 / 49.0 | main layover point | 142.9 (133–152) | 144 | -0.8 | 1.5 | 0.04 | ok |
| 402 | one terminal | M MAZA DE JUAREZ Y CAMPO VERDE (30) | 44.14 / 48.9 | main layover point | 143.0 (132–154) | 144 | -0.7 | 1.0 | 0.07 | ok |
| 501 | one terminal | josemaria escriva de balaguer y blvd progreso (0) | 0.00 / 38.9 | km 0 of the portal polyline | 139.2 (129–151) | 140 | -0.6 | 0.7 | 0.22 | ok |
| 601 | two terminals | Solidaridad patio (12) | 63.21 / 64.8 | layover points (two) | 175.9 (163–188) | 170 | +3.5 | 0.6 | 0.06 | review |
| 801 | one terminal | calle luis encinas y av veracruz (25) | 35.95 / 36.0 | main layover point | 123.9 (113–133) | 126 | -1.7 | 2.2 | 0.24 | ok |
| 901 | one terminal | VILDOSOLA Y MANUEL OJEDA (41) | 13.83 / 39.0 | main layover point | 128.9 (120–137) | 140 | -7.9 | 2.9 | 0.25 | review |
| 902 | one terminal | DR ALBERTO G NORIEGA Y CALLE GARMENDIA (0) | 0.00 / 22.4 | km 0 of the portal polyline | 59.0 (52–67) | 60 | -1.7 | 1.5 | 0.90 | ok |
| 1001 | one terminal | Av. Luis Donaldo Colosio / Calle Guerrero (18) | 9.83 / 42.4 | main layover point | 128.0 (118–138) | 130 | -1.5 | 1.5 | 0.07 | ok |
| 1002 | one terminal | Av. Monterrey  / Calle Guerrero (46) | 32.63 / 43.3 | dispatch point (where headways become regular) | 128.5 (120–137) | 130 | -1.2 | 2.1 | 0.19 | ok |
| 1101 | variant | lazaro mercado y chilpancingo (36) | 43.70 / 47.3 | consensus of layover, start of service and lowest occupancy | 174.9 (163–186) | — | — | 2.5 | 0.09 | ok |
| 1101-ALT (110101) | variant | lazaro mercado y chilpancingo (14) | 43.80 / 47.3 | main layover point | 175.8 (164–186) | — | — | 2.4 | 0.02 | ok |
| 1201 | one terminal | Solidaridad patio (17) | 18.48 / 49.0 | main layover point | 164.8 (152–176) | 166 | -0.7 | 1.8 | 0.09 | ok |
| 1301 | one terminal | francisco i. madero y blvd luz valencia (8) | 14.73 / 38.7 | main layover point | 137.9 (127–148) | 144 | -4.2 | 2.3 | 0.16 | ok |
| 1401 | variant | LAZARO CARDENAS Y MEOQUI (44) | 35.21 / 43.1 | main layover point | 144.9 (133–157) | — | — | 2.7 | 0.47 | ok |
| 1402 | variant | LAZARO CARDENAS Y MEOQUI (44) | 34.96 / 42.9 | main layover point | 143.9 (132–156) | — | — | 2.1 | 0.39 | ok |
| 1501 | two terminals | Av Jose Maria Mendoza Y Calle 4 (138) | 6.43 / 45.7 | dispatch points (two) | 156.9 (147–166) | 158 | -0.7 | 2.5 | 0.12 | ok |
| 1502 | one terminal | Benito Juárez y Luis Donaldo Colosio (35) | 0.38 / 25.4 | dispatch point (where headways become regular) | 94.8 (86–103) | 96 | -1.3 | 2.9 | 0.13 | ok |
| 1601 | two terminals | — (53) | 22.30 / 24.8 | lowest occupancy (two) | 82.8 (75–90) | 80 | +3.5 | 2.1 | 0.11 | ok |
| 1602 | two terminals | — (17) | 2.57 / 24.8 | layover-cluster terminals | 82.8 (74–91) | 81 | +2.2 | 2.6 | 0.20 | ok |
| 1701 | two terminals | Blvd Solidaridad Y Maria Jesus Tanori (0) | 0.00 / 40.0 | geometric extremes of the polyline | 133.4 (124–143) | 135 | -1.2 | 2.9 | 0.07 | ok |
| 1702 | one terminal | MONTERREY Y MATAMOROS (173) | 32.83 / 33.0 | dispatch point (where headways become regular) | 110.2 (102–118) | 112 | -1.6 | 3.2 | 0.14 | ok |
| 1703 | one terminal | Benito Juárez y Luis Donaldo Colosio (5) | 28.15 / 40.7 | consensus of layover, start of service and lowest occupancy | 118.2 (110–126) | 108 | +9.4 | 3.6 | 0.13 | ok |
| 1801 | variant | Calle Margarita Maza de Juarez y Placeritas (1) | 0.00 / 44.1 | km 0 of the portal polyline | 138.0 (127–149) | — | — | 3.3 | 0.11 | ok |
| 1801-ALT (180101) | variant | Calle Margarita Maza de Juarez y Placeritas (1) | 0.00 / 43.5 | km 0 of the portal polyline | 138.0 (127–148) | — | — | 3.8 | 0.08 | ok |
| 1802 | variant | M MAZA DE JUAREZ Y CAMPO VERDE (20) | 45.03 / 45.2 | main layover point | 138.1 (126–149) | — | — | 3.2 | 0.18 | ok |
| 1802-ALT (180201) | variant | M MAZA DE JUAREZ Y CAMPO VERDE (20) | 44.49 / 44.6 | main layover point | 138.0 (127–148) | — | — | 4.0 | 0.18 | ok |
| 1901 | one terminal | Blvd. Enrique Mazon Lopez Y Pilares (0) | 0.00 / 50.3 | km 0 of the portal polyline | 163.4 (154–172) | 160 | +2.1 | 4.2 | 0.11 | ok |
| 1902 | one terminal | Blvd. Enrique Mazon Lopez Y Pilares (0) | 0.00 / 49.6 | km 0 of the portal polyline | 163.9 (154–173) | 165 | -0.7 | 3.9 | 0.13 | ok |
| 20000 | one terminal | BLVD SOLIDARIDAD Y CALLE REAL DEL ARCO (53) | 12.05 / 25.0 | dispatch point (where headways become regular) | 78.1 (69–87) | 85 | -8.1 | 3.6 | 0.19 | ok |

*Notes.* 20 routes have one terminal, 7 two and 8 are variants. The measured round trip is within 2% of the declared duration on 19 of 27 routes and within 5% on 24 (median absolute difference 1.3%; largest 1703, +9.4%). Two-terminal routes wait at the second terminal within the round trip, and that wait is part of the time predicted (Table S16). The duration hardly depends on where the round trip is cut (Table S5).
