# When the pre-registration and its amendments were fixed

This repository is published as a single snapshot, without its git history. The
pre-registration (`PREREGISTRO_VUELTA.md`) says that the proof it was fixed before
any model was run on the round trip is the date of its commit. These are the
commits that created or amended it in the authors' working repository, with
their dates and messages (in Spanish), as recorded by `git log`:

| Date (UTC−7) | Commit | Message |
|---|---|---|
| 2026-10-06 20:01 | `c4efd97` | pre-registro de la corrida con la vuelta: datos, método, inclusión, paneles e inferencia, antes de correr modelos |
| 2026-10-06 23:21 | `2581138` | enmienda E1 del pre-registro: segment_count_ahead ordena las paradas por su avance, no por el largo del tramo |
| 2026-10-07 02:55 | `cd1c8b5` | enmienda E2 del pre-registro: compute_headways vectorizado, la misma salida bit a bit en minutos y no en horas |
| 2026-10-07 16:02 | `6d0e112` | enmienda E3 del pre-registro: la atención del LSTM por trozos en las rutas de más de 512 celdas, la misma función sin pasar de 2^31 elementos |
| 2026-10-08 02:59 | `10035cb` | enmienda E4 del pre-registro: los cubos de horizonte ganan 40-60, 60-90 y > 90 min, porque con la vuelta «> 40» juntaba la mitad de los pares |
| 2026-10-09 20:01 | `cf61876` | enmienda E5 del pre-registro: la escalera temporal se alinea por mes, porque la 901 (AVL desde abril) metía mayo–agosto en las filas de febrero–mayo y entraba al panel constante con cuatro de siete escalones |
| 2026-10-09 21:32 | `252cb83` | fase 0 del manuscrito y enmienda E6: generadores de las cifras sin artefacto, S14 con 35 rutas, y la S9 y el tope de árboles en el servidor |
| 2026-10-10 12:10 | `38a6d14` | enmienda E6, resultados: la S9 en el frame de la vuelta (35 rutas) y la Tabla 17 (tope de 2000 árboles: nada cambia) |

The messages number the tables as the drafts of the time did: their S9 is the
article's Table S14, their S14 is Table S7 and their Table 17 is Table S8.

The job logs inside the three bundles under `artefacts/` carry the dates on which
each model ran. Amendments E1–E6 are described, with their reasons, in §6 of the
pre-registration.
