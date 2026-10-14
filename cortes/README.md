# `cortes/` — dónde se corta la vuelta de cada ruta

Las entradas del corte por cruce de km (`TRIP_CUT=cruce`, `docs/HANDOFF_METODO_V2.md` §9). Las
usa `config.env` §7 por defecto desde la rama `vuelta`, y las fija el pre-registro
(`docs/PREREGISTRO_VUELTA.md` §1).

| archivo | qué es | sha256 |
|---|---|---|
| `cortes_vuelta.json` | `{ruta: [[lat, lon, km]]}`, un punto por ruta (35). El km está sobre la geometría corregida de `data/01_raw/`; lat y lon comprueban que la geometría sea ésa (`CRUCE_KM_TOL_M`) | `d5ca2c92c6e37559df7470dc37b7cb31a0ec3f3d58f9365a927c90432bf835db` |
| `cortes_vuelta.csv` | el mismo punto con su parada, la regla que lo eligió (`metodo`), el tipo de ruta (una o dos terminales, o variante) y por qué | `30ab78b0d0f0805f3d38227251204096fb4d1f50fadd1c028773dd23c9b54d7d` |
| `vueltas_une.csv` | la duración de la vuelta que declara UNE en su portal (`UNE_lineas.json`, metadata `duration`), para validar la vuelta medida | `f44ae1abe0e4481268d89c46d68290d131f2a280a67c94362852be1dd40d1f8f` |

**De dónde salen.** Los tres los escribe `analisis/cortes_vuelta.py` del test
(`../test/hermosillo-bus-test/`). Se copiaron sin cambios el 6-oct-2026 y son los que se
validaron dos veces en el servidor (handoff del proyecto, §4.14). El script lee las salidas del
comparativo de métodos (`../Comparativo_metodos_viajes.md` §10.5), que no viajan en este repo.
Las reglas:
- **las 20 rutas de una terminal**: la regla `recomendado` del comparativo. Es donde coinciden el
  descanso, el arranque del día y el bus vacío;
- **las 7 de dos terminales** (101, 301, 601, 1501, 1601, 1602, 1701): la terminal que da más
  cobertura con la vuelta;
- **las 8 de las familias con variante**: la regla de un corte sin pasajeros, sobre su propia
  corrida del comparativo. La 1401 y la 1402 cortan en Lázaro Cárdenas y Meoqui, su terminal
  sur y su mayor descanso.

**Por qué no importa mucho dónde se corta.** La duración de la vuelta no depende del punto: al
moverlo hasta 22 km, la mediana cambia ≤ 0.7 min. Lo que cambia es la cobertura y qué parada
abre la vuelta. Se probó una sola regla para todas (el punto de mayor descanso) y rompía la 1902
y la 1501 (decisión P2 del plan; evidencia en
`../test/hermosillo-bus-test/reports/plan_articulo_2026-10-06/`).

**Para cambiarlos**: hace falta una enmienda fechada en el pre-registro.
