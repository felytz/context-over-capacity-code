#!/bin/bash
# =====================================================================
# run_pipeline.sh — COMANDO 1 (servidor): encola TODO lo que el análisis
#                   local necesita, y te puedes ir
# =====================================================================
#   bash run_pipeline.sh --dry-run      comprueba y muestra los sbatch; no envía nada
#   bash run_pipeline.sh                lo envía todo
#   bash run_pipeline.sh --skip-data    igual, sin la fase de datos (ya hecha)
#   bash run_pipeline.sh --solo-datos   sólo la fase de datos + el digest (funnels):
#                                       para revisar un cambio del método antes de la GPU
#
# Desde la raíz de la carpeta, en una terminal SSH del login node. Al terminar
# queda UN tarball, data/03_processed/run_bundle_<TAG>_<fecha>.tgz, que se baja
# a la máquina local y se procesa con el comando 2 (`reproduce_tables.py --run`).
#
# ── QUÉ ENCOLA ───────────────────────────────────────────────────────────────
# Con la vuelta (rama `vuelta`, docs/PREREGISTRO_VUELTA.md): 35 rutas, una leg
# `T0->T0` por ruta.
#   job        recurso tareas  qué hace                                  → para
#   datos      CPU      35     pings crudos → df_rec_full por ruta          todo lo demás
#   xgb        CPU      35     frame de evaluación + baselines + XGBoost    Tablas 2-4, 7, S11-S13, S16, S17
#   ablacion   CPU      35     XGBoost sin prior / solo trayectoria         Tabla 5
#   escalera   CPU      35     ventana expansiva por mes (frames `_tw`)     Tablas 6, S15 · Fig. 3
#   lstm_es15  GPU      35     LSTM, early stopping honesto                 Tablas 2-5, 7, S11, S12, S16, S17 · Figs. S2, S3
#   lstm_tw    GPU      35     LSTM sobre la escalera                       Tablas 6, S15 · Fig. 3
#   digest     CPU       1     funnel por ruta                              Tablas 1, S2
#   flota      CPU       1     concentración de flota                       Tabla S2
#   apendices  CPU       1     representatividad de la malla                Tabla S3
#   paquete    CPU       1     todo lo anterior → un .tgz                   (comando 2)
#
# Ya no se encola el LSTM de julio (`es0`, early stopping sobre el fold
# evaluado, ~70 h-GPU): sólo alimentaba la Tabla S16 y las Figs. S2-S3, que pasan
# a `es15` (pre-registro, decisión P5). La corrida de agosto se reproduce con el
# `run_pipeline.sh` de la rama `metodo-v2`.
#
# ── EL ENCADENADO ES POR RUTA ────────────────────────────────────────────────
# `--dependency=aftercorr`: la tarea k de un array arranca cuando la tarea k del
# anterior terminó BIEN. En todos los arrays el índice k es la MISMA ruta, así
# que la 101 entra a la GPU en cuanto tiene frames, sin esperar a las demás. Si
# una ruta falla, sus tareas posteriores se cancelan (--kill-on-invalid-dep):
# sin frames, el LSTM saltaría todas las legs y saldría EN VERDE sin entrenar.
#
# ── SI ALGO FALLA ────────────────────────────────────────────────────────────
# Se vuelve a lanzar el MISMO comando (con --skip-data si los datos salieron):
# todo es reanudable y lo ya calculado se salta como `cached`. Excepción: un
# parquet CORRUPTO se da por hecho para siempre; el job `paquete` los lista (rc=5)
# y hay que BORRARLOS antes de relanzar.
#
# ── NUNCA DESDE JUPYTER ──────────────────────────────────────────────────────
# Un Jupyter de OnDemand es un job de Slurm y hereda variables SLURM_* que
# pisan las directivas #SBATCH (memoria, cpus, partición). Este script las
# limpia, pero un `sbatch` suelto desde una celda no.
#
# Todos los parámetros (intérprete, proyecto, particiones, régimen) salen de
# ./config.env — ese es el archivo a editar.
# =====================================================================
set -uo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" \
    || { echo "✗ no pude situarme en la raíz de la corrida"; exit 2; }
[ -d jobs ] && [ -d func ] || {
    echo "✗ no veo ./jobs y ./func — ¿copiaste la carpeta entera?"; exit 2; }

_inherited=$(compgen -v | grep -E '^(SLURM|SBATCH)_' || true)
if [ -n "$_inherited" ]; then
    echo "· limpiando $(echo "$_inherited" | wc -l) variables SLURM_*/SBATCH_* heredadas"
    for _v in $_inherited; do unset "$_v"; done
fi

. ./config.env
# Con LANG=C, un print("✓") de python revienta con UnicodeEncodeError.
export PYTHONIOENCODING=utf-8

DRY=0; SKIP_DATA=0; SOLO_DATOS=0
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run)    DRY=1; shift ;;
        --skip-data)  SKIP_DATA=1; shift ;;
        --solo-datos) SOLO_DATOS=1; shift ;;
        -h|--help)   sed -n '2,14p' "$0"; exit 0 ;;
        *) echo "✗ opción desconocida: $1  (--help para la lista)"; exit 2 ;;
    esac
done

# Proyecto: obligatorio en el clúster desde el 2026-09-07. Vacío ⇒ no se pasa y
# manda el `#SBATCH --account` de cada .slrm.
ACCT=(); [ -n "${ACCOUNT:-}" ] && ACCT=(--account="$ACCOUNT")

DATA_SLRM=jobs/job_data_pipeline.slrm
CPU_SLRM=jobs/job_xgb_prediction.slrm
GPU_SLRM=jobs/job_lstm_frame.slrm

say() { printf '%s\n' "$*"; }
die() { printf '✗ %s\n' "$*" >&2; exit 1; }

# ═════════════════════════════════════════════════════════════════════════════
#  CHEQUEOS PREVIOS — todos ANTES de encolar nada
# ═════════════════════════════════════════════════════════════════════════════
say "══ chequeos previos ═══════════════════════════════════════════════"

# 1 · CRLF. sbatch RECHAZA un script con \r\n. Pasa cuando la carpeta viaja por
#     Windows. (awk y no `grep $'\r'`: un CR crudo en argv se mangla bajo MSYS.)
_crlf=$(find jobs func . -maxdepth 2 -type f \
            \( -name '*.sh' -o -name '*.slrm' -o -name '*.env' \) \
            -exec awk '/\r$/{print FILENAME; exit}' {} \; 2>/dev/null | sort -u)
if [ -n "$_crlf" ]; then
    say "✗ estos ficheros tienen saltos de línea DOS (\\r\\n) y sbatch los rechaza:"
    printf '%s\n' "$_crlf" | sed 's/^/    /'
    say "  arréglalo con:"
    say "    find . -type f \\( -name '*.sh' -o -name '*.slrm' -o -name '*.py' -o -name '*.env' \\) -exec sed -i 's/\\r\$//' {} +"
    exit 2
fi
say "✓ sin CRLF"

# 2 · El intérprete, por RUTA ABSOLUTA. `conda activate` y `source activate`
#     fallan en silencio en los nodos: el LSTM acaba en CPU y el job sale verde.
[ -x "$PY" ] || die "PY no es ejecutable: '$PY'
  Si es el .venv de esta carpeta y aún no existe:  bash setup_venv.sh
  Si usas otro intérprete, pon su RUTA ABSOLUTA en config.env (ver ENVIRONMENT.md)."
say "✓ python  : $PY"
"$PY" -c "import sys; print('  versión :', sys.version.split()[0])" \
    || die "'$PY' no arranca"

# 3 · Dependencias, todas de una vez: es más barato enterarse ahora.
"$PY" - <<'PYCHK' || die "faltan dependencias (arriba el detalle)"
import importlib.util, sys
req = ["polars", "numpy", "scipy", "pyarrow", "h3", "matplotlib",
       "xgboost", "sklearn"]
falta = [m for m in req if not importlib.util.find_spec(m)]
if falta:
    print("  ✗ faltan:", ", ".join(falta))
    print("    instala con el MISMO intérprete:  <PY> -m pip install -r requirements-server.txt")
    sys.exit(1)
print("  ✓ deps CPU completas")
try:
    import tensorflow as tf
    print(f"  ✓ tensorflow {tf.__version__} (la fase GPU comprueba la GPU en el nodo)")
except Exception as e:
    print(f"  ⚠ tensorflow NO importa aquí ({type(e).__name__}). En un login node")
    print("    sin ROCm es NORMAL; el job de GPU aborta solo si el nodo tampoco.")
PYCHK

# 4 · Datos crudos. Sin esto la fase 0 muere ruta por ruta.
if [ "$SKIP_DATA" = "0" ]; then
    N_ROUTES=$(find data/01_raw/records -mindepth 2 -maxdepth 2 \
                    -name 'df_rec_raw_*.parquet' 2>/dev/null \
               | sed 's|.*/records/\([^/]*\)/.*|\1|' | sort -u | wc -l)
    [ "$N_ROUTES" -gt 0 ] || die "no hay extractos crudos en data/01_raw/records/
  Se esperan data/01_raw/records/<ruta>/df_rec_raw_<ruta>_<rango>.parquet
  y los cinco ficheros de geometría en data/01_raw/ (ver data/01_raw/README.md)."
    for f in polylines.geojson segments.geojson segments_route.json stops.json stops_routes.json; do
        [ -s "data/01_raw/$f" ] || die "falta data/01_raw/$f (geometría de la red)"
    done
    say "✓ datos   : $N_ROUTES rutas con extracto crudo + geometría completa"
else
    [ -d data/02_interim/routes ] || die "--skip-data pero no existe data/02_interim/routes"
    say "· datos   : --skip-data (se reusan los df_rec_full de data/02_interim)"
fi

# 4b · La vuelta (TRIP_CUT=cruce). Una ruta que no esté en TERMINALS_FILE cae EN
#      SILENCIO al corte por dwell (load_terminal_coords: «la ruta no está → dwell»),
#      o sea, otro método. Se exige un punto de corte por cada ruta con crudo.
if [ "${TRIP_CUT:-}" = "cruce" ]; then
    [ "${TERMINAL_RULE:-}" = "file" ] || die "TRIP_CUT=cruce necesita TERMINAL_RULE=file (config.env §7)"
    [ -s "${TERMINALS_FILE:-}" ] || die "TRIP_CUT=cruce pero no encuentro TERMINALS_FILE='${TERMINALS_FILE:-}'"
    "$PY" - <<'PYCHK' || die "el corte de la vuelta no está bien configurado (arriba el detalle)"
import json, os, pathlib, sys
cortes = json.loads(pathlib.Path(os.environ["TERMINALS_FILE"]).read_text(encoding="utf-8"))
rec = pathlib.Path("data/01_raw/records")
rutas = sorted(p.name for p in rec.iterdir()
               if p.is_dir() and any(p.glob("df_rec_raw_*.parquet"))) if rec.is_dir() else []
falta = [r for r in rutas if r not in cortes]
malos = [r for r in rutas if r in cortes and not (len(cortes[r]) >= 1 and all(len(p) == 3 for p in cortes[r]))]
if falta or malos:
    if falta: print("  ✗ rutas con crudo sin punto de corte:", " ".join(falta))
    if malos: print("  ✗ puntos que no son [lat, lon, km]:", " ".join(malos))
    sys.exit(1)
eje = os.environ.get("GRID_AXIS", "recorrido")
if eje not in ("recorrido", "traza"):
    print(f"  ✗ GRID_AXIS={eje!r}: va con recorrido | traza"); sys.exit(1)
n1 = sum(1 for r in rutas if len(cortes[r]) == 1)
print(f"✓ corte   : por cruce de km · {len(rutas)} rutas con su punto "
      f"({n1} con uno: la vuelta) · malla sobre {'la traza' if eje == 'traza' else 'la distancia recorrida'}")
PYCHK
fi

# 5 · Particiones y GPU, antes de encolar: un nombre equivocado haría fallar
#     el sbatch de la GPU con la CPU ya en cola.
if command -v sinfo >/dev/null 2>&1; then
    _parts="$(sinfo -h -o '%R' 2>/dev/null | sort -u | tr '\n' ' ')"
    if [ -n "$_parts" ]; then
        for _p in "$PARTITION_CPU" "$PARTITION_GPU"; do
            case " $_parts " in
                *" $_p "*) ;;
                *) die "la partición '$_p' no existe en este clúster.
  Particiones disponibles: $_parts
  Corrige PARTITION_CPU / PARTITION_GPU en config.env." ;;
            esac
        done
        say "✓ colas   : $PARTITION_CPU (CPU) · $PARTITION_GPU (GPU)"
    fi
    _gres_want="$(printf '%s' "$GPU_GRES" | cut -d: -f2)"
    case "$_gres_want" in
        ''|[0-9]*) ;;
        *)
            _gres_have="$(sinfo -h -o '%G' 2>/dev/null | sort -u | tr '\n' ' ')"
            case " $_gres_have " in
                *"$_gres_want"*) ;;
                *) say "⚠ pediste --gres=$GPU_GRES pero no veo '$_gres_want' en los nodos: $_gres_have" ;;
            esac ;;
    esac
else
    say "· sin sinfo en el PATH: no puedo validar las particiones"
fi

# 6 · ¿Mapean igual índice → ruta los tres scripts de array? Si no, encadenar
#     por ruta emparejaría una ruta con los datos de OTRA: se cae a `afterany`
#     sobre el array entero (más lento, igual de seguro).
_routes_of() {
    sed -n '/^ROUTES=(/,/)/p' "$1" | tr -d '\\\\' \
        | sed -e 's/^ROUTES=(//' -e 's/)//' | tr -s ' \t\n' ' ' \
        | sed -e 's/^ *//' -e 's/ *$//'
}
CPU_ROUTES="$(_routes_of "$CPU_SLRM")"
GPU_ROUTES="$(_routes_of "$GPU_SLRM")"
N_CPU=$(printf '%s' "$CPU_ROUTES" | wc -w)
# Si la fase 0 lleva rutas de más al final (en agosto, la 902 en el índice 26), se
# comparan sus primeros N_CPU. Con la vuelta las tres listas son las mismas 35.
DATA_HEAD="$(_routes_of "$DATA_SLRM" | cut -d' ' -f1-"$N_CPU")"
DEP_MODE="aftercorr"
if [ "$DATA_HEAD" != "$CPU_ROUTES" ] || [ "$CPU_ROUTES" != "$GPU_ROUTES" ]; then
    DEP_MODE="afterany"
    say "⚠ los arrays NO mapean igual índice → ruta: se encadena por array entero"
else
    say "✓ arrays  : las 3 listas mapean igual índice → ruta"
fi

mkdir -p logs route_reports/xgb      # sbatch no crea el directorio de --output

# ═════════════════════════════════════════════════════════════════════════════
#  ENVÍO
# ═════════════════════════════════════════════════════════════════════════════
export PY MODE TERMINALS RANGE BUS_ANALYSIS_RANGE FORCE RAW_DATE_RANGE DATA_ROOT

XGB=(--n-route-quantile "$Q" --min-coverage "$COV" --max-trips "$MAX_TRIPS" --exo)
CPU=(--partition="$PARTITION_CPU")
GPU=(--partition="$PARTITION_GPU" --gres="$GPU_GRES")

say ""
say "══ $([ "$DRY" = 1 ] && echo 'plan (--dry-run: no se envía nada)' || echo 'envío') ═══════════════════════════════════════════"
say "  proyecto: ${ACCOUNT:-(el #SBATCH --account de cada .slrm)}"
say "  régimen : q=$Q  cov=$COV  max_trips=$MAX_TRIPS  →  tag $TAG   ·   $RANGE"
say ""

IDS=()
# submit <etiqueta> <args de sbatch…>  → deja el job id en $JID. Si un sbatch
# falla se cancela lo ya encolado: así relanzar el comando nunca duplica jobs.
submit() {
    local label="$1"; shift
    if [ "$DRY" = "1" ]; then
        JID="<$label>"; IDS+=("$JID")
        say "  $(printf '%-10s' "$label") sbatch ${ACCT[*]-} $*"
        return 0
    fi
    JID="$(sbatch --parsable "${ACCT[@]+"${ACCT[@]}"}" "$@")"
    JID="${JID%%;*}"                       # --parsable puede dar "id;cluster"
    if [ -z "$JID" ]; then
        say "✗ falló el sbatch de '$label'."
        if [ "${#IDS[@]}" -gt 0 ]; then
            scancel "${IDS[@]}" 2>/dev/null
            say "  cancelado lo ya encolado (${IDS[*]}) para no dejar media cadena."
        fi
        exit 1
    fi
    IDS+=("$JID")
    say "  ✓ $(printf '%-10s' "$label") → job $JID"
}
# after <job id>  → DEP con el encadenado por ruta (vacío si no hay job previo)
after() {
    DEP=()
    [ -n "${1:-}" ] && DEP=(--dependency="$DEP_MODE:$1" --kill-on-invalid-dep=yes)
}

D=""
if [ "$SKIP_DATA" = "0" ]; then
    submit datos "${CPU[@]}" --mem="$MEM_DATA" --cpus-per-task="$CPUS_DATA" \
        --time="$TIME_DATA" --array="$DATA_ARRAY" "$DATA_SLRM"
    D="$JID"
fi

# --solo-datos: los funnels (Tabla 1) y nada más, para revisar el método antes de
# gastar GPU. Después, `bash run_pipeline.sh --skip-data` encola el resto.
if [ "$SOLO_DATOS" = "1" ]; then
    [ -n "$D" ] || die "--solo-datos con --skip-data no encola nada"
    submit digest --dependency="afterany:$D" "${CPU[@]}" jobs/job_report_pack.slrm --mode digest
    [ "$DRY" = "1" ] && { say ""; say "(--dry-run: no se envió nada)"; exit 0; }
    cat <<EOF

══ y ahora ════════════════════════════════════════════════════════

  Vigilar:   squeue -u "\$USER" -o '%.10i %.8a %.14j %.8T %.10M %.20E'
  Cancelar:  scancel ${IDS[*]}

  Al terminar, los funnels por ruta (con el método en "method" y el paso del
  orden cíclico en "trips_valid_cyclic"):
      tar -czf funnels_v2.tgz data/02_interim/routes/*/perf/funnel_*.json route_reports/analysis_digest.json
  y en tu máquina:  scp <usuario>@<servidor>:$PWD/funnels_v2.tgz .
  Si convencen:     bash run_pipeline.sh --skip-data
EOF
    exit 0
fi

after "$D"
submit xgb      "${DEP[@]+"${DEP[@]}"}" "${CPU[@]}" --array="$CPU_ARRAY" "$CPU_SLRM" "${XGB[@]}"
X="$JID"
# --no-frames: esta corrida reconstruye el MISMO frame que leen los LSTM, y
# reescribirlo mientras lo leen los rompe. El frame se calcula igual; solo no
# se escribe, así que ningún número cambia.
submit ablacion  "${DEP[@]+"${DEP[@]}"}" "${CPU[@]}" --array="$CPU_ARRAY" "$CPU_SLRM" "${XGB[@]}" \
    --no-prior --trip-only --no-frames
submit escalera  "${DEP[@]+"${DEP[@]}"}" "${CPU[@]}" --array="$CPU_ARRAY" "$CPU_SLRM" "${XGB[@]}" \
    --split expanding
XT="$JID"

# Sin --directional-only: con la vuelta la única leg de cada ruta es `T0->T0`, y ese
# filtro las saltaba todas (run_frame.py sale con rc=4 si no calcula ninguna).
after "$X"
submit lstm_es15 "${DEP[@]+"${DEP[@]}"}" "${GPU[@]}" --array="$GPU_ARRAY" "$GPU_SLRM" \
    --require-frame --es-frac 0.15
after "$XT"
submit lstm_tw   "${DEP[@]+"${DEP[@]}"}" "${GPU[@]}" --array="$GPU_ARRAY" "$GPU_SLRM" \
    --require-frame --split expanding

# Los auxiliares leen dataframes, no predicciones: esperan a la fase 0 entera.
AUX=(); [ -n "$D" ] && AUX=(--dependency="afterany:$D")
submit digest    "${AUX[@]+"${AUX[@]}"}" "${CPU[@]}" jobs/job_report_pack.slrm --mode digest
submit flota     "${AUX[@]+"${AUX[@]}"}" "${CPU[@]}" jobs/job_fleet_concentration.slrm
submit apendices "${AUX[@]+"${AUX[@]}"}" "${CPU[@]}" jobs/job_appendices.slrm

# `afterany` y no `afterok` a propósito: si una ruta falla de madrugada se
# quiere el paquete con lo que SÍ salió; su RUNINFO dice qué falta.
submit paquete --dependency="afterany$(printf ':%s' "${IDS[@]}")" "${CPU[@]}" \
    --export="ALL,TAG=$TAG,RANGE=$RANGE" jobs/job_pack.slrm
PACK="$JID"

[ "$DRY" = "1" ] && { say ""; say "(--dry-run: no se envió nada)"; exit 0; }

cat <<EOF

══ y ahora ════════════════════════════════════════════════════════

  Vigilar (la columna E es la dependencia pendiente):
    squeue -u "\$USER" -o '%.10i %.8a %.14j %.8T %.10M %.20E'

  Cancelar todo:
    scancel ${IDS[*]}

  Al terminar:
    sacct -j $PACK --format=JobID,JobName,State,ExitCode
    tail -30 logs/pack_$PACK.out      ← dónde quedó el .tgz y cómo bajarlo
        rc=5 ⇒ parquet CORRUPTOS: bórralos y relanza con --skip-data
        rc=4 ⇒ algún _overall sin _devhist/_by_horizon, o un funnel más viejo
               que su df_final (ver RUNINFO en el .tgz)
EOF
