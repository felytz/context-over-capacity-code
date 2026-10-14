#!/bin/bash
# =====================================================================
# run_e6.sh — la enmienda E6 (docs/PREREGISTRO_VUELTA.md §6): dos corridas de
#             XGBoost sobre los datos de la fase 4, sólo CPU, y un paquete
# =====================================================================
#   bash run_e6.sh --dry-run    comprueba y muestra los sbatch; no envía nada
#   bash run_e6.sh              lo envía
#
# Se corre en la carpeta de la fase 4 (~/vuelta_completa), que ya tiene los
# df_rec_full de las 35 rutas, el .venv y los resultados publicados. No toca nada
# de eso: escribe archivos NUEVOS con sufijo `_ablate…`.
#
#   ablacion_s9  CPU 35  Tier A + un grupo de contexto, tope 600 (Tabla S14)
#   tope_2000    CPU 35  los niveles de la Tabla 5 y los grupos, tope 2,000 (Tabla S8)
#   paquete      CPU  1  → e6_resultados_<fecha>.tgz
#
# Las dos corridas son independientes y van a la vez. Ninguna necesita GPU.
# =====================================================================
set -uo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" \
    || { echo "✗ no pude situarme en la carpeta"; exit 2; }

_inherited=$(compgen -v | grep -E '^(SLURM|SBATCH)_' || true)
for _v in $_inherited; do unset "$_v"; done

[ -f config.env ] && [ -d jobs ] && [ -d func ] \
    || { echo "✗ aquí no hay config.env, jobs/ y func/: ¿es ~/vuelta_completa?"; exit 2; }
. ./config.env
export PYTHONIOENCODING=utf-8

DRY=0
case "${1:-}" in
    --dry-run) DRY=1 ;;
    "") ;;
    *) echo "✗ opción desconocida: $1  (sólo --dry-run)"; exit 2 ;;
esac

ACCT=(); [ -n "${ACCOUNT:-}" ] && ACCT=(--account="$ACCOUNT")
say() { printf '%s\n' "$*"; }
die() { printf '✗ %s\n' "$*" >&2; exit 1; }

say "══ chequeos ═══════════════════════════════════════════════════════"

# (sólo lo que lee bash o sbatch: a python le dan igual los \r, y varios .py van con CRLF)
_crlf=$(find jobs . -maxdepth 2 -type f \( -name '*.sh' -o -name '*.slrm' -o -name '*.env' \) \
            -exec awk '/\r$/{print FILENAME; exit}' {} \; 2>/dev/null | sort -u)
[ -z "$_crlf" ] || die "con saltos DOS (\\r\\n): $_crlf
  arréglalo con: sed -i 's/\\r\$//' <archivo>"
say "✓ sin CRLF"

[ -x "$PY" ] || die "PY no es ejecutable: '$PY' (¿existe .venv/? ver setup_venv.sh)"
say "✓ python  : $PY"

# El código nuevo: --n-estimators en los dos scripts, y los descriptivos.
grep -q -- '--n-estimators' func/xgb_prediction/run_routes.py \
    && grep -q 'def exo_descriptives' func/xgb_prediction/run_experiment.py \
    || die "func/xgb_prediction/ no es el del paquete E6: descomprime e6_servidor.tgz aquí"
[ -f E6_SHA256SUMS ] && { sha256sum -c --quiet E6_SHA256SUMS || die "los archivos de E6 no verifican"; }
say "✓ código  : el de E6"

# Los datos y los resultados publicados, ruta por ruta.
ROUTES=$(sed -n '/^ROUTES=(/,/)/p' jobs/job_xgb_prediction.slrm | tr -d '\\' \
         | sed -e 's/^ROUTES=(//' -e 's/)//' | tr -s ' \t\n' ' ')
N=0; FALTA=""
for r in $ROUTES; do
    N=$((N + 1))
    [ -s "data/02_interim/routes/$r/dataframes/df_rec_full_${RANGE}.parquet" ] \
        || FALTA="$FALTA $r(df_rec_full)"
    [ -s "data/02_interim/routes/$r/xgb/${RANGE}/legT0_T0_${TAG}_exo_overall.parquet" ] \
        || FALTA="$FALTA $r(_exo)"
done
[ -z "$FALTA" ] || die "faltan:$FALTA"
[ "$N" = 35 ] || die "jobs/job_xgb_prediction.slrm trae $N rutas, se esperan 35"
say "✓ datos   : las 35 rutas con df_rec_full y la corrida publicada (${TAG}_exo)"

YA=$(ls data/02_interim/routes/*/xgb/${RANGE}/legT0_T0_${TAG}_ablate*_overall.parquet 2>/dev/null | wc -l)
[ "$YA" = 0 ] || say "· ya hay $YA resultados _ablate: esas rutas saldrán como 'cached'"

mkdir -p logs route_reports/xgb

COMMON=(--n-route-quantile "$Q" --min-coverage "$COV" --max-trips "$MAX_TRIPS"
        --exo-ablate --no-baselines --no-frames --timeout-min 600)
# 16 hilos, como la corrida publicada (`omp_threads` = 16 en sus `*_timing`): XGBoost
# suma histogramas en paralelo, y con otro número de hilos los árboles cambian en el
# último dígito (probado en local: 8 contra 12 hilos, |ΔMAE| hasta 0.14 s; con los
# mismos, 0). make_table_11_ablation.py exige |ΔMAE| = 0 contra la publicada.
CPU=(--partition="$PARTITION_CPU" --cpus-per-task=16)

say ""
say "══ $([ "$DRY" = 1 ] && echo 'plan (--dry-run)' || echo 'envío') ═══════════════════════════════════════"
IDS=()
submit() {
    local label="$1"; shift
    if [ "$DRY" = 1 ]; then
        JID="<$label>"; IDS+=("$JID")
        say "  $(printf '%-12s' "$label") sbatch ${ACCT[*]-} $*"; return 0
    fi
    JID="$(sbatch --parsable "${ACCT[@]+"${ACCT[@]}"}" "$@")"; JID="${JID%%;*}"
    if [ -z "$JID" ]; then
        say "✗ falló el sbatch de '$label'"
        [ "${#IDS[@]}" -gt 0 ] && scancel "${IDS[@]}" 2>/dev/null
        exit 1
    fi
    IDS+=("$JID"); say "  ✓ $(printf '%-12s' "$label") → job $JID"
}

submit ablacion_s9 "${CPU[@]}" --array="$CPU_ARRAY" --job-name=e6_s9 \
    jobs/job_xgb_prediction.slrm "${COMMON[@]}"
A="$JID"
submit tope_2000 "${CPU[@]}" --array="$CPU_ARRAY" --job-name=e6_n2000 \
    jobs/job_xgb_prediction.slrm "${COMMON[@]}" --no-prior --trip-only --n-estimators 2000
B="$JID"

STAMP=$(date +%Y%m%d_%H%M)
OUT="data/03_processed/e6_resultados_${STAMP}.tgz"
submit paquete "${CPU[@]}" --dependency="afterany:$A:$B" --job-name=e6_pack \
    --time=00:30:00 --mem=4G --cpus-per-task=1 --output="logs/e6_pack_%j.out" \
    --wrap="mkdir -p data/03_processed && \
      sacct -j $A,$B --format=JobID%22,JobName%16,State,Elapsed,MaxRSS,ReqMem > logs/e6_sacct.txt; \
      ls data/02_interim/routes/*/xgb/${RANGE}/legT0_T0_${TAG}_ablate*  > logs/e6_lista.txt; \
      tar -czf $OUT -T logs/e6_lista.txt logs/e6_sacct.txt logs/xgb_*_${A}_* logs/xgb_*_${B}_* \
          route_reports/xgb/xgb_status_${RANGE}_${TAG}_ablate*.json E6_SHA256SUMS 2>/dev/null; \
      echo; echo \"rutas con S9 : \$(grep -c '_ablate_overall' logs/e6_lista.txt) de 35\"; \
      echo \"rutas con 2000: \$(grep -c '_ablate_np_t0_n2000_overall' logs/e6_lista.txt) de 35\"; \
      echo \"paquete: \$PWD/$OUT\"; ls -la $OUT; sha256sum $OUT"

[ "$DRY" = 1 ] && { say ""; say "(--dry-run: no se envió nada)"; exit 0; }

cat <<EOF

══ y ahora ════════════════════════════════════════════════════════

  Vigilar:   squeue -u "\$USER" -o '%.10i %.10j %.8T %.10M %.20E'
  Cancelar:  scancel ${IDS[*]}

  Al terminar (unas horas):
    cat logs/e6_pack_${IDS[2]}.out        ← «35 de 35» dos veces, y dónde quedó el .tgz
  y en tu máquina, dentro de hermosillo-bus-arrival-paper/artefacts/:
    scp <usuario>@<servidor>:$PWD/$OUT .
EOF
