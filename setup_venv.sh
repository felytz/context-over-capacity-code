#!/bin/bash
# =====================================================================
# setup_venv.sh — crea ./.venv con todo lo que el pipeline necesita
# =====================================================================
#   bash setup_venv.sh                      autodetecta el intérprete base
#   bash setup_venv.sh --python /ruta/python3.10
#   bash setup_venv.sh --no-tf              solo CPU (fases 0, 1 y 3)
#   bash setup_venv.sh --rocm 6.4.2         rama de la rueda de AMD
#   bash setup_venv.sh --recreate           borra .venv y empieza de cero
#
# ── LA ÚNICA RESTRICCIÓN QUE IMPORTA ────────────────────────────────────────
# `python -m venv` NO elige la versión de Python: hereda la del intérprete base.
# Y TensorFlow para AMD viene en una rueda con etiqueta de versión:
#
#   tensorflow_rocm-2.18.1-cp310-cp310-manylinux_2_28_x86_64.whl
#                          ^^^^^ esta, solo Python 3.10
#
# Así que el venv sirve perfectamente **si lo montas sobre un Python cuya minor
# tenga rueda**. Este script busca uno, comprueba en el repo de AMD qué ruedas
# existen para esa versión, y aborta con un mensaje claro si no cuadran — en vez
# de dejarte un venv a medias que falla en la fase 2, seis horas después.
#
# NO estás atado a 3.10: en rocm-rel-6.4.2 hay ruedas de TF 2.18.1 para cp39,
# cp310 y cp312 (pero NO cp311). El índice se consulta en vivo, así que si AMD
# publica más, este script las verá.
#
# La rueda tampoco sale de PyPI: la aloja AMD y su rama tiene que casar con el
# ROCm del sistema (los .slrm cargan `module load rocm/6.4.3`). Un
# `pip install tensorflow` a secas trae el build de CUDA, que en una MI210 no ve
# la GPU — y no da error, solo lentitud.
#
# Se corre en el LOGIN NODE: descarga de internet, y los nodos de cómputo suelen
# no tenerlo.
# =====================================================================
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" || exit 2
# Solo para ROCM_MODULE, que se usa en un mensaje. Ojo: config.env define PY
# apuntando a este mismo .venv, que todavía no existe — por eso aquí NO se usa
# esa variable, sino la que se elija como base.
[ -f config.env ] && . ./config.env
export PYTHONIOENCODING=utf-8   # con LANG=C, print("✓") desde python revienta

VENV=".venv"
BASE_PY="${BASE_PY:-}"
ROCM_REL="${ROCM_REL:-6.4.2}"     # rama de la rueda (≠ del módulo rocm/6.4.3)
TF_FALLBACK="2.18.1"              # la que corrió el paper
WITH_TF=1
RECREATE=0

while [ $# -gt 0 ]; do
    case "$1" in
        --python)   BASE_PY="$2"; shift 2 ;;
        --rocm)     ROCM_REL="$2"; shift 2 ;;
        --no-tf)    WITH_TF=0; shift ;;
        --recreate) RECREATE=1; shift ;;
        -h|--help)  sed -n '2,32p' "$0"; exit 0 ;;
        *) echo "✗ opción desconocida: $1  (--help)"; exit 2 ;;
    esac
done

hdr() { printf '\n══ %s ═══════════════════════════════════════════\n' "$*"; }
die() { printf '\n✗ %s\n' "$*" >&2; exit 1; }

# ─────────────────────────────────────────────────────────────────────────────
hdr "1 · intérprete base"

# Versiones con rueda de tensorflow_rocm, en orden de preferencia. Comprobado
# contra el índice de rocm-rel-6.4.2 (2026-08): TF 2.18.1 existe para cp39,
# cp310 y cp312 — pero NO para cp311. 3.10 va primero por ser la del paper.
# La lista es solo una heurística de arranque: quien decide es el chequeo del
# índice, más abajo.
PREF_VERSIONS="3.10 3.12 3.9"

# Candidatos. Los pythons de conda que haya en el sistema sirven perfectamente
# como BASE de un venv: lo que sale es un venv, no un env de conda.
_cands=()
[ -n "$BASE_PY" ] && _cands+=("$BASE_PY")
for _n in python3.10 python3.12 python3.9 python3.11 python3.13; do
    _p="$(command -v "$_n" 2>/dev/null)" && _cands+=("$_p")
done
for _g in ~/.conda/envs/*/bin/python /apps/miniconda3/envs/*/bin/python \
          ~/miniconda3/envs/*/bin/python ~/anaconda3/envs/*/bin/python; do
    [ -x "$_g" ] && _cands+=("$_g")
done
_p="$(command -v python3 2>/dev/null)" && _cands+=("$_p")

[ "${#_cands[@]}" -gt 0 ] || die "no encontré ningún python.
  Prueba \`module avail python\` y pásalo con --python /ruta/al/python"

# Versión de cada candidato, una sola vez (arrancar un intérprete no es gratis).
_paths=(); _vers=()
for _c in "${_cands[@]}"; do
    _v="$("$_c" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null)" || continue
    case " ${_paths[*]-} " in *" $_c "*) continue ;; esac   # sin duplicados
    _paths+=("$_c"); _vers+=("$_v")
done
[ "${#_paths[@]}" -gt 0 ] || die "ningún candidato arranca"

PICK=""; PICK_V=""
if [ -n "$BASE_PY" ]; then
    PICK="${_paths[0]}"; PICK_V="${_vers[0]}"      # --python manda
else
    for _want in $PREF_VERSIONS; do
        for _i in "${!_paths[@]}"; do
            if [ "${_vers[$_i]}" = "$_want" ]; then
                PICK="${_paths[$_i]}"; PICK_V="$_want"; break 2
            fi
        done
    done
fi
if [ -z "$PICK" ]; then
    PICK="${_paths[0]}"; PICK_V="${_vers[0]}"
    echo "⚠ ninguno tiene una versión con rueda conocida ($PREF_VERSIONS)."
    echo "  Se prueba con el primero; el chequeo del índice decide."
    echo
fi

echo "Candidatos encontrados:"
for _i in "${!_paths[@]}"; do
    _mark=" "; [ "${_paths[$_i]}" = "$PICK" ] && _mark="→"
    printf '  %s %-58s %s\n' "$_mark" "${_paths[$_i]}" "${_vers[$_i]}"
done

CP="cp$(printf '%s' "$PICK_V" | tr -d '.')"
echo
echo "Base : $PICK   (Python $PICK_V → etiqueta de rueda $CP)"

# ─────────────────────────────────────────────────────────────────────────────
# Se comprueba ANTES de crear nada: si no hay rueda para esta versión, un venv
# a medias solo sirve para descubrirlo en la fase 2, con horas de cola perdidas.
if [ "$WITH_TF" = "1" ]; then
    hdr "2 · ¿hay rueda de TensorFlow-ROCm para $CP?"
    IDX="https://repo.radeon.com/rocm/manylinux/rocm-rel-${ROCM_REL}/"
    echo "· consultando $IDX"
    WHEEL=""
    if _html="$(curl -fsSL --max-time 30 "$IDX" 2>/dev/null)"; then
        WHEEL="$(printf '%s' "$_html" \
                 | grep -oE "tensorflow_rocm-[0-9][^\"']*-${CP}-${CP}-manylinux[^\"']*\.whl" \
                 | sort -u | tail -1)"
    fi
    if [ -n "$WHEEL" ]; then
        echo "  ✓ $WHEEL"
        TF_URL="${IDX}${WHEEL}"
    elif [ "$CP" = "cp310" ]; then
        TF_URL="${IDX}tensorflow_rocm-${TF_FALLBACK}-cp310-cp310-manylinux_2_28_x86_64.whl"
        echo "  · no pude listar el índice (¿sin internet?); uso la URL exacta que"
        echo "    corrió el paper:"
        echo "    $TF_URL"
    else
        echo
        echo "✗ No hay rueda de tensorflow_rocm para Python $PICK_V en rocm-rel-$ROCM_REL."
        echo
        # Se listan las que SÍ hay: es la información que resuelve el problema,
        # y evita adivinar. (En rocm-rel-6.4.2 son cp39, cp310 y cp312 — 3.11 NO.)
        if [ -n "${_html:-}" ]; then
            echo "  Versiones de Python que SÍ tienen rueda ahí:"
            printf '%s' "$_html" \
                | grep -oE "tensorflow_rocm-[0-9][^\"']*\.whl" | sort -u \
                | sed 's/^/      /'
            echo
        fi
        echo "  Tres salidas, por orden de preferencia:"
        echo "   1) montar el venv sobre un Python de los de arriba:"
        echo "        module avail python"
        echo "        bash setup_venv.sh --python /ruta/al/python3.10"
        echo "      Si existe el env de conda del proyecto, su python vale de base:"
        echo "        bash setup_venv.sh --python /lustre/home/usuario/.conda/envs/tf-local/bin/python"
        echo "   2) probar otra rama de ROCm (mira qué hay en $IDX y en"
        echo "      https://repo.radeon.com/rocm/manylinux/):"
        echo "        bash setup_venv.sh --rocm 6.3.4"
        echo "   3) montar el venv SOLO para CPU y dejar la fase 2 (LSTM) al env"
        echo "      de conda que ya ve la GPU:"
        echo "        bash setup_venv.sh --no-tf"
        exit 1
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
hdr "3 · crear $VENV"
if [ -d "$VENV" ] && [ "$RECREATE" = "1" ]; then
    echo "· --recreate → borrando $VENV"
    rm -rf "$VENV"
fi
if [ -d "$VENV" ]; then
    echo "· ya existe; se reutiliza (--recreate para empezar de cero)"
else
    "$PICK" -m venv "$VENV" || die "no pude crear el venv con $PICK.
  En algunas distros el módulo va aparte: instala python3-venv, o usa
  --python con otro intérprete."
    echo "✓ creado con $PICK"
fi

# `bin/` en Linux, que es el destino. Si sale `Scripts/`, estás en Windows: los
# .slrm y config.env dan por hecho `bin/python`, así que no serviría.
if [ -x "$VENV/bin/python" ]; then
    PY="$(cd "$VENV/bin" && pwd)/python"
elif [ -e "$VENV/Scripts/python.exe" ]; then
    die "el venv salió con Scripts/python.exe (Windows). Esta carpeta se monta
  en el servidor Linux, donde el venv queda en .venv/bin/python — que es lo que
  esperan config.env y los .slrm. Créalo allá, no aquí.
  Borra el intento con:  rm -rf $VENV"
else
    die "no veo $VENV/bin/python tras crear el venv"
fi
"$PY" -m pip install --quiet --upgrade pip setuptools wheel \
    || die "falló el upgrade de pip"
echo "✓ pip al día"

# ─────────────────────────────────────────────────────────────────────────────
hdr "4 · dependencias de CPU"
"$PY" -m pip install --no-input -r requirements-server.txt \
    || die "falló la instalación de requirements-server.txt"
echo "✓ instaladas"

# ─────────────────────────────────────────────────────────────────────────────
if [ "$WITH_TF" = "1" ]; then
    hdr "5 · TensorFlow para AMD"
    echo "· $TF_URL"
    echo "  (son ~450 MB; tarda)"
    # La rueda pide rangos (ml-dtypes>=0.4,<1 · numpy>=1.26,<2.1 · keras>=3.5) y
    # pip toma lo más nuevo: ml-dtypes 0.6 exige numpy>=2, así que numpy subía a
    # 2.0.2 y movía knn/linreg/xgb_mae_b respecto al paper (corrida del 01-oct).
    # Se fija el resto de la pila de TF a la del env del paper (env_capture del
    # 10-ago, tf-local) y -c impide que la rueda mueva lo de requirements-server.
    TF_PINS=(ml_dtypes==0.5.4 keras==3.9.0 h5py==3.13.0 tensorboard==2.18.0
             optree==0.14.1 protobuf==4.25.6)
    "$PY" -m pip install --no-input -c requirements-server.txt \
        "$TF_URL" "${TF_PINS[@]}" \
        || die "falló la instalación de la rueda de ROCm.
  Comprueba que la URL existe y que la rama de ROCm casa con \`module avail rocm\`."
    echo "✓ instalado"
else
    hdr "5 · TensorFlow"
    echo "· --no-tf: NO se instala. Este venv sirve para las fases 0, 1 y 3."
    echo "  La fase 2 (LSTM, GPU) necesitará otro intérprete: pásaselo con"
    echo "     PY=/ruta/al/python/con/tf-rocm sbatch ... jobs/job_lstm_frame.slrm"
fi

# ─────────────────────────────────────────────────────────────────────────────
hdr "6 · comprobación"
"$PY" -c "import polars, numpy, scipy, pyarrow, h3, matplotlib, xgboost, sklearn
print('  ✓ deps CPU: polars', polars.__version__, '· numpy', numpy.__version__)" \
    || die "faltan dependencias de CPU tras instalar"

# Cada `paquete==versión` de requirements-server.txt tiene que estar TAL CUAL: un
# venv que funciona pero con otra versión no falla, solo mueve resultados.
"$PY" - <<'EOF' || die "el venv no casa con requirements-server.txt; recréalo: bash setup_venv.sh --recreate"
import re, sys
from importlib.metadata import version, PackageNotFoundError
bad = []
for line in open("requirements-server.txt", encoding="utf-8"):
    m = re.match(r"\s*([A-Za-z0-9_.-]+)==([^\s#;]+)", line)
    if not m:
        continue
    name, want = m.groups()
    try:
        have = version(name)
    except PackageNotFoundError:
        have = "(ausente)"
    if have != want:
        bad.append(f"    {name}: {have} (requirements-server.txt pide {want})")
if bad:
    print("  ✗ versiones distintas a las fijadas:", *bad, sep="\n")
    sys.exit(1)
print("  ✓ todas las versiones fijadas en requirements-server.txt coinciden")
EOF
"$PY" -m pip check >/dev/null 2>&1 \
    || { echo "  ⚠ pip check encontró dependencias rotas:"; "$PY" -m pip check | sed 's/^/      /'; }

if [ "$WITH_TF" = "1" ]; then
    # NO se silencia el stderr: la RAZÓN del fallo es lo único que distingue
    # "normal" de "roto", y son dos mensajes muy distintos.
    _tf="$("$PY" -c "import tensorflow as tf; print('  ✓ tensorflow', tf.__version__)" 2>&1)"
    if [ $? -eq 0 ]; then
        echo "$_tf"
    elif printf '%s' "$_tf" | grep -q 'libhsa-runtime64'; then
        # La runtime de ROCm solo existe en los nodos con GPU. El env que produjo
        # el paper daba EXACTAMENTE este mensaje en el login node.
        echo "  ✓ tensorflow instalado; no importa aquí por falta de ROCm."
        echo "    ImportError: libhsa-runtime64.so.1 — es NORMAL en un login node."
        echo "    Para verlo importar aquí:  module load ${ROCM_MODULE:-rocm/6.4.3}"
    else
        echo "  ⚠ tensorflow NO importa, y no es por ROCm:"
        printf '%s\n' "$_tf" | tail -3 | sed 's/^/      /'
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
hdr "listo"
cat <<EOF
  El venv está en   $PY

  config.env ya apunta ahí por defecto, así que NO hace falta tocarlo mientras
  el venv viva dentro de esta carpeta. Para fijarlo explícitamente:

      PY="$PY"

  ⚠ NUNCA hagas 'source .venv/bin/activate' antes de un sbatch. Los .slrm hacen
    'unset VIRTUAL_ENV PYTHONPATH PYTHONHOME' y llaman al python por ruta
    absoluta, que es lo correcto: un venv activo en la shell que envía el job se
    cuela por --export=ALL y puede romperle el import al intérprete del job.
    Llamar a $VENV/bin/python directamente funciona sin activar nada — el venv
    se reconoce por su pyvenv.cfg, no por la variable.

  Siguiente:
      bash run_pipeline.sh --dry-run
EOF
