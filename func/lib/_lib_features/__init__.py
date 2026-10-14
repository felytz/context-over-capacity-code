"""
_lib_features — paquete de feature engineering del paso 07.

Antes era un único módulo de ~3300 líneas; se partió por las 4 secciones que ya
existían (enrich · interp · segment_features · extended+histórico+orquestador).
Se re-exporta todo para que `from _lib_features import compute_all_features, ...`
siga funcionando sin cambios en los call sites (step_07).
"""
from .enrich import *            # noqa: F401,F403
from .interp import *            # noqa: F401,F403
from .segment_features import *  # noqa: F401,F403
from .extended import *          # noqa: F401,F403
