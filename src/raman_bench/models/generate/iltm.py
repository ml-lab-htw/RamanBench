"""ILTM: rebind TabArena's own randomised search space (a
``CustomAGConfigGenerator``, ``ConfigSpace``-sampled) onto ``Prep_ILTM``.

``ILTMModel`` (iLTM, https://arxiv.org/abs/2511.15941) lives only in TabArena's
own package (``tabarena.models.iltm.model``) -- there is no AutoGluon-core
counterpart. See ``wrapped_models.py``'s ``_OPTIONAL_TABARENA_MODEL_IMPORTS``
block for that import and its ``ag_key`` override (``"TA-ILTM"`` -> ``"ILTM"``).

Supports binary/multiclass/regression -- no problem-type restriction.

No ``max_features``/``max_rows``/``max_classes`` cap (confirmed via
``_get_default_auxiliary_params()`` on the installed build).

Upstream's search space (``tabarena.models.iltm.hpo.generate_configs_iltm``) has
24 real (mostly continuous/many-valued-categorical) dimensions -- confirmed to
support well over 200 unique random configs without exhaustion, unlike
PERPETUAL_BOOSTER's small all-categorical space (batch 2).

GPU-tier the same "auto-detects a GPU at fit time, falls back to CPU when none
is available" way as XRFM/ModernNCA/RealMLP -- ``ILTMModel._fit`` only raises if
more GPUs are *requested* than are actually available, not merely because none
are present, so it's listed in ``cluster/gpu_models.json``.
"""

from __future__ import annotations

from tabarena.models.iltm.hpo import gen_iltm as _upstream

from raman_bench.models.generate._tabarena_adapter import (
    rebind_tabarena_generator,
    require_available,
)
from raman_bench.preprocessing.wrapped_models import Prep_ILTM

gen_iltm = rebind_tabarena_generator(_upstream, require_available(Prep_ILTM, "ILTM"))
