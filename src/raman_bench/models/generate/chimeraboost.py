"""CHIMERABOOST: rebind TabArena's own search space onto ``Prep_CHIMERABOOST``.

``ChimeraBoostModel`` lives only in TabArena's own package
(``tabarena.models.chimeraboost.model``) -- there is no AutoGluon-core
counterpart. See ``wrapped_models.py``'s ``_OPTIONAL_TABARENA_MODEL_IMPORTS``
block for that import and its ``ag_key`` override (``"CHIMERA"`` -> ``"CHIMERABOOST"``).

CPU-only (``ChimeraBoostModel._get_default_resources`` hardcodes ``num_gpus=0``),
so unlike XRFM this one is NOT listed in ``cluster/gpu_models.json``. Search
space is a real mix of continuous/discrete knobs (13 params), confirmed to
support at least 200 unique random configs without exhaustion.
"""

from __future__ import annotations

from tabarena.models.chimeraboost.hpo import gen_chimeraboost as _upstream

from raman_bench.models.generate._tabarena_adapter import (
    rebind_tabarena_generator,
    require_available,
)
from raman_bench.preprocessing.wrapped_models import Prep_CHIMERABOOST

gen_chimeraboost = rebind_tabarena_generator(
    _upstream, require_available(Prep_CHIMERABOOST, "CHIMERABOOST")
)
