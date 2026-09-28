"""CTBOOST: rebind TabArena's own search space onto ``Prep_CTBOOST``.

``CTBoostModel`` lives only in TabArena's own package
(``tabarena.models.ctboost.model``) -- there is no AutoGluon-core
counterpart. See ``wrapped_models.py``'s ``_OPTIONAL_TABARENA_MODEL_IMPORTS``
block for that import and its ``ag_key`` override (``"CTB"`` -> ``"CTBOOST"``).

CPU-only (``CTBoostModel.default_num_gpus = 0``), so not listed in
``cluster/gpu_models.json``.
"""

from __future__ import annotations

from tabarena.models.ctboost.hpo import gen_ctboost as _upstream

from raman_bench.models.generate._tabarena_adapter import (
    rebind_tabarena_generator,
    require_available,
)
from raman_bench.preprocessing.wrapped_models import Prep_CTBOOST

gen_ctboost = rebind_tabarena_generator(_upstream, require_available(Prep_CTBOOST, "CTBOOST"))
