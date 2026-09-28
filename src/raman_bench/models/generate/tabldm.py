"""TABLDM: rebind TabArena's own search space onto ``Prep_TABLDM``.

``TabLDMModel`` lives only in TabArena's own package
(``tabarena.models.tabldm.model``) -- there is no AutoGluon-core counterpart.
See ``wrapped_models.py``'s ``_OPTIONAL_TABARENA_MODEL_IMPORTS`` block for
that import and its ``ag_key`` override (``"TA-XIAOMI-TABLDM"`` -> ``"TABLDM"``).

GPU-only (``TabLDMModel.default_num_gpus = 1``, ``minimum_num_gpus = 1``), so
listed in ``cluster/gpu_models.json``.
"""

from __future__ import annotations

from tabarena.models.tabldm.hpo import gen_tabldm as _upstream

from raman_bench.models.generate._tabarena_adapter import (
    rebind_tabarena_generator,
    require_available,
)
from raman_bench.preprocessing.wrapped_models import Prep_TABLDM

gen_tabldm = rebind_tabarena_generator(_upstream, require_available(Prep_TABLDM, "TABLDM"))
