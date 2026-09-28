"""APLR: rebind TabArena's own search space onto ``Prep_APLR``.

``APLRModel`` lives only in TabArena's own package
(``tabarena.models.aplr.model``) -- there is no AutoGluon-core counterpart.
See ``wrapped_models.py``'s ``_OPTIONAL_TABARENA_MODEL_IMPORTS`` block for
that import and its ``ag_key`` override (``"TA-APLR"`` -> ``"APLR"``).

CPU-only (``APLRModel.default_resources_physical_cores_only = True``, no GPU
resource declared), so not listed in ``cluster/gpu_models.json``.
"""

from __future__ import annotations

from tabarena.models.aplr.hpo import gen_aplr as _upstream

from raman_bench.models.generate._tabarena_adapter import (
    rebind_tabarena_generator,
    require_available,
)
from raman_bench.preprocessing.wrapped_models import Prep_APLR

gen_aplr = rebind_tabarena_generator(_upstream, require_available(Prep_APLR, "APLR"))
