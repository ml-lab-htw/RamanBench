"""MITRA: rebind TabArena's own (empty -- Mitra has no tunable HPO surface, see
upstream's ``can_hpo=False``) search space onto ``Prep_MITRA``.

``MitraModel`` is one of the optional foundation-model classes that may be
missing from a given AutoGluon build (see ``wrapped_models.py``'s
``_OPTIONAL_AG_MODEL_NAMES``); ``require_available`` turns that into a clear
error instead of a confusing one deep inside ``tabarena``.
"""

from __future__ import annotations

from tabarena.models.mitra.hpo import gen_mitra as _upstream

from raman_bench.models.generate._tabarena_adapter import (
    rebind_tabarena_generator,
    require_available,
)
from raman_bench.preprocessing.wrapped_models import Prep_MITRA

gen_mitra = rebind_tabarena_generator(_upstream, require_available(Prep_MITRA, "MITRA"))
