"""TABM: rebind TabArena's own search space onto ``Prep_TABM``.

``TabMModel`` is one of the optional foundation-model classes that may be missing
from a given AutoGluon build (see ``wrapped_models.py``'s
``_OPTIONAL_AG_MODEL_NAMES``); ``require_available`` turns that into a clear
error instead of a confusing one deep inside ``tabarena``.
"""

from __future__ import annotations

from tabarena.models.tabm.hpo import gen_tabm as _upstream

from raman_bench.models.generate._tabarena_adapter import (
    rebind_tabarena_generator,
    require_available,
)
from raman_bench.preprocessing.wrapped_models import Prep_TABM

gen_tabm = rebind_tabarena_generator(_upstream, require_available(Prep_TABM, "TABM"))
