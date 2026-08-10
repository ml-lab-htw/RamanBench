"""TABDPT: rebind TabArena's own search space onto ``Prep_TABDPT``.

Uses ``tabarena.models.tabdpt.hpo.gen_tabdpt`` (the plain ``TabDPTModel``
generator), not ``gen_tabdpt_turbo`` (a separate ``TabDPTTurboModel``/
``TA-TABDPT-TURBO`` key RamanBench doesn't currently wrap).

``TabDPTModel`` is one of the optional foundation-model classes that may be
missing from a given AutoGluon build (see ``wrapped_models.py``'s
``_OPTIONAL_AG_MODEL_NAMES``); ``require_available`` turns that into a clear
error instead of a confusing one deep inside ``tabarena``.
"""

from __future__ import annotations

from tabarena.models.tabdpt.hpo import gen_tabdpt as _upstream

from raman_bench.models.generate._tabarena_adapter import (
    rebind_tabarena_generator,
    require_available,
)
from raman_bench.preprocessing.wrapped_models import Prep_TABDPT

gen_tabdpt = rebind_tabarena_generator(_upstream, require_available(Prep_TABDPT, "TABDPT"))
