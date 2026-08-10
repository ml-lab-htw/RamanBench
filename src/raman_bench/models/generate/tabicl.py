"""TABICL: rebind TabArena's own search space onto ``Prep_TABICL``.

Uses ``tabarena.models.tabicl.hpo.gen_tabicl`` (the plain ``TabICLModel``
generator), not ``gen_tabiclv2`` (a separate ``TabICLv2Model``/``TA-TABICLv2``
key RamanBench doesn't currently wrap).

``TabICLModel`` is one of the optional foundation-model classes that may be
missing from a given AutoGluon build (see ``wrapped_models.py``'s
``_OPTIONAL_AG_MODEL_NAMES``); ``require_available`` turns that into a clear
error instead of a confusing one deep inside ``tabarena``.
"""

from __future__ import annotations

from tabarena.models.tabicl.hpo import gen_tabicl as _upstream

from raman_bench.models.generate._tabarena_adapter import (
    rebind_tabarena_generator,
    require_available,
)
from raman_bench.preprocessing.wrapped_models import Prep_TABICL

gen_tabicl = rebind_tabarena_generator(_upstream, require_available(Prep_TABICL, "TABICL"))
