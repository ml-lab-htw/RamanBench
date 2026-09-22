"""TABICLV2: rebind TabArena's own search space onto ``Prep_TABICLV2``.

Uses ``tabarena.models.tabicl.hpo.gen_tabiclv2`` (the ``TabICLv2Model``
generator), not ``gen_tabicl`` (the separate ``TabICLModel``/``TABICL`` key --
see ``tabicl.py`` for that one, wired onto ``Prep_TABICL``).

``TabICLv2Model`` is one of the optional foundation-model classes that may be
missing from a given tabarena build (see ``wrapped_models.py``'s
``_OPTIONAL_TABARENA_MODEL_IMPORTS``); ``require_available`` turns that into a
clear error instead of a confusing one deep inside ``tabarena``.
"""

from __future__ import annotations

from tabarena.models.tabicl.hpo import gen_tabiclv2 as _upstream

from raman_bench.models.generate._tabarena_adapter import (
    rebind_tabarena_generator,
    require_available,
)
from raman_bench.preprocessing.wrapped_models import Prep_TABICLV2

gen_tabiclv2 = rebind_tabarena_generator(_upstream, require_available(Prep_TABICLV2, "TABICLV2"))
