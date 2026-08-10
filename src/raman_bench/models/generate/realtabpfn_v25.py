"""REALTABPFN-V2.5: rebind TabArena's own search space onto ``Prep_REALTABPFN_V25``.

Uses ``tabarena.models.tabpfnv2_5.hpo.gen_realtabpfnv25`` (the ``RealTabPFNv25Model``
generator), not ``gen_tabpfnv26`` (a separate ``TabPFNv26Model``/``TABPFN-V2.6``
key RamanBench doesn't currently wrap).

``RealTabPFNv25Model`` is one of the optional foundation-model classes that may be
missing from a given AutoGluon build (see ``wrapped_models.py``'s
``_OPTIONAL_AG_MODEL_NAMES``); ``require_available`` turns that into a clear
error instead of a confusing one deep inside ``tabarena``.
"""

from __future__ import annotations

from tabarena.models.tabpfnv2_5.hpo import gen_realtabpfnv25 as _upstream

from raman_bench.models.generate._tabarena_adapter import (
    rebind_tabarena_generator,
    require_available,
)
from raman_bench.preprocessing.wrapped_models import Prep_REALTABPFN_V25

gen_realtabpfn_v25 = rebind_tabarena_generator(
    _upstream, require_available(Prep_REALTABPFN_V25, "REALTABPFN-V2.5")
)
