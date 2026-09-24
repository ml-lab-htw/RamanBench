"""TABPFN-V3.5: rebind TabArena's own (empty -- no tunable HPO surface, manual-config-only,
same shape as TabPFN-3) search space onto ``Prep_TABPFN_V3_5``.

Uses ``tabarena.models.tabpfn_3_5.hpo.gen_tabpfn_3_5`` (the plain ``TabPFN35Model``
generator), not ``gen_tabpfn_3_5_fast`` (a separate ``TabPFN35FastModel``/
``TABPFN-V3.5-FAST`` key RamanBench doesn't currently wrap).

``TabPFN35Model`` is one of the optional TabArena-native foundation-model classes
that may be missing from a given tabarena build (see ``wrapped_models.py``'s
``_OPTIONAL_TABARENA_MODEL_IMPORTS``); ``require_available`` turns that into a
clear error instead of a confusing one deep inside ``tabarena``.

See ``run_experiment.py::_import_generator`` for the ``model_key ->
gen_<module_key>`` naming convention this file name/symbol satisfies:
``"TABPFN-V3.5".lower().replace("-", "_").replace(".", "")`` == ``"tabpfn_v35"``.
"""

from __future__ import annotations

from tabarena.models.tabpfn_3_5.hpo import gen_tabpfn_3_5 as _upstream

from raman_bench.models.generate._tabarena_adapter import (
    rebind_tabarena_generator,
    require_available,
)
from raman_bench.preprocessing.wrapped_models import Prep_TABPFN_V3_5

gen_tabpfn_v35 = rebind_tabarena_generator(
    _upstream, require_available(Prep_TABPFN_V3_5, "TABPFN-V3.5")
)
