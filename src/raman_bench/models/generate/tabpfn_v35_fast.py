"""TABPFN-V3.5-FAST: rebind TabArena's own (empty -- no tunable HPO surface,
manual-config-only, same shape as TabPFN-3.5) search space onto
``Prep_TABPFN_V3_5_FAST``.

Uses ``tabarena.models.tabpfn_3_5.hpo.gen_tabpfn_3_5_fast`` (the
``TabPFN35FastModel`` generator, a sibling of ``gen_tabpfn_3_5`` -- see
``tabpfn_v35.py`` for the plain TabPFN-3.5 wrapper this one sits next to).

``TabPFN35FastModel`` is one of the optional TabArena-native foundation-model
classes that may be missing from a given tabarena build (see
``wrapped_models.py``'s ``_OPTIONAL_TABARENA_MODEL_IMPORTS``); ``require_available``
turns that into a clear error instead of a confusing one deep inside ``tabarena``.

See ``run_experiment.py::_import_generator`` for the ``model_key ->
gen_<module_key>`` naming convention this file name/symbol satisfies:
``"TABPFN-V3.5-FAST".lower().replace("-", "_").replace(".", "")`` == ``"tabpfn_v35_fast"``.
"""

from __future__ import annotations

from tabarena.models.tabpfn_3_5.hpo import gen_tabpfn_3_5_fast as _upstream

from raman_bench.models.generate._tabarena_adapter import (
    rebind_tabarena_generator,
    require_available,
)
from raman_bench.preprocessing.wrapped_models import Prep_TABPFN_V3_5_FAST

gen_tabpfn_v35_fast = rebind_tabarena_generator(
    _upstream, require_available(Prep_TABPFN_V3_5_FAST, "TABPFN-V3.5-FAST")
)
