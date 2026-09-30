"""TABDPT-V1.3: rebind TabArena's own (empty -- ``can_hpo=False``, single default
config) search space onto ``Prep_TABDPT_V13``.

Uses ``tabarena.models.tabdpt.hpo.gen_tabdpt_v13`` (the ``TabDPTv13Model``
generator) -- NOT ``gen_tabdpt`` (the plain ``TabDPTModel`` v1.1 generator
``tabdpt.py`` rebinds onto the *separate* ``TABDPT`` key, a different
AutoGluon-core class; see ``wrapped_models.py``'s ``Prep_TABDPT_V13`` comment
for the full v1.1-vs-v1.3 checkpoint-compatibility investigation) and not
``gen_tabdpt_turbo`` (a separate ``TabDPTTurboModel``/v1.2 generator RamanBench
doesn't currently wrap).

``TabDPTv13Model`` is one of the optional TabArena-native foundation-model
classes that may be missing from a given tabarena build (see
``wrapped_models.py``'s ``_OPTIONAL_TABARENA_MODEL_IMPORTS``); ``require_available``
turns that into a clear error instead of a confusing one deep inside ``tabarena``.

See ``run_experiment.py::_import_generator`` for the ``model_key ->
gen_<module_key>`` naming convention this file name/symbol satisfies:
``"TABDPT-V1.3".lower().replace("-", "_").replace(".", "")`` == ``"tabdpt_v13"``.
"""

from __future__ import annotations

from tabarena.models.tabdpt.hpo import gen_tabdpt_v13 as _upstream

from raman_bench.models.generate._tabarena_adapter import (
    rebind_tabarena_generator,
    require_available,
)
from raman_bench.preprocessing.wrapped_models import Prep_TABDPT_V13

gen_tabdpt_v13 = rebind_tabarena_generator(_upstream, require_available(Prep_TABDPT_V13, "TABDPT-V1.3"))
