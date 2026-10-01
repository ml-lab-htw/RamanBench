"""REALTABPFN-V2.6: rebind TabArena's own search space onto ``Prep_REALTABPFN_V26``.

Uses ``tabarena.models.tabpfnv2_5.hpo.gen_tabpfnv26`` -- yes, under the ``tabpfnv2_5``
subpackage: TabArena bundles both ``RealTabPFNv25Model``/``gen_realtabpfnv25`` and
``TabPFNv26Model``/``gen_tabpfnv26`` in the same ``hpo.py``, this isn't a typo.
``gen_tabpfnv26``'s search space is empty (``{}``, manual-config-only) -- same
convention TabArena itself uses for ``DUMMY``/``MITRA``/``REALTABPFN-V2`` (see
``realtabpfn_v2.py``), not a bespoke no-HPO fix: ``configs/v1/scope_default.json``'s
``num_random_configs: 0`` is what actually keeps the real routine sweep safe from
``ExhaustedSearchSpaceError`` (an empty search space asked for N>0 random configs
raises). A direct ``run_experiment.py`` invocation that doesn't also pass
``--num-random-configs 0`` will hit the same error -- deliberately not papered over
here, matching how its empty-search-space siblings already behave.

``TabPFNv26Model`` is one of the optional foundation-model classes that may be
missing from a given AutoGluon build (see ``wrapped_models.py``'s
``_OPTIONAL_AG_MODEL_NAMES``); ``require_available`` turns that into a clear
error instead of a confusing one deep inside ``tabarena``.
"""

from __future__ import annotations

from tabarena.models.tabpfnv2_5.hpo import gen_tabpfnv26 as _upstream

from raman_bench.models.generate._tabarena_adapter import (
    rebind_tabarena_generator,
    require_available,
)
from raman_bench.preprocessing.wrapped_models import Prep_REALTABPFN_V26

gen_realtabpfn_v26 = rebind_tabarena_generator(
    _upstream, require_available(Prep_REALTABPFN_V26, "REALTABPFN-V2.6")
)
