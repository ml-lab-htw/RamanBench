"""REALTABPFN-V2: bespoke handling -- no upstream TabArena generator exists for this key.

TabArena's own package has moved past plain TabPFN v2: ``tabarena.models`` only
ships ``tabpfnv2_5`` (``RealTabPFNv25Model`` / ``TabPFNv26Model`` -- RamanBench's
``REALTABPFN-V2.5``, see ``realtabpfn_v25.py``) -- there is no
``tabarena.models.tabpfnv2`` package defining a search space for the older,
plain ``RealTabPFNv2Model`` that RamanBench's ``Prep_REALTABPFN_V2`` still wraps.
Reusing v2.5's search space isn't safe either: several of its entries (e.g.
``zip_model_path``, built from ``RealTabPFNv25Model.extra_checkpoints_for_tuning``)
are resolved against v2.5-specific checkpoint filenames that don't necessarily
exist for the plain v2 model.

Only the default config is meaningful here until someone hand-authors a real
search space for plain TabPFN v2 specifically (follow-up, not done here) -- see
``dummy.py`` for the same empty-search-space convention TabArena itself uses for
models with no (or no safely reusable) HPO search space.
"""

from __future__ import annotations

from tabarena.utils.config_utils import ConfigGenerator

from raman_bench.models.generate._tabarena_adapter import require_available
from raman_bench.preprocessing.wrapped_models import Prep_REALTABPFN_V2

gen_realtabpfn_v2 = ConfigGenerator(
    model_cls=require_available(Prep_REALTABPFN_V2, "REALTABPFN-V2"),
    manual_configs=[{}],
    search_space={},
)
