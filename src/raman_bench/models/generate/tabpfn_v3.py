"""TABPFN-V3: rebind TabArena's own (empty -- TabPFN-3 has no tunable HPO surface,
manual-config-only) search space onto ``Prep_TABPFN_V3``.

Two different classes both happen to be named ``TabPFN3Model`` -- AutoGluon's own
port (``autogluon.tabular.models.tabpfnv2.tabpfn3_model.TabPFN3Model``) and
TabArena's own, actively-maintained implementation
(``tabarena.models.tabpfn_3.model.TabPFN3Model``, ``ag_key="TA-TABPFN-3"``). This
rebinds the latter's generator (``tabarena.models.tabpfn_3.hpo.gen_tabpfn_3``) --
the one ``Prep_TABPFN_V3`` actually subclasses (see ``wrapped_models.py``) and the
one this search space was tuned against.

``Prep_TABPFN_V3`` overrides ``ag_key`` to ``"TABPFN-V3"``, distinct from
``"TA-TABPFN-3"``, which ``raman_bench.models.custom.ta_tabpfn_3`` already owns
for an un-preprocessed TabArena baseline variant -- the two coexist as separate
registry entries, not a collision (see that module's own docstring, and
``wrapped_models.py``'s comment on ``Prep_TABPFN_V3`` for why).
"""

from __future__ import annotations

from tabarena.models.tabpfn_3.hpo import gen_tabpfn_3 as _upstream

from raman_bench.models.generate._tabarena_adapter import (
    rebind_tabarena_generator,
    require_available,
)
from raman_bench.preprocessing.wrapped_models import Prep_TABPFN_V3

gen_tabpfn_v3 = rebind_tabarena_generator(_upstream, require_available(Prep_TABPFN_V3, "TABPFN-V3"))
