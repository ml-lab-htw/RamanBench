"""TabArena's own stock TabPFN-3 wrapper (key "TA-TABPFN-3" in the registry).

Unlike the other ``models/generate/*.py`` modules, this one has **no RamanBench
preprocessing wrapper** -- it's a direct re-export of TabArena's
``gen_tabpfn_3``, used when RamanBench's own AutoGluon-native
``Prep_TABPFN_V3`` isn't available (confirmed to happen on AutoGluon builds
that don't yet ship ``autogluon.tabular.models.TabPFNv3Model`` under that
exact name -- see the defensive import in ``preprocessing/wrapped_models.py``).
This directly reuses TabArena's logic rather than reimplementing it, matching
the refactor's "use TabArena's models as much as possible" goal.
"""

from tabarena.models.tabpfn_3.generate import gen_tabpfn_3 as gen_ta_tabpfn_3

__all__ = ["gen_ta_tabpfn_3"]
