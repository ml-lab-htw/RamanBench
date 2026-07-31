"""TabArena's own stock TabPFN-3 model, re-exported directly (no Raman wrapper).

Used when RamanBench's own AutoGluon-native ``Prep_TABPFN_V3`` isn't available
(confirmed to happen on AutoGluon builds that don't yet ship
``autogluon.tabular.models.TabPFNv3Model`` under that exact name -- see the
defensive import in ``preprocessing/wrapped_models.py``). This directly reuses
TabArena's logic rather than reimplementing it, matching the refactor's "use
TabArena's models as much as possible" goal. TabArena's own class already
carries ``ag_key = "TA-TABPFN-3"``, so no RamanBench-side renaming is needed.
"""

from __future__ import annotations

from tabarena.models.tabpfn_3.model import TabPFN3Model

__all__ = ["TabPFN3Model"]
