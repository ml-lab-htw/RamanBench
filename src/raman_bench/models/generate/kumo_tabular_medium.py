"""KUMO-TABULAR-MEDIUM: rebind TabArena's own search space onto
``Prep_KUMO_TABULAR_MEDIUM`` (the medium checkpoint, 8 estimators).

See ``kumo_tabular.py`` (the large-checkpoint sibling) for the full context --
same staging situation, same PR #625 provenance, same ``ag_key`` override
convention (``"TA-KUMO-TABULAR-MEDIUM"`` -> ``"KUMO-TABULAR-MEDIUM"``).
"""

from __future__ import annotations

from tabarena.models.kumo_tabular.hpo import gen_kumo_tabular_medium as _upstream

from raman_bench.models.generate._tabarena_adapter import (
    rebind_tabarena_generator,
    require_available,
)
from raman_bench.preprocessing.wrapped_models import Prep_KUMO_TABULAR_MEDIUM

gen_kumo_tabular_medium = rebind_tabarena_generator(
    _upstream, require_available(Prep_KUMO_TABULAR_MEDIUM, "KUMO-TABULAR-MEDIUM")
)
