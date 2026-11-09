"""TABM-A: rebind TabArena's own generator onto ``Prep_TABM_A`` (ag_key ``"TABM-A"``).

Model: Yandex agentic TabM-A (TA-TABM-A). Upstream generator: ``tabarena.models.agentic_tabm.hpo``.
"""

from __future__ import annotations

from tabarena.models.agentic_tabm.hpo import gen_agentic_tabm as _upstream

from raman_bench.models.generate._tabarena_adapter import (
    rebind_tabarena_generator,
    require_available,
)
from raman_bench.preprocessing.wrapped_models import Prep_TABM_A

gen_tabm_a = rebind_tabarena_generator(_upstream, require_available(Prep_TABM_A, "TABM-A"))
