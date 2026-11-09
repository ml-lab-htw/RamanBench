"""SELDON-MINI: rebind TabArena's own generator onto ``Prep_SELDON_MINI`` (ag_key ``"SELDON-MINI"``).

Model: Seldon-Mini (TA-SELDON-MINI). Upstream generator: ``tabarena.models.seldon.hpo``.
"""

from __future__ import annotations

from tabarena.models.seldon.hpo import gen_seldon_mini as _upstream

from raman_bench.models.generate._tabarena_adapter import (
    rebind_tabarena_generator,
    require_available,
)
from raman_bench.preprocessing.wrapped_models import Prep_SELDON_MINI

gen_seldon_mini = rebind_tabarena_generator(_upstream, require_available(Prep_SELDON_MINI, "SELDON-MINI"))
