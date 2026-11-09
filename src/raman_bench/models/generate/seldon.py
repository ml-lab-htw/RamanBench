"""SELDON: rebind TabArena's own generator onto ``Prep_SELDON`` (ag_key ``"SELDON"``).

Model: Seldon (TA-SELDON). Upstream generator: ``tabarena.models.seldon.hpo``.
"""

from __future__ import annotations

from tabarena.models.seldon.hpo import gen_seldon as _upstream

from raman_bench.models.generate._tabarena_adapter import (
    rebind_tabarena_generator,
    require_available,
)
from raman_bench.preprocessing.wrapped_models import Prep_SELDON

gen_seldon = rebind_tabarena_generator(_upstream, require_available(Prep_SELDON, "SELDON"))
