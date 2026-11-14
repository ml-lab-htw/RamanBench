"""TABABAX: rebind TabArena's own generator onto ``Prep_TABABAX`` (ag_key ``"TABABAX"``).

Model: TabAbax (TA-TABABAX). Upstream generator: ``tabarena.models.tababax.hpo``.
"""

from __future__ import annotations

from tabarena.models.tababax.hpo import gen_tababax as _upstream

from raman_bench.models.generate._tabarena_adapter import (
    rebind_tabarena_generator,
    require_available,
)
from raman_bench.preprocessing.wrapped_models import Prep_TABABAX

gen_tababax = rebind_tabarena_generator(_upstream, require_available(Prep_TABABAX, "TABABAX"))
