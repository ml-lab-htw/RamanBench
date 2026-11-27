"""LIMIX2-1016: rebind TabArena's own (empty -- ``can_hpo=False``) search space onto ``Prep_LIMIX2_1016``.

See ``wrapped_models.py`` for the checkpoint and the reasoning. Runs in the same Python 3.12 image as LIMIX2.
"""

from __future__ import annotations

from tabarena.models.limix_2.hpo import gen_limix_2 as _upstream

from raman_bench.models.generate._tabarena_adapter import (
    rebind_tabarena_generator,
    require_available,
)
from raman_bench.preprocessing.wrapped_models import Prep_LIMIX2_1016

gen_limix2_1016 = rebind_tabarena_generator(_upstream, require_available(Prep_LIMIX2_1016, "LIMIX2-1016"))
