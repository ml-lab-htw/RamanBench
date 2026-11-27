"""LIMIX-2M: rebind TabArena's own (empty -- ``can_hpo=False``) search space onto ``Prep_LIMIX_2M``.

See ``wrapped_models.py`` for the checkpoint and the reasoning. Same code path as LIMIX (16M), with the 2M checkpoint and its retrieval configs.
"""

from __future__ import annotations

from tabarena.models.limix.hpo import gen_limix as _upstream

from raman_bench.models.generate._tabarena_adapter import (
    rebind_tabarena_generator,
    require_available,
)
from raman_bench.preprocessing.wrapped_models import Prep_LIMIX_2M

gen_limix_2m = rebind_tabarena_generator(_upstream, require_available(Prep_LIMIX_2M, "LIMIX-2M"))
