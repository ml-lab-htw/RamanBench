"""TABSTAR_V1P1: rebind TabArena's own generator (one default config, ``can_hpo=False``) onto
``Prep_TABSTAR_V1P1``.

``TabSTARv1p1Model`` lives only in TabArena's package (``tabarena.models.tabstar_v1p1.model``,
autogluon/tabarena#674, on our pin as a cherry-pick -- see ``requirements-tabarena-git.txt``).
A separate model from TABSTAR (1.1.15): see ``wrapped_models.py`` for the ``ag_key`` override
(``"TA-TabSTAR-v1.1"`` -> ``"TABSTAR_V1P1"``).
"""

from __future__ import annotations

from tabarena.models.tabstar_v1p1.hpo import gen_tabstar_v1p1 as _upstream

from raman_bench.models.generate._tabarena_adapter import (
    rebind_tabarena_generator,
    require_available,
)
from raman_bench.preprocessing.wrapped_models import Prep_TABSTAR_V1P1

gen_tabstar_v1p1 = rebind_tabarena_generator(
    _upstream, require_available(Prep_TABSTAR_V1P1, "TABSTAR_V1P1")
)
