"""TABPACK: rebind TabArena's own generator (one default config) onto ``Prep_TABPACK``.

``TabPackModel`` lives in TabArena's package (``tabarena.models.tabpack.model``, upstream
autogluon/tabarena main). ag_key ``"TA-TABPACK"`` -> ``"TABPACK"`` (see ``wrapped_models.py``).
"""

from __future__ import annotations

from tabarena.models.tabpack.hpo import gen_tabpack as _upstream

from raman_bench.models.generate._tabarena_adapter import (
    rebind_tabarena_generator,
    require_available,
)
from raman_bench.preprocessing.wrapped_models import Prep_TABPACK

gen_tabpack = rebind_tabarena_generator(_upstream, require_available(Prep_TABPACK, "TABPACK"))
