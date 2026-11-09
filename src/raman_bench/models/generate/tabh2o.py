"""TABH2O: rebind TabArena's own generator (one default config) onto ``Prep_TABH2O``.

``TabH2OModel`` lives in TabArena's package (``tabarena.models.tabh2o.model``, upstream
autogluon/tabarena main). ag_key ``"TA-TABH2O"`` -> ``"TABH2O"`` (see ``wrapped_models.py``).
"""

from __future__ import annotations

from tabarena.models.tabh2o.hpo import gen_tabh2o as _upstream

from raman_bench.models.generate._tabarena_adapter import (
    rebind_tabarena_generator,
    require_available,
)
from raman_bench.preprocessing.wrapped_models import Prep_TABH2O

gen_tabh2o = rebind_tabarena_generator(_upstream, require_available(Prep_TABH2O, "TABH2O"))
