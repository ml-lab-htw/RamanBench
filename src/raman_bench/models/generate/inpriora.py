"""INPRIORA: rebind TabArena's own generator (one default config) onto ``Prep_INPRIORA``.

``InprioraModel`` lives in TabArena's package (``tabarena.models.inpriora.model``, upstream
autogluon/tabarena main). ag_key ``"TA-INPRIORA"`` -> ``"INPRIORA"`` (see ``wrapped_models.py``).
"""

from __future__ import annotations

from tabarena.models.inpriora.hpo import gen_inpriora as _upstream

from raman_bench.models.generate._tabarena_adapter import (
    rebind_tabarena_generator,
    require_available,
)
from raman_bench.preprocessing.wrapped_models import Prep_INPRIORA

gen_inpriora = rebind_tabarena_generator(_upstream, require_available(Prep_INPRIORA, "INPRIORA"))
