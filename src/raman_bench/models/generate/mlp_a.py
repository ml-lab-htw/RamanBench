"""MLP-A: rebind TabArena's own generator onto ``Prep_MLP_A`` (ag_key ``"MLP-A"``).

Model: Yandex agentic MLP-A (TA-MLP-A). Upstream generator: ``tabarena.models.agentic_mlp.hpo``.
"""

from __future__ import annotations

from tabarena.models.agentic_mlp.hpo import gen_agentic_mlp as _upstream

from raman_bench.models.generate._tabarena_adapter import (
    rebind_tabarena_generator,
    require_available,
)
from raman_bench.preprocessing.wrapped_models import Prep_MLP_A

gen_mlp_a = rebind_tabarena_generator(_upstream, require_available(Prep_MLP_A, "MLP-A"))
