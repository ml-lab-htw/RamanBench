"""LR (LinearModel): rebind TabArena's own search space onto ``Prep_LR``.

See ``_tabarena_adapter.rebind_tabarena_generator`` for why this rebind is needed
instead of using ``tabarena.models.lr.hpo.gen_linear`` directly.
"""

from __future__ import annotations

from tabarena.models.lr.hpo import gen_linear as _upstream

from raman_bench.models.generate._tabarena_adapter import rebind_tabarena_generator
from raman_bench.preprocessing.wrapped_models import Prep_LR

gen_lr = rebind_tabarena_generator(_upstream, Prep_LR)
