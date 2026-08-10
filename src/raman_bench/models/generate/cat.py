"""CAT (CatBoost): rebind TabArena's own search space onto ``Prep_CAT``.

See ``_tabarena_adapter.rebind_tabarena_generator`` for why this rebind is needed
instead of using ``tabarena.models.catboost.hpo.gen_catboost`` directly.
"""

from __future__ import annotations

from tabarena.models.catboost.hpo import gen_catboost as _upstream

from raman_bench.models.generate._tabarena_adapter import rebind_tabarena_generator
from raman_bench.preprocessing.wrapped_models import Prep_CAT

gen_cat = rebind_tabarena_generator(_upstream, Prep_CAT)
