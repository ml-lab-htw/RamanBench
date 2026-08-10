"""RF (RandomForest): rebind TabArena's own search space onto ``Prep_RF``.

See ``_tabarena_adapter.rebind_tabarena_generator`` for why this rebind is needed
instead of using ``tabarena.models.random_forest.hpo.gen_randomforest`` directly.
"""

from __future__ import annotations

from tabarena.models.random_forest.hpo import gen_randomforest as _upstream

from raman_bench.models.generate._tabarena_adapter import rebind_tabarena_generator
from raman_bench.preprocessing.wrapped_models import Prep_RF

gen_rf = rebind_tabarena_generator(_upstream, Prep_RF)
