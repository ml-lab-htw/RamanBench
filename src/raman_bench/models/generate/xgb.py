"""XGB (XGBoost): rebind TabArena's own search space onto ``Prep_XGB``.

See ``_tabarena_adapter.rebind_tabarena_generator`` for why this rebind is needed
instead of using ``tabarena.models.xgboost.hpo.gen_xgboost`` directly.
"""

from __future__ import annotations

from tabarena.models.xgboost.hpo import gen_xgboost as _upstream

from raman_bench.models.generate._tabarena_adapter import rebind_tabarena_generator
from raman_bench.preprocessing.wrapped_models import Prep_XGB

gen_xgb = rebind_tabarena_generator(_upstream, Prep_XGB)
