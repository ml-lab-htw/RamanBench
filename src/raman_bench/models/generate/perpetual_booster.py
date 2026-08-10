"""PERPETUAL_BOOSTER: rebind TabArena's own search space onto ``Prep_PERPETUAL_BOOSTER``.

``PerpetualBoosterModel`` lives only in TabArena's own package
(``tabarena.models.perpetual_booster.model``) -- there is no AutoGluon-core
counterpart. See ``wrapped_models.py``'s ``_OPTIONAL_TABARENA_MODEL_IMPORTS``
block for that import and its ``ag_key`` override (``"PB"`` -> ``"PERPETUAL_BOOSTER"``).

Search space is a single categorical knob (``budget``, 5 discrete values --
straight from TabArena's own tuning notes, see ``perpetual_booster/hpo.py``'s
comment), so ``LocalRandomSearcher._get_num_configs()`` reports exactly 5 total
configs: ``gen_perpetual_booster.get_searcher_configs(n)`` raises
``ExhaustedSearchSpaceError`` for any ``n > 5`` (confirmed empirically against the
installed tabarena build). A default-config-only run (``--config-indices 0``)
never hits this -- ``--num-random-configs 0`` generates just the manual config
regardless of search-space size -- but any future HPO sweep for this model MUST
cap ``--num-random-configs`` at 5.
"""

from __future__ import annotations

from tabarena.models.perpetual_booster.hpo import gen_perpetual_booster as _upstream

from raman_bench.models.generate._tabarena_adapter import (
    rebind_tabarena_generator,
    require_available,
)
from raman_bench.preprocessing.wrapped_models import Prep_PERPETUAL_BOOSTER

gen_perpetual_booster = rebind_tabarena_generator(
    _upstream, require_available(Prep_PERPETUAL_BOOSTER, "PERPETUAL_BOOSTER")
)
