"""PRISMBOOST: rebind TabArena's own search space onto ``Prep_PRISMBOOST``.

``PrismBoostModel`` (https://github.com/PrismBoost/PrismBoost, MIT) lives only in
TabArena's package (``tabarena.models.prismboost.model``, autogluon/tabarena#629, on our
pin as a cherry-pick -- see ``requirements-tabarena-git.txt``). Config 0 is upstream's
default (``manual_configs=[{}]``: PrismBoost's "auto" capacity rules, round count by early
stopping); the search space is upstream's, for an HPO sweep. CPU-only.
"""

from __future__ import annotations

from tabarena.models.prismboost.hpo import gen_prismboost as _upstream

from raman_bench.models.generate._tabarena_adapter import (
    rebind_tabarena_generator,
    require_available,
)
from raman_bench.preprocessing.wrapped_models import Prep_PRISMBOOST

gen_prismboost = rebind_tabarena_generator(_upstream, require_available(Prep_PRISMBOOST, "PRISMBOOST"))
