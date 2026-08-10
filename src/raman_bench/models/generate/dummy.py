"""DUMMY: bespoke handling -- no upstream TabArena generator exists for this key.

``DummyModel`` (``autogluon.core.models.DummyModel``) is AutoGluon-core's own
sanity-check baseline (predicts the training mean / majority class); it isn't part
of TabArena's benchmarked model roster, so there is no ``tabarena.models.dummy``
package to reuse a search space from. There is nothing to tune, so this mirrors
the same empty-search-space convention TabArena itself uses for HPO-less models
(e.g. ``tabarena.models.mitra.hpo.gen_mitra``): only the default config
(``--config-index 0``) is meaningful here.
"""

from __future__ import annotations

from tabarena.utils.config_utils import ConfigGenerator

from raman_bench.preprocessing.wrapped_models import Prep_DUMMY

gen_dummy = ConfigGenerator(
    model_cls=Prep_DUMMY,
    manual_configs=[{}],
    search_space={},
)
