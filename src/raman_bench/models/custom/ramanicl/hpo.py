from __future__ import annotations

from tabarena.utils.config_utils import ConfigGenerator

from raman_bench.models.custom.ramanicl.model import Prep_RAMANICL, _RamanICLBridge


class _NoHPOConfigGenerator(ConfigGenerator):
    """A generator with no random-search pool: the manual default config is the only one.

    RamanICL is a pretrained checkpoint, so there is genuinely nothing to tune per task
    -- the weights are fixed, and searching ``n_channels`` or ``max_context`` would tune
    the harness rather than the method. Its search space is therefore empty.

    An empty search space is not otherwise expressible here. ``ModelInfo`` has no "no
    HPO" flag and ``run_experiment.py`` defaults ``--num-random-configs`` to 50, so the
    base generator reaches ``get_random_searcher({})``, asks it for 50 configs from zero
    dimensions, and the searcher raises ``ExhaustedSearchSpaceError`` while building the
    pool. That happens inside ``generate_all_bag_experiments``, before any fit, so it
    fires even for ``--config-index 0``, which only ever needed the manual config --
    i.e. the default invocation of this model crashes without this override. Every other
    custom model here has a non-empty space, so nothing else hits it.

    Returning no random configs keeps the fix inside this package: callers need no
    special flag, ``--config-index 0`` resolves to the single manual config as usual,
    and nothing in the shared harness changes. ``--config-index >= 1`` then has no
    config to resolve, which is the correct behaviour for a fixed checkpoint.
    """

    def get_searcher_configs(self, num_configs: int) -> list[dict]:
        return []


gen_ramanicl = _NoHPOConfigGenerator(
    model_cls=Prep_RAMANICL,
    manual_configs=[{}],
    search_space=_RamanICLBridge._get_default_searchspace(_RamanICLBridge),
)
