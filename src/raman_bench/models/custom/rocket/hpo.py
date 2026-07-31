from __future__ import annotations

from tabarena.utils.config_utils import ConfigGenerator

from raman_bench.models.custom.rocket.model import Prep_ROCKET, _RocketBridge

gen_rocket = ConfigGenerator(
    model_cls=Prep_ROCKET,
    manual_configs=[{}],
    search_space=_RocketBridge._get_default_searchspace(_RocketBridge),
)
