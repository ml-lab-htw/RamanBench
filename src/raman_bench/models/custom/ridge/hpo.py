from __future__ import annotations

from tabarena.utils.config_utils import ConfigGenerator

from raman_bench.models.custom.ridge.model import Prep_RIDGE, _RidgeBridge

gen_ridge = ConfigGenerator(
    model_cls=Prep_RIDGE,
    manual_configs=[{}],
    search_space=_RidgeBridge._get_default_searchspace(_RidgeBridge),
)
