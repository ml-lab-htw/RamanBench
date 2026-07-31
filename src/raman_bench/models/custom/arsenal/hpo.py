from __future__ import annotations

from tabarena.utils.config_utils import ConfigGenerator

from raman_bench.models.custom.arsenal.model import Prep_ARSENAL, _ArsenalBridge

gen_arsenal = ConfigGenerator(
    model_cls=Prep_ARSENAL,
    manual_configs=[{}],
    search_space=_ArsenalBridge._get_default_searchspace(_ArsenalBridge),
)
