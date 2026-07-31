from __future__ import annotations

from tabarena.utils.config_utils import ConfigGenerator

from raman_bench.models.custom.ramannet.model import Prep_RAMANNET, _RamanNetBridge

gen_ramannet = ConfigGenerator(
    model_cls=Prep_RAMANNET,
    manual_configs=[{}],
    search_space=_RamanNetBridge._get_default_searchspace(_RamanNetBridge),
)
