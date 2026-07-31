from __future__ import annotations

from tabarena.utils.config_utils import ConfigGenerator

from raman_bench.models.custom.coatnet.model import Prep_COATNET, _CoAtNetBridge

gen_coatnet = ConfigGenerator(
    model_cls=Prep_COATNET,
    manual_configs=[{}],
    search_space=_CoAtNetBridge._get_default_searchspace(_CoAtNetBridge),
)
