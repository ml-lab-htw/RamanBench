from __future__ import annotations

from tabarena.utils.config_utils import ConfigGenerator

from raman_bench.models.custom.rezeronet.model import Prep_REZERONET, _ReZeroNetBridge

gen_rezeronet = ConfigGenerator(
    model_cls=Prep_REZERONET,
    manual_configs=[{}],
    search_space=_ReZeroNetBridge._get_default_searchspace(_ReZeroNetBridge),
)
