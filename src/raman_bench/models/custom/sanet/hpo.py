from __future__ import annotations

from tabarena.utils.config_utils import ConfigGenerator

from raman_bench.models.custom.sanet.model import Prep_SANET, _SANetBridge

gen_sanet = ConfigGenerator(
    model_cls=Prep_SANET,
    manual_configs=[{}],
    search_space=_SANetBridge._get_default_searchspace(_SANetBridge),
)
