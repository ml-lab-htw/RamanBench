from __future__ import annotations

from tabarena.utils.config_utils import ConfigGenerator

from raman_bench.models.custom.pcr.model import Prep_PCR, _PCRBridge

gen_pcr = ConfigGenerator(
    model_cls=Prep_PCR,
    manual_configs=[{}],
    search_space=_PCRBridge._get_default_searchspace(_PCRBridge),
)
