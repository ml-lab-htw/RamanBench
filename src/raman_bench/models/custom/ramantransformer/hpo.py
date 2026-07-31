from __future__ import annotations

from tabarena.utils.config_utils import ConfigGenerator

from raman_bench.models.custom.ramantransformer.model import (
    Prep_RAMANTRANSFORMER,
    _RamanTransformerBridge,
)

gen_ramantransformer = ConfigGenerator(
    model_cls=Prep_RAMANTRANSFORMER,
    manual_configs=[{}],
    search_space=_RamanTransformerBridge._get_default_searchspace(_RamanTransformerBridge),
)
