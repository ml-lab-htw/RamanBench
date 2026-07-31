from __future__ import annotations

from tabarena.utils.config_utils import ConfigGenerator

from raman_bench.models.custom.ramanformer.model import Prep_RAMANFORMER, _RamanFormerBridge

gen_ramanformer = ConfigGenerator(
    model_cls=Prep_RAMANFORMER,
    manual_configs=[{}],
    search_space=_RamanFormerBridge._get_default_searchspace(_RamanFormerBridge),
)
