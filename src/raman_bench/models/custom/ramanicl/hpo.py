from __future__ import annotations

from tabarena.utils.config_utils import ConfigGenerator

from raman_bench.models.custom.ramanicl.model import Prep_RAMANICL, _RamanICLBridge

gen_ramanicl = ConfigGenerator(
    model_cls=Prep_RAMANICL,
    manual_configs=[{}],
    search_space=_RamanICLBridge._get_default_searchspace(_RamanICLBridge),
)
