from __future__ import annotations

from tabarena.utils.config_utils import ConfigGenerator

from raman_bench.models.custom.pls.model import Prep_PLS, _PLSBridge

gen_pls = ConfigGenerator(
    model_cls=Prep_PLS,
    manual_configs=[{}],
    search_space=_PLSBridge._get_default_searchspace(_PLSBridge),
)
