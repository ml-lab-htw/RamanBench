from __future__ import annotations

from tabarena.utils.config_utils import ConfigGenerator

from raman_bench.models.custom.fcresnext.model import Prep_FCRESNEXT, _FCResNeXtBridge

gen_fcresnext = ConfigGenerator(
    model_cls=Prep_FCRESNEXT,
    manual_configs=[{}],
    search_space=_FCResNeXtBridge._get_default_searchspace(_FCResNeXtBridge),
)
