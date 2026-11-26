from __future__ import annotations

from tabarena.utils.config_utils import ConfigGenerator

from raman_bench.models.custom.causilo15.model import (
    Prep_CAUSILO15,
    Prep_CAUSILO15_LIGHT,
    _Causilo15Bridge,
    _Causilo15LightBridge,
)

gen_causilo15 = ConfigGenerator(
    model_cls=Prep_CAUSILO15,
    manual_configs=[{}],
    search_space=_Causilo15Bridge._get_default_searchspace(_Causilo15Bridge),
)

gen_causilo15_light = ConfigGenerator(
    model_cls=Prep_CAUSILO15_LIGHT,
    manual_configs=[{}],
    search_space=_Causilo15LightBridge._get_default_searchspace(_Causilo15LightBridge),
)
