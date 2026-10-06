from __future__ import annotations

from tabarena.utils.config_utils import ConfigGenerator

from raman_bench.models.custom.aeon.model import (
    Prep_ARSENAL,
    Prep_DRCIF,
    Prep_HIVECOTEV2,
    Prep_STC,
    Prep_TDE,
)

# Default configs only: HC2's published component settings, no search space.
gen_hivecotev2 = ConfigGenerator(model_cls=Prep_HIVECOTEV2, manual_configs=[{}], search_space={})
gen_stc = ConfigGenerator(model_cls=Prep_STC, manual_configs=[{}], search_space={})
gen_drcif = ConfigGenerator(model_cls=Prep_DRCIF, manual_configs=[{}], search_space={})
gen_arsenal = ConfigGenerator(model_cls=Prep_ARSENAL, manual_configs=[{}], search_space={})
gen_tde = ConfigGenerator(model_cls=Prep_TDE, manual_configs=[{}], search_space={})
