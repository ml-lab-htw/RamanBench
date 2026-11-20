from __future__ import annotations

from tabarena.utils.config_utils import ConfigGenerator

from raman_bench.models.custom.tabomics.model import Prep_TABOMICS, _TabOmicsBridge

gen_tabomics = ConfigGenerator(
    model_cls=Prep_TABOMICS,
    manual_configs=[{}],
    search_space=_TabOmicsBridge._get_default_searchspace(_TabOmicsBridge),
)
