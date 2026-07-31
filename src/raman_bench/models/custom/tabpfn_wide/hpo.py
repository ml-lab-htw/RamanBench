from __future__ import annotations

from tabarena.utils.config_utils import ConfigGenerator

from raman_bench.models.custom.tabpfn_wide.model import Prep_TABPFN_WIDE, _TabPFNWideBridge

gen_tabpfn_wide = ConfigGenerator(
    model_cls=Prep_TABPFN_WIDE,
    manual_configs=[{}],
    search_space=_TabPFNWideBridge._get_default_searchspace(_TabPFNWideBridge),
)
