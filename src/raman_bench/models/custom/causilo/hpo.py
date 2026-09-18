from __future__ import annotations

from tabarena.utils.config_utils import ConfigGenerator

from raman_bench.models.custom.causilo.model import Prep_CAUSILO, _CausiloBridge

gen_causilo = ConfigGenerator(
    model_cls=Prep_CAUSILO,
    manual_configs=[{}],
    search_space=_CausiloBridge._get_default_searchspace(_CausiloBridge),
)
