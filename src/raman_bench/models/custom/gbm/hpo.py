from __future__ import annotations

from tabarena.models.lightgbm.hpo import generate_configs_lightgbm
from tabarena.utils.config_utils import CustomAGConfigGenerator

from raman_bench.models.custom.gbm.model import Prep_GBM

gen_gbm = CustomAGConfigGenerator(
    model_cls=Prep_GBM,
    search_space_func=generate_configs_lightgbm,
    manual_configs=[{}],
)
