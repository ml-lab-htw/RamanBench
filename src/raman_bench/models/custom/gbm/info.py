from __future__ import annotations

from raman_bench.models._model_info import ModelInfo
from raman_bench.models.custom.gbm.hpo import gen_gbm
from raman_bench.models.custom.gbm.model import Prep_GBM

gbm_info = ModelInfo(
    model_cls=Prep_GBM,
    search_space=gen_gbm,
    display_name="GBM",
    compute="cpu",
)
