from __future__ import annotations

from raman_bench.models._model_info import ModelInfo
from raman_bench.models.custom.pcalda.hpo import gen_pcalda
from raman_bench.models.custom.pcalda.model import Prep_PCALDA

pcalda_info = ModelInfo(
    model_cls=Prep_PCALDA,
    search_space=gen_pcalda,
    display_name="PCA-LDA",
    compute="cpu",
)
