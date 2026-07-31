from __future__ import annotations

from raman_bench.models._model_info import ModelInfo
from raman_bench.models.custom.ramanformer.hpo import gen_ramanformer
from raman_bench.models.custom.ramanformer.model import Prep_RAMANFORMER

ramanformer_info = ModelInfo(
    model_cls=Prep_RAMANFORMER,
    search_space=gen_ramanformer,
    display_name="RamanFormer",
    compute="gpu",
    reference_url="https://doi.org/10.1021/acsomega.3c09247",
)
