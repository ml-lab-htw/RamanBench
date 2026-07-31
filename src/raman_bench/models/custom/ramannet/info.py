from __future__ import annotations

from raman_bench.models._model_info import ModelInfo
from raman_bench.models.custom.ramannet.hpo import gen_ramannet
from raman_bench.models.custom.ramannet.model import Prep_RAMANNET

ramannet_info = ModelInfo(
    model_cls=Prep_RAMANNET,
    search_space=gen_ramannet,
    display_name="RamanNet",
    compute="gpu",
    reference_url="https://arxiv.org/abs/2307.07312",
)
