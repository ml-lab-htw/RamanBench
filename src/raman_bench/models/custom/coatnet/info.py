from __future__ import annotations

from raman_bench.models._model_info import ModelInfo
from raman_bench.models.custom.coatnet.hpo import gen_coatnet
from raman_bench.models.custom.coatnet.model import Prep_COATNET

coatnet_info = ModelInfo(
    model_cls=Prep_COATNET,
    search_space=gen_coatnet,
    display_name="CoAtNet",
    compute="gpu",
    reference_url="https://arxiv.org/abs/2106.04803",
)
