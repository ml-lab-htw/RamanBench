from __future__ import annotations

from raman_bench.models._model_info import ModelInfo
from raman_bench.models.custom.rezeronet.hpo import gen_rezeronet
from raman_bench.models.custom.rezeronet.model import Prep_REZERONET

rezeronet_info = ModelInfo(
    model_cls=Prep_REZERONET,
    search_space=gen_rezeronet,
    display_name="ReZeroNet",
    compute="gpu",
)
