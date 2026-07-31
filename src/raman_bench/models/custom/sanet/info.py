from __future__ import annotations

from raman_bench.models._model_info import ModelInfo
from raman_bench.models.custom.sanet.hpo import gen_sanet
from raman_bench.models.custom.sanet.model import Prep_SANET

sanet_info = ModelInfo(
    model_cls=Prep_SANET,
    search_space=gen_sanet,
    display_name="SANet",
    compute="gpu",
    reference_url="https://doi.org/10.1109/JBHI.2021.3113700",
)
