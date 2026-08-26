from __future__ import annotations

from raman_bench.models._model_info import ModelInfo
from raman_bench.models.custom.hydra.hpo import gen_hydra
from raman_bench.models.custom.hydra.model import Prep_HYDRA

hydra_info = ModelInfo(
    model_cls=Prep_HYDRA,
    search_space=gen_hydra,
    display_name="Hydra",
    compute="gpu",
)
