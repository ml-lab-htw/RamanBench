from __future__ import annotations

from raman_bench.models._model_info import ModelInfo
from raman_bench.models.custom.arsenal.hpo import gen_arsenal
from raman_bench.models.custom.arsenal.model import Prep_ARSENAL

arsenal_info = ModelInfo(
    model_cls=Prep_ARSENAL,
    search_space=gen_arsenal,
    display_name="ARSENAL",
    compute="cpu",
    reference_url="https://doi.org/10.1007/s10994-021-06055-9",
)
