from __future__ import annotations

from raman_bench.models._model_info import ModelInfo
from raman_bench.models.custom.ridge.hpo import gen_ridge
from raman_bench.models.custom.ridge.model import Prep_RIDGE

ridge_info = ModelInfo(
    model_cls=Prep_RIDGE,
    search_space=gen_ridge,
    display_name="Ridge",
    compute="cpu",
)
