from __future__ import annotations

from raman_bench.models._model_info import ModelInfo
from raman_bench.models.custom.ramanicl.hpo import gen_ramanicl
from raman_bench.models.custom.ramanicl.model import Prep_RAMANICL

ramanicl_info = ModelInfo(
    model_cls=Prep_RAMANICL,
    search_space=gen_ramanicl,
    display_name="RamanICL",
    compute="gpu",
)
