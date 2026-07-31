from __future__ import annotations

from raman_bench.models._model_info import ModelInfo
from raman_bench.models.custom.rocket.hpo import gen_rocket
from raman_bench.models.custom.rocket.model import Prep_ROCKET

rocket_info = ModelInfo(
    model_cls=Prep_ROCKET,
    search_space=gen_rocket,
    display_name="ROCKET",
    compute="cpu",
    reference_url="https://doi.org/10.1007/s10618-020-00701-z",
)
