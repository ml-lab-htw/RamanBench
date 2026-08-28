from __future__ import annotations

from raman_bench.models._model_info import ModelInfo
from raman_bench.models.custom.pcr.hpo import gen_pcr
from raman_bench.models.custom.pcr.model import Prep_PCR

pcr_info = ModelInfo(
    model_cls=Prep_PCR,
    search_space=gen_pcr,
    display_name="PCR",
    compute="cpu",
)
