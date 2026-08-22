from __future__ import annotations

from raman_bench.models._model_info import ModelInfo
from raman_bench.models.custom.ramanpfn.hpo import gen_ramanpfn
from raman_bench.models.custom.ramanpfn.model import Prep_RAMANPFN

ramanpfn_info = ModelInfo(
    model_cls=Prep_RAMANPFN,
    search_space=gen_ramanpfn,
    display_name="RamanPFN",
    compute="gpu",
)
