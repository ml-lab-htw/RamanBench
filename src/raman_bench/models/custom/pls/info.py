from __future__ import annotations

from raman_bench.models._model_info import ModelInfo
from raman_bench.models.custom.pls.hpo import gen_pls
from raman_bench.models.custom.pls.model import Prep_PLS

pls_info = ModelInfo(
    model_cls=Prep_PLS,
    search_space=gen_pls,
    display_name="PLS",
    compute="cpu",
    reference_url="https://doi.org/10.1137/0905052",
)
