from __future__ import annotations

from raman_bench.models._model_info import ModelInfo
from raman_bench.models.custom.ramantransformer.hpo import gen_ramantransformer
from raman_bench.models.custom.ramantransformer.model import Prep_RAMANTRANSFORMER

ramantransformer_info = ModelInfo(
    model_cls=Prep_RAMANTRANSFORMER,
    search_space=gen_ramantransformer,
    display_name="RamanTransformer",
    compute="gpu",
    reference_url="https://doi.org/10.1038/s41598-023-28730-w",
)
