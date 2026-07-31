from __future__ import annotations

from raman_bench.models._model_info import ModelInfo
from raman_bench.models.custom.fcresnext.hpo import gen_fcresnext
from raman_bench.models.custom.fcresnext.model import Prep_FCRESNEXT

fcresnext_info = ModelInfo(
    model_cls=Prep_FCRESNEXT,
    search_space=gen_fcresnext,
    display_name="FC-ResNeXt",
    compute="gpu",
    reference_url="https://arxiv.org/abs/2402.03970v1",
)
