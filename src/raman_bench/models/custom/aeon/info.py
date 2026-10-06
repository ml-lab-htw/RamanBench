from __future__ import annotations

from raman_bench.models._model_info import ModelInfo
from raman_bench.models.custom.aeon.hpo import gen_hivecotev2
from raman_bench.models.custom.aeon.model import Prep_HIVECOTEV2

_PIP = ("aeon>=1.6.0,<1.7",)
_HC2_PAPER = "https://doi.org/10.1007/s10994-021-06057-9"

hivecotev2_info = ModelInfo(
    model_cls=Prep_HIVECOTEV2,
    search_space=gen_hivecotev2,
    display_name="HIVE-COTE 2",
    compute="cpu",
    reference_url=_HC2_PAPER,
    pip_extra=_PIP,
)
