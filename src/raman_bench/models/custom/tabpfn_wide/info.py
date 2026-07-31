from __future__ import annotations

from raman_bench.models._model_info import ModelInfo
from raman_bench.models.custom.tabpfn_wide.hpo import gen_tabpfn_wide
from raman_bench.models.custom.tabpfn_wide.model import Prep_TABPFN_WIDE

tabpfn_wide_info = ModelInfo(
    model_cls=Prep_TABPFN_WIDE,
    search_space=gen_tabpfn_wide,
    display_name="TabPFN-Wide",
    compute="gpu",
    reference_url="https://doi.org/10.48550/arXiv.2510.06162",
    pip_extra=("tabpfnwide",),
)
