from __future__ import annotations

from raman_bench.models._model_info import ModelInfo
from raman_bench.models.custom.tabomics.hpo import gen_tabomics
from raman_bench.models.custom.tabomics.model import Prep_TABOMICS

tabomics_info = ModelInfo(
    model_cls=Prep_TABOMICS,
    search_space=gen_tabomics,
    display_name="TabOmics",
    compute="gpu",
    reference_url="https://github.com/lbotecur/tabomics",
    pip_extra=("tabomics @ git+https://github.com/lbotecur/tabomics.git",),
)
