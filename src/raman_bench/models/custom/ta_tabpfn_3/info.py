from __future__ import annotations

from raman_bench.models._model_info import ModelInfo
from raman_bench.models.custom.ta_tabpfn_3.hpo import gen_ta_tabpfn_3
from raman_bench.models.custom.ta_tabpfn_3.model import TabPFN3Model

ta_tabpfn_3_info = ModelInfo(
    model_cls=TabPFN3Model,
    search_space=gen_ta_tabpfn_3,
    display_name="TabPFN-3 (TabArena)",
    compute="gpu",
)
