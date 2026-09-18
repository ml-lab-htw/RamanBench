from __future__ import annotations

from raman_bench.models._model_info import ModelInfo
from raman_bench.models.custom.ta_mitra_v2.hpo import gen_ta_mitra_v2
from raman_bench.models.custom.ta_mitra_v2.model import Prep_MITRA_V2

ta_mitra_v2_info = ModelInfo(
    model_cls=Prep_MITRA_V2,
    search_space=gen_ta_mitra_v2,
    display_name="Mitra-v2",
    compute="gpu",
    reference_url="https://arxiv.org/abs/2609.04540",
    # Matches tabarena.models.mitra_v2.info's own pip_extra metadata, so the wrapped
    # estimator is the same runtime TabArena's own search space was evaluated against.
    # flash-attn is optional (needs a prebuilt wheel: `pip install flash-attn
    # --no-build-isolation`) and not required for correctness, only for speed/memory.
    pip_extra=("autogluon.tabular[mitra]>=1.6,<1.7",),
)
