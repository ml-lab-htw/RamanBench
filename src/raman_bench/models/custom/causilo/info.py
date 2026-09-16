from __future__ import annotations

from raman_bench.models._model_info import ModelInfo
from raman_bench.models.custom.causilo.hpo import gen_causilo
from raman_bench.models.custom.causilo.model import Prep_CAUSILO

causilo_info = ModelInfo(
    model_cls=Prep_CAUSILO,
    search_space=gen_causilo,
    display_name="Causilo",
    compute="gpu",
    reference_url="https://github.com/nums-ai/causilo",
    pip_extra=("causilo",),
)
