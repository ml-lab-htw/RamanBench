from __future__ import annotations

from raman_bench.models._model_info import ModelInfo
from raman_bench.models.custom.causilo15.hpo import gen_causilo15, gen_causilo15_light
from raman_bench.models.custom.causilo15.model import Prep_CAUSILO15, Prep_CAUSILO15_LIGHT

causilo15_info = ModelInfo(
    model_cls=Prep_CAUSILO15,
    search_space=gen_causilo15,
    display_name="Causilo 1.5",
    compute="gpu",
    reference_url="https://github.com/nums-ai/causilo",
    pip_extra=("causilo>=1.5.0",),
)

causilo15_light_info = ModelInfo(
    model_cls=Prep_CAUSILO15_LIGHT,
    search_space=gen_causilo15_light,
    display_name="Causilo 1.5 (Light)",
    compute="gpu",
    reference_url="https://github.com/nums-ai/causilo",
    pip_extra=("causilo>=1.5.0",),
)
