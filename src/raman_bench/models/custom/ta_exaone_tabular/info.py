from __future__ import annotations

from raman_bench.models._model_info import ModelInfo
from raman_bench.models.custom.ta_exaone_tabular.hpo import gen_ta_exaone_tabular
from raman_bench.models.custom.ta_exaone_tabular.model import Prep_EXAONE_TABULAR

ta_exaone_tabular_info = ModelInfo(
    model_cls=Prep_EXAONE_TABULAR,
    search_space=gen_ta_exaone_tabular,
    display_name="EXAONE-Tabular",
    compute="gpu",
    reference_url="https://github.com/LGAI-Research/EXAONE-Tabular",
    # Git-only (no PyPI release) -- see requirements-models-git.txt, not
    # pyproject.toml's `[models]` extra (PyPI forbids direct-URL deps in an
    # uploaded package). Pin matches tabarena.models.exaone_tabular.info's own
    # pip_extra metadata, so the wrapped estimator is the same checkpoint/runtime
    # TabArena's own search space was evaluated against.
    pip_extra=(
        "exaonetabular @ git+https://github.com/LGAI-Research/EXAONE-Tabular.git"
        "@cf55bd2d74aeb9c0b5d5d4f509d05831251a827e",
    ),
)
