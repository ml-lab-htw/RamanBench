from __future__ import annotations

from raman_bench.models._model_info import ModelInfo
from raman_bench.models.custom.aeon.hpo import (
    gen_arsenal,
    gen_drcif,
    gen_hivecotev2,
    gen_stc,
    gen_tde,
)
from raman_bench.models.custom.aeon.model import (
    Prep_ARSENAL,
    Prep_DRCIF,
    Prep_HIVECOTEV2,
    Prep_STC,
    Prep_TDE,
)

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
stc_info = ModelInfo(
    model_cls=Prep_STC,
    search_space=gen_stc,
    display_name="STC",
    compute="cpu",
    reference_url=_HC2_PAPER,
    pip_extra=_PIP,
)
drcif_info = ModelInfo(
    model_cls=Prep_DRCIF,
    search_space=gen_drcif,
    display_name="DrCIF",
    compute="cpu",
    reference_url=_HC2_PAPER,
    pip_extra=_PIP,
)
arsenal_info = ModelInfo(
    model_cls=Prep_ARSENAL,
    search_space=gen_arsenal,
    display_name="Arsenal",
    compute="cpu",
    reference_url=_HC2_PAPER,
    pip_extra=_PIP,
)
tde_info = ModelInfo(
    model_cls=Prep_TDE,
    search_space=gen_tde,
    display_name="TDE",
    compute="cpu",
    reference_url="https://doi.org/10.1007/978-3-030-67658-2_38",
    pip_extra=_PIP,
)
