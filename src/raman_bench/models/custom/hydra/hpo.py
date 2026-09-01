from __future__ import annotations

from tabarena.utils.config_utils import ConfigGenerator

from raman_bench.models.custom.hydra.model import Prep_HYDRA, _HydraBridge

gen_hydra = ConfigGenerator(
    model_cls=Prep_HYDRA,
    manual_configs=[{}],
    search_space=_HydraBridge._get_default_searchspace(_HydraBridge),
)
