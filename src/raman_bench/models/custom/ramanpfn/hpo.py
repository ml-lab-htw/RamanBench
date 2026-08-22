from __future__ import annotations

from tabarena.utils.config_utils import ConfigGenerator

from raman_bench.models.custom.ramanpfn.model import Prep_RAMANPFN, _RamanPFNBridge

gen_ramanpfn = ConfigGenerator(
    model_cls=Prep_RAMANPFN,
    manual_configs=[{}],
    search_space=_RamanPFNBridge._get_default_searchspace(_RamanPFNBridge),
)
