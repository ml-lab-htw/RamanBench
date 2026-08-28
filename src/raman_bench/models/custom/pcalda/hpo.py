from __future__ import annotations

from tabarena.utils.config_utils import ConfigGenerator

from raman_bench.models.custom.pcalda.model import Prep_PCALDA, _PCALDABridge

gen_pcalda = ConfigGenerator(
    model_cls=Prep_PCALDA,
    manual_configs=[{}],
    search_space=_PCALDABridge._get_default_searchspace(_PCALDABridge),
)
