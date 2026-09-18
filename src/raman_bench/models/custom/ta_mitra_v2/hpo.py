from __future__ import annotations

from tabarena.utils.config_utils import ConfigGenerator

from raman_bench.models.custom.ta_mitra_v2.model import Prep_MITRA_V2

# Mirrors tabarena.models.mitra_v2.hpo.gen_mitra_v2: Mitra-v2's fine-tuning recipe is
# frozen (see tabarena.models.mitra_v2._internal.recipe) -- the default configuration
# is the method, so the search space is empty and the only config is the default.
gen_ta_mitra_v2 = ConfigGenerator(
    model_cls=Prep_MITRA_V2,
    search_space={},
    manual_configs=[{}],
)
