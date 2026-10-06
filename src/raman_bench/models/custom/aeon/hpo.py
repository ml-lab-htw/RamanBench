from __future__ import annotations

from tabarena.utils.config_utils import ConfigGenerator

from raman_bench.models.custom.aeon.model import Prep_HIVECOTEV2

# Default config only: aeon's HIVECOTEV2 defaults, no search space.
gen_hivecotev2 = ConfigGenerator(model_cls=Prep_HIVECOTEV2, manual_configs=[{}], search_space={})
