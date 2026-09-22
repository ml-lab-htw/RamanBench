from __future__ import annotations

from tabarena.utils.config_utils import ConfigGenerator

from raman_bench.models.custom.ta_exaone_tabular.model import Prep_EXAONE_TABULAR

# Mirrors tabarena.models.exaone_tabular.hpo.gen_exaone_tabular: EXAONE-Tabular is an
# in-context-learning foundation model with no tunable hyperparameters worth searching
# (TabArena itself ships ``can_hpo=False``), so the search space is empty and the only
# config is the default.
gen_ta_exaone_tabular = ConfigGenerator(
    model_cls=Prep_EXAONE_TABULAR,
    search_space={},
    manual_configs=[{}],
)
