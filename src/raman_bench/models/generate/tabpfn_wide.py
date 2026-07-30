from tabarena.utils.config_utils import ConfigGenerator

from raman_bench.preprocessing.wrapped_models import Prep_TABPFN_WIDE

# No tunable hyperparameters exposed today -- default config only (_c1).
search_space: dict = {}

gen_tabpfn_wide = ConfigGenerator(model_cls=Prep_TABPFN_WIDE, manual_configs=[{}], search_space=search_space)
