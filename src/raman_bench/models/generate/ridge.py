from tabarena.utils.config_utils import ConfigGenerator

from raman_bench.models.generate._common import bridge_search_space
from raman_bench.preprocessing.wrapped_models import Prep_RIDGE, _RidgeBridge

search_space = bridge_search_space(_RidgeBridge)

gen_ridge = ConfigGenerator(model_cls=Prep_RIDGE, manual_configs=[{}], search_space=search_space)
