from tabarena.utils.config_utils import ConfigGenerator

from raman_bench.models.generate._common import bridge_search_space
from raman_bench.preprocessing.wrapped_models import Prep_ROCKET, _RocketBridge

search_space = bridge_search_space(_RocketBridge)

gen_rocket = ConfigGenerator(model_cls=Prep_ROCKET, manual_configs=[{}], search_space=search_space)
