from tabarena.utils.config_utils import ConfigGenerator

from raman_bench.models.generate._common import bridge_search_space
from raman_bench.preprocessing.wrapped_models import Prep_RAMANNET, _RamanNetBridge

search_space = bridge_search_space(_RamanNetBridge)

gen_ramannet = ConfigGenerator(model_cls=Prep_RAMANNET, manual_configs=[{}], search_space=search_space)
