from tabarena.utils.config_utils import ConfigGenerator

from raman_bench.models.generate._common import bridge_search_space
from raman_bench.preprocessing.wrapped_models import Prep_COATNET, _CoAtNetBridge

search_space = bridge_search_space(_CoAtNetBridge)

gen_coatnet = ConfigGenerator(model_cls=Prep_COATNET, manual_configs=[{}], search_space=search_space)
