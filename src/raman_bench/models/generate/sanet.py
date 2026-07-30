from tabarena.utils.config_utils import ConfigGenerator

from raman_bench.models.generate._common import bridge_search_space
from raman_bench.preprocessing.wrapped_models import Prep_SANET, _SANetBridge

search_space = bridge_search_space(_SANetBridge)

gen_sanet = ConfigGenerator(model_cls=Prep_SANET, manual_configs=[{}], search_space=search_space)
