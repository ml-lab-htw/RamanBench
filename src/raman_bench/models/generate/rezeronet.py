from tabarena.utils.config_utils import ConfigGenerator

from raman_bench.models.generate._common import bridge_search_space
from raman_bench.preprocessing.wrapped_models import Prep_REZERONET, _ReZeroNetBridge

search_space = bridge_search_space(_ReZeroNetBridge)

gen_rezeronet = ConfigGenerator(model_cls=Prep_REZERONET, manual_configs=[{}], search_space=search_space)
