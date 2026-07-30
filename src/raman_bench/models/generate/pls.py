from tabarena.utils.config_utils import ConfigGenerator

from raman_bench.models.generate._common import bridge_search_space
from raman_bench.preprocessing.wrapped_models import Prep_PLS, _PLSBridge

search_space = bridge_search_space(_PLSBridge)

gen_pls = ConfigGenerator(model_cls=Prep_PLS, manual_configs=[{}], search_space=search_space)
