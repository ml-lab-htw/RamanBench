from tabarena.utils.config_utils import ConfigGenerator

from raman_bench.models.generate._common import bridge_search_space
from raman_bench.preprocessing.wrapped_models import Prep_ARSENAL, _ArsenalBridge

search_space = bridge_search_space(_ArsenalBridge)

gen_arsenal = ConfigGenerator(model_cls=Prep_ARSENAL, manual_configs=[{}], search_space=search_space)
