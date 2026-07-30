from tabarena.utils.config_utils import ConfigGenerator

from raman_bench.models.generate._common import bridge_search_space
from raman_bench.preprocessing.wrapped_models import Prep_FCRESNEXT, _FCResNeXtBridge

search_space = bridge_search_space(_FCResNeXtBridge)

gen_fcresnext = ConfigGenerator(model_cls=Prep_FCRESNEXT, manual_configs=[{}], search_space=search_space)
