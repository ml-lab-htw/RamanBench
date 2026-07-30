from tabarena.utils.config_utils import ConfigGenerator

from raman_bench.models.generate._common import bridge_search_space
from raman_bench.preprocessing.wrapped_models import Prep_RAMANFORMER, _RamanFormerBridge

search_space = bridge_search_space(_RamanFormerBridge)

gen_ramanformer = ConfigGenerator(model_cls=Prep_RAMANFORMER, manual_configs=[{}], search_space=search_space)
