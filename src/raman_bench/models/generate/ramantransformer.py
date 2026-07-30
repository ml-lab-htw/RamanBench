from tabarena.utils.config_utils import ConfigGenerator

from raman_bench.models.generate._common import bridge_search_space
from raman_bench.preprocessing.wrapped_models import Prep_RAMANTRANSFORMER, _RamanTransformerBridge

search_space = bridge_search_space(_RamanTransformerBridge)

gen_ramantransformer = ConfigGenerator(
    model_cls=Prep_RAMANTRANSFORMER, manual_configs=[{}], search_space=search_space
)
