from tabarena.utils.config_utils import ConfigGenerator

from raman_bench.models.generate._common import bridge_search_space
from raman_bench.preprocessing.wrapped_models import Prep_DEEPCNN, _DeepCNNBridge

search_space = bridge_search_space(_DeepCNNBridge)

gen_deepcnn = ConfigGenerator(model_cls=Prep_DEEPCNN, manual_configs=[{}], search_space=search_space)
