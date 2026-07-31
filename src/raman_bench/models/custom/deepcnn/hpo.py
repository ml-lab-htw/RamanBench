from __future__ import annotations

from tabarena.utils.config_utils import ConfigGenerator

from raman_bench.models.custom.deepcnn.model import Prep_DEEPCNN, _DeepCNNBridge

gen_deepcnn = ConfigGenerator(
    model_cls=Prep_DEEPCNN,
    manual_configs=[{}],
    search_space=_DeepCNNBridge._get_default_searchspace(_DeepCNNBridge),
)
