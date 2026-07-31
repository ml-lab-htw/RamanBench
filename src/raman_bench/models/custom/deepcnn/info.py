from __future__ import annotations

from raman_bench.models._model_info import ModelInfo
from raman_bench.models.custom.deepcnn.hpo import gen_deepcnn
from raman_bench.models.custom.deepcnn.model import Prep_DEEPCNN

deepcnn_info = ModelInfo(
    model_cls=Prep_DEEPCNN,
    search_space=gen_deepcnn,
    display_name="DeepCNN",
    compute="gpu",
    reference_url="https://doi.org/10.1039/C7AN01042G",
)
