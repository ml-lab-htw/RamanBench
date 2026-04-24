"""Raman-specific neural network models.

All models are AutoGluon-compatible (inherit from
:class:`~autogluon.core.models.AbstractModel`) and implement the
:class:`~raman_bench.models.custom.base.BaseCustomModel` training loop with
early stopping and optional per-epoch augmentation.
"""

from raman_bench.models.custom.base import BaseCustomModel
from raman_bench.models.custom.coatnet import CoAtNetModel
from raman_bench.models.custom.deepcnn import DeepCNNModel
from raman_bench.models.custom.fcresnext import FCResNeXtModel
from raman_bench.models.custom.pls import FlexiblePipelinePLS, PLSModel, PreprocessingPLS
from raman_bench.models.custom.ramanformer import RamanFormerModel
from raman_bench.models.custom.ramannet import RamanNetModel
from raman_bench.models.custom.ramantransformer import RamanTransformerModel
from raman_bench.models.custom.rezeronet import ReZeroNetModel
from raman_bench.models.custom.sanet import SANetModel
from raman_bench.models.custom.sktime_models import ArsenalModel, RocketModel

CUSTOM_MODELS = {
    "PLS": PLSModel,
    "DEEPCNN": DeepCNNModel,
    "RAMANNET": RamanNetModel,
    "SANET": SANetModel,
    "RAMANFORMER": RamanFormerModel,
    "RAMANTRANSFORMER": RamanTransformerModel,
    "PREPROCESSINGPLS": PreprocessingPLS,
    "PipelinePLS": FlexiblePipelinePLS,
    "REZERONET": ReZeroNetModel,
    "FCRESNEXT": FCResNeXtModel,
    "COATNET": CoAtNetModel,
    "ROCKET": RocketModel,
    "ARSENAL": ArsenalModel,
}

__all__ = [
    "BaseCustomModel",
    "CoAtNetModel",
    "DeepCNNModel",
    "FCResNeXtModel",
    "FlexiblePipelinePLS",
    "PLSModel",
    "PreprocessingPLS",
    "RamanFormerModel",
    "RamanNetModel",
    "RamanTransformerModel",
    "ReZeroNetModel",
    "SANetModel",
    "ArsenalModel",
    "RocketModel",
    "CUSTOM_MODELS",
]
