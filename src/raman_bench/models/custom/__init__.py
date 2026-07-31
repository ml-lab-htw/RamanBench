"""Raman-specific models with a scikit-learn-compatible API.

All models expose ``fit(X, y)`` and ``predict(X)`` and require only
``torch`` (PyTorch-based models), ``sktime`` (ROCKET/Arsenal), or the
respective package (TabPFN, RealMLP, TabM via pytabkit, TabDPT).
AutoGluon is **not** required.
"""

from raman_bench.models.custom.base import BaseRamanEstimator
from raman_bench.models.custom.coatnet import CoAtNetModel
from raman_bench.models.custom.deepcnn import DeepCNNModel
from raman_bench.models.custom.fcresnext import FCResNeXtModel
from raman_bench.models.custom.pls import PLSModel
from raman_bench.models.custom.ramanformer import RamanFormerModel
from raman_bench.models.custom.ramannet import RamanNetModel
from raman_bench.models.custom.ramantransformer import RamanTransformerModel
from raman_bench.models.custom.rezeronet import ReZeroNetModel
from raman_bench.models.custom.ridge.model import RidgeModel
from raman_bench.models.custom.sanet import SANetModel
from raman_bench.models.custom.sktime_models import ArsenalModel, RocketModel
from raman_bench.models.custom.tabular_foundation import (
    RealMLPModel,
    TabDPTModel,
    TabMModel,
    TabPFNModel,
)

CUSTOM_MODELS = {
    "PLS": PLSModel,
    "DEEPCNN": DeepCNNModel,
    "RAMANNET": RamanNetModel,
    "SANET": SANetModel,
    "RAMANFORMER": RamanFormerModel,
    "RAMANTRANSFORMER": RamanTransformerModel,
    "REZERONET": ReZeroNetModel,
    "FCRESNEXT": FCResNeXtModel,
    "COATNET": CoAtNetModel,
    "ROCKET": RocketModel,
    "ARSENAL": ArsenalModel,
    "RIDGE": RidgeModel,
    "TABPFN": TabPFNModel,
    "REALMLP": RealMLPModel,
    "TABM": TabMModel,
    "TABDPT": TabDPTModel,
}

__all__ = [
    "BaseRamanEstimator",
    "CoAtNetModel",
    "DeepCNNModel",
    "FCResNeXtModel",
    "PLSModel",
    "RamanFormerModel",
    "RamanNetModel",
    "RamanTransformerModel",
    "ReZeroNetModel",
    "SANetModel",
    "ArsenalModel",
    "RocketModel",
    "RidgeModel",
    "TabPFNModel",
    "RealMLPModel",
    "TabMModel",
    "TabDPTModel",
    "CUSTOM_MODELS",
]
