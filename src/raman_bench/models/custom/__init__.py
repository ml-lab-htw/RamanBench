"""Raman-specific models with a scikit-learn-compatible API.

All models expose ``fit(X, y)`` and ``predict(X)`` and require only
``torch`` (PyTorch-based models), ``sktime`` (ROCKET), or the
respective package (TabPFN, RealMLP, TabM via pytabkit, TabDPT).
AutoGluon is **not** required.
"""

from raman_bench.models.custom.base import BaseRamanEstimator
from raman_bench.models.custom.coatnet.model import CoAtNetModel
from raman_bench.models.custom.deepcnn.model import DeepCNNModel
from raman_bench.models.custom.fcresnext.model import FCResNeXtModel
from raman_bench.models.custom.hydra.model import HydraModel
from raman_bench.models.custom.pls.model import PLSModel
from raman_bench.models.custom.ramanformer.model import RamanFormerModel
from raman_bench.models.custom.ramannet.model import RamanNetModel
from raman_bench.models.custom.ramantransformer.model import RamanTransformerModel
from raman_bench.models.custom.rezeronet.model import ReZeroNetModel
from raman_bench.models.custom.ridge.model import RidgeModel
from raman_bench.models.custom.rocket.model import RocketModel
from raman_bench.models.custom.sanet.model import SANetModel
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
    "HYDRA": HydraModel,
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
    "RocketModel",
    "HydraModel",
    "RidgeModel",
    "TabPFNModel",
    "RealMLPModel",
    "TabMModel",
    "TabDPTModel",
    "CUSTOM_MODELS",
]
