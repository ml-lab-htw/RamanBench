"""Preprocessed AutoGluon model subclasses.

Each ``Prep_*`` class combines :class:`~raman_bench.preprocessing.mixin.RamanPreprocessingMixin`
with an AutoGluon or custom model, giving every model tunable Raman preprocessing
hyperparameters (``prep_*_enabled``, ``prep_bl_lam``, etc.).

All preprocessing steps default to **disabled**.  Domain-appropriate defaults
are applied for specific model families:

- :class:`Prep_PLS` enables baseline correction, denoising, and SNV (standard
  in the spectroscopy PLS literature).
- :class:`Prep_KNN` enables SNV (distances should reflect spectral shape, not
  absolute intensity).
- :class:`Prep_LR` enables baseline correction and SNV.
- Raman DL models (:class:`Prep_DEEPCNN`, :class:`Prep_RAMANNET`, etc.) enable
  spectral augmentation (noise + shift) which is the standard regularisation
  strategy for small spectral datasets.
"""

from autogluon.core.models import DummyModel
from autogluon.tabular.models import (
    CatBoostModel,
    KNNModel,
    LGBModel,
    LinearModel,
    MitraModel,
    NNFastAiTabularModel,
    RealMLPModel,
    RealTabPFNv2Model,
    RealTabPFNv25Model,
    RFModel,
    TabDPTModel,
    TabICLModel,
    TabMModel,
    TabularNeuralNetTorchModel,
    XGBoostModel,
    XTModel,
)

from raman_bench.models.custom.coatnet import CoAtNetModel
from raman_bench.models.custom.deepcnn import DeepCNNModel
from raman_bench.models.custom.fcresnext import FCResNeXtModel
from raman_bench.models.custom.pls import PLSModel
from raman_bench.models.custom.ramanformer import RamanFormerModel
from raman_bench.models.custom.ramannet import RamanNetModel
from raman_bench.models.custom.ramantransformer import RamanTransformerModel
from raman_bench.models.custom.rezeronet import ReZeroNetModel
from raman_bench.models.custom.sanet import SANetModel
from raman_bench.models.custom.sktime_models import ArsenalModel, RocketModel
from raman_bench.preprocessing.mixin import RamanPreprocessingMixin


class _NoAugBase(RamanPreprocessingMixin):
    """Disable preprocessing augmentation by default."""

    def _set_default_params(self):
        self._set_default_param_value("prep_aug_enabled", False)
        super()._set_default_params()


# ---------------------------------------------------------------------------
# Built-in AutoGluon models
# ---------------------------------------------------------------------------

class Prep_GBM(_NoAugBase, LGBModel):  # noqa: N801
    pass


class Prep_XGB(_NoAugBase, XGBoostModel):  # noqa: N801
    pass


class Prep_CAT(_NoAugBase, CatBoostModel):  # noqa: N801
    pass


class Prep_RF(_NoAugBase, RFModel):  # noqa: N801
    pass


class Prep_XT(_NoAugBase, XTModel):  # noqa: N801
    pass


class Prep_NN_TORCH(_NoAugBase, TabularNeuralNetTorchModel):  # noqa: N801
    pass


class Prep_FASTAI(_NoAugBase, NNFastAiTabularModel):  # noqa: N801
    pass


class Prep_DUMMY(_NoAugBase, DummyModel):  # noqa: N801
    pass


class Prep_REALMLP(_NoAugBase, RealMLPModel):  # noqa: N801
    pass


class Prep_MITRA(_NoAugBase, MitraModel):  # noqa: N801
    pass


class Prep_TABM(_NoAugBase, TabMModel):  # noqa: N801
    pass


class Prep_TABDPT(_NoAugBase, TabDPTModel):  # noqa: N801
    pass


class Prep_TABICL(_NoAugBase, TabICLModel):  # noqa: N801
    pass


class Prep_REALTABPFN_V2(_NoAugBase, RealTabPFNv2Model):  # noqa: N801
    pass


class Prep_REALTABPFN_V25(_NoAugBase, RealTabPFNv25Model):  # noqa: N801
    pass


class Prep_KNN(_NoAugBase, KNNModel):  # noqa: N801
    """SNV normalises intensity scale so Euclidean distances reflect spectral shape."""

    def _set_default_params(self):
        self._set_default_param_value("prep_snv_enabled", True)
        super()._set_default_params()


class Prep_LR(_NoAugBase, LinearModel):  # noqa: N801
    """Baseline correction + SNV are standard practice before linear regression."""

    def _set_default_params(self):
        self._set_default_param_value("prep_bl_enabled", True)
        self._set_default_param_value("prep_snv_enabled", True)
        super()._set_default_params()


# ---------------------------------------------------------------------------
# Custom spectroscopy models
# ---------------------------------------------------------------------------

class Prep_PLS(_NoAugBase, PLSModel):  # noqa: N801
    """Baseline correction + denoising + SNV — standard PLS pre-processing in spectroscopy."""

    def _set_default_params(self):
        self._set_default_param_value("prep_bl_enabled", True)
        self._set_default_param_value("prep_denoise_enabled", True)
        self._set_default_param_value("prep_snv_enabled", True)
        super()._set_default_params()


class _RamanDLBase(RamanPreprocessingMixin):
    """Raman DL base — enables spectral augmentation (standard for small datasets)."""

    def _set_default_params(self):
        self._set_default_param_value("prep_aug_enabled", True)
        self._set_default_param_value("prep_aug_n", 19)
        self._set_default_param_value("prep_aug_noise", 0.01)
        self._set_default_param_value("prep_aug_shift", 0)
        self._set_default_param_value("prep_aug_mixup", 0.0)
        self._set_default_param_value("prep_aug_max_train_samples", 2000)
        super()._set_default_params()


class Prep_DEEPCNN(_RamanDLBase, DeepCNNModel):  # noqa: N801
    pass


class Prep_RAMANNET(_RamanDLBase, RamanNetModel):  # noqa: N801
    pass


class Prep_SANET(_RamanDLBase, SANetModel):  # noqa: N801
    pass


class Prep_RAMANFORMER(_RamanDLBase, RamanFormerModel):  # noqa: N801
    pass


class Prep_RAMANTRANSFORMER(_RamanDLBase, RamanTransformerModel):  # noqa: N801
    pass


class Prep_REZERONET(_RamanDLBase, ReZeroNetModel):  # noqa: N801
    pass


class Prep_FCRESNEXT(_RamanDLBase, FCResNeXtModel):  # noqa: N801
    pass


class Prep_COATNET(_RamanDLBase, CoAtNetModel):  # noqa: N801
    pass


class Prep_ROCKET(_NoAugBase, RocketModel):  # noqa: N801
    pass


class Prep_ARSENAL(_NoAugBase, ArsenalModel):  # noqa: N801
    pass


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

PREPROCESSED_MODELS = {
    # Built-in AutoGluon
    "GBM": Prep_GBM,
    "XGB": Prep_XGB,
    "CAT": Prep_CAT,
    "RF": Prep_RF,
    "XT": Prep_XT,
    "KNN": Prep_KNN,
    "LR": Prep_LR,
    "NN_TORCH": Prep_NN_TORCH,
    "FASTAI": Prep_FASTAI,
    "DUMMY": Prep_DUMMY,
    "REALMLP": Prep_REALMLP,
    "MITRA": Prep_MITRA,
    "TABM": Prep_TABM,
    "TABDPT": Prep_TABDPT,
    "TABICL": Prep_TABICL,
    "REALTABPFN-V2": Prep_REALTABPFN_V2,
    "REALTABPFN-V2.5": Prep_REALTABPFN_V25,
    # Custom spectroscopy models
    "PLS": Prep_PLS,
    "DEEPCNN": Prep_DEEPCNN,
    "RAMANNET": Prep_RAMANNET,
    "SANET": Prep_SANET,
    "RAMANFORMER": Prep_RAMANFORMER,
    "RAMANTRANSFORMER": Prep_RAMANTRANSFORMER,
    "REZERONET": Prep_REZERONET,
    "FCRESNEXT": Prep_FCRESNEXT,
    "COATNET": Prep_COATNET,
    "ROCKET": Prep_ROCKET,
    "ARSENAL": Prep_ARSENAL,
}


def create_preprocessed_hyperparameters(
    model_names: list[str],
    prep_restriction: dict | None = None,
    problem_type: str | None = None,
) -> dict:
    """Build a hyperparameters dict for AutoGluon using ``Prep_*`` wrappers.

    Parameters
    ----------
    model_names : list[str]
        Model names such as ``["GBM", "PLS", "RAMANNET"]``.
    prep_restriction : dict | None
        Step-level restriction dict (step_key → bool).  Only steps with
        ``True`` appear in the HPO search space.  ``None`` allows all.
    problem_type : str | None
        Unused; kept for call-site compatibility.

    Returns
    -------
    dict
        Mapping of ``Prep_*`` class → params dict, ready for
        ``TabularPredictor.fit(hyperparameters=...)``.
    """
    hyperparameters = {}
    for name in model_names:
        upper = name.upper()
        if upper in PREPROCESSED_MODELS:
            cls = PREPROCESSED_MODELS[upper]
            cfg = {}
            if prep_restriction is not None:
                cfg["_prep_restriction"] = prep_restriction
            hyperparameters[cls] = cfg
        else:
            hyperparameters[name] = {}
    return hyperparameters
