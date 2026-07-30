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

Custom sklearn-compatible models (``DeepCNNModel``, ``PLSModel``, etc.) are
connected to AutoGluon via :class:`SklearnAutoGluonBridge`, which adapts
sklearn's ``fit(X, y)`` / ``predict(X)`` API to AutoGluon's internal
``_fit`` / ``_predict_proba`` interface.  This keeps the custom model classes
themselves free of any AutoGluon dependency.
"""

import inspect

import numpy as np

try:
    from autogluon.core.models import AbstractModel, DummyModel
except ImportError as _ag_err:
    raise ImportError(
        "raman_bench.preprocessing.wrapped_models requires autogluon. "
        "Install with: pip install 'raman-bench[autogluon]'"
    ) from _ag_err
from autogluon.tabular.models import (
    CatBoostModel,
    KNNModel,
    LGBModel,
    LinearModel,
    NNFastAiTabularModel,
    RFModel,
    TabularNeuralNetTorchModel,
    XGBoostModel,
    XTModel,
)

# The tabular-foundation-model classes below are NOT reliably present across
# every AutoGluon >=1.5 release/prerelease build -- confirmed in practice on a
# real deployment: a given dated prerelease snapshot may be missing several of
# these (observed missing: RealTabPFNv26Model, TabFMModel, TabPFNv3Model), even
# though the "classic" models imported above have been stable across releases
# for years. Import defensively so a missing foundation-model class doesn't
# crash this whole module (and thus every other model, including plain PLS).
import warnings as _warnings

from autogluon.tabular import models as _ag_tabular_models

_OPTIONAL_AG_MODEL_NAMES = [
    "MitraModel",
    "RealMLPModel",
    "RealTabPFNv2Model",
    "RealTabPFNv25Model",
    "RealTabPFNv26Model",
    "TabDPTModel",
    "TabFMModel",
    "TabICLModel",
    "TabMModel",
    "TabPFNv3Model",
    "TabPFNv3ThinkingModel",
]
_missing_optional_ag_models = []
for _name in _OPTIONAL_AG_MODEL_NAMES:
    globals()[_name] = getattr(_ag_tabular_models, _name, None)
    if globals()[_name] is None:
        _missing_optional_ag_models.append(_name)
if _missing_optional_ag_models:
    _warnings.warn(
        f"This AutoGluon build is missing model classes: {_missing_optional_ag_models}. "
        "The corresponding RamanBench Prep_* models will be unavailable in this "
        "environment (expected -- different AutoGluon releases/prereleases carry "
        "different bleeding-edge model classes).",
        stacklevel=2,
    )
del _name, _ag_tabular_models

from raman_bench.models.custom.coatnet import CoAtNetModel
from raman_bench.models.custom.deepcnn import DeepCNNModel
from raman_bench.models.custom.fcresnext import FCResNeXtModel
from raman_bench.models.custom.pls import PLSModel
from raman_bench.models.custom.ramanformer import RamanFormerModel
from raman_bench.models.custom.ramannet import RamanNetModel
from raman_bench.models.custom.ramantransformer import RamanTransformerModel
from raman_bench.models.custom.rezeronet import ReZeroNetModel
from raman_bench.models.custom.ridge import RidgeModel
from raman_bench.models.custom.sanet import SANetModel
from raman_bench.models.custom.sktime_models import ArsenalModel, RocketModel
from raman_bench.models.custom.tabular_foundation import TabPFNWideModel
from raman_bench.preprocessing.mixin import RamanPreprocessingMixin

# ---------------------------------------------------------------------------
# Bridge: sklearn fit()/predict() ↔ AutoGluon _fit()/_predict_proba()
# ---------------------------------------------------------------------------


class SklearnAutoGluonBridge(AbstractModel):
    """Adapter that connects sklearn-compatible estimators to AutoGluon.

    Concrete subclasses set ``_sklearn_cls`` to a sklearn ``BaseEstimator``
    subclass.  AutoGluon hyperparameters (excluding ``ag.*`` and ``prep_*``
    prefixes) are forwarded verbatim to the sklearn constructor.

    This keeps all custom model classes free of AutoGluon imports while still
    allowing them to participate in the full AutoGluon benchmark pipeline.
    """

    _sklearn_cls = None  # override in subclass

    def _fit(self, X, y, time_limit=None, **kwargs):
        X_np = (
            X.values.astype(np.float32) if hasattr(X, "values") else np.asarray(X, dtype=np.float32)
        )
        y_arr = y.values if hasattr(y, "values") else np.asarray(y)

        # _get_model_params() at this point no longer contains prep_* keys —
        # RamanPreprocessingMixin._fit() strips them before calling super().
        params = {
            k: v
            for k, v in self._get_model_params().items()
            if not k.startswith("ag.") and not k.startswith("_")
        }
        self._estimator = self._sklearn_cls(**params)
        # Only the deep-learning estimators accept a training budget; the sklearn
        # wrappers (PLS, ROCKET, ARSENAL) have fit(self, X, y). Passing time_limit
        # to those raises TypeError, which under HPO fails every trial. Forward it
        # only when the estimator's fit actually accepts it (named param or **kwargs).
        fit_params = inspect.signature(self._estimator.fit).parameters
        accepts_time_limit = "time_limit" in fit_params or any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in fit_params.values()
        )
        if accepts_time_limit:
            self._estimator.fit(X_np, y_arr, time_limit=time_limit)
        else:
            self._estimator.fit(X_np, y_arr)

    def _predict_proba(self, X, **kwargs):
        X_np = (
            X.values.astype(np.float32) if hasattr(X, "values") else np.asarray(X, dtype=np.float32)
        )
        if self.problem_type == "regression":
            return self._estimator.predict(X_np)
        proba = self._estimator.predict_proba(X_np)
        if self.problem_type == "binary":
            # AutoGluon bagging pre-allocates oof_pred_proba as 1-D for binary;
            # return the positive-class column only.
            return proba[:, 1] if proba.ndim == 2 else proba
        return proba

    def _get_default_searchspace(self):
        return {}


# Per-model bridge subclasses (one line each — just set _sklearn_cls).
# ag_key/ag_name identify the model family to TabArena's ConfigGenerator/registry
# (tabarena/tabarena/utils/config_utils.py asserts both are set); ag_key matches
# the string key already used in PREPROCESSED_MODELS below.
class _PLSBridge(SklearnAutoGluonBridge):
    _sklearn_cls = PLSModel
    ag_key = "PLS"
    ag_name = "PLS"

    def _get_default_searchspace(self):
        from autogluon.common import space

        # n_components is PLS's defining hyperparameter (number of latent variables);
        # widened from 50 -> 100 since Raman spectra commonly have hundreds of feature
        # bins. PLSModel.fit() still clips to min(n_components, n_features,
        # n_samples - 1), so this only grants more headroom on datasets big enough to
        # use it; small datasets are unaffected. max_iter/tol control the NIPALS
        # deflation's convergence and were plumbed through PLSModel but previously
        # fixed at their sklearn defaults for every trial.
        return {
            "n_components": space.Int(lower=2, upper=100),
            "scale": space.Categorical(True, False),
            "max_iter": space.Int(lower=200, upper=1000),
            "tol": space.Real(lower=1e-8, upper=1e-3, log=True),
        }


class _RidgeBridge(SklearnAutoGluonBridge):
    _sklearn_cls = RidgeModel
    ag_key = "RIDGE"
    ag_name = "Ridge"

    def _get_default_searchspace(self):
        from autogluon.common import space

        return {
            "alpha": space.Real(lower=1e-3, upper=1e3, log=True),
        }


class _DeepCNNBridge(SklearnAutoGluonBridge):
    _sklearn_cls = DeepCNNModel
    ag_key = "DEEPCNN"
    ag_name = "DeepCNN"

    def _get_default_searchspace(self):
        from autogluon.common import space

        return {
            "lr": space.Real(1e-4, 3e-3, default=1e-3, log=True),
            "weight_decay": space.Real(1e-6, 1e-3, default=1e-4, log=True),
            "dropout": space.Categorical(0.5, 0.2, 0.3, 0.7),
            "initial_channels": space.Categorical(32, 16, 64),
            "dense_dim": space.Categorical(256, 128, 512),
            "aug_noise_sigma": space.Real(1e-4, 1e-1, default=1e-2, log=True),
            "aug_mixup_alpha": space.Real(1e-2, 1e2, default=1, log=True),
        }


class _RamanNetBridge(SklearnAutoGluonBridge):
    _sklearn_cls = RamanNetModel
    ag_key = "RAMANNET"
    ag_name = "RamanNet"

    def _get_default_searchspace(self):
        from autogluon.common import space

        return {
            "lr": space.Real(1e-4, 3e-3, default=1e-3, log=True),
            "weight_decay": space.Real(1e-6, 1e-3, default=1e-4, log=True),
            "fc_dropout": space.Categorical(0.5, 0.2, 0.3, 0.7),
            "fc_dim": space.Categorical(512, 256, 1024),
            "window_size": space.Categorical(50, 25, 100),
            "aug_noise_sigma": space.Real(1e-4, 1e-1, default=1e-2, log=True),
            "aug_mixup_alpha": space.Real(1e-2, 1e2, default=1, log=True),
        }


class _SANetBridge(SklearnAutoGluonBridge):
    _sklearn_cls = SANetModel
    ag_key = "SANET"
    ag_name = "SANet"

    def _get_default_searchspace(self):
        from autogluon.common import space

        # First categorical value is the model default (AutoGluon convention).
        return {
            "lr": space.Real(1e-4, 3e-3, default=1e-3, log=True),
            "weight_decay": space.Real(1e-6, 1e-3, default=1e-4, log=True),
            "num_blocks": space.Int(lower=3, upper=7),
            "channel_factor": space.Categorical(2.0, 1.5, 2.5),
            "initial_channels": space.Categorical(16, 8, 32),
            "num_branches": space.Categorical(6, 4, 8),
            "reduction": space.Categorical(16, 8, 32),
            "aug_noise_sigma": space.Real(1e-4, 1e-1, default=1e-2, log=True),
            "aug_mixup_alpha": space.Real(1e-2, 1e2, default=1, log=True),
        }


class _RamanFormerBridge(SklearnAutoGluonBridge):
    _sklearn_cls = RamanFormerModel
    ag_key = "RAMANFORMER"
    ag_name = "RamanFormer"

    def _get_default_searchspace(self):
        from autogluon.common import space

        # d_model options stay divisible by nhead (8).
        return {
            "patch_size": space.Categorical(64, 128, 256),
            "lr": space.Real(3e-5, 1e-3, default=1e-4, log=True),
            "weight_decay": space.Real(1e-6, 1e-3, default=1e-4, log=True),
            "dropout": space.Categorical(0.1, 0.0, 0.2, 0.3),
            "d_model": space.Categorical(256, 128, 512),
            "nhead": space.Categorical(4, 8, 16),
            "dim_feedforward": space.Categorical(1024, 512, 2048),
            "n_layers": space.Categorical(3, 2, 4),
            "aug_noise_sigma": space.Real(1e-4, 1e-1, default=1e-2, log=True),
            "aug_mixup_alpha": space.Real(1e-2, 1e2, default=1, log=True),
            "post_processing_dim": space.Categorical(256, 512, 1024),
        }


class _RamanTransformerBridge(SklearnAutoGluonBridge):
    _sklearn_cls = RamanTransformerModel
    ag_key = "RAMANTRANSFORMER"
    ag_name = "RamanTransformer"

    def _get_default_searchspace(self):
        from autogluon.common import space

        # Keep d_model/nhead/patch_size fixed (hard divisibility constraints);
        # tune depth, regularisation and optimisation.
        return {
            "patch_size": space.Categorical(16, 32, 64),
            "d_model": space.Categorical(256, 128, 512),
            "lr": space.Real(1e-4, 3e-3, default=1e-3, log=True),
            "weight_decay": space.Real(1e-6, 1e-3, default=1e-4, log=True),
            "dropout": space.Categorical(0.1, 0.0, 0.2),
            "nhead": space.Categorical(4, 8, 16),
            "dim_feedforward": space.Categorical(3072, 1536, 2048),
            "n_layers": space.Categorical(12, 6, 8),
            "aug_noise_sigma": space.Real(1e-4, 1e-1, default=1e-2, log=True),
            "aug_mixup_alpha": space.Real(1e-2, 1e2, default=1, log=True),
        }


class _ReZeroNetBridge(SklearnAutoGluonBridge):
    _sklearn_cls = ReZeroNetModel
    ag_key = "REZERONET"
    ag_name = "ReZeroNet"

    def _get_default_searchspace(self):
        from autogluon.common import space

        # First categorical value is the model default (AutoGluon convention).
        return {
            "lr": space.Real(1e-4, 3e-3, default=1e-3, log=True),
            "weight_decay": space.Real(1e-6, 1e-3, default=1e-4, log=True),
            "fc_dropout": space.Categorical(0.2, 0.0, 0.1, 0.3, 0.5),
            "n_blocks": space.Int(lower=4, upper=8),
            "base_channels": space.Categorical(64, 32, 128),
            "channel_factor": space.Categorical(1.0, 1.25, 1.5),
            "kernel_size": space.Categorical(3, 5, 7),
            "aug_noise_sigma": space.Real(1e-4, 1e-1, default=1e-2, log=True),
            "aug_mixup_alpha": space.Real(1e-2, 1e2, default=1, log=True),
        }


class _FCResNeXtBridge(SklearnAutoGluonBridge):
    _sklearn_cls = FCResNeXtModel
    ag_key = "FCRESNEXT"
    ag_name = "FCResNeXt"

    def _get_default_searchspace(self):
        from autogluon.common import space

        # cardinality choices divide every hidden_dim option.
        return {
            "lr": space.Real(1e-4, 3e-3, default=1e-3, log=True),
            "weight_decay": space.Real(1e-6, 1e-3, default=1e-4, log=True),
            "fc_dropout": space.Categorical(0.2, 0.0, 0.1, 0.3, 0.5),
            "hidden_dim": space.Categorical(256, 128, 512),
            "n_blocks": space.Int(lower=2, upper=6),
            "pool_size": space.Categorical(10, 5, 20),
            "cardinality": space.Categorical(4, 2, 8),
            "aug_noise_sigma": space.Real(1e-4, 1e-1, default=1e-2, log=True),
            "aug_mixup_alpha": space.Real(1e-2, 1e2, default=1, log=True),
        }


class _CoAtNetBridge(SklearnAutoGluonBridge):
    _sklearn_cls = CoAtNetModel
    ag_key = "COATNET"
    ag_name = "CoAtNet"

    def _get_default_searchspace(self):
        from autogluon.common import space

        # base_channels options stay divisible by nhead (4).
        return {
            "lr": space.Real(1e-4, 3e-3, default=1e-3, log=True),
            "weight_decay": space.Real(1e-6, 1e-3, default=1e-4, log=True),
            "fc_dropout": space.Categorical(0.2, 0.0, 0.1, 0.3),
            "attn_dropout": space.Categorical(0.1, 0.0, 0.2),
            "n_blocks": space.Int(lower=3, upper=8),
            "base_channels": space.Categorical(64, 32, 128),
            "channel_factor": space.Categorical(1.0, 1.25, 1.5),
            "nhead": space.Categorical(2, 4, 8),
            "kernel_size": space.Categorical(3, 5, 7),
            "aug_noise_sigma": space.Real(1e-4, 1e-1, default=1e-2, log=True),
            "aug_mixup_alpha": space.Real(1e-2, 1e2, default=1, log=True),
        }


class _RocketBridge(SklearnAutoGluonBridge):
    _sklearn_cls = RocketModel
    ag_key = "ROCKET"
    ag_name = "ROCKET"

    def _get_default_searchspace(self):
        from autogluon.common import space

        return {
            "rocket_transform": space.Categorical("minirocket", "rocket", "multirocket"),
            "num_kernels": space.Int(lower=5_000, upper=20_000),
        }


class _TabPFNWideBridge(SklearnAutoGluonBridge):
    _sklearn_cls = TabPFNWideModel
    ag_key = "TABPFN-WIDE"
    ag_name = "TabPFN-Wide"


class _ArsenalBridge(SklearnAutoGluonBridge):
    _sklearn_cls = ArsenalModel
    ag_key = "ARSENAL"
    ag_name = "ARSENAL"

    def _get_default_searchspace(self):
        from autogluon.common import space

        return {
            "rocket_transform": space.Categorical("rocket", "minirocket"),
            "num_kernels": space.Int(lower=2_000, upper=5_000),
            "n_estimators": space.Int(lower=20, upper=40),
        }


# ---------------------------------------------------------------------------
# Shared mixin bases
# ---------------------------------------------------------------------------


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
    _supports_augmentation: bool = True


class Prep_FASTAI(_NoAugBase, NNFastAiTabularModel):  # noqa: N801
    _supports_augmentation: bool = True


class Prep_DUMMY(_NoAugBase, DummyModel):  # noqa: N801
    pass


def _make_optional_prep_class(name: str, base_model_cls, **class_attrs):
    """Build a ``Prep_*`` class for a possibly-unavailable AutoGluon model class.

    Returns ``None`` (rather than raising ``TypeError: bases must be types``)
    when ``base_model_cls`` is ``None`` -- i.e. this AutoGluon build doesn't
    carry that foundation-model class (see the defensive import above).
    """
    if base_model_cls is None:
        return None
    return type(name, (_NoAugBase, base_model_cls), dict(class_attrs))


Prep_REALMLP = _make_optional_prep_class("Prep_REALMLP", RealMLPModel, _supports_augmentation=True)
Prep_MITRA = _make_optional_prep_class("Prep_MITRA", MitraModel)
Prep_TABM = _make_optional_prep_class("Prep_TABM", TabMModel)
Prep_TABDPT = _make_optional_prep_class("Prep_TABDPT", TabDPTModel)
Prep_TABFM = _make_optional_prep_class("Prep_TABFM", TabFMModel)
Prep_TABICL = _make_optional_prep_class("Prep_TABICL", TabICLModel)
Prep_REALTABPFN_V2 = _make_optional_prep_class("Prep_REALTABPFN_V2", RealTabPFNv2Model)
Prep_REALTABPFN_V25 = _make_optional_prep_class("Prep_REALTABPFN_V25", RealTabPFNv25Model)
Prep_REALTABPFN_V26 = _make_optional_prep_class("Prep_REALTABPFN_V26", RealTabPFNv26Model)
Prep_TABPFN_V3 = _make_optional_prep_class("Prep_TABPFN_V3", TabPFNv3Model)
Prep_TABPFN_V3_THINKING = _make_optional_prep_class("Prep_TABPFN_V3_THINKING", TabPFNv3ThinkingModel)


class Prep_TABPFN_WIDE(_NoAugBase, _TabPFNWideBridge):  # noqa: N801
    """TabPFN-Wide — classification-only, targets wide datasets (many features, few samples).

    Built with PriorLabs-TabPFN.
    """

    pass


class Prep_KNN(_NoAugBase, KNNModel):  # noqa: N801
    """SNV normalises intensity scale so Euclidean distances reflect spectral shape."""

    _optimize_preprocessing = True

    def _set_default_params(self):
        self._set_default_param_value("prep_snv_enabled", True)
        super()._set_default_params()

    def _fit(self, X, y, **kwargs):
        # sklearn requires n_neighbors <= n_samples_fit; clamp for tiny datasets
        n_neighbors = self._get_model_params().get("n_neighbors", 5)
        if n_neighbors > len(X):
            self.params["n_neighbors"] = len(X)
        super()._fit(X, y, **kwargs)


class Prep_LR(_NoAugBase, LinearModel):  # noqa: N801
    """Baseline correction + SNV are standard practice before linear regression."""

    _optimize_preprocessing = True

    def _set_default_params(self):
        self._set_default_param_value("prep_bl_enabled", True)
        self._set_default_param_value("prep_snv_enabled", True)
        super()._set_default_params()


# ---------------------------------------------------------------------------
# Custom spectroscopy models
# ---------------------------------------------------------------------------


class Prep_PLS(_NoAugBase, _PLSBridge):  # noqa: N801
    """Baseline correction + denoising + SNV — standard PLS pre-processing in spectroscopy."""

    _optimize_preprocessing = True

    def _set_default_params(self):
        self._set_default_param_value("prep_bl_enabled", True)
        self._set_default_param_value("prep_denoise_enabled", True)
        self._set_default_param_value("prep_snv_enabled", True)
        super()._set_default_params()


class Prep_RIDGE(_NoAugBase, _RidgeBridge):  # noqa: N801
    """Ridge (L2-regularized linear) regression/classification.

    Added as a model-agent workflow dry-run artifact -- validates the
    add-a-new-model steps end to end, not a benchmark-motivated addition.
    """

    pass


class _RamanDLBase(RamanPreprocessingMixin):
    """Raman DL base — enables spectral augmentation (standard for small datasets)."""

    _supports_augmentation: bool = True

    def _set_default_params(self):
        self._set_default_param_value("prep_aug_enabled", True)
        self._set_default_param_value("prep_aug_n", 19)
        self._set_default_param_value("prep_aug_noise", 0.01)
        self._set_default_param_value("prep_aug_shift", 0)
        self._set_default_param_value("prep_aug_mixup", 0.0)
        self._set_default_param_value("prep_aug_max_train_samples", 2000)
        super()._set_default_params()


class Prep_DEEPCNN(_RamanDLBase, _DeepCNNBridge):  # noqa: N801
    pass


class Prep_RAMANNET(_RamanDLBase, _RamanNetBridge):  # noqa: N801
    pass


class Prep_SANET(_RamanDLBase, _SANetBridge):  # noqa: N801
    pass


class Prep_RAMANFORMER(_RamanDLBase, _RamanFormerBridge):  # noqa: N801
    pass


class Prep_RAMANTRANSFORMER(_RamanDLBase, _RamanTransformerBridge):  # noqa: N801
    pass


class Prep_REZERONET(_RamanDLBase, _ReZeroNetBridge):  # noqa: N801
    pass


class Prep_FCRESNEXT(_RamanDLBase, _FCResNeXtBridge):  # noqa: N801
    pass


class Prep_COATNET(_RamanDLBase, _CoAtNetBridge):  # noqa: N801
    pass


class Prep_ROCKET(_NoAugBase, _RocketBridge):  # noqa: N801
    pass


class Prep_ARSENAL(_NoAugBase, _ArsenalBridge):  # noqa: N801
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
    "TABFM": Prep_TABFM,
    "TABICL": Prep_TABICL,
    "REALTABPFN-V2": Prep_REALTABPFN_V2,
    "REALTABPFN-V2.5": Prep_REALTABPFN_V25,
    "REALTABPFN-V2.6": Prep_REALTABPFN_V26,
    "TABPFN-V3": Prep_TABPFN_V3,
    "TABPFN-V3-THINKING": Prep_TABPFN_V3_THINKING,
    "TABPFN-WIDE": Prep_TABPFN_WIDE,
    # Custom spectroscopy models
    "PLS": Prep_PLS,
    "RIDGE": Prep_RIDGE,
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

# Drop any entry whose AutoGluon base class wasn't available on this build (see
# the defensive import above) rather than exposing a None-valued model class.
_unavailable_models = [key for key, cls in PREPROCESSED_MODELS.items() if cls is None]
for _key in _unavailable_models:
    del PREPROCESSED_MODELS[_key]
if _unavailable_models:
    _warnings.warn(
        f"Skipping model(s) unavailable in this AutoGluon build: {_unavailable_models}",
        stacklevel=2,
    )
del _unavailable_models

CLASSIFICATION_ONLY_MODELS = {"ROCKET", "ARSENAL", "TABPFN-WIDE"}


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
