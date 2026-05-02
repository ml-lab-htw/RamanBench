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

    def _fit(self, X, y, **kwargs):
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
        self._estimator.fit(X_np, y_arr)

    def _predict_proba(self, X, **kwargs):
        X_np = (
            X.values.astype(np.float32) if hasattr(X, "values") else np.asarray(X, dtype=np.float32)
        )
        if self.problem_type == "regression":
            return self._estimator.predict(X_np)
        proba = self._estimator.predict_proba(X_np)
        # AutoGluon expects (n, n_classes); our binary wrappers may return 1-D
        if proba.ndim == 1:
            return np.column_stack([1.0 - proba, proba])
        return proba

    def _get_default_searchspace(self):
        return {}


# Per-model bridge subclasses (one line each — just set _sklearn_cls)
class _PLSBridge(SklearnAutoGluonBridge):
    _sklearn_cls = PLSModel

    def _get_default_searchspace(self):
        from autogluon.common import space

        return {
            "n_components": space.Int(lower=2, upper=50),
            "scale": space.Categorical(True, False),
        }


class _DeepCNNBridge(SklearnAutoGluonBridge):
    _sklearn_cls = DeepCNNModel


class _RamanNetBridge(SklearnAutoGluonBridge):
    _sklearn_cls = RamanNetModel


class _SANetBridge(SklearnAutoGluonBridge):
    _sklearn_cls = SANetModel


class _RamanFormerBridge(SklearnAutoGluonBridge):
    _sklearn_cls = RamanFormerModel


class _RamanTransformerBridge(SklearnAutoGluonBridge):
    _sklearn_cls = RamanTransformerModel


class _ReZeroNetBridge(SklearnAutoGluonBridge):
    _sklearn_cls = ReZeroNetModel


class _FCResNeXtBridge(SklearnAutoGluonBridge):
    _sklearn_cls = FCResNeXtModel


class _CoAtNetBridge(SklearnAutoGluonBridge):
    _sklearn_cls = CoAtNetModel


class _RocketBridge(SklearnAutoGluonBridge):
    _sklearn_cls = RocketModel

    def _get_default_searchspace(self):
        from autogluon.common import space

        return {
            "rocket_transform": space.Categorical("minirocket", "rocket", "multirocket"),
            "num_kernels": space.Int(lower=5_000, upper=20_000),
        }


class _ArsenalBridge(SklearnAutoGluonBridge):
    _sklearn_cls = ArsenalModel

    def _get_default_searchspace(self):
        from autogluon.common import space

        return {
            "rocket_transform": space.Categorical("rocket", "minirocket", "multirocket"),
            "num_kernels": space.Int(lower=2_000, upper=10_000),
            "n_estimators": space.Int(lower=10, upper=50),
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


class _RamanDLBase(RamanPreprocessingMixin):
    """Raman DL base — enables spectral augmentation (standard for small datasets)."""

    _prep_aug_default: bool = True

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
