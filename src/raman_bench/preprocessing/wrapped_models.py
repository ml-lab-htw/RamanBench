"""Preprocessed AutoGluon model subclasses.

Each ``Prep_*`` class combines :class:`~raman_bench.preprocessing.mixin.RamanPreprocessingMixin`
with an AutoGluon model, giving every model tunable Raman preprocessing
hyperparameters (``prep_*_enabled``, ``prep_bl_lam``, etc.).

All preprocessing steps default to **disabled**.  Domain-appropriate defaults
are applied for specific model families:

- :class:`Prep_KNN` enables SNV (distances should reflect spectral shape, not
  absolute intensity).
- :class:`Prep_LR` enables baseline correction and SNV.

The Raman-specific custom architectures (PLS, DeepCNN, RamanNet, SANet,
RamanFormer, RamanTransformer, ReZeroNet, FC-ResNeXt, CoAtNet, ROCKET,
Arsenal, TabPFN-Wide) and GBM/TA-TABPFN-3 have moved to the per-model
``raman_bench/models/custom/<key>/{model.py,hpo.py,info.py}`` convention
(auto-discovered via :mod:`raman_bench.models.discover`, see
``models/custom/ridge/`` for the reference implementation) -- this module now
only holds the built-in-AutoGluon-backed ``Prep_*`` classes that haven't been
migrated there yet, plus the merge point that combines both conventions into
one :data:`PREPROCESSED_MODELS` dict.
"""

import numpy as np

try:
    from autogluon.core.models import DummyModel
except ImportError as _ag_err:
    raise ImportError(
        "raman_bench.preprocessing.wrapped_models requires autogluon. "
        "Install with: pip install 'raman-bench[autogluon]'"
    ) from _ag_err
# The tabular-foundation-model classes below are NOT reliably present across
# every AutoGluon >=1.5 release/prerelease build -- confirmed in practice on a
# real deployment: a given dated prerelease snapshot may be missing several of
# these (observed missing: RealTabPFNv26Model, TabFMModel, TabPFNv3Model), even
# though the "classic" models imported above have been stable across releases
# for years. Import defensively so a missing foundation-model class doesn't
# crash this whole module (and thus every other model, including plain PLS).
import warnings as _warnings

from autogluon.tabular import models as _ag_tabular_models
from autogluon.tabular.models import (
    CatBoostModel,
    KNNModel,
    LinearModel,
    NNFastAiTabularModel,
    RFModel,
    TabularNeuralNetTorchModel,
    XGBoostModel,
    XTModel,
)

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

from raman_bench.models.discover import discover_custom_models
from raman_bench.preprocessing.bridge_bases import _make_optional_prep_class, _NoAugBase

# ---------------------------------------------------------------------------
# Built-in AutoGluon models (not yet migrated to the per-model-directory
# convention -- see the module docstring)
# ---------------------------------------------------------------------------


class Prep_XGB(_NoAugBase, XGBoostModel):  # noqa: N801
    def get_eval_metric(self):
        """Guard XGBoost's custom multiclass metric against flat 1D predictions.

        ``XGBoostModel.get_eval_metric()`` falls back to
        ``xgboost_utils.func_generator`` for any stopping metric without a native
        XGBoost mapping (``autogluon.tabular.models.xgboost.xgboost_utils._ag_to_xgbm_metric_dict``);
        for multiclass, that generated callable unconditionally calls
        ``y_hat.argmax(axis=1)``, which assumes a 2D ``(n_samples, n_classes)`` array.
        XGBoost can instead pass predictions as a flat 1D array of length
        ``n_samples * n_classes``, which crashes there. RamanBench pins log_loss/roc_auc
        as the stopping metric (both natively mapped -- "mlogloss"/"auc" -- so this path
        is not normally reached), but the guard is kept as a defensive fallback for any
        other metric choice (e.g. a custom scorer passed via ``model_extra_params``).
        Reshapes the flat array back to 2D before delegating; no-op otherwise.
        """
        from autogluon.core.constants import MULTICLASS, SOFTCLASS

        eval_metric = super().get_eval_metric()
        if not callable(eval_metric) or self.problem_type not in (MULTICLASS, SOFTCLASS):
            return eval_metric

        base_metric = eval_metric

        def _safe_custom_metric(y_true, y_hat):
            if y_hat.ndim == 1:
                n_classes = len(np.unique(y_true))
                if n_classes > 1 and len(y_hat) == len(y_true) * n_classes:
                    y_hat = y_hat.reshape(len(y_true), n_classes)
            return base_metric(y_true, y_hat)

        _safe_custom_metric.__name__ = base_metric.__name__
        return _safe_custom_metric


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


# Plain upstream AutoGluon caps these four tabular-foundation-model families at
# max_features between 500 and 2000 (see each model's own `_default_auxiliary_params_extra`
# in autogluon.tabular.models.{mitra,tabdpt,tabicl,tabpfnv2}) -- Raman spectra routinely
# run 500-4000 wavenumber points, well above that. RamanBench previously carried a
# patched AutoGluon fork that relaxed these caps upstream-side; that fork is no longer
# maintained (see git history around the AutoGluon 1.6 dependency bump) in favor of
# overriding the cap here instead, using AutoGluon's own supported per-subclass
# extension point (`_default_auxiliary_params_extra`, merged base-most-class-first so
# the most-derived class -- these Prep_* classes -- wins; see
# `AbstractModel._get_default_auxiliary_params` upstream). This is intentionally scoped
# to max_rows/max_features/max_classes only, matching exactly what the fork changed and
# nothing more (e.g. it does NOT touch AutoGluon's constraint-checking mechanism itself,
# `AbstractModel.validate_fit_args`, which stays fully intact for every other model).
#
# Accepted tradeoff: no fork-only extras beyond the cap relief (e.g. Mitra/TabPFN's
# ManyClassClassifier many-class support the fork also carried) are reproduced here, and
# TabICL v2 regression support -- the fork's other reason to exist -- is no longer needed
# at all since upstream 1.6 ships TabICL v2 with regression support natively. Some
# model x dataset combinations that worked under the fork (e.g. >10-class datasets on
# Mitra/TabPFN, which the fork routed through an ECOC many-class wrapper) may now fail
# or be skipped outright -- accepted in favor of depending on plain upstream AutoGluon.
_NO_FOUNDATION_MODEL_FEATURE_CAP = {"max_rows": None, "max_features": None, "max_classes": None}

Prep_REALMLP = _make_optional_prep_class("Prep_REALMLP", RealMLPModel, _supports_augmentation=True)
Prep_MITRA = _make_optional_prep_class(
    "Prep_MITRA", MitraModel, _default_auxiliary_params_extra=_NO_FOUNDATION_MODEL_FEATURE_CAP
)
Prep_TABM = _make_optional_prep_class("Prep_TABM", TabMModel)
Prep_TABDPT = _make_optional_prep_class(
    "Prep_TABDPT", TabDPTModel, _default_auxiliary_params_extra=_NO_FOUNDATION_MODEL_FEATURE_CAP
)
Prep_TABFM = _make_optional_prep_class("Prep_TABFM", TabFMModel)
Prep_TABICL = _make_optional_prep_class(
    "Prep_TABICL", TabICLModel, _default_auxiliary_params_extra=_NO_FOUNDATION_MODEL_FEATURE_CAP
)
Prep_REALTABPFN_V2 = _make_optional_prep_class(
    "Prep_REALTABPFN_V2", RealTabPFNv2Model, _default_auxiliary_params_extra=_NO_FOUNDATION_MODEL_FEATURE_CAP
)
Prep_REALTABPFN_V25 = _make_optional_prep_class(
    "Prep_REALTABPFN_V25", RealTabPFNv25Model, _default_auxiliary_params_extra=_NO_FOUNDATION_MODEL_FEATURE_CAP
)
Prep_REALTABPFN_V26 = _make_optional_prep_class("Prep_REALTABPFN_V26", RealTabPFNv26Model)
Prep_TABPFN_V3 = _make_optional_prep_class("Prep_TABPFN_V3", TabPFNv3Model)
Prep_TABPFN_V3_THINKING = _make_optional_prep_class(
    "Prep_TABPFN_V3_THINKING", TabPFNv3ThinkingModel
)


class Prep_KNN(_NoAugBase, KNNModel):  # noqa: N801
    """SNV normalises intensity scale so Euclidean distances reflect spectral shape."""

    _optimize_preprocessing = True

    def _set_default_params(self):
        self._set_default_param_value("prep_snv_enabled", True)
        super()._set_default_params()

    def _fit(self, X, y, **kwargs):
        # sklearn requires n_neighbors < n_samples_fit (strictly less than) for the
        # leave-one-out OOF path AutoGluon's bagging uses, not <=. On very small
        # datasets -- and the tiny inner CV/bagging folds produced during HPO -- the
        # default or searched n_neighbors can be >= n_samples, which sklearn rejects
        # ("Expected n_neighbors < n_samples_fit"), failing the whole fit. Clamp to
        # max(1, n_samples - 1); this previously clamped to n_samples exactly, which
        # is still one too high and left the same error reachable on tiny folds.
        n_neighbors = self._get_model_params().get("n_neighbors", 5)
        max_n_neighbors = max(1, len(X) - 1)
        if n_neighbors > max_n_neighbors:
            self.params["n_neighbors"] = max_n_neighbors
        super()._fit(X, y, **kwargs)


class Prep_LR(_NoAugBase, LinearModel):  # noqa: N801
    """Baseline correction + SNV are standard practice before linear regression."""

    _optimize_preprocessing = True

    def _set_default_params(self):
        self._set_default_param_value("prep_bl_enabled", True)
        self._set_default_param_value("prep_snv_enabled", True)
        super()._set_default_params()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

PREPROCESSED_MODELS = {
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

# Models migrated to the per-directory convention
# (raman_bench/models/custom/<key>/{model.py,hpo.py,info.py}, discovered via
# raman_bench.models.discover.discover_custom_models()) register themselves
# here instead of being hand-listed above. New models should be added there,
# not as a new dict entry in this file -- see RamanBench/.claude/agents/model-agent.md.
for _key, _info in discover_custom_models().items():
    PREPROCESSED_MODELS[_key] = _info.model_cls
del _key, _info

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
