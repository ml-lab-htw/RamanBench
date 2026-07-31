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

try:
    from autogluon.core.models import DummyModel
except ImportError as _ag_err:
    raise ImportError(
        "raman_bench.preprocessing.wrapped_models requires autogluon. "
        "Install with: pip install 'raman-bench[autogluon]'"
    ) from _ag_err
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

from raman_bench.models.discover import discover_custom_models
from raman_bench.preprocessing.bridge_bases import _make_optional_prep_class, _NoAugBase

# ---------------------------------------------------------------------------
# Built-in AutoGluon models (not yet migrated to the per-model-directory
# convention -- see the module docstring)
# ---------------------------------------------------------------------------


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
