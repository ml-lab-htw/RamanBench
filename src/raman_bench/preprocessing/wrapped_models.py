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
# these (observed missing: RealTabPFNv26Model), even though the "classic"
# models imported above have been stable across releases for years. Import
# defensively so a missing foundation-model class doesn't crash this whole
# module (and thus every other model, including plain PLS).
import warnings as _warnings

from autogluon.tabular import models as _ag_tabular_models

# EBMModel (unlike the foundation-model classes in _OPTIONAL_AG_MODEL_NAMES below)
# is imported unconditionally here despite also having an optional runtime
# dependency (the `interpret` package) -- confirmed via a real installed build:
# autogluon.tabular.models.ebm.ebm_model imports `interpret` lazily inside
# `_fit`/`get_class_from_problem_type`, never at module top-level, so the class
# itself is always importable even without `interpret` installed (same shape as
# CatBoostModel/XGBoostModel above, both also optional-at-fit-time).
from autogluon.tabular.models import (
    CatBoostModel,
    EBMModel,
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
    "TabICLModel",
    "TabMModel",
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

# TabFM, TabPFN-3, and TabSwift are a step earlier in that same graduation pipeline:
# unlike the classes above, they don't exist under *any* name in autogluon.tabular.models
# at all yet (confirmed against a real installed build -- no tabfm/tabswift submodule
# anywhere in the autogluon.tabular package tree). They ship only from TabArena's own
# package, one subpackage per model (tabarena.models.<key>.model). TabPFN-3 is a partial
# exception: autogluon.tabular.models.tabpfnv2.tabpfn3_model also defines a same-named
# `TabPFN3Model` -- a *different*, independently-implemented class, not a re-export --
# which is deliberately NOT used here: tabarena.models.tabpfn_3.hpo's search space (what
# generate/tabpfn_v3.py rebinds onto Prep_TABPFN_V3) is tuned against TabArena's own class,
# not AutoGluon's. Imported the same defensively as the block above: TabArena's own
# package is pulled in via a floating (unpinned) git dependency (see pyproject.toml), so a
# given snapshot could in principle rename/drop one of these just as an AutoGluon
# prerelease can.
_OPTIONAL_TABARENA_MODEL_IMPORTS = {
    "TabFMModel": "tabarena.models.tabfm.model",
    "TabPFN3Model": "tabarena.models.tabpfn_3.model",
    "TabSwiftModel": "tabarena.models.tabswift.model",
    "ModernNCAModel": "tabarena.models.modernnca.model",
    # Batch 2 (EBM, PerpetualBooster, xRFM, ChimeraBoost): same "not yet graduated
    # into AutoGluon core" situation, EXCEPT EBM, which already lives in
    # autogluon.tabular.models (imported unconditionally above) -- it graduated
    # some time ago, unlike these three. None of the three below carry the "TA-"
    # staging-prefix on their own ag_key (PerpetualBoosterModel.ag_key == "PB",
    # XRFMModel.ag_key == "XRFM", ChimeraBoostModel.ag_key == "CHIMERA") -- same
    # as ModernNCAModel above, not TabFM/TabPFN-3/TabSwift.
    "PerpetualBoosterModel": "tabarena.models.perpetual_booster.model",
    "XRFMModel": "tabarena.models.xrfm.model",
    "ChimeraBoostModel": "tabarena.models.chimeraboost.model",
    # Batch 3 (NORI, SAP_RPT_OSS, ORIONMSP, ILTM, LIMIX, TABSTAR) -- the final batch
    # of the 14-model TabArena-native onboarding effort (batches 1/2 above). Same
    # "TabArena-package-only, not (yet) graduated into AutoGluon core" situation as
    # every entry above -- confirmed against a real installed build, none of these
    # six exist under any name in autogluon.tabular.models. All six are pretrained/
    # fine-tuned tabular *foundation* models (in-context learning or LoRA
    # fine-tuning), unlike batch 2's tree/boosting family.
    "NoriModel": "tabarena.models.nori.model",
    "SAPRPTOSSModel": "tabarena.models.sap_rpt_oss.model",
    "OrionMSPModel": "tabarena.models.orionmsp.model",
    "ILTMModel": "tabarena.models.iltm.model",
    "LimiXModel": "tabarena.models.limix.model",
    "TabSTARModel": "tabarena.models.tabstar.model",
}
_missing_optional_tabarena_models = []
for _name, _module_path in _OPTIONAL_TABARENA_MODEL_IMPORTS.items():
    try:
        _module = __import__(_module_path, fromlist=[_name])
        globals()[_name] = getattr(_module, _name)
    except ImportError:
        globals()[_name] = None
        _missing_optional_tabarena_models.append(_name)
if _missing_optional_tabarena_models:
    _warnings.warn(
        f"This tabarena build is missing model classes: {_missing_optional_tabarena_models}. "
        "The corresponding RamanBench Prep_* models will be unavailable in this "
        "environment.",
        stacklevel=2,
    )
del _name, _module_path, _OPTIONAL_TABARENA_MODEL_IMPORTS

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


class Prep_EBM(_NoAugBase, EBMModel):  # noqa: N801
    """EBM (Explainable Boosting Machine) -- graduated AutoGluon-core model
    (``autogluon.tabular.models.EBMModel``), not a TabArena-only class, so it's
    defined here alongside CAT/RF/XT rather than via ``_make_optional_prep_class``.
    ``ag_key`` is already the short, unprefixed ``"EBM"`` -- no override needed.
    """

    def _estimate_memory_usage(self, X, y=None, **kwargs):
        """Strip ``prep_*``/``ag.*`` keys before EBM's own memory estimator sees them.

        Confirmed via a real local run: EBM (unlike CAT/XGB/RF/XT, whose memory
        estimators are purely arithmetic from ``X``'s shape) actually instantiates
        the real ``interpret`` estimator inside
        ``EBMModel._estimate_memory_usage_static`` (``model_cls(**params)``, to
        call its own ``.estimate_mem()``) -- ``construct_ebm_params`` merges
        ``hyperparameters`` in unfiltered (``params.update(hyperparameters)``, no
        allowlist), so any RamanBench-only key raises
        ``TypeError: ExplainableBoostingClassifier.__init__() got an unexpected
        keyword argument 'prep_aug_enabled'``.

        ``RamanPreprocessingMixin._fit()`` normally strips ``prep_*`` from
        ``self.params`` before the underlying library ever sees them, but
        ``AbstractModel.fit()`` calls memory validation (which calls this method)
        *before* ``_fit()`` runs, so at this point they're still present. Mirrors
        ``EBMModel._estimate_memory_usage`` exactly, just with a filtered
        ``hyperparameters`` copy -- ``self.params`` itself is untouched (the real
        strip/restore in ``_fit()`` still needs the full dict).
        """
        clean_params = {
            k: v
            for k, v in self._get_model_params().items()
            if not k.startswith("prep_") and not k.startswith("ag.") and not k.startswith("_")
        }
        return self.estimate_memory_usage_static(
            X=X,
            y=y,
            hyperparameters=clean_params,
            problem_type=self.problem_type,
            num_classes=self.num_classes,
            features=self._features,
            **kwargs,
        )


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
# TabFM, TabPFN-3, TabSwift, and ModernNCA (added below) were checked against this same
# issue -- inspected via each class's own `_get_default_auxiliary_params`/
# `_default_auxiliary_params_extra` in the installed tabarena package -- and, unlike the
# four above, none of them cap max_features (or max_rows) at all: AutoGluon's own
# `AbstractModel._get_default_auxiliary_params` base default is already `None` (uncapped)
# for all three keys, and nothing in any of these four classes' MRO overrides that. TabPFN-3
# does cap `max_classes` at 160 (`tabarena.models.tabpfn_3.model.TabPFN3Model
# ._get_default_auxiliary_params`) -- an intentional, unrelated limit on the number of
# *target classes* a single TabPFN-3 fit supports, not on wavenumber count, so it is left
# untouched here (Raman classification tasks are essentially never anywhere near 160
# classes). None of the four get `_NO_FOUNDATION_MODEL_FEATURE_CAP` applied.
_NO_FOUNDATION_MODEL_FEATURE_CAP = {"max_rows": None, "max_features": None, "max_classes": None}

Prep_REALMLP = _make_optional_prep_class("Prep_REALMLP", RealMLPModel, _supports_augmentation=True)
Prep_MITRA = _make_optional_prep_class(
    "Prep_MITRA", MitraModel, _default_auxiliary_params_extra=_NO_FOUNDATION_MODEL_FEATURE_CAP
)
Prep_TABM = _make_optional_prep_class("Prep_TABM", TabMModel)
Prep_TABDPT = _make_optional_prep_class(
    "Prep_TABDPT", TabDPTModel, _default_auxiliary_params_extra=_NO_FOUNDATION_MODEL_FEATURE_CAP
)
# TabFM/TabPFN-3/TabSwift's ag_key as inherited from their tabarena base class carries a
# "TA-" prefix (e.g. TabFMModel.ag_key == "TA-TABFM") -- TabArena's own marker for a model
# that hasn't (yet) graduated into AutoGluon core, unlike e.g. MitraModel/TabDPTModel/
# TabICLModel, whose ag_key is already the short, unprefixed form these Prep_* classes
# inherit unmodified. Overriding ag_key here to the short form keeps RamanBench's own
# model keys (configs/models/*.json, cluster/gpu_models.json, --model CLI values) short
# and TA-prefix-free like every other model, and -- for TabPFN-3 specifically -- avoids
# colliding with raman_bench.models.custom.ta_tabpfn_3, which already legitimately owns
# ag_key "TA-TABPFN-3" for the un-preprocessed TabArena baseline variant (kept as a
# separate, still-useful registry entry, not superseded by this one).
Prep_TABFM = _make_optional_prep_class("Prep_TABFM", TabFMModel, ag_key="TABFM")
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
# Wraps tabarena.models.tabpfn_3.model.TabPFN3Model (TabArena's own, actively-maintained
# TabPFN-3 implementation -- see the _OPTIONAL_TABARENA_MODEL_IMPORTS block above for why
# this is NOT autogluon.tabular.models.tabpfnv2.tabpfn3_model.TabPFN3Model, a same-named
# but different class). ag_name is also overridden (not just ag_key): left inherited it
# would collide with raman_bench.models.custom.ta_tabpfn_3's ag_name ("TA-TabPFN-3"),
# which shares the same base class -- harmless for lookups by ag_key (still unique) but
# would make infer_model_cls's ag_name-string branch pick whichever of the two classes
# happens to be first in the registry's model list, which is fragile.
Prep_TABPFN_V3 = _make_optional_prep_class(
    "Prep_TABPFN_V3", TabPFN3Model, ag_key="TABPFN-V3", ag_name="RamanBench-TabPFN-3"
)
Prep_TABPFN_V3_THINKING = _make_optional_prep_class(
    "Prep_TABPFN_V3_THINKING", TabPFNv3ThinkingModel
)
Prep_TABSWIFT = _make_optional_prep_class("Prep_TABSWIFT", TabSwiftModel, ag_key="TABSWIFT")
# ModernNCAModel's own ag_key ("MNCA") predates the "TA-" staging-prefix convention (it's
# an older tabarena model than TabFM/TabPFN-3/TabSwift) -- overridden to the spelled-out
# "MODERNNCA" purely for readability/consistency with the other three, not to dodge a
# collision (nothing else in this registry uses "MNCA" or "MODERNNCA").
Prep_MODERNNCA = _make_optional_prep_class("Prep_MODERNNCA", ModernNCAModel, ag_key="MODERNNCA")

# Batch 2 (EBM, PerpetualBooster, xRFM, ChimeraBoost) -- tree/boosting/kernel-family
# models, not tabular foundation models, so the max_features/max_rows/max_classes cap
# that forces _NO_FOUNDATION_MODEL_FEATURE_CAP on Mitra/TabDPT/TabICL/RealTabPFN wasn't
# expected here going in. Verified rather than assumed, the same way as the block above:
# instantiated each class and inspected `_get_default_auxiliary_params()` directly. None
# of the four cap any of the three keys -- Prep_EBM (defined above, next to Prep_XT)
# only overrides `valid_raw_types` via `_default_auxiliary_params_extra`;
# PerpetualBoosterModel and XRFMModel don't override `_get_default_auxiliary_params` at
# all; ChimeraBoostModel overrides it too, also only for `valid_raw_types`. None of the
# four get `_NO_FOUNDATION_MODEL_FEATURE_CAP` applied.
#
# ag_key: PerpetualBoosterModel ("PB") and ChimeraBoostModel ("CHIMERA") predate the
# "TA-" staging-prefix convention too (same situation as ModernNCA above) -- overridden
# to the spelled-out "PERPETUAL_BOOSTER"/"CHIMERABOOST" purely for readability, not to
# dodge a collision (nothing else in this registry uses "PB" or "CHIMERA" either; those
# short keys stay available for TabArena's own un-preprocessed baseline entries in
# raman_bench_model_registry, coexisting the same way "MNCA" does after Prep_MODERNNCA's
# override). XRFMModel's ag_key ("XRFM") already matches the desired short form -- no
# override needed.
Prep_PERPETUAL_BOOSTER = _make_optional_prep_class(
    "Prep_PERPETUAL_BOOSTER", PerpetualBoosterModel, ag_key="PERPETUAL_BOOSTER"
)
Prep_XRFM = _make_optional_prep_class("Prep_XRFM", XRFMModel)
Prep_CHIMERABOOST = _make_optional_prep_class(
    "Prep_CHIMERABOOST", ChimeraBoostModel, ag_key="CHIMERABOOST"
)

# Batch 3 (NORI, SAP_RPT_OSS, ORIONMSP, ILTM, LIMIX, TABSTAR) -- the final batch of
# the 14-model TabArena-native onboarding effort. All six are tabular *foundation*
# models. Checked the same way as every batch above: instantiated each class and
# inspected `_get_default_auxiliary_params()` directly against the installed
# tabarena build, rather than assuming from the class name.
#
# - NoriModel caps `max_rows` at 100_000 (its in-context-learning context-window
#   limit; NORI only supports regression -- see CLASSIFICATION_ONLY_MODELS's mirror
#   below). Checked against RamanBench's own precomputed dataset stats
#   (`data/precomputed/dataset_stats.json`): the largest regression dataset
#   (`sugar_mixtures_low_snr`) has 7,840 rows, nowhere near the cap. No override.
# - SAPRPTOSSModel, OrionMSPModel, ILTMModel, TabSTARModel cap none of
#   max_rows/max_features/max_classes at all.
# - LimiXModel caps `max_classes` at 10 -- and unlike NoriModel's max_rows headroom,
#   this DOES collide with real RamanBench classification datasets: confirmed via
#   `data/precomputed/dataset_stats.json`, `bacteria_identification` (30 classes),
#   `pharmaceutical_ingredients` (32), `rruff_mineral_raw` (79), `mlrod` (16), and
#   the `cancer_cell_*` trio (12 each) all exceed it. UNLIKE Mitra/TabDPT/TabICL/
#   RealTabPFN's cap, though, this one can't be lifted by simply passing
#   `_default_auxiliary_params_extra=_NO_FOUNDATION_MODEL_FEATURE_CAP` to
#   `_make_optional_prep_class` -- confirmed via a real local check (instantiating
#   Prep_LIMIX and calling `_get_default_auxiliary_params()` still returned
#   `max_classes: 10` with that kwarg in place). `LimiXModel._get_default_auxiliary_params`
#   (`tabarena/models/limix/model.py`) doesn't rely on the declarative
#   `_default_auxiliary_params_extra` class-attribute merge
#   `AbstractModel._get_default_auxiliary_params` performs over `type(self).__mro__`
#   at all -- it fully overrides the method itself, calling `super()` and then
#   unconditionally `dict.update()`-ing `max_classes=10` back in, which clobbers
#   any subclass's declared `_default_auxiliary_params_extra` no matter where it
#   sits in the MRO. Same *class* of bug as `Prep_EBM._estimate_memory_usage` in
#   batch 2 (a tabarena model class doing something non-declarative that the
#   generic override hook can't see) -- fixed the same way, with a real method
#   override below instead of a declarative kwarg.
#
# ag_key: NoriModel ("TA-NORI"), OrionMSPModel ("TA-ORION-MSP"), ILTMModel
# ("TA-ILTM"), and LimiXModel ("TA-LIMIX") inherit the "TA-" staging-prefix from
# their tabarena base class (same situation as TabFM/TabPFN-3/TabSwift in batch 1)
# -- overridden to a short form below. SAPRPTOSSModel's own ag_key ("SAP-RPT-OSS")
# already matches RamanBench's naming except for its hyphens -- every other
# multi-word key in this registry uses underscores (PERPETUAL_BOOSTER, NN_TORCH),
# so it's overridden to "SAP_RPT_OSS" purely for that consistency, not to dodge a
# collision. TabSTARModel's own ag_key ("TABSTAR") already matches exactly -- no
# override needed (nor is ag_name: "TabSTAR" doesn't collide with anything already
# in this registry).
Prep_NORI = _make_optional_prep_class("Prep_NORI", NoriModel, ag_key="NORI")
Prep_SAP_RPT_OSS = _make_optional_prep_class(
    "Prep_SAP_RPT_OSS", SAPRPTOSSModel, ag_key="SAP_RPT_OSS"
)
Prep_ORIONMSP = _make_optional_prep_class("Prep_ORIONMSP", OrionMSPModel, ag_key="ORIONMSP")
Prep_ILTM = _make_optional_prep_class("Prep_ILTM", ILTMModel, ag_key="ILTM")

if LimiXModel is None:
    Prep_LIMIX = None
else:

    class Prep_LIMIX(_NoAugBase, LimiXModel):  # noqa: N801
        """LimiX -- see the batch-3 comment block above for why this can't be built
        via ``_make_optional_prep_class`` like the other five models in this batch.
        """

        ag_key = "LIMIX"

        def _get_default_auxiliary_params(self) -> dict:
            """Re-clobber ``LimiXModel``'s own hardcoded ``max_classes=10`` back to
            uncapped, after it (unconditionally, un-overridably via the normal
            declarative hook) sets it. See the batch-3 comment block above for the
            full explanation; mirrors ``Prep_EBM._estimate_memory_usage``'s shape
            for a different non-declarative override in batch 2.
            """
            default_auxiliary_params = super()._get_default_auxiliary_params()
            default_auxiliary_params.update(_NO_FOUNDATION_MODEL_FEATURE_CAP)
            return default_auxiliary_params

Prep_TABSTAR = _make_optional_prep_class("Prep_TABSTAR", TabSTARModel)


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
    "TABSWIFT": Prep_TABSWIFT,
    "MODERNNCA": Prep_MODERNNCA,
    "EBM": Prep_EBM,
    "PERPETUAL_BOOSTER": Prep_PERPETUAL_BOOSTER,
    "XRFM": Prep_XRFM,
    "CHIMERABOOST": Prep_CHIMERABOOST,
    "NORI": Prep_NORI,
    "SAP_RPT_OSS": Prep_SAP_RPT_OSS,
    "ORIONMSP": Prep_ORIONMSP,
    "ILTM": Prep_ILTM,
    "LIMIX": Prep_LIMIX,
    "TABSTAR": Prep_TABSTAR,
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

CLASSIFICATION_ONLY_MODELS = {"ROCKET", "ARSENAL", "TABPFN-WIDE", "ORIONMSP"}

# Mirror of CLASSIFICATION_ONLY_MODELS: NORI (OrionMSPModel's opposite number in
# batch 3) wraps NoriModel, whose own supported_problem_types() returns only
# ["regression"] (tabarena.models.nori.model.NoriModel._fit raises AssertionError
# for anything else). Nothing in this registry needed a regression-only entry
# before batch 3 -- predictions.py's classification-only skip already had a home
# (CLASSIFICATION_ONLY_MODELS); this is the first model needing the reverse.
REGRESSION_ONLY_MODELS = {"NORI"}


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
