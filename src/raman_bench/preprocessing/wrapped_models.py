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

import math

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


_EBM_WIDE_FEATURE_THRESHOLD = 4000


class Prep_EBM(_NoAugBase, EBMModel):  # noqa: N801
    """EBM (Explainable Boosting Machine) -- graduated AutoGluon-core model
    (``autogluon.tabular.models.EBMModel``), not a TabArena-only class, so it's
    defined here alongside CAT/RF/XT rather than via ``_make_optional_prep_class``.
    ``ag_key`` is already the short, unprefixed ``"EBM"`` -- no override needed.
    """

    def _fit(self, X, y, **kwargs):
        """Disable ``interactions`` for wide feature counts (>``_EBM_WIDE_FEATURE_THRESHOLD``).

        ``interpret``'s own default (``interactions="3x"``, i.e. fit ``3 *
        n_features`` pairwise interaction terms, selected via its own FAST
        interaction-ranking pre-scan over ALL candidate feature pairs) scales
        combinatorially with feature count -- confirmed via a real local timed
        run of the actual pipeline (``run_experiment.py --model EBM --dataset
        microgel_synthesis``, 11,084 features): the pre-scan alone produced
        millions of per-pair log lines and had not finished after 9+ minutes
        wall time, well before boosting itself even starts. Unlike the boosting
        rounds -- which DO respect ``EbmCallback``/``time_limit`` (see
        ``autogluon.tabular.models.ebm.ebm_model.construct_ebm_params``) -- this
        pre-scan does not check the time budget at all, so a bigger
        ``time_limit`` alone cannot bound it.

        A *small positive* ``interactions`` count does NOT avoid this --
        confirmed the hard way (a first attempt at this fix set
        ``interactions=20``, and the "Fast interaction strength" flood
        continued unchanged against the real pipeline). ``interpret``'s own
        ``rank_interactions`` FAST algorithm has to rank every candidate pair
        before it can keep only the top-N; only literal ``interactions=0``
        skips the ranking loop entirely (see
        ``interpret.glassbox._ebm._ebm.py``: ``if interactions == 0: break``,
        *before* the ``rank_interactions`` call -- any other value, however
        small, still reaches it). So this disables interaction terms outright
        for wide data rather than merely capping the count -- a real behavior
        change (EBM becomes a pure additive/GAM model, no pairwise terms, for
        these datasets specifically), but a legitimate, ``interpret``-supported
        configuration (not a hack), and the only way to actually remove the
        blowup. The remaining, much cheaper per-round main-effects boosting
        cost (roughly linear in feature count, ~1.5-1.8s/round measured at
        11,084 features) is still bounded normally by ``time_limit``.

        Threshold matches ``wrapped_models._TABSTAR_MAX_FEATURES`` -- both sit in
        the same real, dataset-free gap in RamanBench's own feature-count
        distribution (no target in ``configs/v1/target_list.json`` has between
        ~3,300 and ~5,470 features), so one consistent "wide" boundary applies
        across both fixes rather than two arbitrarily different numbers.

        Same data-shape-adaptive pattern as ``Prep_KNN._fit``'s ``n_neighbors``
        clamp -- only kicks in above the threshold; default AutoGluon/interpret
        behavior (and model quality, including interaction terms) is unchanged
        for RamanBench's more common narrower spectra.
        """
        n_features = X.shape[1]
        if n_features > _EBM_WIDE_FEATURE_THRESHOLD:
            interactions = self._get_model_params().get("interactions", "3x")
            if not (isinstance(interactions, (int, float)) and interactions == 0):
                self.params["interactions"] = 0
        super()._fit(X, y, **kwargs)

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


def _patch_limix_pickle_bug(limix_model_module) -> None:
    """RamanBench-local workaround for a real upstream ``tabarena`` bug.

    ``tabarena.models.limix.model._nan_clean_encoder_cls()`` is a ``functools.cache``d
    factory that builds ``_NaNCleanEncoder`` as a class local to the factory's own
    function body (deliberately -- see that function's docstring -- so importing this
    module doesn't transitively import ``torch``). A class built inside a function gets
    the default qualname ``_nan_clean_encoder_cls.<locals>._NaNCleanEncoder``, which
    ``pickle`` cannot resolve. AutoGluon's bagged-ensemble ``save_child()`` pickles every
    fold child right after it finishes training, so every LIMIX run crashes at that step
    -- confirmed on 4/4 real cluster runs (both classification and regression), always
    *after* training completed successfully::

        AttributeError: Can't pickle local object
        '_nan_clean_encoder_cls.<locals>._NaNCleanEncoder'

    Reported upstream with a fix (rewrite the produced class's ``__qualname__`` to a
    plain, module-resolvable name, and add a module-level ``__getattr__`` (PEP 562) that
    rebuilds/returns the -- ``functools.cache``-stable -- class on demand, so ``pickle``
    can resolve ``tabarena.models.limix.model._NaNCleanEncoder`` both in the same process
    that trained the model and in a cold process that never called the factory, e.g. a
    fresh ``TabularPredictor.load()``): https://github.com/autogluon/tabarena/pull/468.

    This function reproduces that exact fix at runtime, monkeypatching the installed
    ``tabarena`` package in place, so LIMIX runs don't have to wait for that PR to merge
    and release. It is idempotent (a no-op if already patched, including once the fix
    ships upstream and this module already defines its own ``__getattr__``) and verified
    to round-trip pickle/unpickle both within one process and across a cold process that
    never touched the factory, without ``torch`` ending up in ``sys.modules`` merely from
    importing ``tabarena.models.limix.model``.

    Remove once RamanBench's ``tabarena`` pin includes the merged upstream fix.
    """
    if "__getattr__" in vars(limix_model_module):
        return

    original_factory = limix_model_module._nan_clean_encoder_cls

    def _patched_factory():
        cls = original_factory()
        cls.__qualname__ = cls.__name__
        return cls

    limix_model_module._nan_clean_encoder_cls = _patched_factory

    def _module_getattr(name):
        if name == "_NaNCleanEncoder":
            return _patched_factory()
        raise AttributeError(f"module {limix_model_module.__name__!r} has no attribute {name!r}")

    # PEP 562: a module-level `__getattr__` in the module's own namespace dict is enough
    # -- it doesn't need to be defined with `def __getattr__` syntax at parse time.
    limix_model_module.__getattr__ = _module_getattr


if LimiXModel is None:
    Prep_LIMIX = None
else:
    import tabarena.models.limix.model as _limix_model_module

    _patch_limix_pickle_bug(_limix_model_module)
    del _limix_model_module

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

# TabSTAR builds a per-column LM text embedding (see tabstar/arch/arch.py's
# get_textual_embedding): memory scales with FEATURE count, not row count --
# matches upstream's own documented warning about >200-column datasets. Confirmed
# via real batch-3 verification: fine on diabetes_skin_ear_lobe (2,803-3,160
# features depending on how post-preprocessing columns are counted -- see below),
# OOM-killed (`RuntimeError("OOM even with batch size 1!")`) on microgel_synthesis
# (11,084 features), both on the same ~36GB machine, even though
# microgel_synthesis has FEWER rows (14 vs 20) -- ruling out row count as the
# driver. Reproduced directly against tabstar.arch.arch.TabStarModel in
# isolation (synthetic per-cell text, same LoRA freeze scheme as
# tabstar/training/lora.py's `to_freeze = range(6)`, CPU, gradient tracking
# enabled to match real fine-tuning): memory grows steeply with feature count
# even at a few hundred to ~1,000 features (into the tens of GB) -- confirming
# the mechanism (materializing a (batch_rows x n_features x d_model) embedding
# tensor per forward pass, GRADIENT-tracked, not released until backward())
# is real and severe, well beyond what "a bit more time/memory" would fix.
#
# `_NO_FOUNDATION_MODEL_FEATURE_CAP` (used above to LIFT AutoGluon's default
# max_features cap for Mitra/TabDPT/TabICL/RealTabPFN, which top out around
# 500-2000) is the wrong direction here: TabSTAR genuinely cannot handle
# RamanBench's widest spectra, so this goes the other way -- an actual cap,
# using the SAME AutoGluon mechanism (`ag.max_features`, which produces a
# clean `ConstraintViolationError` one-line skip, not a crash -- see
# `autogluon.core.models.abstract.abstract_model.AbstractModel
# .validate_fit_args`/`autogluon.core.utils.exceptions.ConstraintViolationError`).
#
# Cap value (4,000) is chosen from RamanBench's real, current dataset
# distribution (`configs/v1/target_list.json`, cross-referenced against
# `data/precomputed/dataset_stats.json`), not a guess: the 66-target v1 scope
# has a completely dataset-free gap between the widest confirmed-safe target
# (pharmaceutical_ingredients, 3,276 features) and the next-widest target
# (bioprocess_analytes_kaiser, 5,472 features) -- above which sits the
# acid-species/microgel cluster (9 targets, 11,084-11,689 features) that
# actually produced the OOM. 4,000 sits in that empty gap: ~22% of headroom
# above the highest confirmed-working real target, ~27% of margin below the
# next real target, cleanly separating "confirmed-safe-plus-margin" (56 of 66
# targets stay eligible) from "genuinely too wide for this model" (10 of 66:
# bioprocess_analytes_kaiser + the 9-target ultra-wide cluster) without
# guessing at any dataset in between (there isn't one). See
# `wrapped_models.MAX_FEATURES_MODELS` (consumed by
# `scripts/run_experiment.py::run_one()`, mirroring how
# `CLASSIFICATION_ONLY_MODELS`/`REGRESSION_ONLY_MODELS` are consumed) for the
# job-level clean skip -- belt-and-suspenders with the AutoGluon-level
# `max_features` cap below, since RamanBench's own cluster jobs fit exactly one
# model at a time (no other model for AutoGluon to fall back on), and
# AutoGluon's `raise_on_no_models_fitted` default would otherwise turn "the one
# model was cleanly constraint-skipped" into a job-crashing `RuntimeError`.
#
# For the sub-cap-but-still-large cases (pharmaceutical_ingredients at 3,276,
# the diabetes_skin_* family at 3,160), `cluster/profiles/htw.yaml`'s
# `mem_tiers` gets a TABSTAR entry bumped to 128G (matching the other
# foundation models in its tier -- MITRA/TABDPT/TABFM/TABSWIFT) rather than the
# 64G `default_mem`, for headroom beyond what the ~36GB laptop where the
# original OOM was found provides. The 10 excluded-by-cap targets are NOT
# expected to become feasible merely by throwing more memory at them --
# unlike the sub-cap cases, that's not a "needs a bit more headroom" gap, it's
# the regime that produced "OOM even with batch size 1" -- so no amount of
# memory tier is substituted for the cap itself there (see issue writeup /
# CHANGELOG for the full reasoning).
_TABSTAR_MAX_FEATURES = 4000

# GPU memory does NOT reset between AutoGluon's sequential bagged folds -- confirmed as a
# real production failure, not a hypothetical: a real HTW cluster run (job 33535, RamanBench
# default `num_bag_folds=8`) OOM-killed 6/6 tasks on `diabetes_skin_ear_lobe`
# (2,803-3,160 features, 20 rows -- comfortably under the 4,000 cap above, and the exact
# dataset the batch-3 memory characterization above called "confirmed-safe") with
# `torch.OutOfMemoryError: ... 76.32 GiB is allocated by PyTorch` on a 79.25 GiB GPU, for a
# 20-row fit. Cross-job GPU contention was independently ruled out first (4 concurrent
# diagnostic SLURM jobs confirmed SLURM cgroup-isolates each job to its own distinct
# physical GPU).
#
# Root cause: `TabSTARModel._get_default_ag_args_ensemble` (tabarena) forces
# `fold_fitting_strategy: "sequential_local"` (parallel folding isn't safe here yet -- see
# that method's own docstring/TODO: "switch to parallel fitting on one GPU once VRAM memory
# estimation is supported"; `_class_tags` also declares
# `can_estimate_memory_usage_static: False`), so all `num_bag_folds` child fits happen
# sequentially inside ONE process. Each fold's `BaseTabSTAR.fit()`
# (`tabstar/tabstar_model.py`) builds a `TabStarTrainer` (`tabstar/training/trainer.py`)
# whose `self.optimizer`/`self.scheduler`/`self.scaler` hold live references to that fold's
# full parameter set (frozen backbone + LoRA adapters) for the duration of the local
# `trainer` variable's lifetime inside `BaseTabSTAR.fit()`. `TabStarTrainer.load_model()`
# *does* call `gc.collect()`/`torch.cuda.empty_cache()` -- but before the optimizer holding
# the old (pre-averaging) model's tensors is dropped, so that call is a no-op for the
# fold's actual training-time footprint; nothing downstream (`TabSTARModel._fit`, nor
# AutoGluon's own `SequentialLocalFoldFittingStrategy`/`_predict_oof`/
# `_update_bagged_ensemble`, which only plain-dereferences via `fold_model.model = None`)
# ever forces a cyclic-GC sweep after a fold's `trainer` object itself goes out of scope --
# and `nn.Module`/PEFT/autograd object graphs are exactly the kind that commonly form
# reference cycles CPython's refcounting alone won't free promptly, deferring reclamation
# to whenever the interpreter's generational collector happens to run (which is not tied to
# GPU memory pressure at all). Net effect: each sequential fold leaves a growing amount of
# genuinely still-"allocated" (not merely cached) GPU memory behind, accumulating fold over
# fold within the one process instead of staying bounded to ~1 fold's footprint.
#
# This is exactly why earlier batch-3 verification (`results/v1/smoke_resource_fixes/data/
# TabSTAR_c1_BAG_L1/kaiser_ecoli_fermentation__0/0_0/results.pkl`) missed it: that smoke run
# used `num_bag_folds=2` (not RamanBench's real `DEFAULT_NUM_BAG_FOLDS=8` from
# `cluster/submit_job.py`/`scripts/run_experiment.py`) on CPU (`gpu_tracking_enabled:
# False`), where host RAM headroom absorbed 2 folds' worth of un-released memory
# (`peak_mem_cpu` there: 21,076,115,456 bytes = 19.63 GiB, i.e. ~9.81 GiB/fold) without
# incident. Extrapolated *linearly* to production's 8 folds: 9.81 * 8 = 78.51 GiB -- versus
# the real OOM's 76.32 GiB allocated + 2.38 GiB reserved-unallocated = 78.70 GiB. That is a
# <0.3% match on an entirely independent dataset/run, strong quantitative confirmation this
# is genuine per-fold accumulation (roughly linear in fold count), not dataset-specific bad
# luck -- and explains the uniform 6/6 failure (every array task shares the same
# `num_bag_folds=8` default, so all six are equally exposed).
#
# Fix: force a real release point around each fold's own `_fit()` call -- `gc.collect()`
# (to break whatever reference cycle is deferring reclamation) THEN
# `torch.cuda.empty_cache()` (to return the now-actually-freed blocks to the allocator, sos
# fragmentation from differently-shaped folds can't compound either), both before AND after
# `super()._fit()`: the "before" call cleans up whatever the *previous* fold left behind
# before this fold starts consuming budget (the one that actually caps cross-fold growth);
# the "after" call reclaims this fold's own training-time garbage (the `trainer`/`optimizer`
# cycle) before OOF prediction and the next fold begin. This is RamanBench-side only (no
# tabarena/tabstar patching, unlike the LIMIX pickling workaround above) since the hook
# point is a plain method override -- same shape as `Prep_EBM._fit`'s wide-feature
# `interactions=0` override and `Prep_LIMIX._get_default_auxiliary_params`'s re-clobber,
# just at `_fit` instead. No local GPU was available to instrument peak VRAM directly across
# folds; verification is the real before/after cluster run recorded in CHANGELOG.md.
#
# Not filed upstream (yet, pending user sign-off): no existing tabarena or tabstar issue
# covers this (checked both repos' issue trackers). A draft upstream report is warranted --
# real GPU-memory-budget projects (RamanBench included) hit this the moment they combine
# TabSTAR with `num_bag_folds` > ~2-3 on real VRAM limits, and the ~9-10 GiB/fold footprint
# this uncovers is itself worth flagging even independent of the reference-cycle angle,
# since it's far larger than a 20-row fit should plausibly need. See CHANGELOG.md for the
# decision on whether/how it was filed.


if TabSTARModel is None:
    Prep_TABSTAR = None
else:

    class Prep_TABSTAR(_NoAugBase, TabSTARModel):  # noqa: N801
        """TabSTAR -- see the comment block above for why this can't be built via
        ``_make_optional_prep_class`` like most other batch-3 models (needs a real
        ``_fit`` override, not just a declarative ``_default_auxiliary_params_extra``
        merge -- same *class* of exception as ``Prep_LIMIX`` above, for an
        unrelated reason).
        """

        ag_key = "TABSTAR"
        _default_auxiliary_params_extra = {"max_features": _TABSTAR_MAX_FEATURES}

        def _fit(self, X, y, **kwargs):
            """Force a real GPU-memory release point around each bagged fold's fit.

            See the module-level comment block above ``Prep_TABSTAR`` for the full
            root-cause writeup and the quantitative evidence tying this to
            cross-fold GPU memory accumulation under AutoGluon's
            ``sequential_local`` fold-fitting strategy specifically (not a
            single-fit memory *ceiling* problem -- that's what
            ``_TABSTAR_MAX_FEATURES`` already guards against).
            """
            import gc

            import torch

            if torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()
            super()._fit(X, y, **kwargs)
            if torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()


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

# A model this registry has confirmed genuinely cannot handle RamanBench's widest
# spectra (as opposed to CLASSIFICATION_ONLY_MODELS/REGRESSION_ONLY_MODELS'
# problem-type mismatch) -- keyed by model name -> max feature count. Consumed by
# scripts/run_experiment.py::run_one() as a clean, job-level skip (return None,
# log a message, exit 0 -- no results.pkl written, same convention as the
# rare-class-filtering / all-NaN-label skips already there), mirroring exactly how
# CLASSIFICATION_ONLY_MODELS/REGRESSION_ONLY_MODELS are consumed by
# predictions.py. This is belt-and-suspenders with the AutoGluon-level
# `max_features` cap set on the model class itself (see Prep_TABSTAR above): the
# AutoGluon-level cap alone produces a clean `ConstraintViolationError` skip
# in a multi-model `TabularPredictor.fit()` call, but RamanBench's own cluster
# jobs always fit exactly one model, so with no other model to fall back on,
# AutoGluon's `raise_on_no_models_fitted=True` default (see
# `autogluon.tabular.predictor.predictor.TabularPredictor._post_fit`) turns that
# same clean skip into a job-crashing `RuntimeError: No models were trained
# successfully during fit()` -- confirmed with a real local run against a
# too-wide dataset. This dict lets `run_experiment.py` catch that case BEFORE
# ever calling into AutoGluon, for a real clean exit instead.
MAX_FEATURES_MODELS = {"TABSTAR": _TABSTAR_MAX_FEATURES}

# ORIONMSP: a real production CUDA OOM (`pharmaceutical_ingredients`, 3,276
# features, 2,340 train rows, on a 79.25 GiB GPU: "Tried to allocate 94.47
# GiB. ... 63.84 GiB is free") looked at first like the same shape of problem
# as TabSTAR above -- but it is NOT a single-number `max_features` situation,
# confirmed by reading the actual source rather than assuming from the
# traceback alone (`tabtune.models.orionmsp_v15.model.{interaction,
# attention}.py`, installed alongside `tabarena`'s own
# `tabarena.models.orionmsp.model.OrionMSPModel`, which is what
# `Prep_ORIONMSP` above wraps):
#
# - `OrionMSPv15._train_forward` (`orionmsp_v15.py`) feeds the ENTIRE
#   training partition through `RowInteraction` (`interaction.py`) in one
#   forward pass, with no row-chunking at all -- chunking
#   (`InferenceManager`/`mgr_config`) only exists on the separate
#   `_inference_forward` path, used for predict, not fit.
# - `RowInteraction._run_one_scale` builds a per-row attention sequence of
#   length `L = num_special (6 = row_num_cls=4 + row_num_global=2) +
#   ceil(n_features / features_per_group=2)` and runs it through
#   `Encoder`/`multi_head_attention_forward` (`attention.py`), which -- for
#   the arbitrary boolean sparse mask this model builds
#   (`_build_block_sparse_mask`) -- falls back to PyTorch SDPA's dense
#   "math" backend and materializes a full `(n_rows, nhead, L, L)` score
#   tensor. `n_rows` here is `batch_shape[1]` in `multi_head_attention_
#   forward` -- i.e. every row in the forward call is an independent batch
#   element for this attention, not part of a shared L-length sequence, so
#   this allocation scales LINEARLY in row count on top of QUADRATICALLY in
#   feature count. None of OrionMSP's hyperparameters are tunable through
#   this integration (`tabarena.models.orionmsp.hpo.gen_orionmsp`:
#   `search_space={}`, `manual_configs=[{}]`), so `embed_dim=128` ->
#   `row_nhead=8`, `features_per_group=2`, `row_num_cls=4`,
#   `row_num_global=2` are fixed constants for every real config, not
#   defaults that might vary.
#
# Reproducing this formula (`n_rows * nhead(8) * L**2 * 2 bytes` -- 2 bytes
# because `use_amp=True` by default, i.e. fp16 during this forward) against
# the real failure: `L = 6 + ceil(3276/2) = 1644`, predicted
# `2340 * 8 * 1644**2 * 2 / 1024**3 = 94.24 GiB` -- 0.24% off the actual
# "Tried to allocate 94.47 GiB" in the traceback. That is close enough to be
# the literal tensor that failed to allocate, not a coincidental
# order-of-magnitude match, so this formula (not a features-only guess) is
# what the cap below is built from.
#
# Upstream's own `OrionMSPModel._fit` (`tabarena/models/orionmsp/model.py`)
# already carries a real, verified-in-source mitigation -- not a secondhand
# paraphrase, confirmed by reading it directly: `if X.shape[1] > 500:
# hps["batch_size"] = 1  # avoid OOM for wide datasets`, next to the comment
# "Needs up to 400GB VRAM for datasets with 1k features." `batch_size`
# controls how many of the classifier's `n_estimators=64` ensemble members
# are processed together at inference time -- a real lever, but for a
# DIFFERENT dimension (ensemble width) than the one that actually OOM'd here
# (row count during `_train_forward`, which never consults `batch_size` at
# all). Confirmed insufficient by the failure itself: `pharmaceutical_
# ingredients` has 3,276 > 500 features, so this fallback was already
# active, and it still OOM'd.
#
# A TabSTAR-style single `max_features` cap would also be the wrong
# mechanism here, not just an unnecessary extra dimension -- confirmed by
# cross-referencing `configs/v1/target_list.json` against
# `data/precomputed/dataset_stats.json` for all 152 real v1 targets with
# known dataset stats:
# - `mlrod` (1,836 features, 130,061 rows), `wheat_lines` (1,748 features,
#   53,134 rows), and `bacteria_identification` (1,000 features, 78,500
#   rows) all have FEWER features than the already-OOMing `pharmaceutical_
#   ingredients` (3,276), yet predict far larger attention buffers (1,103 /
#   409 / 200 GiB respectively) from row count alone -- a features-only cap
#   set anywhere near 3,276 would let all three straight through to a GPU
#   job that predictably OOMs far worse than the one that triggered this
#   investigation.
# - `sugar_mixtures_high_snr` and `sugar_mixtures_low_snr` share the exact
#   same 2,000 features but differ 4x in row count (1,960 vs 7,840 total
#   instances) and land on opposite sides of any reasonable budget (19.7 vs
#   78.8 GiB predicted) -- proof row count is genuinely load-bearing here,
#   not just noise around a feature-count signal.
#
# So the cap below is a joint predicate (features AND rows), not a second
# `MAX_FEATURES_MODELS` entry. The budget (40 GiB) is chosen the same way
# TabSTAR's 4,000 was: it sits in a real, wide, dataset-free gap in the
# predicted-GiB distribution across all 152 real v1 targets -- nothing
# between 26.30 GiB (`flow_microgel_synthesis`, kept) and 61.22 GiB
# (`bioprocess_substrates`, excluded), so any budget from ~27-60 GiB
# produces the IDENTICAL partition (7 of ~72 real datasets / 17 of 152
# targets excluded: `mlrod`, `wheat_lines`, `bacteria_identification`,
# `pharmaceutical_ingredients`, `sugar_mixtures_low_snr`, `microgel_size_
# raw_global`, `bioprocess_substrates`) -- 40 is not a fragile choice. It
# also leaves real margin below the empirically-observed ceiling: the real
# OOM reported 63.84 GiB free (of 79.25 GiB total) at the moment of
# failure, and this formula deliberately only models the single dominant
# attention-buffer allocation, not the smaller baseline overhead (checkpoint
# weights, the column-embedding activation tensor, CUDA context -- ~15.4 GiB
# in the real failure) that also grows mildly with n_rows/n_features -- 40
# GiB leaves roughly 24 GiB of headroom below 63.84 for that.
_ORIONMSP_ROW_NHEAD = 8
_ORIONMSP_ROW_NUM_SPECIAL = 6  # row_num_cls (4) + row_num_global (2)
_ORIONMSP_FEATURES_PER_GROUP = 2
_ORIONMSP_ATTN_BYTES_PER_ELEMENT = 2  # fp16 (use_amp=True by default)
_ORIONMSP_MAX_PREDICTED_ATTN_GIB = 40.0


def _orionmsp_predicted_attn_gib(n_features: int, n_rows: int) -> float:
    """Predicted peak size (GiB) of OrionMSP's ``RowInteraction`` attention buffer.

    See the comment block above this function for the full derivation and
    the real-failure calibration (94.24 GiB predicted vs. 94.47 GiB actually
    attempted, 0.24% off, for ``pharmaceutical_ingredients``).
    """
    seq_len = _ORIONMSP_ROW_NUM_SPECIAL + math.ceil(n_features / _ORIONMSP_FEATURES_PER_GROUP)
    n_bytes = n_rows * _ORIONMSP_ROW_NHEAD * seq_len * seq_len * _ORIONMSP_ATTN_BYTES_PER_ELEMENT
    return n_bytes / 1024**3


def _orionmsp_exceeds_vram_budget(n_features: int, n_rows: int) -> bool:
    return _orionmsp_predicted_attn_gib(n_features, n_rows) > _ORIONMSP_MAX_PREDICTED_ATTN_GIB


# Joint (features AND rows) counterpart to `MAX_FEATURES_MODELS` -- keyed by
# model name -> a `(n_features, n_rows) -> bool` predicate (True = skip)
# instead of a single int, since (see the comment block above) a single
# number can't express OrionMSP's real constraint. Consumed the same way by
# `scripts/run_experiment.py::run_one()`: a clean, job-level skip (return
# `None`, log, exit 0) BEFORE ever calling into AutoGluon/CUDA, for the same
# `raise_on_no_models_fitted=True` reason `MAX_FEATURES_MODELS`'s own
# docstring explains.
VRAM_CAPPED_MODELS = {"ORIONMSP": _orionmsp_exceeds_vram_budget}


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
