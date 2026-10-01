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
Arsenal, TabPFN-Wide) and GBM/TA-TABPFN-3/TA-MITRA-V2/EXAONE-Tabular have moved to the per-model
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
    # TabPFN-3.5 (September 2026 release, one multitask checkpoint for both
    # classification and regression) -- same staging situation as TabPFN-3
    # above: TabArena-package-only, not (yet) graduated into AutoGluon core.
    # Its own ag_key ("TA-TABPFN-3.5") carries the same "TA-" staging prefix,
    # overridden below (Prep_TABPFN_V3_5) purely for naming consistency with
    # REALTABPFN-V2.5/V2.6 -- no collision to dodge (nothing else in this
    # registry uses "TA-TABPFN-3.5").
    "TabPFN35Model": "tabarena.models.tabpfn_3_5.model",
    "TabSwiftModel": "tabarena.models.tabswift.model",
    "ModernNCAModel": "tabarena.models.modernnca.model",
    # TabICLv2 (tabarena.models.tabicl.model.TabICLv2Model) is a *different* class
    # from the "TabICL" (v1) wrapped above via autogluon.tabular.models.TabICLModel
    # (_OPTIONAL_AG_MODEL_NAMES): v1 has graduated into AutoGluon core, v2 hasn't --
    # it exists only in tabarena's own package, same staging situation as
    # TabFM/TabPFN-3/TabSwift/ModernNCA above (confirmed: no `TabICLv2Model` anywhere
    # in `autogluon.tabular.models`, checked against the pinned tabarena commit in
    # requirements-tabarena-git.txt). Its own hpo module (tabarena.models.tabicl.hpo)
    # already exposes both `gen_tabicl` (bound to `TabICLModel`) and `gen_tabiclv2`
    # (bound to this class) side by side -- see generate/tabiclv2.py.
    "TabICLv2Model": "tabarena.models.tabicl.model",
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
    # Nori-30M (not the base NoriModel above): see Prep_NORI's own comment below for why.
    "Nori30MModel": "tabarena.models.nori.model",
    "SAPRPTOSSModel": "tabarena.models.sap_rpt_oss.model",
    "OrionMSPModel": "tabarena.models.orionmsp.model",
    "ILTMModel": "tabarena.models.iltm.model",
    "LimiXModel": "tabarena.models.limix.model",
    # LimiX-2 (Stable AI, released 2026-09-15) -- a separate model/checkpoint from
    # LimiXModel above, not a version bump of it (different HF repo, different
    # inference package pin, own ``tabarena.models.limix_2`` package). Runs in a
    # dedicated container (Dockerfile.py312) because its inference package
    # (``LimiX @ git+.../LimiX.git@774aa3e``) requires Python >=3.12 and pins
    # torch==2.9.1, both incompatible with the main image's Python 3.11.10 base
    # and shared torch~=2.14 floor (see requirements-limix2-git.txt). The import
    # here still resolves normally in the main image (tabarena's own package
    # doesn't need the LimiX pip package just to define ``LimiX2Model`` -- only
    # actually fitting one does, inside ``LimiX2Model._fit``'s own deferred
    # import), so ``Prep_LIMIX2`` is always registered; only fitting it needs
    # the limix2 image/environment.
    "LimiX2Model": "tabarena.models.limix_2.model",
    "TabSTARModel": "tabarena.models.tabstar.model",
    # Batch 4 (2026-09-28): APLR, CTBoost. Both CPU-only, own third-party pip
    # packages (aplr, ctboost -- see pyproject.toml), TabArena-package-only
    # like every entry above (confirmed: neither exists under any name in
    # autogluon.tabular.models).
    "APLRModel": "tabarena.models.aplr.model",
    "CTBoostModel": "tabarena.models.ctboost.model",
    # TabLDM (2026-09-28): GPU-only tabular foundation model (dual-stream column
    # embedder + MoE backbone), TabArena-package-only like every foundation model
    # above. Own third-party pip package (``tabldm``, git-pinned via TabArena's
    # ``tabldm`` extra -- see ``requirements-tabarena-git.txt``), not on PyPI.
    "TabLDMModel": "tabarena.models.tabldm.model",
    # Kumo Tabular (2026-09-30) -- large/medium/small checkpoint variants (NVIDIA,
    # https://huggingface.co/blog/nvidia/kumo-tabular). PROVISIONAL: sourced from
    # an UNMERGED upstream PR (autogluon/tabarena#625, branch `kumo-tabular`), not
    # a merged-main commit -- see requirements-tabarena-git.txt's own comment on
    # the pin for the full reasoning. GPU-only tabular foundation model
    # (interleaved row/column attention encoder + in-context-learning transformer),
    # TabArena-package-only like every foundation model above. Own third-party pip
    # package (``structured-data-models``, imported as ``sdm``, git-pinned via
    # TabArena's own PR-local ``kumo_tabular`` extra), not on PyPI.
    "KumoTabularModel": "tabarena.models.kumo_tabular.model",
    "KumoTabularMediumModel": "tabarena.models.kumo_tabular.model",
    "KumoTabularSmallModel": "tabarena.models.kumo_tabular.model",
    # TabPFN-3.5-Fast (2026-09-30) -- the smaller/faster sibling of TabPFN-3.5
    # (Prior Labs reports up to 6x faster inference; marked alpha upstream),
    # released from the same Hugging Face repo alongside TabPFN-3.5 itself.
    # Subclasses TabPFN35Model (same file/module) -- same estimator surface,
    # limits and license as TabPFN-3.5; only the checkpoint differs.
    "TabPFN35FastModel": "tabarena.models.tabpfn_3_5.model",
    # TabDPT v1.3 (2026-09-30) -- tabarena.models.tabdpt.model.TabDPTv13Model,
    # NOT the same class as the already-wrapped TABDPT below (that one is
    # autogluon.tabular.models.TabDPTModel, a *different*, already-graduated-
    # into-AutoGluon-core class -- see Prep_TABDPT's own comment). This class
    # subclasses TabDPTTurboModel (v1.2), itself a sibling of TabDPTModel (v1.1)
    # under tabarena's own TabDPTModelBase.
    #
    # Checkpoint-compatibility check performed before onboarding (2026-09-30):
    # tabarena's own `tabdpt` extra (see its pyproject.toml, pinned via
    # requirements-tabarena-git.txt) requires `tabdpt>=1.3.1` -- the first
    # `tabdpt` release with a separable `TabDPTEstimator._load_model` (the
    # shared-weights loader `TabDPTv13Model` needs) and the release that
    # renamed the network's internal label encoders (`y_encoders` ->
    # `cls_y_encoders`/`reg_y_encoders`), making the 1.2 and 1.3 checkpoint
    # formats mutually unreadable. Verified live on a running k8s pod
    # (`rb-realtabpfn-v2-*`, 2026-09-30): the actually-installed `tabdpt` there
    # is already 1.3.1, matching this requirement. The existing `TABDPT` entry
    # below wraps AutoGluon-core's OWN `TabDPTModel`, which (unlike tabarena's
    # own v1.1 `TabDPTModel`, hardcoded to the `tabdpt1_1.safetensors`
    # checkpoint and pinned to `tabdpt<1.2` upstream) leaves `_checkpoint_filename`
    # at `None` and simply uses whatever checkpoint the installed `tabdpt`
    # package defaults to -- i.e. it floats with the installed package version
    # rather than hardcoding a specific one. So `TABDPT` is NOT stuck on a
    # v1.1/v1.2-only checkpoint format that `tabdpt>=1.3.1` would break: 2708+
    # real `results.pkl` files for `TabDPT_c1_BAG_L1` on the cluster PVC are
    # dated 2026-09-24/25 (checked via `find -printf '%T@'`), i.e. AFTER this
    # tabarena pin (and its `tabdpt>=1.3.1` floor) was already in place, and a
    # spot-checked result (`alzheimer__0/0_0`) has a sane, non-NaN
    # `metric_error` (0.0044, roc_auc). So `TABDPT` already empirically works
    # against tabdpt 1.3.1 -- no genuine coexistence conflict, and the existing
    # `TABDPT` entry is left untouched.
    "TabDPTv13Model": "tabarena.models.tabdpt.model",
}
_missing_optional_tabarena_models = []
for _name, _module_path in _OPTIONAL_TABARENA_MODEL_IMPORTS.items():
    try:
        _module = __import__(_module_path, fromlist=[_name])
        globals()[_name] = getattr(_module, _name)
    except (ImportError, AttributeError):
        # AttributeError: the module itself imports fine (tabarena is installed) but an
        # older build doesn't yet define this class -- e.g. tabdpt.model exists without
        # TabDPTv13Model. Without catching this too, a stale tabarena takes down
        # raman_bench.models.registry entirely (every model fails, not just this one),
        # with a traceback naming an unrelated class and no hint the real cause is a
        # version mismatch. Same degrade-gracefully intent as the ImportError case.
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
from raman_bench.preprocessing.mixin import RamanPreprocessingMixin

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

    def _fit(self, X, y, **kwargs):
        """Disable ``interactions`` unconditionally (see UPDATE below for why
        this is no longer feature-count-gated).

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

        UPDATE 2026-09-18: this used to be gated behind a >4000-feature
        threshold (matching ``wrapped_models._TABSTAR_MAX_FEATURES``'s own
        "wide" boundary) -- that gate turned out NOT a sufficient condition,
        confirmed by a real production hang on a k8s cluster:
        ``cancer_cell_(cooh)2`` (627 rows, 2091 features -- well under the
        4000 threshold, closer to RamanBench's median feature count than to the
        wide extreme this threshold was calibrated against) stuck in "Fast
        interaction strength" for 2+ hours with zero results, identical
        symptom to the originally-diagnosed wide-feature blowup. Squared
        feature-count scaling alone doesn't explain a >100x slowdown between
        this and the (comparably sized) ``alzheimer`` dataset (885 features)
        that fits in under a minute, so this looks data-dependent (e.g.
        near-duplicate/collinear columns hitting a worse-case branch in
        ``interpret``'s ``rank_interactions``), not simply a function of raw
        feature count -- meaning no static threshold can be trusted to catch
        every risky case. Disabling ``interactions`` unconditionally (removing
        the threshold gate) is the only change that's actually bounded: EBM
        becomes a pure additive/GAM model (no pairwise terms) for every
        RamanBench dataset, not just "wide" ones. A real, deliberate quality
        trade-off (documented, not silent), chosen over leaving a
        non-deterministic multi-hour hang risk on every single EBM task.
        """
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
# UPDATE 2026-09-19: the ManyClassClassifier (ECOC) many-class support this
# comment used to say wasn't reproduced is, as of the current AutoGluon
# pin, natively built into upstream itself -- MitraModel._fit and
# TabPFNModel._fit (autogluon.tabular.models.{mitra.mitra_model,
# tabpfnv2.tabpfnv2_5_model}, the latter shared by RealTabPFN-V2/V2.5/V2.6)
# both auto-wrap with tabpfn_extensions.many_class.ManyClassClassifier once
# num_classes exceeds their many_class_threshold (10). No RamanBench code
# needed for these two model families specifically -- confirmed working
# once `setuptools<80` is pinned (see pyproject.toml's `setuptools` comment
# for why: tabpfn_extensions eagerly imports hyperopt, which breaks on any
# setuptools that has removed pkg_resources, hard-crashing this exact path).
# TabICL v2 regression support -- the fork's other reason to exist -- is no longer needed
# at all since upstream 1.6 ships TabICL v2 with regression support natively.
#
# UPDATE 2026-09-19 (2): RamanBench's own custom Causilo and TabPFN-Wide wrappers
# (models/custom/{causilo,tabpfn_wide}/model.py) now also use ECOC, matching
# Mitra/RealTabPFN's native pattern -- Causilo unconditionally, TabPFN-Wide only under
# ``_ECOC_MAX_FEATURES`` (a real OOM was found combining ECOC's per-sub-model cost with
# wide Raman spectra even at 256G; see that constant's own docstring for the width cap
# and its rationale). Also opened a fix upstream for LimiX, which had the same hard
# class-count cap with no ECOC fallback at all: https://github.com/autogluon/tabarena/pull/594
# (unmerged as of this note).
#
# UPDATE 2026-09-19 (3): TabDPT was previously listed here as a "genuine gap" with no
# many-class handling at all -- that was wrong. Verified directly against the installed
# `tabdpt` package (v1.1.12): the checkpoint's classification head is fixed-width
# (`max_num_classes`, 10 for both shipped checkpoints), but `TabDPTClassifier` itself
# already falls back to a native digit-decomposition scheme once `num_classes` exceeds
# that (`_predict_large_cls` in `tabdpt/classifier.py`: encodes each class as a
# base-`max_num_classes` "digit string", runs one forward pass per digit position, then
# recombines) -- no ECOC/ManyClassClassifier wrapper needed or wanted. Confirmed live: a
# 15-class fit against the 10-class checkpoint produces valid, correctly-normalized
# per-class probabilities. So there is nothing left to fix for TabDPT.
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
# TabDPT v1.3 -- tabarena.models.tabdpt.model.TabDPTv13Model, a DIFFERENT sibling class
# from the plain TabDPTModel wrapped just above (that one is AutoGluon-core's own
# checkpoint-version-agnostic class, not tabarena's). See this module's
# _OPTIONAL_TABARENA_MODEL_IMPORTS entry for TabDPTv13Model for the full checkpoint-
# compatibility investigation (tabdpt>=1.3.1 required, confirmed already installed and
# empirically working on the cluster) that cleared this for coexistence with TABDPT.
# Checked (like the TabFM/TabPFN-3/TabSwift/ModernNCA batch above) whether
# _NO_FOUNDATION_MODEL_FEATURE_CAP is needed: inspected tabarena's own
# TabDPTModelBase/TabDPTTurboModel/TabDPTv13Model (the whole MRO) for a
# _get_default_auxiliary_params override -- none of the three define one, so
# max_rows/max_features/max_classes are already AutoGluon's uncapped default
# (None); applying the extra here would be a no-op, so (matching that batch's own
# convention) it's deliberately left off.
# ag_key overridden to the spelled-out "TABDPT-V1.3" for consistency with TABPFN-V3.5
# above (inherited ag_key/ag_name would otherwise carry tabarena's "TA-" staging prefix).
Prep_TABDPT_V13 = _make_optional_prep_class("Prep_TABDPT_V13", TabDPTv13Model, ag_key="TABDPT-V1.3")
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
# TabICL's own memory estimate scales close to the node's total available RAM on
# RamanBench's widest spectra (confirmed live on BHT k8s: a real fit logged "Estimated to
# require 792.209 GB out of 926.205 GB available memory (85.533%)" and proceeded anyway --
# AutoGluon's default `max_memory_usage_ratio` of 1.0 only blocks a fit once the estimate
# EXCEEDS the available memory outright, so 85.5% sailed through as a warning, not a
# skip). A later task on the same long-running pod then exhausted the node for real and
# the whole pod was hard `OOMKilled` -- losing every task still in flight, not just the
# one that pushed it over. Capping the ratio at 0.8 (below the 85.5% already observed)
# makes AutoGluon itself raise a catchable `NotEnoughMemoryError` and skip the model
# before it starts, matching how every other graceful failure in this pipeline behaves,
# instead of gambling on the OS OOM-killer. This is a genuine `params_aux` key
# (`AbstractModel._get_default_auxiliary_params`'s own `max_memory_usage_ratio`, NOT the
# `ag.`-prefixed form -- that prefix only matters for a raw user-supplied
# `ag_args_fit` dict, see `AbstractModel._init_user_params`), so the existing
# `_default_auxiliary_params_extra` declarative merge is the right, and only, place for
# it -- no separate `ag_args_fit` wiring needed.
_TABICL_MEMORY_SAFETY = {**_NO_FOUNDATION_MODEL_FEATURE_CAP, "max_memory_usage_ratio": 0.8}
Prep_TABICL = _make_optional_prep_class(
    "Prep_TABICL", TabICLModel, _default_auxiliary_params_extra=_TABICL_MEMORY_SAFETY
)
# TabICLv2 (tabarena.models.tabicl.model.TabICLv2Model, see the
# _OPTIONAL_TABARENA_MODEL_IMPORTS block above) hasn't graduated into AutoGluon
# core, so -- like TabFM/TabPFN-3/TabSwift/ModernNCA -- its ag_key still carries
# tabarena's "TA-" staging prefix (TabICLv2Model.ag_key == "TA-TABICLv2"),
# overridden here to the short form for the same reason as Prep_TABFM above.
# Feature/row/class cap: checked TabICLv2Model's own MRO (TabICLModelBase ->
# AbstractTorchModel -> AbstractModel) for a `_default_auxiliary_params_extra`
# override the way v1's plain-AutoGluon TabICLModel has -- there isn't one, so
# max_features/max_rows/max_classes are already uncapped (AutoGluon's base
# default), same situation as TabFM/TabPFN-3/TabSwift/ModernNCA, not
# Mitra/TabDPT/TabICL(v1); _NO_FOUNDATION_MODEL_FEATURE_CAP intentionally not
# applied here for that reason.
# Memory safety: NOT applying `_TABICL_MEMORY_SAFETY`'s `max_memory_usage_ratio`
# cap here (unlike v1 immediately above) -- that cap was added in response to a
# confirmed live OOM incident specific to v1's memory estimator
# (`TabICLModelBase._estimate_memory_usage_static`, shared by both v1 and v2 via
# the same base class). No equivalent incident has been observed for v2 yet, and
# v2 overrides that estimator with its own, deliberately simpler one
# (`TabICLv2Model._estimate_memory_usage_static`, whose own docstring says memory
# estimation for v2 on large data "is not supported yet... we ignore it for
# now") -- so the same OOM risk plausibly exists here too, but fabricating a cap
# without a confirmed incident to calibrate it against would just be guessing.
# Revisit if/when a real v2 OOM is observed, same as v1's own history.
Prep_TABICLV2 = _make_optional_prep_class("Prep_TABICLV2", TabICLv2Model, ag_key="TABICLV2")

# REALTABPFN-V2/V2.5 many-class support: the installed tabpfn>=9.0.0's own
# tabpfn.validation.validate_num_classes hard-crashes (TabPFNValidationError,
# no bypass -- unlike tabpfn's separate row/feature ignore_pretraining_limits
# flag, this check takes no such kwarg at all, confirmed by reading
# tabpfn/validation.py directly) whenever a dataset has more classes than this
# checkpoint's InferenceConfig.MAX_NUMBER_OF_CLASSES (10 for both the v2 and
# v2.5 checkpoints -- confirmed via tabpfn/inference_config.py's
# _get_v2_config/_get_v2_5_config). Confirmed live on the cluster: every
# REALTABPFN-V2/V2.5 attempt at bacteria_identification (30 classes),
# pharmaceutical_ingredients (32), rruff_mineral_raw (79), mlrod (16), and the
# cancer_cell_* trio (12 each) failed exactly this way.
#
# tabpfn-extensions (already a hard dependency, see pyproject.toml's
# setuptools<80 comment) ships tabpfn_extensions.many_class.ManyClassClassifier,
# an ECOC-style meta-estimator built specifically to decompose a >max_classes
# problem into an ensemble of in-range sub-problems for exactly this situation
# -- same class of fix as LIMIX's max_classes=10 override above, but LIMIX's
# own model class doesn't hard-crash without it (AutoGluon's declarative cap
# was the only thing stopping it), whereas TabPFN's crash lives inside the
# `tabpfn` library itself and needs the actual estimator swapped, not just an
# AutoGluon-level cap lifted. Verified end-to-end on the real cluster image,
# against real RamanBench data (bacteria_identification, 30 classes): fit,
# predict_proba (correct (n_samples, 30) shape), and predict all succeed
# (0.82 accuracy), and with integer class labels (AutoGluon's own y encoding
# convention, confirmed via a synthetic 25-class check) `classes_` comes back
# as exactly `np.arange(n_classes)` -- i.e. predict_proba's column order
# already matches what AutoGluon's `_convert_proba_to_unified_form` assumes
# (that function does no reindexing of its own; it trusts the column order
# outright), so no relabeling step is needed on top of this.
#
# Implementation: `_fit` temporarily monkeypatches `tabpfn.TabPFNClassifier`
# (module-global, but scoped to this one call via try/finally -- safe because
# RamanBench's job runner fits exactly one model per process, never
# concurrently) to `_ManyClassTabPFNProxy`, a constructor-compatible stand-in
# that only wraps with `ManyClassClassifier` when the actual fit-time class
# count exceeds `_TABPFN_OFFICIAL_MAX_CLASSES`; below that, it constructs and
# fits the real `TabPFNClassifier` exactly as before, so this is a pure
# extension, never a behavior change, for every dataset that already worked.
# `_get_memory_size` needs its own override alongside `_fit`: the base
# class's version (`TabPFNModel._get_memory_size`) reaches directly into
# `estimator.executor_`/`estimator.models_` for a weightless-pickle size
# optimisation, attributes `ManyClassClassifier` doesn't expose (it holds a
# per-codeword list of separately-fitted estimators, not one `executor_`) --
# falls back to `AbstractModel`'s generic accounting in that case instead of
# crashing with `AttributeError`. `_narrow_inference_context` (a separate
# memory optimisation, halving stored float precision) doesn't need an
# override: it already reads `self.model`'s TabPFN-specific attributes via
# `getattr(..., None)` defensively, so it already no-ops harmlessly on a
# wrapped model, just skipping that optimisation.
#
# Real performance tradeoff, confirmed live (not assumed): `ManyClassClassifier`
# does NOT eagerly fit its per-codeword sub-estimators in `.fit()` -- it stores
# the codebook and training data, then fits each sub-estimator lazily, inside
# the first `predict`/`predict_proba` call. Against the one dataset in this
# batch large enough to matter (`bacteria_identification`, capped to 10,000
# rows by `max_train_samples_overrides`), that lazy fit completed successfully
# (0.86 accuracy, correct proba shape) but took ~15 minutes wall-clock in a
# real cluster test -- past the routine sweep's 600s per-task `time_limit`,
# though that test ran on a GPU shared with another live job, so an isolated
# production pod may well come in faster. Worst case, this dataset trades one
# failure mode for another (`TabPFNValidationError` crash -> `TimeLimitExceeded`
# timeout) rather than producing a result -- still strictly no worse than
# before it (no `results.pkl` either way), so left enabled rather than
# special-cased. The other four confirmed many-class datasets this fixes
# (`pharmaceutical_ingredients` 3,510 rows, `rruff_mineral_raw` 1,162,
# `cancer_cell_nh2`/`cancer_cell_cooh` ~632 each) are all far smaller and
# were not observed to have this problem in testing.
_TABPFN_OFFICIAL_MAX_CLASSES = 10


class _ManyClassTabPFNProxy:
    """Constructor-compatible stand-in for ``tabpfn.TabPFNClassifier``.

    See the many-class comment block above this class's two call sites
    (``Prep_REALTABPFN_V2``/``Prep_REALTABPFN_V25``) for why this exists and
    how it's wired in (a `_fit`-scoped monkeypatch, not a global change).
    """

    def __init__(self, _real_cls, **hps):
        self._real_cls = _real_cls
        self._hps = hps
        self._impl = None

    def fit(self, X, y):
        from tabpfn_extensions.many_class import ManyClassClassifier

        base = self._real_cls(**self._hps)
        n_classes = len(np.unique(np.asarray(y)))
        if n_classes > _TABPFN_OFFICIAL_MAX_CLASSES:
            self._impl = ManyClassClassifier(
                estimator=base, random_state=self._hps.get("random_state")
            )
        else:
            self._impl = base
        self._impl.fit(X, y)
        return self

    def predict(self, X):
        return self._impl.predict(X)

    def predict_proba(self, X):
        return self._impl.predict_proba(X)

    @property
    def classes_(self):
        return self._impl.classes_

    def __getattr__(self, name):
        # Delegates everything else (executor_, models_,
        # forced_inference_dtype_, ...) to the underlying real
        # TabPFNClassifier when NOT many-class-wrapped. When wrapped, this
        # naturally raises AttributeError (ManyClassClassifier doesn't have
        # these either) -- exactly what every real call site here
        # (_narrow_inference_context) already handles via getattr(..., None).
        return getattr(self._impl, name)

    @property
    def devices_(self):
        # TabPFNModel.get_device() reads self.model.devices_[0].type directly
        # (no getattr fallback, unlike _narrow_inference_context), so this one
        # needs a real answer even when wrapped. ManyClassClassifier does not
        # persist fitted per-codeword estimator instances when a real codebook
        # is in use (self.estimators_ = None, confirmed by reading
        # tabpfn_extensions.many_class.fit directly -- sub-fits happen lazily
        # inside predict/predict_proba, not stored), so there is no wrapped
        # TabPFNClassifier instance to introspect for this. Constructed
        # directly from the same device string AutoGluon already resolved and
        # passed into our constructor kwargs instead -- correct regardless of
        # whether ManyClassClassifier ends up in play.
        import torch

        device = self._hps.get("device", "cpu")
        if isinstance(device, list):
            return tuple(torch.device(d) for d in device)
        return (torch.device(device),)


def _many_class_tabpfn_fit(self, X, y, **kwargs):
    """``_fit`` override for the many-class-capable REALTABPFN-V2/V2.5 classes.

    Must call ``RamanPreprocessingMixin._fit`` here, NOT the raw TabPFN model
    class's own ``_fit`` directly -- confirmed as a real bug via a live
    cluster failure (``TypeError: TabPFNClassifier.__init__() got an
    unexpected keyword argument 'prep_aug_enabled'``): calling the raw
    model's ``_fit`` skips ``RamanPreprocessingMixin._fit``'s own
    ``prep_*``-key stripping/preprocessing setup entirely (this class's MRO
    puts the mixin before the raw TabPFN model class, and overriding ``_fit``
    directly on this class -- the same declarative-dict mechanism
    ``_make_optional_prep_class`` uses everywhere else -- shadows the
    mixin's ``_fit`` rather than chaining through it). Calling the mixin's
    ``_fit`` here instead reproduces the normal MRO: it does its own
    preprocessing/prep_*-stripping work and then calls ``super()._fit(...)``
    itself, which (since ``self``'s real class still has the raw TabPFN
    model class right after the mixin in the MRO) correctly reaches that
    class's ``_fit`` next -- exactly the normal chain, just entered here
    instead of at the top.
    """
    import tabpfn

    original_cls = tabpfn.TabPFNClassifier
    tabpfn.TabPFNClassifier = lambda **hps: _ManyClassTabPFNProxy(original_cls, **hps)
    try:
        return RamanPreprocessingMixin._fit(self, X, y, **kwargs)
    finally:
        tabpfn.TabPFNClassifier = original_cls


def _many_class_get_memory_size(self, **kwargs):
    from tabpfn_extensions.many_class import ManyClassClassifier

    if isinstance(self.model, _ManyClassTabPFNProxy) and isinstance(
        self.model._impl, ManyClassClassifier
    ):
        return DummyModel._get_memory_size(self, **kwargs)
    return type(self).__mro__[1]._get_memory_size(self, **kwargs)


Prep_REALTABPFN_V2 = _make_optional_prep_class(
    "Prep_REALTABPFN_V2",
    RealTabPFNv2Model,
    _default_auxiliary_params_extra=_NO_FOUNDATION_MODEL_FEATURE_CAP,
    _fit=_many_class_tabpfn_fit,
    _get_memory_size=_many_class_get_memory_size,
)
Prep_REALTABPFN_V25 = _make_optional_prep_class(
    "Prep_REALTABPFN_V25",
    RealTabPFNv25Model,
    _default_auxiliary_params_extra=_NO_FOUNDATION_MODEL_FEATURE_CAP,
    _fit=_many_class_tabpfn_fit,
    _get_memory_size=_many_class_get_memory_size,
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
# Wraps tabarena.models.tabpfn_3_5.model.TabPFN35Model (TabArena's own TabPFN-3.5
# integration -- see the _OPTIONAL_TABARENA_MODEL_IMPORTS block above). ag_key
# overridden to the spelled-out "TABPFN-V3.5" for consistency with
# REALTABPFN-V2.5/V2.6 and TABPFN-V3 above -- no collision to dodge, unlike
# Prep_TABPFN_V3's ag_name override.
Prep_TABPFN_V3_5 = _make_optional_prep_class("Prep_TABPFN_V3_5", TabPFN35Model, ag_key="TABPFN-V3.5")
# TabPFN-3.5-Fast: the smaller/faster sibling of TabPFN-3.5, released alongside it from
# the same Hugging Face repo (Prior Labs reports up to 6x faster inference; the model is
# marked alpha upstream). Subclasses TabPFN35Model directly (same file) -- same limits,
# license and estimator surface as TabPFN-3.5, only the checkpoint differs. ag_key
# overridden to the spelled-out "TABPFN-V3.5-FAST" for consistency with TABPFN-V3.5 above
# (inherited ag_key/ag_name would otherwise carry tabarena's "TA-" staging prefix).
Prep_TABPFN_V3_5_FAST = _make_optional_prep_class(
    "Prep_TABPFN_V3_5_FAST", TabPFN35FastModel, ag_key="TABPFN-V3.5-FAST"
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
# APLR (ag_key "TA-APLR" -- overridden to the spelled-out short form, no collision
# to dodge) and CTBoost (ag_key "CTB" -- overridden to "CTBOOST" purely for
# readability, same reasoning as CHIMERABOOST/PERPETUAL_BOOSTER above; "CTB" stays
# available for TabArena's own un-preprocessed baseline entries). Both CPU-only
# (verified: neither declares any GPU resource requirement in its own class body),
# neither caps max_features/max_rows/max_classes (checked _get_default_auxiliary_params/
# _default_auxiliary_params_extra directly against the installed classes -- APLR
# declares no auxiliary-params override at all; CTBoost's only overrides
# valid_raw_types/ignored_type_group_special, same shape as Prep_EBM in batch 2).
Prep_APLR = _make_optional_prep_class("Prep_APLR", APLRModel, ag_key="APLR")
Prep_CTBOOST = _make_optional_prep_class("Prep_CTBOOST", CTBoostModel, ag_key="CTBOOST")

# TabLDM (ag_key "TA-XIAOMI-TABLDM" -- overridden to the shorter "TABLDM", same
# "TA-" staging-prefix strip as TabFM/TabPFN-3/TabSwift above). GPU-only
# (``default_num_gpus = 1``, ``minimum_num_gpus = 1``). Checked the same way as
# TabFM/TabPFN-3/TabSwift/ModernNCA (see the batch-3 comment above): TabLDMModel
# declares no ``_default_auxiliary_params_extra`` at all, and its base class
# (``AbstractTorchModel``) doesn't cap max_features/max_rows/max_classes either
# (confirmed against ``AuxiliaryParams.base_defaults()`` directly -- no such keys
# there, so they resolve to uncapped/None) -- so, like those four, it does NOT get
# ``_NO_FOUNDATION_MODEL_FEATURE_CAP`` applied here. No row cap
# (``model_max_train_samples_overrides`` in ``configs/v1/scope_default.json``)
# either at onboarding time -- every existing entry there (PERPETUAL_BOOSTER,
# MITRA, TABFM, REALTABPFN-V2/V2.5, TABICL) was added only after a real observed
# cluster failure (OOM or TimeLimitExceeded), not speculatively; same policy here,
# revisit if TabLDM shows the same failure mode in production.
Prep_TABLDM = _make_optional_prep_class("Prep_TABLDM", TabLDMModel, ag_key="TABLDM")

# Kumo Tabular (2026-09-30) -- large/medium/small checkpoint variants of NVIDIA's
# pretrained in-context-learning tabular foundation model
# (https://huggingface.co/blog/nvidia/kumo-tabular). PROVISIONAL onboarding:
# sourced from autogluon/tabarena PR #625 (still open/unmerged as of this
# onboarding -- see requirements-tabarena-git.txt's pin comment) rather than a
# merged-main release; revisit the ``ag_key`` overrides below (harmless either
# way -- they just strip the "TA-" staging prefix, same convention as
# TabFM/TabPFN-3/TabSwift/TabLDM above) once #625 merges.
#
# GPU-only (``default_num_gpus = 1``, ``minimum_num_gpus = 1`` on all three
# sizes -- ``class_settings_per_subclass = True`` upstream, each size re-declares
# its own ``shared_weights``/``ag_key``/``ag_name`` but all three inherit the same
# GPU requirement from the base ``KumoTabularModel``). Checked the same way as
# TabLDM immediately above: no ``_default_auxiliary_params_extra`` on any of the
# three classes, and the shared base (``AbstractTorchModel``) doesn't cap
# max_features/max_rows/max_classes either -- so, like TabLDM, none of the three
# get ``_NO_FOUNDATION_MODEL_FEATURE_CAP`` applied here, and no row cap
# (``model_max_train_samples_overrides`` in ``configs/v1/scope_default.json``) is
# added preemptively (same reactive-only policy as TabLDM's comment above --
# watch the first real cluster runs for OOM/TimeLimitExceeded).
#
# Own third-party pip package (``structured-data-models``, imported as ``sdm``,
# git-pinned via TabArena's own PR-local ``kumo_tabular`` extra in
# requirements-tabarena-git.txt), not on PyPI. ``requires-python = ">=3.11"`` and
# ``torch>=2.7`` -- both satisfied by the main image's Python 3.11.10 base and
# torch~=2.14 floor, so this runs in the shared main image, no dedicated
# container needed (unlike LIMIX2's Dockerfile.limix2).
Prep_KUMO_TABULAR = _make_optional_prep_class("Prep_KUMO_TABULAR", KumoTabularModel, ag_key="KUMO-TABULAR")
Prep_KUMO_TABULAR_MEDIUM = _make_optional_prep_class(
    "Prep_KUMO_TABULAR_MEDIUM", KumoTabularMediumModel, ag_key="KUMO-TABULAR-MEDIUM"
)
Prep_KUMO_TABULAR_SMALL = _make_optional_prep_class(
    "Prep_KUMO_TABULAR_SMALL", KumoTabularSmallModel, ag_key="KUMO-TABULAR-SMALL"
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
# Wraps Nori30MModel, not the base NoriModel -- confirmed as a real production
# failure on the k8s cluster: NoriModel leaves NoriRegressor's `model=` variant
# kwarg unset, and synthefy_nori.hf.download_checkpoint's own auto-selection
# only works for some datasets (deterministic per-dataset, not flaky -- e.g.
# alzheimer/cancer_cell_cooh/parkinson always succeeded, amino_acids_glycine/
# ecoli_fermentation/fuel_benchtop always failed with "ValueError:
# download_checkpoint requires model= ('nori-6m'/'nori-30m'/'nori-100m') or an
# explicit repo_id="). Nori30MModel is tabarena's own fix for exactly this --
# its `_set_default_params` explicitly sets `model="nori-30m"`, sidestepping
# the broken auto-selection entirely. ag_key overridden to "NORI" (not
# Nori30MModel's own "TA-NORI-30M") since RamanBench only wraps one Nori
# variant, matching every other single-variant entry in this registry.
Prep_NORI = _make_optional_prep_class("Prep_NORI", Nori30MModel, ag_key="NORI")


def _patch_sap_rpt_oss_single_row_predict_bug(sap_rpt_oss_rpt_module) -> None:
    """Fix ``SAP_RPT_OSS_Regressor.predict``'s crash on a single-row test set.

    Confirmed live on the cluster (``tg_ecoli_fermentation`` target 2, an
    8-row dataset after NaN-label rows are dropped -- its 3-fold CV leaves a
    1-row test fold on one split) and reproduced directly against the
    installed ``sap_rpt_oss`` package outside AutoGluon entirely::

        File ".../sap_rpt_oss/rpt.py", line 378, in predict
            preds = np.concatenate(preds)
        ValueError: zero-dimensional arrays cannot be concatenated

    Root cause: ``predict()`` chunks ``X`` into ``test_chunk_size``-sized
    pieces, calls ``self._predict(chunk)`` per chunk, and concatenates the
    per-chunk results -- but ``_predict`` does
    ``np.mean(all_preds, axis=0)`` over a per-bagging-model prediction list,
    which numpy collapses to a bare 0-d scalar (not a 1-element 1-d array)
    when the chunk has exactly one row, so ``np.concatenate`` on a list of
    0-d arrays fails. Filed upstream: reported (see
    https://github.com/SAP-samples/sap-rpt-1-oss/issues/33), still open as
    of this note.

    The classifier's own ``predict``/``predict_proba`` do NOT share this bug
    -- they concatenate with ``torch.cat`` over softmax output, which always
    keeps the batch dimension even for a single row (checked directly against
    the installed package). Regressor-only fix.

    This function reproduces the obvious fix (wrap each chunk's prediction in
    ``np.atleast_1d`` before concatenating) at runtime, monkeypatching the
    installed ``sap_rpt_oss`` package in place, so SAP_RPT_OSS runs don't have
    to wait for the upstream fix to merge and release. Idempotent (a no-op if
    already patched). Remove once RamanBench's ``sap_rpt_oss`` pin includes
    the merged upstream fix.
    """
    regressor_cls = sap_rpt_oss_rpt_module.SAP_RPT_OSS_Regressor
    if getattr(regressor_cls, "_ramanbench_patched_single_row_predict", False):
        return

    def _patched_predict(self, X):
        import numpy as np
        import pandas as pd

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.X_.columns)
        preds = []
        for start in range(0, len(X), self.test_chunk_size):
            end = start + self.test_chunk_size
            preds.append(np.atleast_1d(self._predict(X.iloc[start:end])))
        return np.concatenate(preds)

    regressor_cls.predict = _patched_predict
    regressor_cls._ramanbench_patched_single_row_predict = True


if SAPRPTOSSModel is not None:
    # `SAPRPTOSSModel is not None` only means tabarena's own thin wrapper class
    # imported cleanly -- it does NOT guarantee the actual third-party
    # `sap_rpt_oss` inference package is installed (same situation as every
    # other foundation model here: the tabarena wrapper class can be defined
    # without its backing package). Confirmed as a real failure building a
    # minimal environment that installs tabarena but not
    # requirements-models-git.txt (e.g. Dockerfile.py312, a LIMIX2-only
    # image): this module-level `import sap_rpt_oss.rpt` crashed the entire
    # `wrapped_models` import with a bare `ModuleNotFoundError`, taking every
    # other Prep_* model down with it -- exactly the failure mode the "keep
    # optional imports out of module top-level" convention (see this module's
    # docstring) exists to prevent. Guarded the same way as every other
    # optional import in this file; Prep_SAP_RPT_OSS itself still degrades to
    # unavailable below via the same _unavailable_models sweep.
    try:
        import sap_rpt_oss.rpt as _sap_rpt_oss_module
    except ImportError:
        SAPRPTOSSModel = None
    else:
        _patch_sap_rpt_oss_single_row_predict_bug(_sap_rpt_oss_module)
        del _sap_rpt_oss_module

Prep_SAP_RPT_OSS = _make_optional_prep_class(
    "Prep_SAP_RPT_OSS", SAPRPTOSSModel, ag_key="SAP_RPT_OSS"
)


def _patch_orionmsp_v15_sklearn_compat_bug(preprocessing_module) -> None:
    """Fix ``tabtune.models.orionmsp_v15.sklearn.preprocessing``'s own sklearn
    version-compat shim, which is itself broken against the sklearn actually
    installed here.

    That module ships a backport of ``BaseEstimator._validate_data`` (applied as a
    global monkeypatch onto ``sklearn.base.BaseEstimator`` when the running sklearn
    doesn't have it natively -- true for the sklearn pinned here), hardcoding the
    ``force_all_finite`` kwarg to ``check_array``/``check_X_y``. That kwarg was
    renamed to ``ensure_all_finite`` upstream (sklearn deprecates then removes it
    across versions), so on an installed sklearn past that point the shim itself
    raises. Confirmed live on the cluster -- every ORIONMSP task on any
    non-skipped (classification) dataset failed identically::

        Xc, yc = check_X_y(
        TypeError: check_X_y() got an unexpected keyword argument
        'force_all_finite'

    Fix: monkeypatch just the ``check_array``/``check_X_y`` names inside this one
    module (not sklearn globally -- several *other* installed packages, e.g.
    tabpfn/skrub/lightgbm, reference the same deprecated kwarg via their own compat
    shims and should not be touched here) to translate ``force_all_finite`` to
    whichever kwarg name the installed sklearn's ``check_array`` actually accepts
    before delegating to the real function. ``_validate_data`` (defined in this same
    module) looks up ``check_array``/``check_X_y`` as module globals at call time, so
    patching the module's own names here takes effect for every call site.
    Idempotent (a no-op if already patched). Remove once upstream fixes this shim (or
    drops it once the installed sklearn always has a native ``_validate_data``).
    """
    if getattr(preprocessing_module, "_ramanbench_patched_sklearn_compat", False):
        return

    import inspect

    from sklearn.utils.validation import check_array as _real_check_array
    from sklearn.utils.validation import check_X_y as _real_check_X_y

    _finite_kwarg = (
        "ensure_all_finite"
        if "ensure_all_finite" in inspect.signature(_real_check_array).parameters
        else "force_all_finite"
    )

    def _translate_finite_kwarg(kwargs):
        if "force_all_finite" in kwargs:
            kwargs[_finite_kwarg] = kwargs.pop("force_all_finite")
        return kwargs

    def _compat_check_array(*args, **kwargs):
        return _real_check_array(*args, **_translate_finite_kwarg(kwargs))

    def _compat_check_X_y(*args, **kwargs):
        return _real_check_X_y(*args, **_translate_finite_kwarg(kwargs))

    preprocessing_module.check_array = _compat_check_array
    preprocessing_module.check_X_y = _compat_check_X_y
    preprocessing_module._ramanbench_patched_sklearn_compat = True


if OrionMSPModel is not None:
    # Same reasoning as SAPRPTOSSModel above: the tabarena wrapper class importing
    # cleanly doesn't guarantee the backing `tabtune` package is installed.
    try:
        import tabtune.models.orionmsp_v15.sklearn.preprocessing as _orionmsp_v15_preprocessing_module
    except ImportError:
        pass
    else:
        _patch_orionmsp_v15_sklearn_compat_bug(_orionmsp_v15_preprocessing_module)
        del _orionmsp_v15_preprocessing_module

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


# LimiX-2 -- Stable AI's second-generation tabular foundation model
# (tabarena.models.limix_2.model.LimiX2Model), distinct from LimiXModel above
# (different checkpoint, different HF repo, own inference package pin). Unlike
# LimiXModel, LimiX2Model caps max_classes=10 via the normal DECLARATIVE
# `_default_auxiliary_params_extra` class attribute (confirmed by reading
# tabarena/models/limix_2/model.py directly, not assumed from LimiXModel's
# shape) -- so, unlike Prep_LIMIX, this one lifts the cap the same way
# Mitra/TabDPT/TabICL/RealTabPFN do, via `_make_optional_prep_class`'s
# `_default_auxiliary_params_extra=_NO_FOUNDATION_MODEL_FEATURE_CAP` kwarg,
# with no method override needed. ag_key overridden from LimiX2Model's own
# staging-prefixed "TA-LIMIX-2" to the short "LIMIX2" for consistency with
# every other entry in this registry (no collision to dodge).
#
# GPU-tier, and its own inference package requires Python >=3.12 and pins
# torch==2.9.1 -- incompatible with the main image (Python 3.11.10,
# torch~=2.14 floor). Runs in a dedicated container built from
# Dockerfile.py312 instead (see requirements-limix2-git.txt and
# cluster/submit_job.py's model_image_overrides). A LIMIX2 job submitted
# against the main image's environment will fail to import the LimiX
# inference package at fit time (a clean ModuleNotFoundError/ImportError from
# LimiX2Model's own deferred import inside `_fit`, not a hang or crash
# elsewhere) -- always route LIMIX2 through the limix2 image.
Prep_LIMIX2 = _make_optional_prep_class(
    "Prep_LIMIX2",
    LimiX2Model,
    ag_key="LIMIX2",
    _default_auxiliary_params_extra=_NO_FOUNDATION_MODEL_FEATURE_CAP,
)


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
    "TABDPT-V1.3": Prep_TABDPT_V13,
    "TABFM": Prep_TABFM,
    "TABICL": Prep_TABICL,
    "TABICLV2": Prep_TABICLV2,
    "REALTABPFN-V2": Prep_REALTABPFN_V2,
    "REALTABPFN-V2.5": Prep_REALTABPFN_V25,
    "REALTABPFN-V2.6": Prep_REALTABPFN_V26,
    "TABPFN-V3": Prep_TABPFN_V3,
    "TABPFN-V3-THINKING": Prep_TABPFN_V3_THINKING,
    "TABPFN-V3.5": Prep_TABPFN_V3_5,
    "TABPFN-V3.5-FAST": Prep_TABPFN_V3_5_FAST,
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
    "LIMIX2": Prep_LIMIX2,
    "TABSTAR": Prep_TABSTAR,
    "APLR": Prep_APLR,
    "CTBOOST": Prep_CTBOOST,
    "TABLDM": Prep_TABLDM,
    "KUMO-TABULAR": Prep_KUMO_TABULAR,
    "KUMO-TABULAR-MEDIUM": Prep_KUMO_TABULAR_MEDIUM,
    "KUMO-TABULAR-SMALL": Prep_KUMO_TABULAR_SMALL,
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

# ROCKET and ARSENAL used to be classification-only, but that no longer applies
# here: ARSENAL's custom model was removed entirely (see
# src/raman_bench/models/custom/arsenal/ deletion), and ROCKET gained a
# regression branch (RocketRegressor) in the same upstream change that dropped
# it from this set -- see git history on wrapped_models.py / rocket/model.py.
# PCALDA (added on this branch, 2026-08-28) is a genuine classification-only
# addition: LDA has no regression analogue, and PCALDAModel.fit() raises on a
# continuous target as a backstop -- see its docstring.
CLASSIFICATION_ONLY_MODELS = {"TABPFN-WIDE", "ORIONMSP", "PCALDA"}

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
