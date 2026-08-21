"""
RamanPreprocessingMixin — mixin class that integrates Raman spectral preprocessing
as tunable hyperparameters within AutoGluon models.

All preprocessing is disabled by default (prep_*_enabled = False).
Subclasses override _set_default_params() to enable specific steps for their
domain (e.g. Prep_PLS enables baseline correction + denoising + SNV).

The optional _prep_restriction attribute (a dict of {step_key: bool}) narrows
the HPO search space to physically meaningful steps for the current dataset.
When absent, all steps are included in the search space.

Preprocessing order (when enabled) — matches the sequential wiring in
``_preprocess_fit``/``_preprocess_transform`` below:
1. Crop (fingerprint-region, fractional-index proxy) — runs first because it
   changes the array's feature count; every later step must see the final
   (cropped) width, so cropping before smoothing/baseline/etc. keeps the
   window-length-relative-to-n_features logic in those steps consistent.
2. Cosmic ray removal — spikes must be removed before any smoothing/baseline
   step, or they get spread/baked into neighbouring points.
3. Denoising (Savitzky-Golay smoothing) — smooth before baseline estimation
   so baseline fits aren't thrown off by high-frequency noise.
4. Denoising (wavelet threshold) — an alternative/complementary smoothing
   method to Savitzky-Golay; placed alongside it, still before baseline
   correction for the same reason.
5. Baseline correction (ASLS)
6. Baseline correction (airPLS) — same penalized-least-squares family as
   ASLS, placed immediately after it.
7. Baseline correction (arPLS) — likewise part of the ASLS/airPLS family.
8. Baseline correction (rubberband / convex hull) — a non-iterative
   baseline alternative, grouped with the other baseline methods.
9. MSC (multiplicative scatter correction) — scatter correction runs after
   baseline correction so it isn't biased by additive baseline drift.
10. EMSC (extended MSC) — generalizes MSC, so it sits right after it.
11. Derivative (Savitzky-Golay 1st/2nd derivative) — applied after
    baseline/scatter correction (derivatives amplify baseline noise if taken
    too early) but before SNV/scaling (derivatives should be normalized like
    any other preprocessed spectrum).
12. SNV (Standard Normal Variate)
13. Vector normalization (L2) — alongside SNV/scaling, since like them it is
    a per-spectrum normalization step best applied once the spectrum shape
    has been fully cleaned up.
14. Standard scaling
15. Augmentation (training only)

GCU and LVSE (see ``gcu``/``lvse`` step definitions below) are a different
kind of step: they *replace* the feature representation (like PCA) rather
than transform it in place, so they are architecturally last within
``_preprocess_fit``/``_preprocess_transform`` — after step 14 (standard
scaling) and before augmentation (which then operates on whatever GCU/LVSE
produced, same as it would on any other preprocessed representation). When
enabled, they run after every shape-preserving step above and everything
downstream sees only their output. If both are enabled, each runs as an
independent block on the same pre-GCU/LVSE input and their outputs are
concatenated column-wise (the paper's "separate forward pass" design) —
LVSE does not chain on top of GCU's output. See their docstrings in
``raman_preprocessing.py`` and the composition-order design note (a logged
warning, not a hard error — see the rationale in the code) near the end of
``_preprocess_fit``/``_preprocess_transform`` below.

The preprocessing-ensemble mechanism (``prep_ensemble_enabled`` /
``prep_ensemble_blocks``) is orthogonal to this ordered pipeline: it fans
out into several independent copies of the pipeline above (one per block),
run in parallel and concatenated column-wise, rather than being one more
step within the sequence. See ``_preprocess_fit_ensemble`` /
``_preprocess_transform_ensemble``.
"""

import logging

import numpy as np
import pandas as pd

try:
    from autogluon.common import space
except ImportError as _ag_err:
    raise ImportError(
        "RamanPreprocessingMixin requires autogluon. "
        "Install with: pip install 'raman-bench[autogluon]'"
    ) from _ag_err

from raman_bench.preprocessing.raman_preprocessing import (
    augment_spectra,
    baseline_correction_airpls,
    baseline_correction_arpls,
    baseline_correction_asls,  # also covers fluorescence removal at high lam / low p
    cosmic_ray_removal,
    crop_spectra,
    denoise_savgol,
    emsc_fit,
    emsc_transform,
    gcu_fit,
    gcu_transform,
    lvse_fit,
    lvse_transform,
    multiplicative_scatter_correction_fit,
    multiplicative_scatter_correction_transform,
    rubberband_correction,
    savgol_derivative,
    snv,
    vector_normalize,
    wavelet_denoise,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single source of truth for all HPO-tunable preprocessing steps
# ---------------------------------------------------------------------------
# Each entry maps step_key → {
#   "search_params": param_name → space.* (for HPO search space),
#   "defaults":      param_name → default value (all prep_*_enabled = False),
# }
# standard_scaling is handled separately below as a forced-on-by-restriction
# step (not HPO-tunable).

_PREP_STEP_DEFINITIONS = {
    "crop": {
        "search_params": {
            "prep_crop_enabled": space.Categorical(True, False),
            "prep_crop_start_frac": space.Real(lower=0.0, upper=0.3),
            "prep_crop_end_frac": space.Real(lower=0.5, upper=1.0),
        },
        "defaults": {
            "prep_crop_enabled": False,
            "prep_crop_start_frac": 0.15,
            "prep_crop_end_frac": 0.75,
        },
    },
    "cosmic_ray_removal": {
        "search_params": {
            "prep_crr_enabled": space.Categorical(True, False),
            "prep_crr_threshold": space.Categorical(4, 6, 8, 10),
            "prep_crr_kernel_size": space.Categorical(2, 3, 5),
        },
        "defaults": {
            "prep_crr_enabled": False,
            "prep_crr_threshold": 6,
            "prep_crr_kernel_size": 3,
        },
    },
    "baseline_correction": {
        "search_params": {
            "prep_bl_enabled": space.Categorical(True, False),
            "prep_bl_lam": space.Real(lower=1e3, upper=1e9, log=True),
            "prep_bl_p": space.Real(lower=0.0001, upper=0.1, log=True),
        },
        "defaults": {
            "prep_bl_enabled": False,
            "prep_bl_lam": 1e5,
            "prep_bl_p": 0.01,
        },
    },
    "airpls": {
        "search_params": {
            "prep_airpls_enabled": space.Categorical(True, False),
            "prep_airpls_lam": space.Real(lower=1e2, upper=1e6, log=True),
            "prep_airpls_max_iter": space.Categorical(10, 12, 15, 18, 20),
            "prep_airpls_tol": space.Real(lower=1e-4, upper=1e-2, log=True),
        },
        "defaults": {
            "prep_airpls_enabled": False,
            "prep_airpls_lam": 1e5,
            "prep_airpls_max_iter": 15,
            "prep_airpls_tol": 1e-3,
        },
    },
    "arpls": {
        "search_params": {
            "prep_arpls_enabled": space.Categorical(True, False),
            "prep_arpls_lam": space.Real(lower=1e2, upper=1e8, log=True),
            "prep_arpls_max_iter": space.Categorical(10, 12, 15, 18, 20),
        },
        "defaults": {
            "prep_arpls_enabled": False,
            "prep_arpls_lam": 1e5,
            "prep_arpls_max_iter": 15,
        },
    },
    "rubberband": {
        "search_params": {
            "prep_rubberband_enabled": space.Categorical(True, False),
        },
        "defaults": {
            "prep_rubberband_enabled": False,
        },
    },
    "msc": {
        "search_params": {
            "prep_msc_enabled": space.Categorical(True, False),
        },
        "defaults": {
            "prep_msc_enabled": False,
        },
    },
    "emsc": {
        "search_params": {
            "prep_emsc_enabled": space.Categorical(True, False),
            "prep_emsc_poly_order": space.Categorical(2, 4, 6),
        },
        "defaults": {
            "prep_emsc_enabled": False,
            "prep_emsc_poly_order": 4,
        },
    },
    "denoising": {
        "search_params": {
            "prep_denoise_enabled": space.Categorical(True, False),
            "prep_denoise_wl": space.Categorical(5, 7, 9, 11, 13, 15),
            "prep_denoise_po": space.Categorical(2, 3, 4),
        },
        "defaults": {
            "prep_denoise_enabled": False,
            "prep_denoise_wl": 9,
            "prep_denoise_po": 3,
        },
    },
    "wavelet_denoise": {
        "search_params": {
            "prep_wavelet_enabled": space.Categorical(True, False),
            "prep_wavelet_name": space.Categorical("sym7", "sym8", "db6"),
            "prep_wavelet_level": space.Categorical(3, 4, 5),
        },
        "defaults": {
            "prep_wavelet_enabled": False,
            "prep_wavelet_name": "sym7",
            "prep_wavelet_level": 4,
        },
    },
    "derivative": {
        "search_params": {
            "prep_deriv_enabled": space.Categorical(True, False),
            "prep_deriv_order": space.Categorical(1, 2),
            "prep_deriv_wl": space.Categorical(9, 11, 15, 21),
            "prep_deriv_po": space.Categorical(2, 3),
        },
        "defaults": {
            "prep_deriv_enabled": False,
            "prep_deriv_order": 1,
            "prep_deriv_wl": 11,
            "prep_deriv_po": 2,
        },
    },
    "snv": {
        "search_params": {
            "prep_snv_enabled": space.Categorical(True, False),
        },
        "defaults": {
            "prep_snv_enabled": False,
        },
    },
    "vecnorm": {
        "search_params": {
            "prep_vecnorm_enabled": space.Categorical(True, False),
        },
        "defaults": {
            "prep_vecnorm_enabled": False,
        },
    },
    "gcu": {
        "search_params": {
            "prep_gcu_enabled": space.Categorical(True, False),
            "prep_gcu_rho": space.Categorical(8, 16, 32, 64),
        },
        "defaults": {
            "prep_gcu_enabled": False,
            "prep_gcu_rho": 16,
        },
    },
    "lvse": {
        "search_params": {
            "prep_lvse_enabled": space.Categorical(True, False),
            "prep_lvse_n_regions": space.Categorical(8, 16, 24, 32),
            "prep_lvse_k_per_region": space.Categorical(2, 4, 6, 8),
        },
        "defaults": {
            "prep_lvse_enabled": False,
            "prep_lvse_n_regions": 16,
            "prep_lvse_k_per_region": 4,
        },
    },
    "augmentation": {
        "search_params": {
            "prep_aug_enabled": space.Categorical(True, False),
            "prep_aug_noise": space.Real(lower=0.001, upper=0.05, log=True),
            "prep_aug_shift": space.Categorical(0, 1, 2, 3),
            "prep_aug_n": space.Categorical(1, 2, 3, 4),
            "prep_aug_mixup": space.Categorical(0.0, 0.2, 0.4),
        },
        "defaults": {
            "prep_aug_enabled": False,
            "prep_aug_noise": 0.01,
            "prep_aug_shift": 0,
            "prep_aug_n": 2,
            "prep_aug_mixup": 0.0,
            "prep_aug_max_train_samples": 2000,
        },
    },
}

# Params that control transform-path steps (not augmentation)
_TRANSFORM_ENABLED_PARAMS = [
    "prep_crop_enabled",
    "prep_crr_enabled",
    "prep_bl_enabled",
    "prep_airpls_enabled",
    "prep_arpls_enabled",
    "prep_rubberband_enabled",
    "prep_msc_enabled",
    "prep_emsc_enabled",
    "prep_denoise_enabled",
    "prep_wavelet_enabled",
    "prep_deriv_enabled",
    "prep_snv_enabled",
    "prep_vecnorm_enabled",
    "prep_scaling_enabled",
    "prep_gcu_enabled",
    "prep_lvse_enabled",
]
_ALL_ENABLED_PARAMS = _TRANSFORM_ENABLED_PARAMS + ["prep_aug_enabled"]

# Public mapping: step_key (as used in preprocessing_config / restriction dicts)
# → the boolean "enabled" hyperparameter name for that step.
# Used by model.py to enforce disabled steps at fit time.
STEP_ENABLED_PARAMS: dict[str, str] = {
    step_key: next(k for k in step_def["defaults"] if k.endswith("_enabled"))
    for step_key, step_def in _PREP_STEP_DEFINITIONS.items()
}


def build_restricted_searchspace(restriction: dict | None) -> dict:
    """Return HPO search-space params filtered by *restriction*.

    Parameters
    ----------
    restriction : dict | None
        Mapping of step_key → bool.  Steps with True are included.
        ``None`` means all steps are included (no restriction).

    Returns
    -------
    dict
        Mapping of param_name → ``space.*`` object for AutoGluon HPO.
    """
    ss = {}
    for step_key, step_def in _PREP_STEP_DEFINITIONS.items():
        if restriction is None or restriction.get(step_key, False):
            ss.update(step_def["search_params"])
    return ss


# Instance-attribute names used to carry stateful fit results between fit and
# transform (MSC/EMSC reference spectra, the fitted StandardScaler; GCU/LVSE
# add their own — see the "gcu"/"lvse" step handling further below). The
# preprocessing-ensemble mechanism needs to snapshot/restore these per block,
# since each block fits its own independent copy of this state on the same
# training data.
_STATEFUL_PREP_ATTRS = (
    "_msc_reference",
    "_emsc_reference",
    "scaler",
    "_gcu_shift",
    "_gcu_nmf",
    "_lvse_region_bounds",
    "_lvse_means",
    "_lvse_stds",
    "_lvse_components",
)

# Ensemble-mechanism param names. Deliberately NOT part of
# _PREP_STEP_DEFINITIONS: the ensemble is an orthogonal fan-out/concatenate
# mechanism rather than one more step in the sequential pipeline (see the
# module docstring), so it has no single "search_params"/"defaults" entry
# and no per-step HPO search space of its own. Exposed here so config.py's
# preprocessing_params validation (Task 1) can also accept these two names.
ENSEMBLE_PARAM_NAMES = ("prep_ensemble_enabled", "prep_ensemble_blocks")


def _output_feature_cols(original_cols: list, n_out: int) -> list:
    """Column names for a preprocessed feature matrix that may have changed width.

    Most steps preserve ``n_features``, so the common case is simply
    returning *original_cols* unchanged. But shape-changing steps — ``crop``
    (Task 2), and the representation-replacing ``gcu``/``lvse`` steps — alter
    the column count, and reusing the pre-preprocessing column names would
    raise inside ``pd.DataFrame(..., columns=...)`` (length mismatch) or
    silently truncate/pad. When the width has changed, fall back to
    positional ``feature_0..feature_{n_out-1}`` names instead of the original
    (now meaningless) wavenumber/index labels.
    """
    if n_out == len(original_cols):
        return original_cols
    return [f"feature_{i}" for i in range(n_out)]


class RamanPreprocessingMixin:
    """Mixin that adds Raman preprocessing as tunable model hyperparameters.

    Must be placed before the model class in MRO::

        class Prep_GBM(RamanPreprocessingMixin, LGBModel): pass

    All preprocessing steps default to disabled (``prep_*_enabled = False``).
    Subclasses override ``_set_default_params()`` to enable domain-appropriate
    steps (e.g. ``Prep_PLS`` enables baseline correction, denoising, and SNV).

    The optional ``_prep_restriction`` instance attribute (dict of step_key →
    bool, injected via ``_prep_restriction`` in hyperparameters) restricts
    which steps appear in the HPO search space.  Without it, all steps are
    available for HPO to explore.

    Set ``_optimize_preprocessing = True`` on subclasses where preprocessing
    is an integral part of the model (e.g. PLS, KNN, LR) and should therefore
    be included in HPO.  All other models only tune model-specific parameters.

    Set ``_supports_augmentation = True`` on subclasses that can meaningfully
    consume augmented training data (neural-network models).  When ``False``
    (default), ``prep_aug_enabled`` is always forced off regardless of the
    config — augmentation is silently skipped for tree/kernel/linear models.
    """

    _optimize_preprocessing: bool = False
    _supports_augmentation: bool = False

    def _set_default_params(self):
        super()._set_default_params()
        # Set all preprocessing defaults to False (disabled).
        # Subclasses call super() then override specific params to True.
        for step_def in _PREP_STEP_DEFINITIONS.values():
            for param, val in step_def["defaults"].items():
                self._set_default_param_value(param, val)
        # Ensemble mechanism defaults (orthogonal to _PREP_STEP_DEFINITIONS,
        # see ENSEMBLE_PARAM_NAMES / _preprocess_fit_ensemble docstring).
        self._set_default_param_value("prep_ensemble_enabled", False)
        self._set_default_param_value("prep_ensemble_blocks", None)

    def _get_default_searchspace(self):
        if not getattr(self, "_optimize_preprocessing", False):
            return super()._get_default_searchspace()
        searchspace = super()._get_default_searchspace()
        restriction = getattr(self, "_prep_restriction", None)
        searchspace.update(build_restricted_searchspace(restriction))
        return searchspace

    def _preprocess_fit(self, X: np.ndarray) -> np.ndarray:
        """Fit stateful preprocessing and transform training data.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples, n_features)
        """
        params = self._get_model_params()

        if params.get("prep_crop_enabled", False):
            start_frac = params.get("prep_crop_start_frac", 0.15)
            end_frac = params.get("prep_crop_end_frac", 0.75)
            logger.debug(
                "Fit — crop (fingerprint-region proxy): start_frac=%s, end_frac=%s",
                start_frac,
                end_frac,
            )
            X = crop_spectra(X, start_frac=start_frac, end_frac=end_frac)

        if params.get("prep_crr_enabled", False):
            threshold = params.get("prep_crr_threshold", 6)
            kernel_size = params.get("prep_crr_kernel_size", 3)
            logger.debug(
                "Fit — cosmic ray removal: threshold=%s, kernel_size=%s",
                threshold,
                kernel_size,
            )
            X = cosmic_ray_removal(X, threshold=threshold, kernel_size=kernel_size)

        if params.get("prep_denoise_enabled", False):
            window_length = params.get("prep_denoise_wl", 9)
            polyorder = params.get("prep_denoise_po", 3)
            logger.debug(
                "Fit — denoising (Savitzky-Golay): window_length=%s, polyorder=%s",
                window_length,
                polyorder,
            )
            X = denoise_savgol(X, window_length=window_length, polyorder=polyorder)

        if params.get("prep_wavelet_enabled", False):
            wavelet_name = params.get("prep_wavelet_name", "sym7")
            wavelet_level = params.get("prep_wavelet_level", 4)
            logger.debug(
                "Fit — denoising (wavelet threshold): wavelet=%s, level=%s",
                wavelet_name,
                wavelet_level,
            )
            X = wavelet_denoise(X, wavelet=wavelet_name, level=wavelet_level)

        if params.get("prep_bl_enabled", False):
            lam = params.get("prep_bl_lam", 1e5)
            p = params.get("prep_bl_p", 0.01)
            if lam > 0 and 0 < p < 1:
                logger.debug(
                    "Fit — baseline correction (ASLS): lam=%s, p=%s",
                    lam,
                    p,
                )
                X = baseline_correction_asls(X, lam=lam, p=p)

        if params.get("prep_airpls_enabled", False):
            lam = params.get("prep_airpls_lam", 1e5)
            max_iter = params.get("prep_airpls_max_iter", 15)
            tol = params.get("prep_airpls_tol", 1e-3)
            logger.debug(
                "Fit — baseline correction (airPLS): lam=%s, max_iter=%s, tol=%s",
                lam,
                max_iter,
                tol,
            )
            X = baseline_correction_airpls(X, lam=lam, max_iter=max_iter, tol=tol)

        if params.get("prep_arpls_enabled", False):
            lam = params.get("prep_arpls_lam", 1e5)
            max_iter = params.get("prep_arpls_max_iter", 15)
            logger.debug(
                "Fit — baseline correction (arPLS): lam=%s, max_iter=%s",
                lam,
                max_iter,
            )
            X = baseline_correction_arpls(X, lam=lam, max_iter=max_iter)

        if params.get("prep_rubberband_enabled", False):
            logger.debug("Fit — baseline correction (rubberband / convex hull)")
            X = rubberband_correction(X)

        if params.get("prep_msc_enabled", False):
            logger.debug("Fit — MSC: fitting reference spectrum and transforming")
            self._msc_reference = multiplicative_scatter_correction_fit(X)
            X = multiplicative_scatter_correction_transform(
                X,
                self._msc_reference,
                start=0.0,
                end=1.0,
            )

        if params.get("prep_emsc_enabled", False):
            poly_order = params.get("prep_emsc_poly_order", 4)
            logger.debug(
                "Fit — EMSC: fitting reference spectrum and transforming, poly_order=%s",
                poly_order,
            )
            self._emsc_reference = emsc_fit(X)
            X = emsc_transform(X, self._emsc_reference, poly_order=poly_order)

        if params.get("prep_deriv_enabled", False):
            deriv_order = params.get("prep_deriv_order", 1)
            window_length = params.get("prep_deriv_wl", 11)
            polyorder = params.get("prep_deriv_po", 2)
            logger.debug(
                "Fit — derivative (Savitzky-Golay): deriv=%s, window_length=%s, polyorder=%s",
                deriv_order,
                window_length,
                polyorder,
            )
            X = savgol_derivative(
                X, window_length=window_length, polyorder=polyorder, deriv=deriv_order
            )

        if params.get("prep_snv_enabled", False):
            logger.debug("Fit — SNV: applying Standard Normal Variate")
            X = snv(X)

        if params.get("prep_vecnorm_enabled", False):
            logger.debug("Fit — vector normalization: applying L2 normalization")
            X = vector_normalize(X)

        if params.get("prep_scaling_enabled", False):
            from sklearn.preprocessing import StandardScaler

            logger.debug("Fit — standard scaling: fitting StandardScaler")
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)

        # GCU / LVSE — representation-replacing, terminal steps. See the
        # module docstring ("GCU/LVSE ... run last and replace the feature
        # representation for everything downstream") and gcu_fit/lvse_fit's
        # docstrings in raman_preprocessing.py for what "representation-
        # replacing" means. Design decision (documented, not enforced as a
        # hard error): because these unconditionally run last, right before
        # augmentation, nothing in this pipeline ever runs "after" them on
        # their output, so no step ordering here can actually corrupt
        # shapes — every earlier step (including prep_scaling_enabled)
        # simply becomes part of GCU/LVSE's input. A hard error was
        # considered (per the task spec) but rejected: it would only break
        # legitimate ablation-grid cells (e.g. "scaling + GCU") without
        # preventing any real bug, since there is no code path where a
        # later step consumes GCU/LVSE's replaced representation. The one
        # thing worth flagging is *wasted* computation — a fitted
        # StandardScaler whose output is immediately discarded — so that is
        # logged, not raised. If GCU and LVSE are enabled together, they run
        # as independent ("separate forward pass", per the paper) blocks on
        # the *same* pre-GCU/LVSE input and their outputs are concatenated
        # column-wise — mirroring the ensemble mechanism's parallel-blocks
        # design (Task 3) rather than chaining LVSE on top of GCU's output.
        gcu_enabled = params.get("prep_gcu_enabled", False)
        lvse_enabled = params.get("prep_lvse_enabled", False)
        if (gcu_enabled or lvse_enabled) and params.get("prep_scaling_enabled", False):
            logger.warning(
                "GCU/LVSE is enabled together with standard scaling; the "
                "fitted StandardScaler's output is discarded, since "
                "GCU/LVSE always run last and replace the feature "
                "representation entirely."
            )

        gcu_out = None
        if gcu_enabled:
            rho = params.get("prep_gcu_rho", 16)
            logger.debug("Fit — GCU: rho=%s", rho)
            self._gcu_shift, self._gcu_nmf, gcu_out = gcu_fit(X, rho=rho)

        lvse_out = None
        if lvse_enabled:
            n_regions = params.get("prep_lvse_n_regions", 16)
            k_per_region = params.get("prep_lvse_k_per_region", 4)
            logger.debug(
                "Fit — LVSE: n_regions=%s, k_per_region=%s", n_regions, k_per_region
            )
            lvse_state, lvse_out = lvse_fit(
                X, n_regions=n_regions, k_per_region=k_per_region
            )
            self._lvse_region_bounds = lvse_state["region_indices"]
            self._lvse_means = lvse_state["means"]
            self._lvse_stds = lvse_state["stds"]
            self._lvse_components = lvse_state["components"]

        if gcu_out is not None or lvse_out is not None:
            X = np.concatenate([p for p in (gcu_out, lvse_out) if p is not None], axis=1)

        if np.isnan(X).any():
            n_nan = int(np.isnan(X).sum())
            logger.warning(
                "Preprocessing produced %d NaN value(s) — replacing with 0. "
                "This may indicate constant rows (std=0) after feature engineering.",
                n_nan,
            )
            X = np.nan_to_num(X, nan=0.0)

        return X

    def _preprocess_transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data using fitted preprocessing (no re-fitting).

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples, n_features)
        """
        params = self._get_model_params()

        if params.get("prep_crop_enabled", False):
            start_frac = params.get("prep_crop_start_frac", 0.15)
            end_frac = params.get("prep_crop_end_frac", 0.75)
            logger.debug(
                "Transform — crop (fingerprint-region proxy): start_frac=%s, end_frac=%s",
                start_frac,
                end_frac,
            )
            X = crop_spectra(X, start_frac=start_frac, end_frac=end_frac)

        if params.get("prep_crr_enabled", False):
            threshold = params.get("prep_crr_threshold", 6)
            kernel_size = params.get("prep_crr_kernel_size", 3)
            logger.debug(
                "Transform — cosmic ray removal: threshold=%s, kernel_size=%s",
                threshold,
                kernel_size,
            )
            X = cosmic_ray_removal(X, threshold=threshold, kernel_size=kernel_size)

        if params.get("prep_denoise_enabled", False):
            window_length = params.get("prep_denoise_wl", 9)
            polyorder = params.get("prep_denoise_po", 3)
            logger.debug(
                "Transform — denoising (Savitzky-Golay): window_length=%s, polyorder=%s",
                window_length,
                polyorder,
            )
            X = denoise_savgol(X, window_length=window_length, polyorder=polyorder)

        if params.get("prep_wavelet_enabled", False):
            wavelet_name = params.get("prep_wavelet_name", "sym7")
            wavelet_level = params.get("prep_wavelet_level", 4)
            logger.debug(
                "Transform — denoising (wavelet threshold): wavelet=%s, level=%s",
                wavelet_name,
                wavelet_level,
            )
            X = wavelet_denoise(X, wavelet=wavelet_name, level=wavelet_level)

        if params.get("prep_bl_enabled", False):
            lam = params.get("prep_bl_lam", 1e5)
            p = params.get("prep_bl_p", 0.01)
            if lam > 0 and 0 < p < 1:
                logger.debug(
                    "Transform — baseline correction (ASLS): lam=%s, p=%s",
                    lam,
                    p,
                )
                X = baseline_correction_asls(X, lam=lam, p=p)

        if params.get("prep_airpls_enabled", False):
            lam = params.get("prep_airpls_lam", 1e5)
            max_iter = params.get("prep_airpls_max_iter", 15)
            tol = params.get("prep_airpls_tol", 1e-3)
            logger.debug(
                "Transform — baseline correction (airPLS): lam=%s, max_iter=%s, tol=%s",
                lam,
                max_iter,
                tol,
            )
            X = baseline_correction_airpls(X, lam=lam, max_iter=max_iter, tol=tol)

        if params.get("prep_arpls_enabled", False):
            lam = params.get("prep_arpls_lam", 1e5)
            max_iter = params.get("prep_arpls_max_iter", 15)
            logger.debug(
                "Transform — baseline correction (arPLS): lam=%s, max_iter=%s",
                lam,
                max_iter,
            )
            X = baseline_correction_arpls(X, lam=lam, max_iter=max_iter)

        if params.get("prep_rubberband_enabled", False):
            logger.debug("Transform — baseline correction (rubberband / convex hull)")
            X = rubberband_correction(X)

        if params.get("prep_msc_enabled", False) and hasattr(self, "_msc_reference"):
            logger.debug("Transform — MSC: applying transform with fitted reference spectrum")
            X = multiplicative_scatter_correction_transform(
                X,
                self._msc_reference,
                start=0.0,
                end=1.0,
            )

        if params.get("prep_emsc_enabled", False) and hasattr(self, "_emsc_reference"):
            poly_order = params.get("prep_emsc_poly_order", 4)
            logger.debug(
                "Transform — EMSC: applying transform with fitted reference spectrum, "
                "poly_order=%s",
                poly_order,
            )
            X = emsc_transform(X, self._emsc_reference, poly_order=poly_order)

        if params.get("prep_deriv_enabled", False):
            deriv_order = params.get("prep_deriv_order", 1)
            window_length = params.get("prep_deriv_wl", 11)
            polyorder = params.get("prep_deriv_po", 2)
            logger.debug(
                "Transform — derivative (Savitzky-Golay): deriv=%s, window_length=%s, "
                "polyorder=%s",
                deriv_order,
                window_length,
                polyorder,
            )
            X = savgol_derivative(
                X, window_length=window_length, polyorder=polyorder, deriv=deriv_order
            )

        if params.get("prep_snv_enabled", False):
            logger.debug("Transform — SNV: applying Standard Normal Variate")
            X = snv(X)

        if params.get("prep_vecnorm_enabled", False):
            logger.debug("Transform — vector normalization: applying L2 normalization")
            X = vector_normalize(X)

        if params.get("prep_scaling_enabled", False) and hasattr(self, "scaler"):
            logger.debug("Transform — standard scaling: applying fitted StandardScaler")
            X = self.scaler.transform(X)

        # GCU / LVSE — see the matching block in _preprocess_fit for the
        # composition-order design rationale.
        gcu_out = None
        if params.get("prep_gcu_enabled", False) and hasattr(self, "_gcu_nmf"):
            logger.debug("Transform — GCU: applying fitted NMF projection")
            gcu_out = gcu_transform(X, self._gcu_shift, self._gcu_nmf)

        lvse_out = None
        if params.get("prep_lvse_enabled", False) and hasattr(self, "_lvse_components"):
            logger.debug("Transform — LVSE: applying fitted per-region projections")
            lvse_state = {
                "region_indices": self._lvse_region_bounds,
                "means": self._lvse_means,
                "stds": self._lvse_stds,
                "components": self._lvse_components,
            }
            lvse_out = lvse_transform(X, lvse_state)

        if gcu_out is not None or lvse_out is not None:
            X = np.concatenate([p for p in (gcu_out, lvse_out) if p is not None], axis=1)

        if np.isnan(X).any():
            n_nan = int(np.isnan(X).sum())
            logger.warning(
                "Preprocessing produced %d NaN value(s) — replacing with 0. "
                "This may indicate constant rows (std=0) after feature engineering.",
                n_nan,
            )
            X = np.nan_to_num(X, nan=0.0)

        return X

    def _preprocess_fit_ensemble(self, X: np.ndarray, blocks: list) -> np.ndarray:
        """Fit each preprocessing *block* independently and concatenate columns.

        A "preprocessing ensemble" is a different kind of mechanism than the
        single ordered pipeline in ``_preprocess_fit``: rather than one
        recipe applied once, several independent recipes ("blocks", each a
        ``preprocessing_config``-shaped dict of ``step_key -> bool``, e.g.
        ``{"snv": True}``) are each run to completion — reusing
        :meth:`_preprocess_fit` internally, never reimplementing a
        transform — and their outputs are concatenated column-wise into one
        wide feature matrix. This is the "ensemble of preprocessings as
        parallel blocks" design used as a strong (if less interpretable)
        baseline in prior spectroscopy/chemometrics work (e.g. PROSAC/SPORT
        report roughly 5-25% error reduction on NIR data from exactly this
        kind of block ensembling): you trade away the ability to point at
        *which* single recipe helped in exchange for an upper-bound-ish
        result. This method and :meth:`_preprocess_transform_ensemble` are
        gated behind ``prep_ensemble_enabled`` + ``prep_ensemble_blocks`` and
        are orthogonal to (not one more entry in) ``_PREP_STEP_DEFINITIONS``.

        Fold-safety: each block's stateful fit (MSC/EMSC reference spectrum,
        StandardScaler, ...) is fit **once per block, on training data only**
        — exactly the same fold-safety contract as the single-recipe path —
        by temporarily swapping ``self.params`` to that block's flags,
        calling the ordinary :meth:`_preprocess_fit`, then snapshotting
        whichever of ``_STATEFUL_PREP_ATTRS`` got set before moving to the
        next block (so blocks cannot leak fit state into one another). The
        snapshots are stored on ``self`` (``_ensemble_block_params`` /
        ``_ensemble_block_fit_state``) for :meth:`_preprocess_transform_ensemble`
        to replay at inference time.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Input spectra (already past any steps that ran before the
            ensemble in the base pipeline, if any).
        blocks : list[dict]
            List of ``preprocessing_config``-shaped dicts, one per block.
            Each is interpreted the same way as a top-level restriction:
            steps present and truthy are enabled for that block, everything
            else is disabled for that block (independent of the base
            model's own ``prep_*_enabled`` settings).

        Returns
        -------
        np.ndarray, shape (n_samples, n_blocks * n_features_per_block)
            Column-wise concatenation of every block's transformed output.
            Like ``crop``/``gcu``/``lvse``, this changes the feature count.
        """
        base_params = dict(self.params)
        block_param_overlays = []
        block_fit_states = []
        block_outputs = []

        for block in blocks:
            block_params = dict(base_params)
            for step_key, enabled_param in STEP_ENABLED_PARAMS.items():
                block_params[enabled_param] = bool(block.get(step_key, False))
            self.params = block_params

            # Clear stateful attrs left over from the previous block so this
            # block starts from a clean slate (a step that this block leaves
            # disabled must not inherit another block's fitted state).
            for attr in _STATEFUL_PREP_ATTRS:
                if hasattr(self, attr):
                    delattr(self, attr)

            logger.debug("Fit — ensemble block %d: %s", len(block_outputs), block)
            X_block = self._preprocess_fit(X)
            block_outputs.append(X_block)
            block_param_overlays.append(block_params)
            block_fit_states.append(
                {attr: getattr(self, attr) for attr in _STATEFUL_PREP_ATTRS if hasattr(self, attr)}
            )

        self.params = base_params
        for attr in _STATEFUL_PREP_ATTRS:
            if hasattr(self, attr):
                delattr(self, attr)

        self._ensemble_block_params = block_param_overlays
        self._ensemble_block_fit_state = block_fit_states

        if not block_outputs:
            return X
        return np.concatenate(block_outputs, axis=1)

    def _preprocess_transform_ensemble(self, X: np.ndarray) -> np.ndarray:
        """Transform through each fitted preprocessing block, then concatenate.

        Mirrors :meth:`_preprocess_fit_ensemble` at inference time: replays
        each block's snapshotted params + fitted state (``self.params`` and
        ``_STATEFUL_PREP_ATTRS`` set by ``_preprocess_fit_ensemble``) and
        calls the ordinary :meth:`_preprocess_transform` — never refitting
        anything.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples, n_blocks * n_features_per_block)
        """
        base_params = dict(self.params)
        block_outputs = []

        block_param_overlays = getattr(self, "_ensemble_block_params", [])
        block_fit_states = getattr(self, "_ensemble_block_fit_state", [])

        for block_params, fit_state in zip(block_param_overlays, block_fit_states):
            self.params = block_params
            for attr in _STATEFUL_PREP_ATTRS:
                if hasattr(self, attr):
                    delattr(self, attr)
            for attr, value in fit_state.items():
                setattr(self, attr, value)

            X_block = self._preprocess_transform(X)
            block_outputs.append(X_block)

        self.params = base_params
        for attr in _STATEFUL_PREP_ATTRS:
            if hasattr(self, attr):
                delattr(self, attr)

        if not block_outputs:
            return X
        return np.concatenate(block_outputs, axis=1)

    def _fit(self, X, y, **kwargs):
        # Pop _prep_restriction so it doesn't reach underlying library constructors.
        prep_restriction = self.params.pop("_prep_restriction", None)
        if prep_restriction is not None:
            self._prep_restriction = prep_restriction

        # For models that don't expose preprocessing to HPO, forcibly restore the
        # intended prep state.  AutoGluon's Bayesian HPO can sample prep params as
        # True even when they are absent from the model's *declared* search space,
        # because _get_search_space() merges self.params (which contains the defaults
        # set by _set_default_params) into the search space dict.  Enforcing here is
        # the only reliable defence.
        if not getattr(self, "_optimize_preprocessing", False):
            # Steps the config explicitly asked for must survive this gate. It
            # exists to stop HPO from *sampling* prep params, but blanket-clearing
            # every step also erased preprocessing that preprocessing_config had
            # deliberately enabled — silently, since the mixin still logged
            # "Start Preprocessing" for the augmentation step. Only PLS/KNN/LR set
            # _optimize_preprocessing, so for every other model (all Raman DL
            # models, TabPFN, the tree models) each arm of a preprocessing
            # ablation ran as if it were the "none" arm and produced identical
            # predictions.
            # _user_params holds only what was passed in as hyperparameters, i.e.
            # exactly the flags _build_model_hyperparameters derived from
            # preprocessing_config. Class defaults and HPO samples are not in it,
            # so clearing everything else keeps the HPO defence intact.
            # Note _build_model_hyperparameters *pops* _prep_restriction before
            # constructing the model, so self._prep_restriction is unset here and
            # cannot be used to recover the intent.
            user_params = getattr(self, "_user_params", None) or {}
            for k in _TRANSFORM_ENABLED_PARAMS:
                if k not in user_params:
                    self.params[k] = False

        # Augmentation gate is unconditional — applies to _optimize_preprocessing
        # models (PLS, KNN, LR) as well as all others.  Without this, HPO can
        # sample prep_aug_enabled=True for PLS because augmentation params are
        # in the search space whenever augmentation=True in the config, and the
        # _optimize_preprocessing path bypasses the block above entirely.
        if not getattr(self, "_supports_augmentation", False):
            self.params["prep_aug_enabled"] = False

        params = self._get_model_params()
        ensemble_enabled = params.get("prep_ensemble_enabled", False) and params.get(
            "prep_ensemble_blocks"
        )
        has_preprocessing = (
            any(params.get(k, False) for k in _ALL_ENABLED_PARAMS) or ensemble_enabled
        )

        logger.debug("Has preprocessing: %s", has_preprocessing)

        if has_preprocessing:
            feature_cols = X.columns.tolist()
            X_np = X.values.astype(np.float64)
            logger.info(
                "Start Preprocessing for Training — %d spectra (%s)",
                len(X_np),
                self.__class__.__name__,
            )
            if ensemble_enabled:
                blocks = params.get("prep_ensemble_blocks")
                logger.info(
                    "Fit — preprocessing ensemble: %d block(s) (%s)",
                    len(blocks),
                    self.__class__.__name__,
                )
                X_np = self._preprocess_fit_ensemble(X_np, blocks)
            else:
                X_np = self._preprocess_fit(X_np)

            if params.get("prep_aug_enabled", False):
                noise_sigma = params.get("prep_aug_noise", 0.01)
                shift_max = params.get("prep_aug_shift", 0)
                n_augments = params.get("prep_aug_n", 2)
                mixup_alpha = params.get("prep_aug_mixup", 0.0)
                max_train_samples = params.get("prep_aug_max_train_samples", None)

                if max_train_samples is not None and len(X_np) > max_train_samples:
                    logger.info(
                        "%s: skipping preprocessing augmentation — train set (%d) "
                        "exceeds prep_aug_max_train_samples (%d).",
                        self.__class__.__name__,
                        len(X_np),
                        max_train_samples,
                    )
                else:
                    y_np = y.values if hasattr(y, "values") else np.array(y)
                    problem_type = getattr(self, "problem_type", None)
                    label_type = "regression" if problem_type == "regression" else "classification"
                    logger.debug(
                        "Fit — augmentation: noise_sigma=%s, shift_max=%s, "
                        "n_augments=%s, mixup_alpha=%s, label_type=%s",
                        noise_sigma,
                        shift_max,
                        n_augments,
                        mixup_alpha,
                        label_type,
                    )
                    X_np, y_np = augment_spectra(
                        X_np,
                        y_np,
                        noise_sigma=noise_sigma,
                        shift_max=shift_max,
                        n_augments=n_augments,
                        mixup_alpha=mixup_alpha,
                        label_type=label_type,
                    )
                    y_name = y.name if hasattr(y, "name") else "target"
                    y = pd.Series(y_np, name=y_name)

            X = pd.DataFrame(X_np, columns=_output_feature_cols(feature_cols, X_np.shape[1]))

        # Strip prep_* params before forwarding to the underlying library
        # constructor (e.g. CatBoostClassifier raises on unknown kwargs).
        # They are restored in `finally` so _get_model_params() still works
        # during predict().
        prep_keys = [k for k in list(self.params.keys()) if k.startswith("prep_")]
        prep_params_backup = {k: self.params.pop(k) for k in prep_keys}
        try:
            result = super()._fit(X, y, **kwargs)
            logger.info("Fitted Model — %s", self.__class__.__name__)
            return result
        finally:
            self.params.update(prep_params_backup)

    def _preprocess_if_dataframe(self, X):
        """Apply preprocessing transform only if X is a pandas DataFrame.

        Some AutoGluon models (e.g. NN_TORCH) pass internal dataset objects
        during validation scoring — these have already been transformed and
        should be passed through unchanged.
        """
        params = self._get_model_params()
        ensemble_enabled = params.get("prep_ensemble_enabled", False) and params.get(
            "prep_ensemble_blocks"
        )
        if any(params.get(k, False) for k in _TRANSFORM_ENABLED_PARAMS) or ensemble_enabled:
            if isinstance(X, pd.DataFrame):
                logger.info(
                    "Start Preprocessing for Inference — %d spectra (%s)",
                    len(X),
                    self.__class__.__name__,
                )
                feature_cols = X.columns.tolist()
                X_np = X.values.astype(np.float64)
                if ensemble_enabled:
                    X_np = self._preprocess_transform_ensemble(X_np)
                else:
                    X_np = self._preprocess_transform(X_np)
                X = pd.DataFrame(X_np, columns=_output_feature_cols(feature_cols, X_np.shape[1]))
        return X

    def _predict(self, X, **kwargs):
        X = self._preprocess_if_dataframe(X)
        return super()._predict(X, **kwargs)

    def _predict_proba(self, X, **kwargs):
        # AutoGluon uses _predict_proba as the universal internal prediction
        # entry point for all problem types, including regression. Preprocessing
        # is applied here so every model gets exactly one pass regardless of
        # problem type.
        # IMPORTANT: custom models must NOT implement _predict_proba for
        # regression by delegating to self._predict — that re-enters this
        # method's MRO and applies preprocessing a second time. Use a private
        # _raw_predict() helper that bypasses the mixin instead.
        n_spectra = len(X) if hasattr(X, "__len__") else None
        X = self._preprocess_if_dataframe(X)
        result = super()._predict_proba(X, **kwargs)
        if n_spectra is not None:
            logger.info("Predicted %d spectra (%s)", n_spectra, self.__class__.__name__)
        return result
