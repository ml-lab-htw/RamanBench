"""
RamanPreprocessingMixin — mixin class that integrates Raman spectral preprocessing
as tunable hyperparameters within AutoGluon models.

All preprocessing is disabled by default (prep_*_enabled = False).
Subclasses override _set_default_params() to enable specific steps for their
domain (e.g. Prep_PLS enables baseline correction + denoising + SNV).

The optional _prep_restriction attribute (a dict of {step_key: bool}) narrows
the HPO search space to physically meaningful steps for the current dataset.
When absent, all steps are included in the search space.

Preprocessing order (when enabled):
1. Cosmic ray removal
2. Baseline correction (ASLS)
3. MSC (multiplicative scatter correction)
4. Denoising (Savitzky-Golay)
5. SNV (Standard Normal Variate)
6. Standard scaling
7. Augmentation (training only)
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
    baseline_correction_asls,  # also covers fluorescence removal at high lam / low p
    cosmic_ray_removal,
    denoise_savgol,
    multiplicative_scatter_correction_fit,
    multiplicative_scatter_correction_transform,
    snv,
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
    "msc": {
        "search_params": {
            "prep_msc_enabled": space.Categorical(True, False),
        },
        "defaults": {
            "prep_msc_enabled": False,
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
    "snv": {
        "search_params": {
            "prep_snv_enabled": space.Categorical(True, False),
        },
        "defaults": {
            "prep_snv_enabled": False,
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
    "prep_crr_enabled",
    "prep_bl_enabled",
    "prep_msc_enabled",
    "prep_denoise_enabled",
    "prep_snv_enabled",
    "prep_scaling_enabled",
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

        if params.get("prep_msc_enabled", False):
            logger.debug("Fit — MSC: fitting reference spectrum and transforming")
            self._msc_reference = multiplicative_scatter_correction_fit(X)
            X = multiplicative_scatter_correction_transform(
                X,
                self._msc_reference,
                start=0.0,
                end=1.0,
            )

        if params.get("prep_snv_enabled", False):
            logger.debug("Fit — SNV: applying Standard Normal Variate")
            X = snv(X)

        if params.get("prep_scaling_enabled", False):
            from sklearn.preprocessing import StandardScaler

            logger.debug("Fit — standard scaling: fitting StandardScaler")
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)

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

        if params.get("prep_msc_enabled", False) and hasattr(self, "_msc_reference"):
            logger.debug("Transform — MSC: applying transform with fitted reference spectrum")
            X = multiplicative_scatter_correction_transform(
                X,
                self._msc_reference,
                start=0.0,
                end=1.0,
            )

        if params.get("prep_snv_enabled", False):
            logger.debug("Transform — SNV: applying Standard Normal Variate")
            X = snv(X)

        if params.get("prep_scaling_enabled", False) and hasattr(self, "scaler"):
            logger.debug("Transform — standard scaling: applying fitted StandardScaler")
            X = self.scaler.transform(X)

        if np.isnan(X).any():
            n_nan = int(np.isnan(X).sum())
            logger.warning(
                "Preprocessing produced %d NaN value(s) — replacing with 0. "
                "This may indicate constant rows (std=0) after feature engineering.",
                n_nan,
            )
            X = np.nan_to_num(X, nan=0.0)

        return X

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
        has_preprocessing = any(params.get(k, False) for k in _ALL_ENABLED_PARAMS)

        logger.debug("Has preprocessing: %s", has_preprocessing)

        if has_preprocessing:
            feature_cols = X.columns.tolist()
            X_np = X.values.astype(np.float64)
            logger.info(
                "Start Preprocessing for Training — %d spectra (%s)",
                len(X_np),
                self.__class__.__name__,
            )
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

            X = pd.DataFrame(X_np, columns=feature_cols)

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
        if any(params.get(k, False) for k in _TRANSFORM_ENABLED_PARAMS):
            if isinstance(X, pd.DataFrame):
                logger.info(
                    "Start Preprocessing for Inference — %d spectra (%s)",
                    len(X),
                    self.__class__.__name__,
                )
                feature_cols = X.columns.tolist()
                X_np = X.values.astype(np.float64)
                X_np = self._preprocess_transform(X_np)
                X = pd.DataFrame(X_np, columns=feature_cols)
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
