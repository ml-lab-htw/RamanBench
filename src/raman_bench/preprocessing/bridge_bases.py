"""Shared AutoGluon bridge infrastructure for custom sklearn/PyTorch models.

Split out of ``wrapped_models.py`` so per-model packages (``models/custom/<key>/``,
see ``raman_bench.models.discover``) can import these bases without importing
``wrapped_models`` itself and its ``PREPROCESSED_MODELS`` merge step -- that
merge step imports ``discover_custom_models()``, which imports every model
package's ``model.py``, which needs these bases; importing them from
``wrapped_models`` instead would be a circular import.
"""

from __future__ import annotations

import inspect

import numpy as np

try:
    from autogluon.core.models import AbstractModel
except ImportError as _ag_err:
    raise ImportError(
        "raman_bench.preprocessing.bridge_bases requires autogluon. "
        "Install with: pip install 'raman-bench[autogluon]'"
    ) from _ag_err

from raman_bench.preprocessing.mixin import RamanPreprocessingMixin


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


class _NoAugBase(RamanPreprocessingMixin):
    """Disable preprocessing augmentation by default."""

    def _set_default_params(self):
        self._set_default_param_value("prep_aug_enabled", False)
        super()._set_default_params()


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


def _make_optional_prep_class(name: str, base_model_cls, **class_attrs):
    """Build a ``Prep_*`` class for a possibly-unavailable AutoGluon model class.

    Returns ``None`` (rather than raising ``TypeError: bases must be types``)
    when ``base_model_cls`` is ``None`` -- i.e. this AutoGluon build doesn't
    carry that foundation-model class.
    """
    if base_model_cls is None:
        return None
    return type(name, (_NoAugBase, base_model_cls), dict(class_attrs))
