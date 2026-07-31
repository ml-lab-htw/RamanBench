import numpy as np
from sklearn.base import BaseEstimator

from raman_bench.preprocessing.bridge_bases import SklearnAutoGluonBridge, _NoAugBase


def _to_3d(X):
    """Reshape tabular X (n_samples, n_features) to sktime numpy3D format."""
    arr = X.values if hasattr(X, "values") else np.asarray(X)
    return arr.reshape(arr.shape[0], 1, arr.shape[1])


def _ensure_2d_proba(proba: np.ndarray, n_samples: int) -> np.ndarray:
    """Normalise sktime predict_proba output to (n_samples, n_classes).

    Handles several shapes produced by different sktime versions:
    - (n_classes, n_samples, 1)       → drop trailing dim, transpose
    - (n_classes, n_timepoints, n_samples) → collapse middle dim, transpose
    - (n_samples, n_timepoints, n_classes) → collapse middle dim
    """
    if proba.ndim == 3:
        if proba.shape[-1] == 1:
            proba = proba[..., 0]  # (n_classes, n_samples) or (n_samples, n_classes)
        else:
            # e.g. (n_classes, n_timepoints, n_samples) — average over middle axis
            proba = proba.mean(axis=1)
        if proba.shape[0] != n_samples:
            proba = proba.T
    return proba


class RocketModel(BaseEstimator):
    """ROCKET classifier for Raman spectra — sklearn-compatible estimator.

    Classification-only (binary and multiclass).

    Reference:
        Dempster, A., Petitjean, F., & Webb, G. I. (2020). ROCKET: exceptionally
        fast and accurate time series classification using random convolutional kernels.
        Data Mining and Knowledge Discovery, 34(5), 1454–1495.
        https://doi.org/10.1007/s10618-020-00701-z
    """

    def __init__(self, rocket_transform="minirocket", num_kernels=10_000):
        self.rocket_transform = rocket_transform
        self.num_kernels = num_kernels

    def fit(self, X, y):
        try:
            from sktime.classification.kernel_based import RocketClassifier
        except ImportError:
            raise ImportError(
                "RocketModel requires sktime. Install with: pip install 'raman-bench[models]'"
            )
        self.model_ = RocketClassifier(
            rocket_transform=self.rocket_transform,
            num_kernels=self.num_kernels,
            n_jobs=1,
        )
        self.model_.fit(_to_3d(X), np.asarray(y))
        self.classes_ = self.model_.classes_
        return self

    def predict(self, X):
        return self.model_.predict(_to_3d(X))

    def predict_proba(self, X):
        proba = self.model_.predict_proba(_to_3d(X))
        proba = _ensure_2d_proba(proba, np.asarray(X).shape[0])
        if len(self.classes_) == 2:
            return proba[:, 1]
        return proba


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


class Prep_ROCKET(_NoAugBase, _RocketBridge):  # noqa: N801
    pass
