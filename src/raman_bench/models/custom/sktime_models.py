import logging

import numpy as np
from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)

_CLF_ONLY_MSG = "{cls} only supports classification tasks (binary/multiclass)."


def _to_3d(X):
    """Reshape tabular X (n_samples, n_features) to sktime numpy3D format."""
    arr = X.values if hasattr(X, "values") else np.asarray(X)
    return arr.reshape(arr.shape[0], 1, arr.shape[1])


def _ensure_2d_proba(proba: np.ndarray, n_samples: int) -> np.ndarray:
    """Normalise sktime predict_proba output to (n_samples, n_classes).

    Newer sktime versions may return (n_classes, n_samples, 1) instead of
    the sklearn-standard (n_samples, n_classes).
    """
    if proba.ndim == 3:
        proba = proba.squeeze(-1)          # drop trailing size-1 dim
        if proba.shape[0] != n_samples:    # got (n_classes, n_samples)
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


class ArsenalModel(BaseEstimator):
    """Arsenal ensemble classifier for Raman spectra — sklearn-compatible estimator.

    Classification-only (binary and multiclass).

    Reference:
        Middlehurst, M., et al. (2021). HIVE-COTE 2.0: a new meta ensemble for
        time series classification. Machine Learning, 110(11), 3211–3243.
        https://doi.org/10.1007/s10994-021-06055-9
    """

    def __init__(self, num_kernels=2_000, n_estimators=25, rocket_transform="rocket"):
        self.num_kernels = num_kernels
        self.n_estimators = n_estimators
        self.rocket_transform = rocket_transform

    def fit(self, X, y):
        try:
            from sktime.classification.kernel_based import Arsenal
        except ImportError:
            raise ImportError(
                "ArsenalModel requires sktime. Install with: pip install 'raman-bench[models]'"
            )
        self.model_ = Arsenal(
            num_kernels=self.num_kernels,
            n_estimators=self.n_estimators,
            rocket_transform=self.rocket_transform,
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
