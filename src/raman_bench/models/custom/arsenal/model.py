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


class _ArsenalBridge(SklearnAutoGluonBridge):
    _sklearn_cls = ArsenalModel
    ag_key = "ARSENAL"
    ag_name = "ARSENAL"

    def _get_default_searchspace(self):
        from autogluon.common import space

        return {
            "rocket_transform": space.Categorical("rocket", "minirocket"),
            "num_kernels": space.Int(lower=2_000, upper=5_000),
            "n_estimators": space.Int(lower=20, upper=40),
        }


class Prep_ARSENAL(_NoAugBase, _ArsenalBridge):  # noqa: N801
    pass
