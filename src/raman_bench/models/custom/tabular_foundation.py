"""Sklearn-compatible wrappers for tabular foundation models.

All models auto-dispatch to the appropriate classifier or regressor variant
based on ``y``'s dtype: float arrays → regression, integer/string arrays →
binary or multiclass classification.

Required extras::

    pip install 'raman-bench[models]'

which installs ``tabpfn``, ``pytabkit``, and ``tabdpt``.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator

_DEEP_INSTALL = "pip install 'raman-bench[models]'"


def _to_numpy(X) -> np.ndarray:
    return X.values if hasattr(X, "values") else np.asarray(X)


def _infer_problem_type(y) -> str:
    arr = np.asarray(y)
    if np.issubdtype(arr.dtype, np.floating):
        return "regression"
    return "binary" if len(np.unique(arr)) == 2 else "multiclass"


class TabPFNModel(BaseEstimator):
    """TabPFN v2 for Raman spectra — sklearn-compatible (classification + regression).

    Requires the ``tabpfn`` package (``pip install 'raman-bench[models]'``).
    Works on datasets with any number of features.

    Reference:
        Hollmann N, et al. TabPFN v2: Improved In-Context Learning for
        Tabular Data. arXiv:2501.02945 (2025).
    """

    def __init__(self, n_estimators: int = 4, device: str = "cpu"):
        self.n_estimators = n_estimators
        self.device = device

    def fit(self, X, y):
        try:
            from tabpfn import TabPFNClassifier, TabPFNRegressor
        except ImportError as e:
            raise ImportError(f"TabPFNModel requires tabpfn. Install with: {_DEEP_INSTALL}") from e

        X_arr, y_arr = _to_numpy(X), np.asarray(y)
        self.problem_type_ = _infer_problem_type(y)
        kwargs = dict(n_estimators=self.n_estimators, device=self.device)

        if self.problem_type_ == "regression":
            self.model_ = TabPFNRegressor(**kwargs)
        else:
            self.classes_ = np.unique(y_arr)
            self.model_ = TabPFNClassifier(**kwargs)

        self.model_.fit(X_arr, y_arr)
        return self

    def predict(self, X):
        return self.model_.predict(_to_numpy(X))

    def predict_proba(self, X):
        if self.problem_type_ == "regression":
            raise ValueError("predict_proba is not available for regression.")
        proba = self.model_.predict_proba(_to_numpy(X))
        return proba[:, 1] if self.problem_type_ == "binary" else proba


class RealMLPModel(BaseEstimator):
    """RealMLP (pytabkit) for Raman spectra — sklearn-compatible (classification + regression).

    Requires the ``pytabkit`` package (``pip install 'raman-bench[models]'``).

    Reference:
        Gorishniy Y, et al. RealMLP-TD: A Powerful Simple Tabular Deep Learning
        Method. arXiv:2407.04491 (2024).
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

    def fit(self, X, y):
        try:
            from pytabkit import RealMLP_TD_Classifier, RealMLP_TD_Regressor
        except ImportError as e:
            raise ImportError(
                f"RealMLPModel requires pytabkit. Install with: {_DEEP_INSTALL}"
            ) from e

        X_arr, y_arr = _to_numpy(X), np.asarray(y)
        self.problem_type_ = _infer_problem_type(y)

        if self.problem_type_ == "regression":
            self.model_ = RealMLP_TD_Regressor(device=self.device)
        else:
            self.classes_ = np.unique(y_arr)
            self.model_ = RealMLP_TD_Classifier(device=self.device)

        self.model_.fit(X_arr, y_arr)
        return self

    def predict(self, X):
        return self.model_.predict(_to_numpy(X))

    def predict_proba(self, X):
        if self.problem_type_ == "regression":
            raise ValueError("predict_proba is not available for regression.")
        proba = self.model_.predict_proba(_to_numpy(X))
        return proba[:, 1] if self.problem_type_ == "binary" else proba


class TabMModel(BaseEstimator):
    """TabM (pytabkit) for Raman spectra — sklearn-compatible (classification + regression).

    Requires the ``pytabkit`` package (``pip install 'raman-bench[models]'``).

    Reference:
        Gorishniy Y, et al. TabM: Advancing Tabular Deep Learning with
        Parameter-Efficient Ensembling. arXiv:2410.24210 (2024).
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

    def fit(self, X, y):
        try:
            from pytabkit import TabM_D_Classifier, TabM_D_Regressor
        except ImportError as e:
            raise ImportError(f"TabMModel requires pytabkit. Install with: {_DEEP_INSTALL}") from e

        X_arr, y_arr = _to_numpy(X), np.asarray(y)
        self.problem_type_ = _infer_problem_type(y)

        if self.problem_type_ == "regression":
            self.model_ = TabM_D_Regressor(device=self.device)
        else:
            self.classes_ = np.unique(y_arr)
            self.model_ = TabM_D_Classifier(device=self.device)

        self.model_.fit(X_arr, y_arr)
        return self

    def predict(self, X):
        return self.model_.predict(_to_numpy(X))

    def predict_proba(self, X):
        if self.problem_type_ == "regression":
            raise ValueError("predict_proba is not available for regression.")
        proba = self.model_.predict_proba(_to_numpy(X))
        return proba[:, 1] if self.problem_type_ == "binary" else proba


class TabDPTModel(BaseEstimator):
    """TabDPT for Raman spectra — sklearn-compatible (classification + regression).

    Requires the ``tabdpt`` package (``pip install 'raman-bench[models]'``).

    Reference:
        Ma J, et al. TabDPT: Scaling Tabular Foundation Models.
        arXiv:2410.18164 (2024).
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

    def fit(self, X, y):
        try:
            from tabdpt import TabDPTClassifier, TabDPTRegressor
        except ImportError as e:
            raise ImportError(f"TabDPTModel requires tabdpt. Install with: {_DEEP_INSTALL}") from e

        X_arr, y_arr = _to_numpy(X), np.asarray(y)
        self.problem_type_ = _infer_problem_type(y)

        if self.problem_type_ == "regression":
            self.model_ = TabDPTRegressor(device=self.device)
            self.model_.fit(X_arr, y_arr)
        else:
            self.classes_ = np.unique(y_arr)
            self._class_to_idx = {c: i for i, c in enumerate(self.classes_)}
            y_enc = np.array([self._class_to_idx[v] for v in y_arr], dtype=np.int64)
            self.model_ = TabDPTClassifier(device=self.device)
            self.model_.fit(X_arr, y_enc)

        return self

    def predict(self, X):
        raw = self.model_.predict(_to_numpy(X))
        if self.problem_type_ == "regression":
            return raw
        return self.classes_[raw.astype(int)]

    def predict_proba(self, X):
        if self.problem_type_ == "regression":
            raise ValueError("predict_proba is not available for regression.")
        proba = self.model_.predict_proba(_to_numpy(X))
        return proba[:, 1] if self.problem_type_ == "binary" else proba
