"""PCR (principal component regression) / PCR-DA.

Unsupervised latent-projection linear model: PCA to ``n_components`` then
ordinary least squares on the scores. Added 2026-08-28 as the classic
counterpart to PLS for the RamanPreprocessing study — PCA is *not* scale
invariant and *not* supervised, so a high-variance but analyte-irrelevant
region (e.g. residual fluorescence) can dominate the leading components unless
removed by preprocessing. PCR is therefore predicted to be *more* sensitive to
baseline correction and scaling than PLS, whose projection is supervised.

Regression wraps ``PCA`` + ``LinearRegression``. Classification is PCR-DA:
one-hot encode the target, fit on the encoded matrix, argmax at inference —
the same discriminant-analysis construction ``PLSModel`` uses for PLS-DA.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

from raman_bench.preprocessing.bridge_bases import SklearnAutoGluonBridge, _NoAugBase


class PCRModel(BaseEstimator):
    """Principal component regression / PCR-DA — sklearn-compatible."""

    def __init__(self, n_components=20):
        self.n_components = n_components

    def fit(self, X, y):
        X_np = X.values if hasattr(X, "values") else np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y)

        k = min(self.n_components, X_np.shape[1], X_np.shape[0] - 1)
        self.pca_ = PCA(n_components=max(k, 1))
        scores = self.pca_.fit_transform(X_np)

        if np.issubdtype(y_arr.dtype, np.floating):
            self.problem_type_ = "regression"
            self.model_ = LinearRegression().fit(scores, y_arr)
        else:
            self.classes_ = np.unique(y_arr)
            self.problem_type_ = "binary" if len(self.classes_) == 2 else "multiclass"
            idx = {c: i for i, c in enumerate(self.classes_)}
            y_oh = np.zeros((len(y_arr), len(self.classes_)))
            for i, val in enumerate(y_arr):
                y_oh[i, idx[val]] = 1.0
            self.model_ = LinearRegression().fit(scores, y_oh)
        return self

    def predict(self, X):
        X_np = X.values if hasattr(X, "values") else np.asarray(X, dtype=np.float64)
        scores = self.pca_.transform(X_np)
        if self.problem_type_ == "regression":
            return self.model_.predict(scores).ravel()
        return self.classes_[np.argmax(self.model_.predict(scores), axis=1)]

    def predict_proba(self, X):
        X_np = X.values if hasattr(X, "values") else np.asarray(X, dtype=np.float64)
        raw = self.model_.predict(self.pca_.transform(X_np))
        raw = raw - raw.max(axis=1, keepdims=True)
        exp = np.exp(raw)
        proba = exp / exp.sum(axis=1, keepdims=True)
        return proba[:, 1] if self.problem_type_ == "binary" else proba


class _PCRBridge(SklearnAutoGluonBridge):
    _sklearn_cls = PCRModel
    ag_key = "PCR"
    ag_name = "PCR"

    def _get_default_searchspace(self):
        from autogluon.common import space

        return {"n_components": space.Int(lower=5, upper=50)}


class Prep_PCR(_NoAugBase, _PCRBridge):  # noqa: N801
    """No model-level preprocessing default: the recipe config is the sole controlled
    preprocessing factor in this study (matches ``Prep_RF``/``Prep_GBM``)."""
