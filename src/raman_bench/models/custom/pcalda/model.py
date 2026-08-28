"""PCA-LDA: principal-component reduction followed by linear discriminant analysis.

The classic chemometric classifier. Plain LDA is undefined for Raman spectra
(p >> n makes the within-class scatter singular), so the standard remedy —
PCA to ``n_components`` first, then LDA on the scores — is what this wraps.
Added 2026-08-28 for the RamanPreprocessing study as the discriminant-analysis
member of the chemometric family; it is a linear, class-covariance method and
so, like PCR, is predicted to be scale- and baseline-sensitive.

**Classification only** — LDA has no regression analogue. The key ``PCALDA`` is
in ``CLASSIFICATION_ONLY_MODELS`` (``preprocessing/wrapped_models.py``); ``fit``
also raises on a continuous target as a backstop.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from raman_bench.preprocessing.bridge_bases import SklearnAutoGluonBridge, _NoAugBase


class PCALDAModel(BaseEstimator):
    """PCA-LDA classifier — sklearn-compatible."""

    def __init__(self, n_components=30):
        self.n_components = n_components

    def fit(self, X, y):
        X_np = X.values if hasattr(X, "values") else np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y)
        if np.issubdtype(y_arr.dtype, np.floating):
            raise ValueError("PCALDAModel is classification-only; got a continuous target.")

        self.classes_ = np.unique(y_arr)
        self.problem_type_ = "binary" if len(self.classes_) == 2 else "multiclass"
        # LDA needs at least (n_classes) samples and n_components < n_samples;
        # PCA rank is bounded by min(n-1, p) and, usefully for LDA, by n_classes-1
        # after the fact -- but keep PCA at the requested width so the model still
        # sees non-discriminant structure.
        k = min(self.n_components, X_np.shape[1], X_np.shape[0] - 1)
        self.pca_ = PCA(n_components=max(k, 1))
        scores = self.pca_.fit_transform(X_np)
        self.lda_ = LinearDiscriminantAnalysis(solver="svd").fit(scores, y_arr)
        return self

    def predict(self, X):
        X_np = X.values if hasattr(X, "values") else np.asarray(X, dtype=np.float64)
        return self.lda_.predict(self.pca_.transform(X_np))

    def predict_proba(self, X):
        X_np = X.values if hasattr(X, "values") else np.asarray(X, dtype=np.float64)
        proba = self.lda_.predict_proba(self.pca_.transform(X_np))
        return proba[:, 1] if self.problem_type_ == "binary" else proba


class _PCALDABridge(SklearnAutoGluonBridge):
    _sklearn_cls = PCALDAModel
    ag_key = "PCALDA"
    ag_name = "PCA-LDA"

    def _get_default_searchspace(self):
        from autogluon.common import space

        return {"n_components": space.Int(lower=5, upper=60)}


class Prep_PCALDA(_NoAugBase, _PCALDABridge):  # noqa: N801
    """No model-level preprocessing default: the recipe config is the sole controlled
    preprocessing factor in this study (matches ``Prep_RF``/``Prep_GBM``)."""
