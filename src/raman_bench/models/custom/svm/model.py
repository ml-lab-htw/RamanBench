"""SVM (RBF-kernel support vector machine) regression / classification.

Kernel / distance-based nonlinear model. Added 2026-08-28 for the
RamanPreprocessing preprocessing-effects study: it is the kernel-method
counterpart to the tree ensembles (RF/GBM) — predicted to be *highly*
sensitive to per-channel scaling and per-spectrum normalisation, the opposite
of the trees' monotone-transform invariance.

Regression wraps ``sklearn.svm.SVR``; classification wraps ``sklearn.svm.SVC``
(hard-label ``predict``; ``predict_proba`` is a softmax of ``decision_function``
rather than Platt scaling, so no internal CV — a documented simplification,
adequate because the classification metric is F1 on argmax predictions).

Compute guard: RBF SVM training is between quadratic and cubic in the number
of samples. RamanBench's own subsample guard already caps training sets at
10,000 samples, and this model additionally subsamples (seeded, stratified for
classification) to ``rbf_max_train`` (default 6000) so a single fit stays
bounded on the shared CPU nodes. Documented, not silent.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.svm import SVC, SVR

from raman_bench.preprocessing.bridge_bases import SklearnAutoGluonBridge, _NoAugBase

_RBF_MAX_TRAIN = 6000


class SVMModel(BaseEstimator):
    """RBF-kernel SVM regression / classification — sklearn-compatible."""

    def __init__(self, C=1.0, gamma="scale", epsilon=0.1, rbf_max_train=_RBF_MAX_TRAIN):
        self.C = C
        self.gamma = gamma
        self.epsilon = epsilon
        self.rbf_max_train = rbf_max_train

    def _subsample(self, X, y, stratify):
        if len(X) <= self.rbf_max_train:
            return X, y
        rng = np.random.default_rng(0)
        if stratify:
            idx = []
            per = self.rbf_max_train // len(np.unique(y))
            for cls in np.unique(y):
                pool = np.flatnonzero(y == cls)
                idx.append(rng.choice(pool, size=min(per, len(pool)), replace=False))
            sel = np.concatenate(idx)
        else:
            sel = rng.choice(len(X), size=self.rbf_max_train, replace=False)
        return X[sel], y[sel]

    def fit(self, X, y):
        X_np = X.values if hasattr(X, "values") else np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y)

        if np.issubdtype(y_arr.dtype, np.floating):
            self.problem_type_ = "regression"
        else:
            self.classes_ = np.unique(y_arr)
            self.problem_type_ = "binary" if len(self.classes_) == 2 else "multiclass"

        X_fit, y_fit = self._subsample(
            X_np, y_arr, stratify=self.problem_type_ != "regression"
        )

        if self.problem_type_ == "regression":
            self.model_ = SVR(kernel="rbf", C=self.C, gamma=self.gamma,
                              epsilon=self.epsilon)
        else:
            self.model_ = SVC(kernel="rbf", C=self.C, gamma=self.gamma,
                              decision_function_shape="ovr")
        self.model_.fit(X_fit, y_fit)
        return self

    def predict(self, X):
        X_np = X.values if hasattr(X, "values") else np.asarray(X, dtype=np.float64)
        return self.model_.predict(X_np).ravel()

    def predict_proba(self, X):
        X_np = X.values if hasattr(X, "values") else np.asarray(X, dtype=np.float64)
        scores = self.model_.decision_function(X_np)
        if self.problem_type_ == "binary":
            return 1.0 / (1.0 + np.exp(-scores))
        scores = scores - scores.max(axis=1, keepdims=True)
        exp = np.exp(scores)
        return exp / exp.sum(axis=1, keepdims=True)


class _SVMBridge(SklearnAutoGluonBridge):
    _sklearn_cls = SVMModel
    ag_key = "SVM"
    ag_name = "SVM"

    def _get_default_searchspace(self):
        from autogluon.common import space

        return {
            "C": space.Real(lower=1e-2, upper=1e2, log=True),
            "gamma": space.Categorical("scale", "auto"),
        }


class Prep_SVM(_NoAugBase, _SVMBridge):  # noqa: N801
    """No model-level preprocessing default: in this study the recipe config is the sole
    controlled preprocessing factor (matches ``Prep_RF``/``Prep_GBM``)."""
