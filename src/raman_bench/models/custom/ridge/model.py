"""Ridge (L2-regularized linear) regression/classification.

Added as a model-agent workflow dry-run artifact -- validates the
add-a-new-model steps end to end, not a benchmark-motivated addition. Also
the reference implementation of the ``models/custom/<key>/{model.py,hpo.py,
info.py}`` onboarding convention (see ``info.py`` in this package and
``RamanBench/.claude/agents/model-agent.md``).
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import Ridge

from raman_bench.preprocessing.bridge_bases import SklearnAutoGluonBridge, _NoAugBase


class RidgeModel(BaseEstimator):
    """Ridge (L2-regularized linear) regression / classification — sklearn-compatible.

    For regression wraps ``sklearn.linear_model.Ridge`` directly. For
    classification implements a Ridge-DA analogous to PLSModel's PLS-DA: the
    target is one-hot encoded, Ridge is fit on the encoded matrix, and
    inference uses argmax of predicted scores.
    """

    def __init__(self, alpha=1.0, max_iter=None, tol=1e-4):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y):
        X_np = X.values if hasattr(X, "values") else np.asarray(X)
        y_arr = np.asarray(y)

        if np.issubdtype(y_arr.dtype, np.floating):
            self.problem_type_ = "regression"
        else:
            unique = np.unique(y_arr)
            self.problem_type_ = "binary" if len(unique) == 2 else "multiclass"

        if self.problem_type_ in ("multiclass", "binary"):
            self.classes_ = np.unique(y_arr)
            self._class_to_idx = {c: i for i, c in enumerate(self.classes_)}
            y_encoded = np.zeros((len(y_arr), len(self.classes_)))
            for i, val in enumerate(y_arr):
                y_encoded[i, self._class_to_idx[val]] = 1.0
            self.model_ = Ridge(alpha=self.alpha, max_iter=self.max_iter, tol=self.tol)
            self.model_.fit(X_np, y_encoded)
        else:
            self.model_ = Ridge(alpha=self.alpha, max_iter=self.max_iter, tol=self.tol).fit(
                X_np, y_arr
            )

        return self

    def predict(self, X):
        X_np = X.values if hasattr(X, "values") else np.asarray(X)
        if self.problem_type_ in ("multiclass", "binary"):
            indices = np.argmax(self.model_.predict(X_np), axis=1)
            return self.classes_[indices]
        return self.model_.predict(X_np).ravel()

    def predict_proba(self, X):
        X_np = X.values if hasattr(X, "values") else np.asarray(X)
        y_pred = self.model_.predict(X_np)
        exp_scores = np.exp(y_pred - np.max(y_pred, axis=1, keepdims=True))
        proba = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        if self.problem_type_ == "binary":
            return proba[:, 1]
        return proba


class _RidgeBridge(SklearnAutoGluonBridge):
    _sklearn_cls = RidgeModel
    ag_key = "RIDGE"
    ag_name = "Ridge"

    def _get_default_searchspace(self):
        from autogluon.common import space

        return {
            "alpha": space.Real(lower=1e-3, upper=1e3, log=True),
        }


class Prep_RIDGE(_NoAugBase, _RidgeBridge):  # noqa: N801
    pass
