import numpy as np
from sklearn.base import BaseEstimator
from sklearn.cross_decomposition import PLSRegression


class PLSModel(BaseEstimator):
    """PLS regression / PLS-DA — sklearn-compatible estimator.

    For regression wraps ``sklearn.cross_decomposition.PLSRegression``
    directly.  For classification implements PLS-DA by one-hot encoding the
    target, fitting PLSRegression on the encoded matrix, and using argmax of
    predicted scores for inference.

    Reference:
        Wold, S., et al. (1984). The collinearity problem in linear regression:
        the PLS approach. SIAM Journal on Scientific and Statistical Computing.
        https://doi.org/10.1137/0905052
    """

    def __init__(self, n_components=10, max_iter=500, scale=True, tol=1e-6):
        self.n_components = n_components
        self.max_iter = max_iter
        self.scale = scale
        self.tol = tol

    def fit(self, X, y):
        X_np = X.values if hasattr(X, "values") else np.asarray(X)
        y_arr = np.asarray(y)

        n_components = min(self.n_components, X_np.shape[1], X_np.shape[0] - 1)

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
            self.model_ = PLSRegression(
                n_components=n_components, max_iter=self.max_iter, scale=self.scale,
                tol=self.tol,
            )
            self.model_.fit(X_np, y_encoded)
        else:
            self.model_ = PLSRegression(
                n_components=n_components, max_iter=self.max_iter, scale=self.scale,
                tol=self.tol,
            ).fit(X_np, y_arr)

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
