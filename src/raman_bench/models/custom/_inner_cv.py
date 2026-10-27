"""Inner cross-validation for the chemometric models' complexity parameters.

Used when a model is given ``n_components="cv"`` (PLS, PCR, PCA-LDA) or ``alpha="cv"`` (ridge):
the parameter is chosen on the training data only, by ``n_splits``-fold CV, and the model is then
refitted on all training data with the chosen value. This is the usual chemometric practice
(number of latent variables or components chosen by cross-validation) and keeps the outer test
fold untouched.

Selection criterion: RMSE for regression, log loss of the model's own ``predict_proba`` for
classification, i.e. the metric the benchmark reports. Each search fits the largest model once per
inner fold and reads every smaller model off that fit (PLS, PCR: nested components; ridge: one SVD
for all penalties), so a search costs about ``n_splits`` ordinary fits.
"""

from __future__ import annotations

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold, StratifiedKFold

PLS_MAX_COMPONENTS = 30
PCR_MAX_COMPONENTS = 50
PCALDA_MAX_COMPONENTS = 60
RIDGE_ALPHAS = np.logspace(-6, 6, 25)
INNER_SPLITS = 5
INNER_SEED = 0


def _softmax(scores: np.ndarray) -> np.ndarray:
    scores = scores - scores.max(axis=1, keepdims=True)
    e = np.exp(scores)
    return e / e.sum(axis=1, keepdims=True)


def _log_loss(proba: np.ndarray, y_idx: np.ndarray) -> float:
    p = np.clip(proba[np.arange(len(y_idx)), y_idx], 1e-15, 1.0)
    return float(-np.mean(np.log(p)))


def _rmse(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - y) ** 2)))


def inner_splits(y: np.ndarray, classification: bool) -> list[tuple[np.ndarray, np.ndarray]]:
    """Stratified for classification (fewer folds if a class is small), shuffled KFold otherwise."""
    n = len(y)
    if classification:
        _, counts = np.unique(y, return_counts=True)
        k = int(min(INNER_SPLITS, counts.min()))
        if k >= 2:
            return list(StratifiedKFold(k, shuffle=True, random_state=INNER_SEED).split(np.zeros(n), y))
    k = int(min(INNER_SPLITS, n))
    return list(KFold(k, shuffle=True, random_state=INNER_SEED).split(np.zeros(n)))


def _targets(y: np.ndarray, classes: np.ndarray | None):
    """Regression: y as float column. Classification: one-hot matrix and class indices."""
    if classes is None:
        return y.astype(float).reshape(-1, 1), None
    idx = np.searchsorted(classes, y)
    onehot = np.zeros((len(y), len(classes)))
    onehot[np.arange(len(y)), idx] = 1.0
    return onehot, idx


def _score(pred: np.ndarray, y_val: np.ndarray, idx_val: np.ndarray | None) -> float:
    if idx_val is None:
        return _rmse(pred.ravel(), y_val.ravel())
    return _log_loss(_softmax(pred), idx_val)


def pls_truncated_predictions(model: PLSRegression, X: np.ndarray, max_k: int) -> list[np.ndarray]:
    """Predictions of the first 1..max_k components of a fitted PLSRegression.

    NIPALS extracts components sequentially, and ``x_rotations_ = W (P'W)^-1`` with ``P'W`` upper
    triangular, so its first k columns are those of a k-component fit; the k-component prediction
    is therefore ``(X - mean) / std @ R_k Q_k' * y_std + y_mean``.
    """
    Xs = (X - model._x_mean) / model._x_std
    T = Xs @ model.x_rotations_
    out = []
    for k in range(1, max_k + 1):
        out.append(T[:, :k] @ model.y_loadings_[:, :k].T * model._y_std + model._y_mean)
    return out


def select_pls(X: np.ndarray, y: np.ndarray, classes: np.ndarray | None, max_iter: int, tol: float):
    """Return (n_components, scale) minimising inner-CV error over 1..30 components x {scaled, not}."""
    Y, idx = _targets(y, classes)
    folds = inner_splits(y, classes is not None)
    k_max = int(max(1, min(PLS_MAX_COMPONENTS, X.shape[1], min(len(tr) for tr, _ in folds) - 1)))
    errors = {}
    for scale in (False, True):
        per_k = np.zeros(k_max)
        for tr, va in folds:
            m = PLSRegression(n_components=k_max, scale=scale, max_iter=max_iter, tol=tol).fit(X[tr], Y[tr])
            preds = pls_truncated_predictions(m, X[va], k_max)
            per_k += [_score(p, Y[va], None if idx is None else idx[va]) for p in preds]
        for k in range(k_max):
            errors[(k + 1, scale)] = per_k[k] / len(folds)
    return min(errors, key=errors.get)


def select_pcr(X: np.ndarray, y: np.ndarray, classes: np.ndarray | None) -> int:
    """Number of principal components (1..50) minimising inner-CV error."""
    Y, idx = _targets(y, classes)
    folds = inner_splits(y, classes is not None)
    k_max = int(max(1, min(PCR_MAX_COMPONENTS, X.shape[1], min(len(tr) for tr, _ in folds) - 1)))
    per_k = np.zeros(k_max)
    for tr, va in folds:
        pca = PCA(n_components=k_max).fit(X[tr])
        S_tr, S_va = pca.transform(X[tr]), pca.transform(X[va])
        for k in range(1, k_max + 1):
            A = np.column_stack([np.ones(len(tr)), S_tr[:, :k]])
            coef, *_ = np.linalg.lstsq(A, Y[tr], rcond=None)
            pred = np.column_stack([np.ones(len(va)), S_va[:, :k]]) @ coef
            per_k[k - 1] += _score(pred, Y[va], None if idx is None else idx[va])
    return int(np.argmin(per_k) + 1)


def select_pcalda(X: np.ndarray, y: np.ndarray) -> int:
    """Number of principal components (1..60) before LDA minimising inner-CV log loss."""
    classes = np.unique(y)
    folds = inner_splits(y, True)
    k_max = int(max(1, min(PCALDA_MAX_COMPONENTS, X.shape[1], min(len(tr) for tr, _ in folds) - 1)))
    per_k = np.zeros(k_max)
    for tr, va in folds:
        pca = PCA(n_components=k_max).fit(X[tr])
        S_tr, S_va = pca.transform(X[tr]), pca.transform(X[va])
        idx_va = np.searchsorted(classes, y[va])
        for k in range(1, k_max + 1):
            lda = LinearDiscriminantAnalysis(solver="svd").fit(S_tr[:, :k], y[tr])
            proba = np.zeros((len(va), len(classes)))
            proba[:, np.searchsorted(classes, lda.classes_)] = lda.predict_proba(S_va[:, :k])
            per_k[k - 1] += _log_loss(proba, idx_va)
    return int(np.argmin(per_k) + 1)


def select_ridge(X: np.ndarray, y: np.ndarray, classes: np.ndarray | None) -> float:
    """Ridge penalty (1e-6..1e6, 25 log-spaced values) minimising inner-CV error, via one SVD per fold."""
    Y, idx = _targets(y, classes)
    folds = inner_splits(y, classes is not None)
    per_a = np.zeros(len(RIDGE_ALPHAS))
    for tr, va in folds:
        x_mean, y_mean = X[tr].mean(axis=0), Y[tr].mean(axis=0)
        U, s, Vt = np.linalg.svd(X[tr] - x_mean, full_matrices=False)
        UtY = U.T @ (Y[tr] - y_mean)
        Xv = (X[va] - x_mean) @ Vt.T
        for i, a in enumerate(RIDGE_ALPHAS):
            pred = Xv @ ((s / (s ** 2 + a))[:, None] * UtY) + y_mean
            per_a[i] += _score(pred, Y[va], None if idx is None else idx[va])
    return float(RIDGE_ALPHAS[int(np.argmin(per_a))])
