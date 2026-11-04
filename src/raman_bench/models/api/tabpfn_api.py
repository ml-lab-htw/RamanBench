"""scikit-learn estimators for the hosted TabPFN API (``tabpfn-client``).

The API runs the model on Prior Labs' servers, so the training and test data are
uploaded. The token is read from the ``TABPFN_TOKEN`` environment variable by
``tabpfn_client`` itself; nothing here stores it.

Groups: when ``fit`` receives ``groups``, they are passed to the API as ``group_col``
(the API's grouped mode). The benchmark's protocol drops the group column from X for
every other model (see ``raman_bench.splitting.RamanBenchTaskWrapper``), so results
that use groups are a different variant and must be reported as one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

GROUP_COL = "_group_id"


THINKING_MAX_COLUMNS = 2000


def _bin_columns(X: np.ndarray, n_bins: int) -> np.ndarray:
    """Average runs of neighbouring columns (spectral bins) down to *n_bins*.

    Stateless: it depends only on the column count, so training and test rows are
    transformed identically and the labels are never used.
    """
    edges = np.linspace(0, X.shape[1], n_bins + 1).round().astype(int)
    return np.stack([X[:, a:b].mean(axis=1) for a, b in zip(edges[:-1], edges[1:])], axis=1)


def _frame(X, groups=None, thinking=False) -> pd.DataFrame:
    X = np.asarray(X, dtype=np.float32)
    if thinking and X.shape[1] > THINKING_MAX_COLUMNS:
        X = _bin_columns(X, THINKING_MAX_COLUMNS).astype(np.float32)
    cols = [f"f{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=cols)
    if groups is not None:
        df[GROUP_COL] = np.asarray(groups).astype(str)
    return df


class _TabPFNAPIMixin:
    def _api_kwargs(self, groups):
        kwargs = {"random_state": self.random_state}
        if self.thinking:
            kwargs["thinking_mode"] = True
            kwargs["thinking_effort"] = self.thinking_effort
            if self.thinking_timeout_s is not None:
                kwargs["thinking_timeout_s"] = self.thinking_timeout_s
        if groups is not None:
            kwargs["group_col"] = GROUP_COL
        return kwargs


class TabPFNAPIClassifier(_TabPFNAPIMixin, ClassifierMixin, BaseEstimator):
    def __init__(self, random_state=0, thinking=False, thinking_effort="medium", thinking_timeout_s=None, use_groups=True):
        self.random_state = random_state
        self.thinking = thinking
        self.thinking_effort = thinking_effort
        self.thinking_timeout_s = thinking_timeout_s
        self.use_groups = use_groups

    def fit(self, X, y, groups=None):
        from tabpfn_client import TabPFNClassifier

        groups = groups if self.use_groups else None
        self.classes_ = np.unique(y)
        self._model = TabPFNClassifier(**self._api_kwargs(groups))
        self._model.fit(_frame(X, groups, self.thinking), np.asarray(y))
        return self

    def predict_proba(self, X, groups=None):
        groups = groups if self.use_groups else None
        return self._model.predict_proba(_frame(X, groups, self.thinking))

    def predict(self, X, groups=None):
        groups = groups if self.use_groups else None
        return self._model.predict(_frame(X, groups, self.thinking))


class TabPFNAPIRegressor(_TabPFNAPIMixin, RegressorMixin, BaseEstimator):
    def __init__(self, random_state=0, thinking=False, thinking_effort="medium", thinking_timeout_s=None, use_groups=True):
        self.random_state = random_state
        self.thinking = thinking
        self.thinking_effort = thinking_effort
        self.thinking_timeout_s = thinking_timeout_s
        self.use_groups = use_groups

    def fit(self, X, y, groups=None):
        from tabpfn_client import TabPFNRegressor

        groups = groups if self.use_groups else None
        self._model = TabPFNRegressor(**self._api_kwargs(groups))
        self._model.fit(_frame(X, groups, self.thinking), np.asarray(y, dtype=np.float32))
        return self

    def predict(self, X, groups=None):
        groups = groups if self.use_groups else None
        return np.asarray(self._model.predict(_frame(X, groups, self.thinking))).ravel()
