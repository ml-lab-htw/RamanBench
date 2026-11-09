"""scikit-learn estimators for the hosted TabPFN API (``tabpfn-client``).

The API runs the model on Prior Labs' servers, so the training and test data are
uploaded. The token is read from the ``TABPFN_TOKEN`` environment variable by
``tabpfn_client`` itself; nothing here stores it.

Groups: when ``fit`` receives ``groups``, they are passed to the API as ``group_col``
(the API's grouped mode). The benchmark's protocol drops the group column from X for
every other model (see ``raman_bench.splitting.RamanBenchTaskWrapper``), so results
that use groups are a different variant and must be reported as one.

Rate limit: the API allows at most 10 thinking fits per minute per account. In thinking
mode every ``fit`` waits until ``min_fit_interval_s`` has passed since the previous thinking
fit in this process. The timer is per process: parallel processes must share the limit
themselves.
"""

from __future__ import annotations

import re
import time

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

GROUP_COL = "_group_id"


THINKING_MAX_COLUMNS = 2000

# 10 thinking fits per minute, the API's limit: 60 / 10 = 6 s between fits.
THINKING_MIN_FIT_INTERVAL_S = 6.0

_last_thinking_fit = float("-inf")

# The API refuses a fit with "HTTP 429 ... Retry in <N>s" when the rate limit is hit.
# Those refusals are retried after the server's wait (plus 1 s), at most this many times.
RATE_LIMIT_RETRIES = 3
_RETRY_IN_RE = re.compile(r"Retry in (\d+(?:\.\d+)?)s")


def _bin_columns(X: np.ndarray, n_bins: int) -> np.ndarray:
    """Average runs of neighbouring columns (spectral bins) down to *n_bins*.

    Stateless: it depends only on the column count, so training and test rows are
    transformed identically and the labels are never used.
    """
    edges = np.linspace(0, X.shape[1], n_bins + 1).round().astype(int)
    return np.stack([X[:, a:b].mean(axis=1) for a, b in zip(edges[:-1], edges[1:])], axis=1)


def _frame(X, groups=None, max_columns=None) -> pd.DataFrame:
    X = np.asarray(X, dtype=np.float32)
    if max_columns is not None and X.shape[1] > max_columns:
        X = _bin_columns(X, max_columns).astype(np.float32)
    cols = [f"f{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=cols)
    if groups is not None:
        df[GROUP_COL] = np.asarray(groups).astype(str)
    return df


class _TabPFNAPIMixin:
    def _max_columns(self):
        """Column cap for the API: thinking mode only; ``None`` means no binning."""
        return self.max_columns if self.thinking else None

    def _fit_with_retry(self, fit):
        """Run one API fit call under the rate limit, retrying the API's 429 refusals."""
        for attempt in range(RATE_LIMIT_RETRIES + 1):
            self._wait_for_rate_limit()
            try:
                return fit()
            except RuntimeError as err:
                match = _RETRY_IN_RE.search(str(err))
                if match is None or "429" not in str(err) or attempt == RATE_LIMIT_RETRIES:
                    raise
                time.sleep(float(match.group(1)) + 1.0)

    def _wait_for_rate_limit(self):
        """Sleep until the thinking fit rate limit allows the next fit (thinking mode only)."""
        global _last_thinking_fit
        if not self.thinking:
            return
        wait = _last_thinking_fit + self.min_fit_interval_s - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last_thinking_fit = time.monotonic()

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
    def __init__(
        self,
        random_state=0,
        thinking=False,
        thinking_effort="medium",
        thinking_timeout_s=None,
        use_groups=True,
        max_columns=THINKING_MAX_COLUMNS,
        min_fit_interval_s=THINKING_MIN_FIT_INTERVAL_S,
    ):
        self.random_state = random_state
        self.thinking = thinking
        self.thinking_effort = thinking_effort
        self.thinking_timeout_s = thinking_timeout_s
        self.use_groups = use_groups
        self.max_columns = max_columns
        self.min_fit_interval_s = min_fit_interval_s

    def fit(self, X, y, groups=None):
        from tabpfn_client import TabPFNClassifier

        groups = groups if self.use_groups else None
        self.classes_ = np.unique(y)
        self._model = TabPFNClassifier(**self._api_kwargs(groups))
        frame = _frame(X, groups, self._max_columns())
        self._fit_with_retry(lambda: self._model.fit(frame, np.asarray(y)))
        return self

    def predict_proba(self, X, groups=None):
        groups = groups if self.use_groups else None
        return self._model.predict_proba(_frame(X, groups, self._max_columns()))

    def predict(self, X, groups=None):
        groups = groups if self.use_groups else None
        return self._model.predict(_frame(X, groups, self._max_columns()))


class TabPFNAPIRegressor(_TabPFNAPIMixin, RegressorMixin, BaseEstimator):
    def __init__(
        self,
        random_state=0,
        thinking=False,
        thinking_effort="medium",
        thinking_timeout_s=None,
        use_groups=True,
        max_columns=THINKING_MAX_COLUMNS,
        min_fit_interval_s=THINKING_MIN_FIT_INTERVAL_S,
    ):
        self.random_state = random_state
        self.thinking = thinking
        self.thinking_effort = thinking_effort
        self.thinking_timeout_s = thinking_timeout_s
        self.use_groups = use_groups
        self.max_columns = max_columns
        self.min_fit_interval_s = min_fit_interval_s

    def fit(self, X, y, groups=None):
        from tabpfn_client import TabPFNRegressor

        groups = groups if self.use_groups else None
        self._model = TabPFNRegressor(**self._api_kwargs(groups))
        frame = _frame(X, groups, self._max_columns())
        target = np.asarray(y, dtype=np.float32)
        self._fit_with_retry(lambda: self._model.fit(frame, target))
        return self

    def predict(self, X, groups=None):
        groups = groups if self.use_groups else None
        return np.asarray(self._model.predict(_frame(X, groups, self._max_columns()))).ravel()
