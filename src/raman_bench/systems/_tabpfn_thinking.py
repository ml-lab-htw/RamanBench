"""TabPFN-3.5-Thinking (Prior Labs' hosted API) as a system on the v1 protocol.

One thinking fit per outer fold on the whole training split, the system setting Prior Labs asked for. Not
wrapped in AutoGluon's bagging: a bagged run costs about four API fits per fold (three bag children and the
refit), which under the account's limit of 30 thinking fits an hour is four times slower and spends four times
the token budget.

Based on the system in autogluon/tabarena#732 (``tabarena.systems.tabpfn_3_5_thinking.system`` at e1700d8,
Apache-2.0), with these changes for RamanBench:

* the task's group column is not sent to the API (``group_col``), and ``_group_id`` is dropped from ``X``: the
  v1 protocol gives no model the groups, so a grouped fit would be a different variant;
* the API's rate limit is handled here: the thinking fits of all processes on a host are paced through a lock
  file, at least ``TABPFN_FIT_INTERVAL_S`` seconds apart (default ``MIN_FIT_INTERVAL_S``, 6 s for the limit of 10
  fits a minute; use 125 s for the limit of 30 fits an hour). A refused fit ("HTTP 429 ... Retry in N s") is still
  repeated after the server's wait, but with the pacing set to the hourly limit no fit is refused. A daily or
  monthly token limit is not retried, it fails the fold, so that a rerun picks it up after the reset;
* spectra wider than ``MAX_COLUMNS`` (23000) are binned down to that by averaging neighbouring columns, the same for
  training and test rows (``raman_bench.preprocessing.raman_preprocessing.bin_columns``), as the API refuses more;
* the protocol's 600 s budget is not sent as ``thinking_timeout_s`` (the API's maximum for the effort is, 7200 s
  at high), so a long thinking fit is not cut off;
* ``effort`` is ``"high"`` by default.

The key is read from ``TABPFN_TOKEN`` at fit time.
"""

from __future__ import annotations

import fcntl
import os
import re
import tempfile
import time
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import pandas as pd
from tabarena.benchmark.exec_models import ExternalSystemModel

from raman_bench.preprocessing.raman_preprocessing import bin_columns
from raman_bench.splitting import GROUP_COL

if TYPE_CHECKING:
    from collections.abc import Callable

    from autogluon.core.metrics import Scorer
    from tabarena.benchmark.task.metadata import ValidationMetadata

MAX_COLUMNS = 23000  # spectra wider than this are binned (the API takes up to 23000 columns at v3.5)
_KEY_ENV = "TABPFN_TOKEN"
_MAX_ATTEMPTS = 3  # transport errors and 5xx the client gave up on
MIN_FIT_INTERVAL_S = 6.0  # the API allows 10 thinking fits a minute per account
RATE_LIMIT_ATTEMPTS = 8  # refused fits repeated after the server's wait (a refusal can name a wait of up to an hour)
_RETRY_IN = re.compile(r"Retry in (\d+(?:\.\d+)?)\s*s")
INTERVAL_ENV = "TABPFN_FIT_INTERVAL_S"
PACING_FILE_ENV = "TABPFN_PACING_FILE"


def wait_for_fit_slot() -> None:
    """Block until this host may submit the next thinking fit, then take the slot.

    The time of the last submission lives in a lock file, so the worker processes of a run share one pace.
    The lock is held while waiting, which lines the waiters up one fit per interval.
    """
    interval = float(os.environ.get(INTERVAL_ENV, MIN_FIT_INTERVAL_S))
    path = os.environ.get(PACING_FILE_ENV, os.path.join(tempfile.gettempdir(), "tabpfn_thinking_pacing"))
    with open(path, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        text = f.read().strip()
        wait = (float(text) if text else 0.0) + interval - time.time()
        if wait > 0:
            time.sleep(wait)
        f.seek(0)
        f.truncate()
        f.write(str(time.time()))


def _bin(X: pd.DataFrame) -> pd.DataFrame:
    """``X`` with more than ``MAX_COLUMNS`` columns averaged down to that many (unchanged otherwise)."""
    if X.shape[1] <= MAX_COLUMNS:
        return X
    binned = bin_columns(X.to_numpy(dtype=np.float32), MAX_COLUMNS)
    return pd.DataFrame(binned, index=X.index, columns=[f"f{i}" for i in range(MAX_COLUMNS)])


def _model_limit():
    """The API's per-call caps (the client's settings call; ``None`` when the server sends none)."""
    from tabpfn_client.client import ServiceClient

    settings = ServiceClient.get_settings()
    return None if settings is None else settings.max_model_limit


def _timeout_cap(effort: str) -> float | None:
    """The longest ``thinking_timeout_s`` the API takes at ``effort`` (``None`` when the server does not say)."""
    from tabpfn_client.client import ServiceClient

    settings = ServiceClient.get_settings()
    if settings is None:
        return None
    limits = (settings.thinking_model_limits or {}).get(settings.thinking_default_model_version)
    cap = None if limits is None or limits.max_timeout_secs is None else limits.max_timeout_secs.get(effort)
    return settings.thinking_max_timeout_secs if cap is None else cap


def rate_limit_wait(error: Exception) -> float | None:
    """Seconds to wait before repeating a call the API refused for its rate limit, else ``None``.

    Only a per-minute or per-hour request limit counts (``HTTP 429 ... Rate limit exceeded ... Retry in N s``).
    A token (usage) limit is a 429 too, but waiting does not help within a run, so it is not retried.
    """
    text = str(error)
    match = _RETRY_IN.search(text)
    if "429" not in text or "Rate limit" not in text or match is None:
        return None
    return float(match.group(1)) + 1.0


class TabPFN35ThinkingSystemModel(ExternalSystemModel):
    """TabPFN-3.5-Thinking through ``tabpfn-client``: ``fit`` uploads the training table, ``predict`` the test features.

    Init hyperparameter ``effort``: the thinking effort (``"micro"``, ``"mini"``, ``"low"``, ``"medium"``, ``"high"``,
    ``"xhigh"``, ``"max"``). The task's metric goes to the API as ``thinking_metric``, the split's seed as
    ``random_state``. ``thinking_timeout_s`` is the longest the API takes at this effort, not the protocol's 600 s.
    A prediction larger than one API call allows is sent as several calls over consecutive row blocks.
    ``get_metadata`` holds the client version, the server's timings and service version and every call's attempts,
    which the runner stores with the result.
    """

    warmup_modules: ClassVar[tuple[str, ...]] = ("tabpfn_client",)

    def __init__(self, *, effort: str = "high", **kwargs):
        super().__init__(**kwargs)
        self.effort = effort
        self._estimator = None
        self._metric: str | None = None
        self._timeout_s: float | None = None
        self._n_train = 0
        self._api_calls: list[dict] = []

    def _fit_system(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        target_name: str,
        problem_type: str,
        eval_metric: Scorer,
        validation_metadata: ValidationMetadata,
        num_cpus: int | None,
        num_gpus: int | None,
        memory_limit: float | None,
        time_limit: float | None,
        random_state: int | None,
    ):
        key = os.environ.get(_KEY_ENV)
        if not key:
            raise RuntimeError(f"TabPFN-3.5-Thinking needs its API key in the {_KEY_ENV} environment variable.")
        from tabpfn_client import TabPFNClassifier, TabPFNRegressor, set_access_token

        set_access_token(key)
        X = _bin(X.drop(columns=[GROUP_COL], errors="ignore"))
        self._metric = eval_metric.name
        estimator_cls = TabPFNRegressor if problem_type == "regression" else TabPFNClassifier
        if problem_type != "regression" and y.dtype == object:
            y = y.infer_objects()
        # The protocol's 600 s budget is not sent: a thinking fit at effort high takes 20+ minutes, and the API
        # stops a fit at ``thinking_timeout_s``. The API's own maximum for the effort is used (``None`` lets the
        # server apply its default when it names none).
        self._timeout_s = _timeout_cap(self.effort)
        self._estimator = estimator_cls(
            thinking_effort=self.effort,
            thinking_timeout_s=self._timeout_s,
            thinking_metric=self._metric,
            random_state=0 if random_state is None else int(random_state),
        )
        self._n_train = len(X)
        self._call("fit", len(X), lambda: self._estimator.fit(X, y), paced=True)
        return self

    def _row_blocks(self, X: pd.DataFrame) -> list[pd.DataFrame]:
        """``X`` in consecutive row blocks that each fit one prediction call (one block when it fits already)."""
        limit = _model_limit()
        if limit is None:
            return [X]
        rows = min(
            limit.test_set_max_rows,
            limit.predict_row_pairs_budget // max(self._n_train, 1),
            limit.test_set_max_cells // max(X.shape[1], 1),
        )
        rows = max(rows, 1)
        return [X.iloc[i : i + rows] for i in range(0, len(X), rows)]

    def _predict_blocks(self, what: str, X: pd.DataFrame, predict: Callable[[pd.DataFrame], object]) -> np.ndarray:
        X = _bin(X.drop(columns=[GROUP_COL], errors="ignore"))
        blocks = self._row_blocks(X)
        out = [np.asarray(self._call(what, len(block), lambda b=block: predict(b))) for block in blocks]
        return out[0] if len(out) == 1 else np.concatenate(out, axis=0)

    def _call(self, what: str, rows: int, call: Callable[[], object], *, paced: bool = False) -> object:
        """Run one client call. A transient failure is repeated, and so is a rate-limit refusal (after its wait)."""
        import httpx
        from tabpfn_client.errors import CappedRetryableServerError, EmptyResponseError, RetryableServerError

        transient = (RetryableServerError, CappedRetryableServerError, EmptyResponseError, httpx.TransportError)
        failure, start, attempt, refused = None, time.monotonic(), 1, 0
        while True:
            if paced:
                wait_for_fit_slot()
            try:
                out = call()
            except transient as exc:
                failure = f"{type(exc).__name__}: {str(exc)[:300]}"
                if attempt < _MAX_ATTEMPTS:
                    time.sleep(30 * attempt)
                    attempt += 1
                    continue
                self._record(what, rows, attempt, refused, start, failure)
                raise RuntimeError(
                    f"TabPFN-3.5-Thinking API {what} failed after {attempt} attempts: {failure}"
                ) from exc
            except RuntimeError as exc:
                wait = rate_limit_wait(exc)
                if wait is None or refused >= RATE_LIMIT_ATTEMPTS:
                    self._record(what, rows, attempt, refused, start, f"{type(exc).__name__}: {str(exc)[:300]}")
                    raise
                refused += 1
                time.sleep(wait)
                continue
            self._record(what, rows, attempt, refused, start, failure)
            return out

    def _record(self, what: str, rows: int, attempts: int, refused: int, start: float, failure: str | None) -> None:
        self._api_calls.append(
            {
                "call": what,
                "rows": rows,
                "attempts": attempts,
                "rate_limit_refusals": refused,
                "wall_s": time.monotonic() - start,
                "failure": failure,
            },
        )

    def _predict(self, X: pd.DataFrame) -> pd.Series:
        pred = self._predict_blocks("predict", X, self._estimator.predict)
        return pd.Series(pred, index=X.index)

    def _predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        proba = self._predict_blocks("predict_proba", X, self._estimator.predict_proba)
        return pd.DataFrame(np.asarray(proba, dtype=float), index=X.index, columns=list(self._estimator.classes_))

    def get_metadata(self) -> dict:
        """What the API reported about the fit and the last prediction, and each call, for ``results.pkl``."""
        from tabpfn_client import __version__ as client_version

        est = self._estimator
        timings = est.get_timings() if est is not None else {}
        return {
            "client": f"tabpfn-client {client_version}",
            "effort": self.effort,
            "thinking_metric": self._metric,
            "thinking_timeout_s": self._timeout_s,
            "api_model_id": None if est is None else str(getattr(est, "model_id_", None)),
            "api_timings": timings,
            "api_fit_s": (timings.get("fit") or {}).get("fit_s"),
            "api_predict_s": (timings.get("predict") or {}).get("predict_s"),
            "api_server_info": None if est is None else est._last_meta,
            "api_calls": self._api_calls,
        }

    def cleanup(self):
        self._estimator = None
