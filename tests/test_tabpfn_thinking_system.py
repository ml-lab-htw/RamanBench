"""TabPFN-3.5-Thinking as a system (raman_bench.systems._tabpfn_thinking) against a fake tabpfn-client.

Nothing is sent: the fake client records the calls. The point of the system is one thinking fit per outer fold
(no AutoGluon bagging), no groups sent to the API, and the API's rate limit waited out and not counted as a failure.
"""

import sys
import types

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("tabarena")
pytest.importorskip("httpx")
pytest.importorskip("pyarrow")

from raman_bench.experiment_utils import build_task, run_cached  # noqa: E402
from raman_bench.systems import _tabpfn_thinking as system_module  # noqa: E402
from raman_bench.systems import get_system  # noqa: E402

RATE_LIMIT_HOUR = (
    "Fail to call submit_fit_job: [HTTP 429] Rate limit exceeded: at most 30 thinking_fit requests per hour "
    "are allowed. Retry in 3538s.. Report trace ID: abc"
)
USAGE_LIMIT = "Fail to call submit_fit_job: [HTTP 429] Monthly usage limit exceeded. Retry in 100s."


class _Calls:
    def __init__(self):
        self.inits, self.fits, self.predicts, self.sleeps = [], [], [], []
        self.fit_errors = []  # raised one per fit call, front first


@pytest.fixture
def fake_client(monkeypatch):
    calls = _Calls()

    class _Estimator:
        def __init__(self, **kwargs):
            calls.inits.append(kwargs)
            self._last_meta = {"service": "fake"}
            self.model_id_ = "fake-model"

        def fit(self, X, y):
            if calls.fit_errors:
                raise calls.fit_errors.pop(0)
            calls.fits.append(list(X.columns))
            self._y = pd.Series(y)
            self.classes_ = sorted(self._y.unique())
            return self

        def predict(self, X):
            calls.predicts.append(len(X))
            return np.full(len(X), self._y.iloc[0] if self.classes_ else 0.0)

        def predict_proba(self, X):
            calls.predicts.append(len(X))
            return np.full((len(X), len(self.classes_)), 1.0 / len(self.classes_))

        def get_timings(self):
            return {"fit": {"fit_s": 2.0}, "predict": {"predict_s": 0.5}}

    class _Classifier(_Estimator):
        pass

    class _Regressor(_Estimator):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.classes_ = []

        def fit(self, X, y):
            super().fit(X, y)
            self.classes_ = []
            self._y = pd.Series(y.astype(float))
            return self

        def predict(self, X):
            calls.predicts.append(len(X))
            return np.full(len(X), float(self._y.mean()))

    class _Service:
        @classmethod
        def get_settings(cls):
            return None

    class _Retryable(Exception):
        pass

    root = types.ModuleType("tabpfn_client")
    root.TabPFNClassifier, root.TabPFNRegressor = _Classifier, _Regressor
    root.set_access_token = lambda token: None
    root.__version__ = "0.0-fake"
    client = types.ModuleType("tabpfn_client.client")
    client.ServiceClient = _Service
    errors = types.ModuleType("tabpfn_client.errors")
    errors.CappedRetryableServerError = errors.EmptyResponseError = errors.RetryableServerError = _Retryable
    monkeypatch.setitem(sys.modules, "tabpfn_client", root)
    monkeypatch.setitem(sys.modules, "tabpfn_client.client", client)
    monkeypatch.setitem(sys.modules, "tabpfn_client.errors", errors)
    monkeypatch.setenv("TABPFN_TOKEN", "test-token")
    monkeypatch.setattr(system_module, "MIN_FIT_INTERVAL_S", 0.0)
    monkeypatch.setattr(system_module.time, "sleep", lambda s: calls.sleeps.append(s))
    return calls


def _task(problem_type, n=60, grouped=False, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n, 8)), columns=[f"{400 + i}" for i in range(8)])
    if grouped:  # the label must stay the last column (build_task reads it from there)
        df["_group_id"] = np.repeat(np.arange(n // 3), 3)
    if problem_type == "classification":
        df["target"] = np.tile(["a", "b", "c"], n // 3)
    else:
        df["target"] = df["400"] * 2 + rng.normal(size=n)
    return build_task(
        dataset_name="toy", target_idx=0, df=df, raw_targets=df[["target"]].to_numpy(),
        problem_type=problem_type, n_repeats=1, n_splits=3,
    )


def _run(tmp_path, problem_type, grouped=False):
    from tabarena.benchmark.experiment.experiment_constructor import ExternalSystemExperiment

    system = get_system("tabpfn-v3.5-thinking")
    task_name, task = _task(problem_type, grouped=grouped)
    experiment = ExternalSystemExperiment(
        name=system.result_dir,
        system_cls=system.load_cls(),
        system_hyperparameters=dict(system.hyperparameters),
        method_kwargs={"fit_kwargs": {"time_limit": 600, "num_cpus": 1}},
        experiment_kwargs={"require_warmup": False},
    )
    return run_cached(experiment, task_name=task_name, task_wrapper=task, repeat=0, fold=1, results_dir=str(tmp_path))


@pytest.mark.parametrize("problem_type", ["classification", "regression"])
def test_one_thinking_fit_per_outer_fold(fake_client, tmp_path, problem_type):
    out = _run(tmp_path, problem_type)

    assert len(fake_client.fits) == 1 and len(fake_client.inits) == 1  # one fit: no bag children, no refit
    (init,) = fake_client.inits
    assert init["thinking_effort"] == "high" and init["thinking_timeout_s"] == 600.0
    assert init["thinking_metric"] and isinstance(init["random_state"], int)
    assert out["framework"] == "TabPFN-V3.5-Thinking-high"
    assert out["method_metadata"]["effort"] == "high" and out["method_metadata"]["api_fit_s"] == 2.0
    assert out["method_metadata"]["api_calls"][0]["call"] == "fit"
    sa = out["simulation_artifacts"]  # test predictions are stored like every other v1 result
    assert len(sa["y_test_idx"]) == sum(fake_client.predicts)


def test_groups_are_not_sent(fake_client, tmp_path):
    _run(tmp_path, "regression", grouped=True)

    (columns,) = fake_client.fits
    assert "_group_id" not in columns and columns == [str(400 + i) for i in range(8)]
    (init,) = fake_client.inits
    assert not {"group_col", "time_col", "group_time_col"} & set(init)


def test_rate_limit_refusals_are_waited_out(fake_client, tmp_path):
    fake_client.fit_errors = [RuntimeError(RATE_LIMIT_HOUR), RuntimeError(RATE_LIMIT_HOUR)]
    out = _run(tmp_path, "regression")

    assert len(fake_client.fits) == 1
    assert fake_client.sleeps.count(3539.0) == 2  # the server's wait plus a second, twice
    assert out["method_metadata"]["api_calls"][0]["rate_limit_refusals"] == 2


def test_usage_limit_is_not_retried(fake_client, tmp_path):
    fake_client.fit_errors = [RuntimeError(USAGE_LIMIT)]
    with pytest.raises(RuntimeError, match="usage limit"):
        _run(tmp_path, "regression")

    assert not fake_client.fits and 101.0 not in fake_client.sleeps


def test_rate_limit_wait_only_for_request_limits():
    assert system_module.rate_limit_wait(RuntimeError(RATE_LIMIT_HOUR)) == 3539.0
    assert system_module.rate_limit_wait(RuntimeError(USAGE_LIMIT)) is None
    assert system_module.rate_limit_wait(RuntimeError("HTTP 500")) is None
