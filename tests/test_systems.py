"""raman_bench.systems: Chakra-Tab on the v1 protocol against a fake API (nothing is sent)."""

import base64
import io

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("tabarena")
requests = pytest.importorskip("requests")
pytest.importorskip("pyarrow")

from raman_bench.aggregation import aggregate_results  # noqa: E402
from raman_bench.experiment_utils import build_task, run_cached  # noqa: E402
from raman_bench.fold_metrics import fold_metrics, to_error  # noqa: E402
from raman_bench.systems import SYSTEMS, get_system  # noqa: E402


class _Response:
    status_code = 200
    text = "ok"

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


@pytest.fixture
def fake_api(monkeypatch):
    """A stand-in for the API: nearest-centroid predictions, recording each request."""
    monkeypatch.setenv("CHAKRA_TAB_KEY", "test-key")
    calls = []

    def post(url, json=None, headers=None, timeout=None):
        train = pd.read_parquet(io.BytesIO(base64.b64decode(json["train"]["bytes"])))
        test = pd.read_parquet(io.BytesIO(base64.b64decode(json["test"]["bytes"])))
        y = train.pop(json["target"])
        calls.append({**{k: json[k] for k in ("preset", "problem_type", "eval_metric", "time_limit")},
                      "auth": headers["Authorization"], "n_train": len(train), "n_test": len(test)})
        fit = {"version": "fake", "fit_s": 1.5, "predict_s": 0.1, "total_s": 1.6}
        if json["problem_type"] == "regression":
            return _Response({"predictions": [float(y.mean())] * len(test), "fit": fit})
        classes = sorted(y.unique())
        centroids = np.stack([train[y == c].mean().to_numpy() for c in classes])
        d = ((test.to_numpy()[:, None, :] - centroids[None]) ** 2).sum(-1)
        proba = np.exp(-(d - d.min(1, keepdims=True)) / d.std())
        proba /= proba.sum(1, keepdims=True)
        return _Response({"classes": classes, "probabilities": proba.tolist(), "fit": fit})

    monkeypatch.setattr(requests, "post", post)
    return calls


def _task(problem_type, n=60, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n, 8)), columns=[f"{400 + i}" for i in range(8)])
    if problem_type == "classification":
        df["target"] = np.tile(["a", "b", "c"], n // 3)
        df.loc[df["target"] == "b", "401"] += 2.0
    else:
        df["target"] = df["400"] * 2 + rng.normal(size=n)
    return build_task(
        dataset_name="toy", target_idx=0, df=df, raw_targets=df[["target"]].to_numpy(),
        problem_type=problem_type, n_repeats=1, n_splits=3,
    )


@pytest.mark.parametrize("problem_type", ["classification", "regression"])
def test_chakra_tab_result_matches_the_v1_result_format(fake_api, tmp_path, problem_type):
    from tabarena.benchmark.experiment.experiment_constructor import ExternalSystemExperiment

    system = get_system("chakra-tab")
    task_name, task = _task(problem_type)
    experiment = ExternalSystemExperiment(
        name=system.result_dir,
        system_cls=system.load_cls(),
        system_hyperparameters=dict(system.hyperparameters),
        method_kwargs={"fit_kwargs": {"time_limit": 600, "num_cpus": 1}},
        experiment_kwargs={"require_warmup": False},
    )
    out = run_cached(experiment, task_name=task_name, task_wrapper=task, repeat=0, fold=1, results_dir=str(tmp_path))

    (call,) = fake_api
    assert call["preset"] == "medium" and call["time_limit"] == 600 and call["auth"] == "Bearer test-key"
    assert call["n_train"] + call["n_test"] == 60
    assert out["framework"] == "Chakra-Tab-medium"
    assert out["method_metadata"]["api_fit_s"] == 1.5

    # Test predictions and rows are stored like every other v1 result ...
    sa = out["simulation_artifacts"]
    assert len(sa["y_test_idx"]) == call["n_test"] and sa["problem_type"] == out["problem_type"]
    _, test_pos = task.get_split_indices(fold=1, repeat=0)
    np.testing.assert_array_equal(np.sort(sa["y_test_idx"]), np.sort(test_pos))
    # ... so the per-fold metrics recompute the stored score
    fm = fold_metrics(tmp_path).iloc[0]
    assert fm["ta_name"] == "Chakra-Tab-medium"
    assert to_error([fm[out["metric"]]], out["metric"])[0] == pytest.approx(out["metric_error"], rel=1e-5)
    # ... and it aggregates under the system's model key, with no validation error
    _, hpo = aggregate_results([out])
    row = hpo[hpo["method_subtype"] == "default"].iloc[0]
    assert row["config_type"] == "CHAKRA-TAB" and row["metric_error"] == pytest.approx(out["metric_error"])


def test_metadata_of_a_system_that_reports_none():
    from tabarena.benchmark.exec_models.external import ExternalSystemModel

    from raman_bench.systems import with_test_artifacts

    class Bare(ExternalSystemModel):
        pass

    assert not hasattr(Bare, "get_metadata")
    cls = with_test_artifacts(Bare, "BARE", "Bare", {"a": 1})
    meta = cls.get_metadata(object.__new__(cls))
    assert meta == {"model_type": "BARE", "name_prefix": "Bare", "model_hyperparameters": {"a": 1}}


def test_every_system_names_its_result_dir_and_key():
    assert {s.result_dir for s in SYSTEMS.values()} == {"Chakra-Tab-medium", "Chakra-Tab-full", "TabFM-Plus"}
    assert all(k == s.key for k, s in SYSTEMS.items())
    with pytest.raises(KeyError, match="Unknown system"):
        get_system("nope")



def test_chakra_tab_request_outlasts_the_fit_budget():
    from raman_bench.systems._chakra_tab_vendored import _request_timeout

    assert _request_timeout(600) == 7200  # TabArena's own timeout covers the default budget
    assert _request_timeout(10800) == 14400  # mlrod: the fit budget plus an hour
    assert _request_timeout(None) == 7200


def test_chakra_tab_sdk_path_runs_a_fold_through_the_library(monkeypatch, tmp_path):
    """CHAKRA_TAB_SDK=1 swaps the plain POST for yhatlabs.Client.predict, which sends wide tables as async jobs."""
    import sys
    import types

    from tabarena.benchmark.experiment.experiment_constructor import ExternalSystemExperiment

    monkeypatch.setenv("CHAKRA_TAB_KEY", "test-key")
    monkeypatch.setenv("CHAKRA_TAB_SDK", "1")
    calls = []

    class Client:
        def __init__(self, api_key=None, timeout=None):
            calls.append({"key": api_key})

        def predict(self, train, target, test, *, task, preset, eval_metric, time_limit, seed, _refs):
            calls.append({"task": task, "preset": preset, "n_train": len(train), "n_test": len(test),
                          "target_in_train": target in train.columns, "target_in_test": target in test.columns, "refs": _refs})
            classes = sorted(train[target].unique())
            proba = np.full((len(test), len(classes)), 1 / len(classes))
            return types.SimpleNamespace(
                predictions=pd.Series([classes[0]] * len(test)), fit={"version": "fake", "fit_s": 2.5},
                probabilities=pd.DataFrame(proba, columns=[f"p_{c}" for c in classes]), classes=classes,
            )

    monkeypatch.setitem(sys.modules, "yhatlabs", types.SimpleNamespace(Client=Client))
    system = get_system("chakra-tab")
    task_name, task = _task("classification")
    experiment = ExternalSystemExperiment(
        name=system.result_dir,
        system_cls=system.load_cls(),
        system_hyperparameters=dict(system.hyperparameters),
        method_kwargs={"fit_kwargs": {"time_limit": 600, "num_cpus": 1}},
        experiment_kwargs={"require_warmup": False},
    )
    out = run_cached(experiment, task_name=task_name, task_wrapper=task, repeat=0, fold=1, results_dir=str(tmp_path))

    assert calls[0] == {"key": "test-key"}
    assert calls[1]["preset"] == "medium" and calls[1]["task"] == "multiclass"
    assert calls[1]["target_in_train"] and not calls[1]["target_in_test"]
    assert calls[1]["n_train"] + calls[1]["n_test"] == 60
    assert calls[1]["refs"] == {}, "the table must go in the request body, not through /v1/uploads (422 above 8 MB)"
    assert out["method_metadata"]["api_fit_s"] == 2.5
