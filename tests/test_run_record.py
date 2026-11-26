"""Failure labels and the setup record next to each result (raman_bench.run_record, experiment_utils.run_cached)."""

import json
import logging

import pandas as pd
import pytest

from raman_bench import run_record
from raman_bench.compare import _warn_unverified_fit
from raman_bench.experiment_utils import run_cached


class OutOfMemoryError(RuntimeError):  # torch.cuda.OutOfMemoryError by name
    pass


class NotEnoughMemoryError(Exception):  # autogluon.core.utils.exceptions.NotEnoughMemoryError by name
    pass


class TimeLimitExceeded(Exception):  # autogluon.core.utils.exceptions.TimeLimitExceeded by name
    pass


@pytest.mark.parametrize(
    "exc, label",
    [
        (OutOfMemoryError("CUDA out of memory. Tried to allocate 2 GiB"), "oom_gpu"),
        (RuntimeError("CUDA out of memory"), "oom_gpu"),
        (MemoryError(), "oom_cpu"),
        (NotEnoughMemoryError("not enough"), "oom_cpu"),
        (TimeLimitExceeded("no time left"), "timeout"),
        (TimeoutError(), "timeout"),
        (ValueError("bad"), "error"),
    ],
)
def test_classify_failure(exc, label):
    assert run_record.classify_failure(exc) == label


def test_classify_failure_follows_the_cause():
    try:
        try:
            raise TimeLimitExceeded("inner")
        except TimeLimitExceeded as inner:
            raise RuntimeError("wrapped") from inner
    except RuntimeError as outer:
        assert run_record.classify_failure(outer) == "timeout"


class _Experiment:
    name = "FAKE_c1_BAG_L1"

    def __init__(self, exc=None):
        self.exc = exc

    def run(self, **kwargs):
        if self.exc:
            raise self.exc
        return {"metric_error": 0.5}


def _run(tmp_path, experiment, setup=None):
    return run_cached(
        experiment, task_name="t__0", task_wrapper=None, repeat=0, fold=1, results_dir=str(tmp_path), setup=setup
    )


def test_a_failed_fold_leaves_a_labelled_record_and_still_raises(tmp_path):
    with pytest.raises(OutOfMemoryError):
        _run(tmp_path, _Experiment(OutOfMemoryError("CUDA out of memory")), setup={"time_limit": 600})
    record = json.loads((tmp_path / "FAKE_c1_BAG_L1/t__0/0_1/failure.json").read_text())
    assert record["label"] == "oom_gpu" and record["exception"] == "OutOfMemoryError"
    assert record["fit_mode"] == "bagged" and record["setup"] == {"time_limit": 600}


def test_a_finished_fold_records_its_setup_and_clears_an_old_failure(tmp_path):
    with pytest.raises(ValueError):
        _run(tmp_path, _Experiment(ValueError("x")))
    fold_dir = tmp_path / "FAKE_c1_BAG_L1/t__0/0_1"
    assert (fold_dir / "failure.json").exists()
    _run(tmp_path, _Experiment(), setup={"time_limit": 10800, "max_train_samples": 30000})
    record = json.loads((fold_dir / "setup.json").read_text())
    assert not (fold_dir / "failure.json").exists()
    assert record["setup"] == {"time_limit": 10800, "max_train_samples": 30000}
    assert {"raman-bench", "tabarena", "torch"} <= set(record["packages"]) and "gpu" in record["hardware"]


def test_collect_folds_and_summary(tmp_path):
    def fold(framework, task, rf, files):
        d = tmp_path / framework / task / rf
        d.mkdir(parents=True)
        for name, content in files.items():
            (d / name).write_text(content)

    fold("M_c1_BAG_L1", "a__0", "0_0", {"results.pkl": "", "setup.json": json.dumps({"fit_mode": "bagged"})})
    fold("M_c1_BAG_L1", "a__0", "0_1", {"failure.json": json.dumps({"label": "timeout", "exception": "TimeLimitExceeded"})})
    fold("M_c1_BAG_L1", "a__0", "0_2", {})
    fold("N_c1_BAG_L1", "a__0", "0_0", {"results.pkl": ""})
    folds = run_record.collect_folds(tmp_path)
    assert folds.set_index(["framework", "fold"]).status.to_dict() == {
        ("M_c1_BAG_L1", 0): "ok", ("M_c1_BAG_L1", 1): "failed", ("M_c1_BAG_L1", 2): "missing", ("N_c1_BAG_L1", 0): "ok",
    }
    summary = run_record.failure_summary(folds).set_index("framework")
    assert summary.loc["M_c1_BAG_L1", ["ok", "timeout", "missing"]].tolist() == [1, 1, 1]
    assert summary.loc["N_c1_BAG_L1", "ok"] == 1


def test_compare_warns_about_results_it_cannot_check(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        _warn_unverified_fit(pd.DataFrame())
    assert "cannot check how they were fitted" in caplog.text
    caplog.clear()
    fold = tmp_path / "M_c1_BAG_L1/a__0/0_0"
    fold.mkdir(parents=True)
    (fold / "results.pkl").write_text("")
    with caplog.at_level(logging.WARNING):
        _warn_unverified_fit(tmp_path)
    assert "no setup.json" in caplog.text
