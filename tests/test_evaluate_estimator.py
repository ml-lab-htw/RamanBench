"""Tests for ``raman_bench.evaluate``: scikit-learn estimators on the v1 protocol.

The end-to-end test runs real (tiny) AutoGluon bagged fits on synthetic spectra; the
dataset loader is patched, the same convention as ``tests/test_run_experiment_recipe.py``.
"""

from __future__ import annotations

import shlex
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from raman_bench.evaluate import protocol_commands

CLF_TASK = "diabetes_skin_ear_lobe__0"
REG_TASK = "tg_ecoli_fermentation__2"


def test_protocol_commands_carry_the_protocol_settings():
    cmds = protocol_commands("MY_MODEL")
    assert len(cmds) == 135 * 3
    mlrod = [shlex.split(c) for c in protocol_commands("MY_MODEL", tasks=["mlrod"])]
    assert [a[a.index("--fold") + 1] for a in mlrod] == ["0", "1", "2"]
    args = mlrod[0]
    assert args[args.index("--time-limit") + 1] == "10800"
    assert args[args.index("--max-train-samples") + 1] == "10000"
    assert args[args.index("--num-bag-folds") + 1] == "3"
    small = shlex.split(protocol_commands("MY_MODEL", tasks=[REG_TASK])[0])
    assert small[small.index("--time-limit") + 1] == "600"
    assert "--max-train-samples" not in small
    assert len(protocol_commands("X", task_type="classification")) == 21 * 3


class _FakeDataset:
    def __init__(self, task_type, df):
        self.task_type = task_type
        self._df = df
        self.targets = df[df.columns[-1]].to_numpy().reshape(-1, 1)

    def to_dataframe(self, target_idx):
        return self._df.copy()


def _spectra(n: int, classification: bool, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = rng.normal(5.0, 1.0, size=(n, 30))
    y = (X[:, 0] > 5.0).astype(int) if classification else X[:, 0] * 2 + rng.normal(0, 0.1, n)
    return pd.DataFrame({**{f"{400 + i}": X[:, i] for i in range(30)}, "target": y})


def test_evaluate_estimator_validates_its_input():
    pytest.importorskip("tabarena")
    from sklearn.svm import LinearSVC

    from raman_bench.evaluate import evaluate_estimator

    with pytest.raises(ValueError, match="classifier, a regressor"):
        evaluate_estimator("X")
    with pytest.raises(ValueError, match="predict_proba"):
        evaluate_estimator("X", classifier=LinearSVC())
    with pytest.raises(ValueError, match="letters, digits"):
        evaluate_estimator("my model/1", regressor=object())


def test_evaluate_estimator_end_to_end(tmp_path):
    pytest.importorskip("tabarena")
    pytest.importorskip("bencheval")
    from raman_data import TASK_TYPE
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.linear_model import LogisticRegression

    from raman_bench.compare import compare
    from raman_bench.evaluate import evaluate_estimator

    fakes = {
        CLF_TASK.split("__")[0]: _FakeDataset(TASK_TYPE.Classification, _spectra(60, True)),
        REG_TASK.split("__")[0]: _FakeDataset(TASK_TYPE.Regression, _spectra(40, False)),
    }

    def load(self, name):
        return fakes[name]

    results_dir = str(tmp_path / "results")
    with patch("raman_bench.benchmark.RamanBenchmark._load_raman_dataset", load):
        res = evaluate_estimator(
            "TEST-EST",
            classifier=LogisticRegression(max_iter=500),
            regressor=PLSRegression(n_components=2),
            tasks=[CLF_TASK, REG_TASK],
            results_dir=results_dir,
            hyperparameters={"prep_snv_enabled": True},
            cache_dir=str(tmp_path / "cache"),
        )
    assert (tmp_path / "results" / "TEST-EST_c1_BAG_L1" / CLF_TASK / "0_2" / "results.pkl").exists()
    assert sorted(res["dataset"].unique()) == sorted([CLF_TASK, REG_TASK])
    assert len(res) == 6 and set(res["model"]) == {"TEST-EST"}
    assert res["metric_error"].notna().all()
    assert (res["preprocessing"] == "snv").all()

    lb = compare(results_dir, tasks="own", bootstrap_rounds=5)["all"].leaderboard
    assert "TEST-EST" in lb.index
    assert lb.loc["TEST-EST", "preprocessing"] == "snv"
