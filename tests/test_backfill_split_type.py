"""Tests for scripts/backfill_split_type.py (see issue #6's backfill requirement)."""

import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest
from raman_data import TASK_TYPE, RamanDataset
from raman_data.types import DatasetInfo

from raman_bench.benchmark import RamanBenchmark

# backfill_split_type.py imports raman_bench.predictions, which unconditionally
# requires autogluon (not part of the `dev` extra CI installs) -- see
# test_split_info_predictions.py for the same guard.
pytest.importorskip("autogluon")

_SPEC = importlib.util.spec_from_file_location(
    "backfill_split_type",
    os.path.join(os.path.dirname(__file__), "..", "scripts", "backfill_split_type.py"),
)
backfill_split_type = importlib.util.module_from_spec(_SPEC)
sys.modules["backfill_split_type"] = backfill_split_type
_SPEC.loader.exec_module(backfill_split_type)


def _dataset(task_type, targets, n_features=4):
    n = len(targets)
    rng = np.random.RandomState(0)
    return RamanDataset(
        spectra=rng.rand(n, n_features).astype(np.float32),
        targets=np.asarray(targets),
        raman_shifts=np.linspace(400, 1800, n_features),
        target_names=["target"],
        info=DatasetInfo(id="stub", name="stub", loader=lambda: None, metadata={}, task_type=task_type),
    )


@pytest.fixture
def fake_run(tmp_path, monkeypatch):
    """A run directory with predictions for two keys, no split_info yet."""
    output_dir = tmp_path / "results"
    predictions_dir = output_dir / "seed_0" / "predictions"
    predictions_dir.mkdir(parents=True)

    # reg_stub_0: a real prediction exists -> should be backfilled.
    pd.DataFrame({"target": [1.0, 2.0]}).to_csv(predictions_dir / "reg_stub_0_PLS_predictions.csv")
    # clf_stub_0: also has a prediction -> should be backfilled too.
    pd.DataFrame({"target": [0, 1]}).to_csv(predictions_dir / "clf_stub_0_PLS_predictions.csv")

    datasets = {
        "reg_stub": _dataset(TASK_TYPE.Regression, [1.0, 2.0, 3.0, 4.0] * 4),
        "clf_stub": _dataset(TASK_TYPE.Classification, [0, 1] * 15),
    }
    monkeypatch.setattr(
        RamanBenchmark, "_load_raman_dataset", lambda self, name: datasets[name]
    )

    config = {
        "output_dir": str(output_dir),
        "n_repetitions": 1,
        "test_size": 0.2,
        "random_state": 0,
        "cache_dir": str(tmp_path / ".cache"),
        "min_samples_per_class": 0,
        "group_regression_splits": True,
        "dataset_names_classification": ["clf_stub"],
        "dataset_names_regression": ["reg_stub"],
        "use_mirror": True,
        "mirror_repo": "unused/repo",
    }
    return config, predictions_dir


def test_backfill_writes_split_info_for_every_predicted_key(fake_run):
    config, predictions_dir = fake_run
    stats = backfill_split_type.backfill(config)

    assert stats["written"] == 2
    assert stats["already_had_it"] == 0
    assert stats["no_predictions"] == 0

    with open(predictions_dir / "reg_stub_0_split_info.json") as f:
        reg_info = json.load(f)
    assert reg_info["split_type"] == "grouped"

    with open(predictions_dir / "clf_stub_0_split_info.json") as f:
        clf_info = json.load(f)
    assert clf_info["split_type"] == "stratified"


def test_backfill_is_idempotent(fake_run):
    config, predictions_dir = fake_run
    backfill_split_type.backfill(config)
    stats = backfill_split_type.backfill(config)

    assert stats["written"] == 0
    assert stats["already_had_it"] == 2


def test_backfill_skips_keys_with_no_predictions(fake_run):
    """A dataset the benchmark knows about but that was never predicted (e.g.
    added to the config after the run finished) must not be backfilled."""
    config, predictions_dir = fake_run
    config["dataset_names_regression"] = config["dataset_names_regression"] + ["never_run"]

    import numpy as np

    datasets_patch = RamanBenchmark._load_raman_dataset

    def _loader(self, name):
        if name == "never_run":
            return _dataset(TASK_TYPE.Regression, np.random.RandomState(1).rand(8).tolist())
        return datasets_patch(self, name)

    RamanBenchmark._load_raman_dataset = _loader
    try:
        stats = backfill_split_type.backfill(config)
    finally:
        RamanBenchmark._load_raman_dataset = datasets_patch

    assert stats["written"] == 2  # only the two keys with real predictions
    assert stats["no_predictions"] == 1  # never_run_0 has no prediction file
    assert not (predictions_dir / "never_run_0_split_info.json").exists()
