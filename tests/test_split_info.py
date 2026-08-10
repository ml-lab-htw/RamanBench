"""Tests for split_type tracking end-to-end through _load_dataset_from_key.

See GitHub issue #6: RamanBenchmark's train/test split silently mixes three
different regimes (grouped, iid, stratified) with no record of which one a
given key used. These tests exercise the full classification, persistence
(get_split_info), and backfill-relevant paths, not just the low-level
_grouped_train_test_split splitter (see test_grouped_split.py for that).
"""

import json
import shutil
import tempfile

import numpy as np
import pandas as pd
import pytest
from raman_data import TASK_TYPE, RamanDataset
from raman_data.types import DatasetInfo

from raman_bench.benchmark import RamanBenchmark


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


def _bench(tmp_path, group_regression_splits=True, min_samples_per_class=0):
    b = RamanBenchmark.__new__(RamanBenchmark)
    b.test_size = 0.34
    b.random_state = 0
    b.min_samples_per_class = min_samples_per_class
    b.group_regression_splits = group_regression_splits
    b.dataset_names_classification = ["clf_stub"]
    b.dataset_names_regression = ["reg_stub"]
    b.cache_dir_processed = str(tmp_path)
    return b


@pytest.fixture
def tmp_cache():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_classification_split_type_is_stratified_not_iid(tmp_cache, monkeypatch):
    """Classification never enters the grouped path -- must be its own value,
    not silently folded into "iid" (issue #6's explicit ask)."""
    b = _bench(tmp_cache)
    y = [0, 1] * 15
    monkeypatch.setattr(b, "_load_raman_dataset", lambda name: _dataset(TASK_TYPE.Classification, y))

    _, _, split_info = b._load_dataset_from_key("clf_stub_0")

    assert split_info["split_type"] == "stratified"
    assert split_info["n_groups"] is None
    assert split_info["largest_group_size"] is None
    assert split_info["n_train"] + split_info["n_test"] == len(y)


def test_regression_grouping_disabled_is_iid(tmp_cache, monkeypatch):
    """group_regression_splits=False must not attempt grouping at all."""
    b = _bench(tmp_cache, group_regression_splits=False)
    y = [1.0, 2.0] * 4 + [1.0, 2.0] * 4  # would group if grouping were attempted
    monkeypatch.setattr(b, "_load_raman_dataset", lambda name: _dataset(TASK_TYPE.Regression, y))

    _, _, split_info = b._load_dataset_from_key("reg_stub_0")

    assert split_info["split_type"] == "iid"
    assert split_info["n_groups"] is None


def test_regression_with_replicates_is_grouped(tmp_cache, monkeypatch):
    b = _bench(tmp_cache, group_regression_splits=True)
    y = [1.0, 2.0, 3.0, 4.0] * 4  # each value repeats 4x -> real groups
    monkeypatch.setattr(b, "_load_raman_dataset", lambda name: _dataset(TASK_TYPE.Regression, y))

    _, _, split_info = b._load_dataset_from_key("reg_stub_0")

    assert split_info["split_type"] == "grouped"
    assert split_info["n_groups"] == 4
    assert split_info["largest_group_size"] == 4


def test_regression_without_replicates_falls_back_to_iid(tmp_cache, monkeypatch):
    b = _bench(tmp_cache, group_regression_splits=True)
    rng = np.random.RandomState(2)
    y = rng.rand(16).tolist()  # all-unique -> no groups
    monkeypatch.setattr(b, "_load_raman_dataset", lambda name: _dataset(TASK_TYPE.Regression, y))

    _, _, split_info = b._load_dataset_from_key("reg_stub_0")

    assert split_info["split_type"] == "iid"
    assert split_info["n_groups"] == len(y)


def test_split_info_persists_and_is_readable_without_resplitting(tmp_cache, monkeypatch):
    """get_split_info must work from the cache alone -- no re-derivation."""
    b = _bench(tmp_cache, group_regression_splits=True)
    y = [1.0, 2.0, 3.0, 4.0] * 4
    monkeypatch.setattr(b, "_load_raman_dataset", lambda name: _dataset(TASK_TYPE.Regression, y))

    train, test, split_info = b._load_dataset_from_key("reg_stub_0")
    b._save_dataset("reg_stub_0", train, test, split_info)

    # A fresh benchmark object (no _load_raman_dataset stub at all) can still
    # read it back -- proves this doesn't re-derive the split.
    b2 = RamanBenchmark.__new__(RamanBenchmark)
    b2.cache_dir_processed = b.cache_dir_processed
    read_back = b2.get_split_info("reg_stub_0")

    assert read_back == split_info

    with open(b._split_info_path("reg_stub_0")) as f:
        on_disk = json.load(f)
    assert on_disk == split_info


def test_get_split_info_returns_none_for_missing_key(tmp_cache):
    b = _bench(tmp_cache)
    assert b.get_split_info("never_computed_0") is None
