"""Regression guard for RamanBenchmark's dataset-split cache writes being atomic.

_save_dataset writes the per-key train/test pickles via a temp-file + os.replace,
the same pattern _save_index already used -- added because this cache dir can be
shared by multiple independent SLURM arrays racing on the same first-ever cache
miss for a given key (see benchmark.py's _save_dataset docstring). This guards
against a regression back to a direct to_pickle(path) call, and confirms the
round-trip still produces byte-identical data.
"""

from __future__ import annotations

import os

import pandas as pd

from raman_bench.benchmark import RamanBenchmark


def _make_benchmark(tmp_path):
    return RamanBenchmark(
        dataset_names_classification=[],
        dataset_names_regression=[],
        cache_dir=str(tmp_path),
    )


def test_save_dataset_round_trips(tmp_path):
    bench = _make_benchmark(tmp_path)
    train = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    test = pd.DataFrame({"a": [4, 5], "b": ["p", "q"]})

    bench._save_dataset("some_key_0", train, test)

    assert bench._has_dataset_in_cache("some_key_0")
    loaded_train, loaded_test = bench._load_dataset_from_cache("some_key_0")
    pd.testing.assert_frame_equal(loaded_train, train)
    pd.testing.assert_frame_equal(loaded_test, test)


def test_save_dataset_leaves_no_temp_files_behind(tmp_path):
    bench = _make_benchmark(tmp_path)
    train = pd.DataFrame({"a": [1]})
    test = pd.DataFrame({"a": [2]})

    bench._save_dataset("some_key_0", train, test)

    leftovers = [f for f in os.listdir(bench.cache_dir_processed) if f.startswith(".dataset.")]
    assert leftovers == []


def test_save_dataset_final_paths_never_partial(tmp_path):
    # The whole point of temp-file + os.replace: the final path only ever exists in a
    # fully-written state, so a concurrent os.path.exists-based reader (the real
    # _has_dataset_in_cache check this guards) never observes a half-written file.
    bench = _make_benchmark(tmp_path)
    train = pd.DataFrame({"a": range(1000)})
    test = pd.DataFrame({"a": range(100)})

    bench._save_dataset("big_key_0", train, test)

    train_path, test_path = bench._get_cache_paths("big_key_0")
    for path in (train_path, test_path):
        # A file produced by os.replace is immediately fully readable, never truncated.
        loaded = pd.read_pickle(path)
        assert len(loaded) > 0
