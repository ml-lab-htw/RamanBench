"""Tests for scripts/run_experiment.py's class-imbalance bag-fold scaling.

This is the narrower, class-imbalance-aware sibling of TabArena's own
built-in flat small-dataset regime (ValidationProtocol's tiny_num_bag_folds/
tiny_max_group_instances, wired into run_one's own ValidationProtocol(...)
construction) -- this one reacts to min_class_count/n_train_est, which a flat
row-count threshold alone wouldn't catch on an otherwise-large, severely
imbalanced dataset. Both are floored at TabArena's own hard minimum of 2 bag
folds -- true single-holdout (0 or 1) isn't reachable in v1's pipeline.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_run_experiment():
    spec = importlib.util.spec_from_file_location(
        "run_experiment_under_test", REPO_ROOT / "scripts" / "run_experiment.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


run_experiment = _load_run_experiment()
resolve_effective_bag_folds = run_experiment.resolve_effective_bag_folds


def test_large_dataset_unaffected():
    assert (
        resolve_effective_bag_folds(
            num_bag_folds=3, n_rows=10_000, n_splits=3, problem_type="regression"
        )
        == 3
    )


def test_never_goes_below_tabarena_hard_minimum_of_2():
    # A pathologically tiny dataset where the formula would want to go below
    # 2 (n_train_est // 4 == 0) still floors at 2, not 0/1 -- ValidationProtocol
    # itself rejects anything below 2.
    assert (
        resolve_effective_bag_folds(
            num_bag_folds=8, n_rows=5, n_splits=3, problem_type="regression"
        )
        == 2
    )


def test_regression_scales_down_for_small_training_partition():
    # n_train_est = 40 * 2/3 = 26, //4 = 6 -- below the default 8, so scaled down.
    assert (
        resolve_effective_bag_folds(
            num_bag_folds=8, n_rows=40, n_splits=3, problem_type="regression"
        )
        == 6
    )


def test_classification_uses_min_class_count():
    # min_class_count=6 -> //2 = 3
    assert (
        resolve_effective_bag_folds(
            num_bag_folds=8,
            n_rows=1000,
            n_splits=3,
            problem_type="classification",
            min_class_count=6,
        )
        == 3
    )


def test_classification_requires_min_class_count():
    with pytest.raises(ValueError):
        resolve_effective_bag_folds(
            num_bag_folds=3, n_rows=200, n_splits=3, problem_type="classification"
        )
