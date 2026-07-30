"""Tests for the TabArena-based grouped splitting/UserTask construction.

Regression guards for the refactor away from
``RamanBenchmark._grouped_train_test_split``'s target-value-hash group
inference (regression-only) toward ``splitting.build_user_task``'s explicit
group-id column (available to classification and regression alike).
"""

import numpy as np
import pandas as pd
import pytest

from raman_bench.splitting import (
    GROUP_COL,
    RamanBenchTaskWrapper,
    TooFewClassesError,
    build_user_task,
    filter_rare_classes,
    infer_group_ids_from_targets,
)


def _grouped_dataset(n=40, n_wn=15, n_groups=10, problem_type="regression", seed=0):
    rng = np.random.RandomState(seed)
    group_ids = np.repeat(np.arange(n_groups), n // n_groups)
    X = pd.DataFrame(rng.randn(n, n_wn), columns=[f"{200 + i * 5}" for i in range(n_wn)])
    df = X.copy()
    df[GROUP_COL] = group_ids
    if problem_type == "classification":
        df["target"] = rng.choice(["a", "b", "c"], size=n)
    else:
        df["target"] = X.iloc[:, 0] * 2 + rng.randn(n) * 0.1
    return df


def test_grouped_classification_no_leakage_across_seeds():
    """Grouping now covers classification too -- previously regression-only."""
    df = _grouped_dataset(problem_type="classification")
    seeds = [0, 1, 2]
    _, task_obj = build_user_task(
        task_name="test_grouped_clf", df=df, label_col="target",
        problem_type="classification", seeds=seeds, test_size=0.3,
    )
    wrapper = RamanBenchTaskWrapper(task=task_obj)
    for seed in seeds:
        X_tr, _, X_te, _ = wrapper.get_train_test_split(fold=0, repeat=seed)
        assert GROUP_COL not in X_tr.columns
        assert GROUP_COL not in X_te.columns
        train_groups = set(df.loc[X_tr.index, GROUP_COL])
        test_groups = set(df.loc[X_te.index, GROUP_COL])
        assert not (train_groups & test_groups), f"group leak at seed {seed}"


def test_grouped_regression_no_leakage():
    df = _grouped_dataset(problem_type="regression")
    _, task_obj = build_user_task(
        task_name="test_grouped_reg", df=df, label_col="target",
        problem_type="regression", seeds=[0], test_size=0.3,
    )
    wrapper = RamanBenchTaskWrapper(task=task_obj)
    X_tr, _, X_te, _ = wrapper.get_train_test_split(fold=0, repeat=0)
    train_groups = set(df.loc[X_tr.index, GROUP_COL])
    test_groups = set(df.loc[X_te.index, GROUP_COL])
    assert not (train_groups & test_groups)


def test_group_id_column_always_stripped_from_features():
    """TabArena's group_on is metadata-only -- it is not auto-excluded from X.

    Confirmed by direct testing against a plain OpenMLTaskWrapper (which does
    leak the column); RamanBenchTaskWrapper must remove it.
    """
    df = _grouped_dataset(problem_type="regression")
    _, task_obj = build_user_task(
        task_name="test_strip", df=df, label_col="target",
        problem_type="regression", seeds=[0], test_size=0.3,
    )
    wrapper = RamanBenchTaskWrapper(task=task_obj)
    assert GROUP_COL not in wrapper.X.columns
    X_tr, _, X_te, _ = wrapper.get_train_test_split(fold=0, repeat=0)
    assert GROUP_COL not in X_tr.columns
    assert GROUP_COL not in X_te.columns


def test_ungrouped_regression_fallback():
    rng = np.random.RandomState(1)
    X = pd.DataFrame(rng.randn(25, 10), columns=[f"{100 + i * 5}" for i in range(10)])
    df = X.copy()
    df["target"] = X.iloc[:, 0] * 2 + rng.randn(25) * 0.1

    _, task_obj = build_user_task(
        task_name="test_ungrouped_reg", df=df, label_col="target",
        problem_type="regression", seeds=[0, 1], test_size=0.2, group_col=None,
    )
    wrapper = RamanBenchTaskWrapper(task=task_obj)
    for seed in (0, 1):
        X_tr, _, X_te, _ = wrapper.get_train_test_split(fold=0, repeat=seed)
        assert len(X_tr) == 20
        assert len(X_te) == 5


def test_ungrouped_classification_fallback_is_stratified():
    rng = np.random.RandomState(2)
    X = pd.DataFrame(rng.randn(30, 8), columns=[f"{100 + i * 5}" for i in range(8)])
    df = X.copy()
    df["target"] = rng.choice(["x", "y"], size=30, p=[0.6, 0.4])

    _, task_obj = build_user_task(
        task_name="test_ungrouped_clf", df=df, label_col="target",
        problem_type="classification", seeds=[0], test_size=0.3, group_col=None,
    )
    wrapper = RamanBenchTaskWrapper(task=task_obj)
    X_tr, y_tr, X_te, y_te = wrapper.get_train_test_split(fold=0, repeat=0)
    # Stratification should roughly preserve the 60/40 ratio in both splits.
    train_frac_x = (y_tr == "x").mean()
    test_frac_x = (y_te == "x").mean()
    assert abs(train_frac_x - 0.6) < 0.15
    assert abs(test_frac_x - 0.6) < 0.2


def test_filter_rare_classes_drops_below_threshold():
    df = pd.DataFrame({"x": range(30), "target": ["a"] * 15 + ["b"] * 12 + ["c"] * 3})
    filtered = filter_rare_classes(df, label_col="target", min_samples_per_class=9)
    assert sorted(filtered["target"].unique()) == ["a", "b"]


def test_filter_rare_classes_raises_when_fewer_than_two_remain():
    df = pd.DataFrame({"x": range(5), "target": ["a"] * 3 + ["b"] * 2})
    with pytest.raises(TooFewClassesError):
        filter_rare_classes(df, label_col="target", min_samples_per_class=9)


def test_filter_rare_classes_disabled_when_threshold_zero():
    df = pd.DataFrame({"x": range(5), "target": ["a"] * 3 + ["b"] * 2})
    filtered = filter_rare_classes(df, label_col="target", min_samples_per_class=0)
    assert len(filtered) == 5


def test_infer_group_ids_finds_real_replicates():
    targets = np.array([
        [1.5, 2.0],
        [3.3, 0.0],
        [1.5, 2.0],  # replicate of row 0
        [9.9, 1.1],
        [3.3, 0.0],  # replicate of row 1
    ])
    group_ids = infer_group_ids_from_targets(targets)
    assert group_ids is not None
    assert group_ids[0] == group_ids[2]
    assert group_ids[1] == group_ids[4]
    assert group_ids[3] not in (group_ids[0], group_ids[1])


def test_infer_group_ids_returns_none_when_no_structure():
    targets = np.array([[1.0], [2.0], [3.0], [4.0]])
    assert infer_group_ids_from_targets(targets) is None


def test_infer_group_ids_ignores_all_zero_rows():
    # An all-zero target row carries no group signal (0 = "not measured" in
    # this domain) and must not be spuriously grouped with other zero rows.
    targets = np.array([[0.0, 0.0], [1.0, 2.0], [0.0, 0.0], [1.0, 2.0]])
    group_ids = infer_group_ids_from_targets(targets)
    assert group_ids is not None
    assert group_ids[1] == group_ids[3]
    assert group_ids[0] != group_ids[2]  # both all-zero, but not grouped with each other
