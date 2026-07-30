"""Tests for the TabArena-based grouped splitting/UserTask construction.

Regression guards for the refactor away from
``RamanBenchmark._grouped_train_test_split``'s target-value-hash group
inference (regression-only) toward ``splitting.build_user_task``'s explicit
group-id column (available to classification and regression alike).
"""

import numpy as np
import pandas as pd

from raman_bench.splitting import GROUP_COL, RamanBenchTaskWrapper, build_user_task


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
