"""Group-aware inner bagging folds (issue #162).

The outer folds are group-disjoint in every mode. With ``inner_group_folds`` on, the
group id stays in the frame TabArena's splitter reads, and every model drops it before
fitting. With it off (v1), the group id never reaches the models.
"""

import os

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("tabarena")

from raman_bench.splitting import GROUP_COL, RamanBenchTaskWrapper, build_user_task  # noqa: E402


def _grouped_task(n_groups=24, per_group=5, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        rng.normal(size=(n_groups * per_group, 20)), columns=[f"f{i}" for i in range(20)]
    )
    df[GROUP_COL] = np.repeat(np.arange(n_groups), per_group).astype(str)
    df["y"] = np.repeat(np.arange(n_groups) % 2, per_group)
    _, task = build_user_task(
        task_name="synth__0",
        df=df,
        label_col="y",
        problem_type="classification",
        n_repeats=1,
        n_splits=3,
        group_col=GROUP_COL,
    )
    return task


def test_outer_folds_are_group_disjoint_and_keep_the_group_column_when_asked():
    wrapper = RamanBenchTaskWrapper(task=_grouped_task(), keep_group_col=True)
    for fold in range(3):
        X_train, _, X_test, _ = wrapper.get_train_test_split(fold=fold, repeat=0)
        assert not set(X_train[GROUP_COL]) & set(X_test[GROUP_COL])


def test_group_column_is_hidden_from_models_by_default():
    wrapper = RamanBenchTaskWrapper(task=_grouped_task())
    assert GROUP_COL not in wrapper.X.columns
    X_train, _, _, _ = wrapper.get_train_test_split(fold=0, repeat=0)
    assert GROUP_COL not in X_train.columns


def test_models_drop_the_group_column_before_fitting():
    from raman_bench.models.custom.ridge.model import Prep_RIDGE

    X = pd.DataFrame({"f0": [1.0, 2.0], "f1": [3.0, 4.0], GROUP_COL: ["a", "b"]})
    model = Prep_RIDGE()
    assert list(model._preprocess(X).columns) == ["f0", "f1"]


def test_inner_folds_are_group_disjoint_when_kept():
    pytest.importorskip("data_foundry")
    from tabarena.benchmark.exec_models.autogluon_utils import resolve_validation_splits
    from tabarena.benchmark.validation_protocol import ValidationProtocol

    wrapper = RamanBenchTaskWrapper(task=_grouped_task(), keep_group_col=True)
    X_train, y_train, _, _ = wrapper.get_train_test_split(fold=0, repeat=0)
    protocol = ValidationProtocol(num_bag_folds=3, task_specific_validation=True)
    custom_splits, n_folds, n_repeats = resolve_validation_splits(
        metadata=wrapper.get_validation_metadata(),
        protocol=protocol,
        X=X_train,
        y=y_train,
        clamps=[],
    )
    assert n_folds == 3 and n_repeats == 1
    assert len(custom_splits) == 3
    for train_idx, val_idx in custom_splits:
        train_groups = set(X_train[GROUP_COL].iloc[train_idx])
        val_groups = set(X_train[GROUP_COL].iloc[val_idx])
        assert not train_groups & val_groups


@pytest.mark.skipif(
    os.environ.get("RAMANBENCH_INTEGRATION") != "1",
    reason="downloads flow_microgel_synthesis from the HF mirror; set RAMANBENCH_INTEGRATION=1",
)
def test_flow_microgel_inner_folds_are_group_disjoint():
    from tabarena.benchmark.exec_models.autogluon_utils import resolve_validation_splits
    from tabarena.benchmark.validation_protocol import ValidationProtocol

    from raman_bench.experiment_utils import build_task, load_dataframe

    dataset, df, sample_idx, problem_type = load_dataframe(
        "flow_microgel_synthesis", 0, cache_dir=".cache_v1"
    )
    _, wrapper = build_task(
        dataset_name="flow_microgel_synthesis",
        target_idx=0,
        df=df,
        raw_targets=dataset.targets,
        problem_type=problem_type,
        n_repeats=1,
        n_splits=3,
        sample_idx=sample_idx,
        inner_group_folds=True,
    )
    protocol = ValidationProtocol(num_bag_folds=3, task_specific_validation=True)
    for fold in range(3):
        X_train, y_train, X_test, _ = wrapper.get_train_test_split(fold=fold, repeat=0)
        assert not set(X_train[GROUP_COL]) & set(X_test[GROUP_COL])
        custom_splits, n_folds, _ = resolve_validation_splits(
            metadata=wrapper.get_validation_metadata(),
            protocol=protocol,
            X=X_train,
            y=y_train,
            clamps=[],
        )
        assert n_folds == 3
        for train_idx, val_idx in custom_splits:
            assert not set(X_train[GROUP_COL].iloc[train_idx]) & set(X_train[GROUP_COL].iloc[val_idx])
