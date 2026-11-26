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


def test_group_column_leaves_the_feature_metadata_too():
    pytest.importorskip("autogluon.tabular")
    from autogluon.common.features.feature_metadata import FeatureMetadata
    from raman_bench.models.custom.ridge.model import Prep_RIDGE

    X = pd.DataFrame({"f0": [1.0, 2.0], "f1": [3.0, 4.0], GROUP_COL: ["a", "b"]})
    model = Prep_RIDGE()
    model._feature_metadata = FeatureMetadata.from_df(X)
    assert GROUP_COL in model._feature_metadata.get_features()
    model._preprocess(X)
    assert GROUP_COL not in model._feature_metadata.get_features()  # TabPack read its categorical list from here


def test_group_column_leaves_the_feature_list_too():
    from raman_bench.models.custom.ridge.model import Prep_RIDGE

    X = pd.DataFrame({"f0": [1.0, 2.0], "f1": [3.0, 4.0], GROUP_COL: ["a", "b"]})
    model = Prep_RIDGE()
    model._features_internal = list(X.columns)
    model._preprocess(X)
    assert model._features == ["f0", "f1"]  # EBM passes this list to interpret as feature_names


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


def _classification_task(group_labels_mixed=False, n_classes=4, maps_per_class=6, seed=0):
    """One class per group (a map) with maps of very different size, the way RamanSTS is built; with
    ``group_labels_mixed`` one group holds two classes."""
    rng = np.random.default_rng(seed)
    n_groups = n_classes * maps_per_class
    sizes = rng.integers(5, 60, size=n_groups)
    group = np.repeat(np.arange(n_groups), sizes)
    label = np.repeat(np.arange(n_groups) % n_classes, sizes)
    if group_labels_mixed:
        label[:3] = (label[:3] + 1) % n_classes
    df = pd.DataFrame(rng.normal(size=(len(group), 10)), columns=[f"f{i}" for i in range(10)])
    df[GROUP_COL] = group.astype(str)
    df["y"] = label
    return build_user_task(task_name="synth__0", df=df, label_col="y", problem_type="classification",
                           n_repeats=1, n_splits=3, group_col=GROUP_COL)[1]


def test_single_class_groups_use_per_group_inner_folds():
    from tabarena.benchmark.task.metadata import GroupLabelTypes

    assert _classification_task().group_labels == GroupLabelTypes.PER_GROUP
    assert _classification_task(group_labels_mixed=True).group_labels == GroupLabelTypes.PER_SAMPLE
    assert _grouped_task().group_labels == GroupLabelTypes.PER_GROUP  # the helper's groups hold one class each


def test_regression_groups_stay_per_sample():
    from tabarena.benchmark.task.metadata import GroupLabelTypes

    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(60, 5)), columns=[f"f{i}" for i in range(5)])
    df[GROUP_COL] = np.repeat(np.arange(12), 5).astype(str)
    df["y"] = rng.normal(size=60)
    task = build_user_task(task_name="synth__1", df=df, label_col="y", problem_type="regression",
                           n_repeats=1, n_splits=3, group_col=GROUP_COL)[1]
    assert task.group_labels == GroupLabelTypes.PER_SAMPLE


@pytest.mark.parametrize("seed", range(12))
def test_every_class_reaches_the_training_part_of_every_inner_fold(seed):
    """The RamanSTS failure: sklearn's approximate stratification put all 7 maps of a class into one inner fold."""
    pytest.importorskip("data_foundry")
    from tabarena.benchmark.exec_models.autogluon_utils import resolve_validation_splits
    from tabarena.benchmark.validation_protocol import ValidationProtocol

    wrapper = RamanBenchTaskWrapper(task=_classification_task(seed=seed), keep_group_col=True)
    for fold in range(3):
        X_train, y_train, _, _ = wrapper.get_train_test_split(fold=fold, repeat=0)
        for n_bag_folds in (2, 3):
            custom_splits, _, _ = resolve_validation_splits(
                metadata=wrapper.get_validation_metadata(),
                protocol=ValidationProtocol(num_bag_folds=n_bag_folds, task_specific_validation=True),
                X=X_train, y=y_train, clamps=[],
            )  # raises when an inner training part misses a class
            for train_idx, val_idx in custom_splits:
                assert set(y_train.iloc[train_idx]) == set(y_train)
                assert not set(X_train[GROUP_COL].iloc[train_idx]) & set(X_train[GROUP_COL].iloc[val_idx])
