"""Grouped train/test splitting and TabArena ``UserTask`` construction.

Replaces :meth:`~raman_bench.benchmark.RamanBenchmark._grouped_train_test_split`'s
group-key inference (hashing each row's *target values* into a frozenset —
fragile: coincidentally-identical targets are falsely grouped, noisy repeat
measurements with slightly different targets are missed, and classification
datasets got no grouping at all) with splitting on an **explicit** group-id
column (populated by ``raman_data`` loaders that know their dataset's
replicate structure — see :attr:`raman_data.types.RamanDataset.group_ids`),
available to classification and regression alike.

Each (dataset, target) pair becomes one TabArena
:class:`~tabarena.benchmark.task.user_task.UserTask` covering every seed as a
separate "repeat" (fold is always 0 — RamanBench has no k-fold CV concept,
only repeated random splits, one per seed). A cluster job for seed ``s``
calls ``experiment.run(task=wrapped_task, fold=0, repeat=s, ...)``.

Confirmed by direct testing (see the Phase 0 spike in the refactor plan):
TabArena's ``group_on`` is used only for split/metadata bookkeeping — it is
**not** stripped from the feature matrix automatically. Fitting a raw
``OpenMLTaskWrapper``-wrapped task would leak the group-id column into the
model as a feature. :class:`RamanBenchTaskWrapper` fixes this.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit, StratifiedShuffleSplit
from tabarena.benchmark.task.openml.task_wrapper import OpenMLTaskWrapper
from tabarena.benchmark.task.user_task import (
    GroupLabelTypes,
    UserTask,
    from_sklearn_splits_to_user_task_splits,
)

GROUP_COL = "_group_id"


class TooFewClassesError(ValueError):
    """Raised when rare-class filtering leaves fewer than 2 classes.

    Callers (e.g. ``scripts/run_experiment.py``) should treat this as a clean,
    expected skip for this (dataset, target) -- not a crash -- matching how
    the old ``RamanBenchmark._filter_rare_classes`` silently excluded such
    keys from the benchmark entirely.
    """


def filter_rare_classes(
    df: pd.DataFrame, *, label_col: str, min_samples_per_class: int = 9
) -> pd.DataFrame:
    """Drop classes with fewer than ``min_samples_per_class`` rows.

    Port of :meth:`~raman_bench.benchmark.RamanBenchmark._filter_rare_classes`
    for the new pipeline. Only meaningful for classification; call only when
    ``problem_type == "classification"``. Raises :class:`TooFewClassesError`
    if fewer than 2 classes remain after filtering.
    """
    if min_samples_per_class <= 0:
        return df
    counts = df[label_col].value_counts()
    rare = counts[counts < min_samples_per_class].index.tolist()
    if rare:
        df = df[~df[label_col].isin(rare)]
    if df[label_col].nunique() < 2:
        raise TooFewClassesError(
            f"Fewer than 2 classes remain in {label_col!r} after dropping classes with "
            f"< {min_samples_per_class} samples (dropped: {rare})."
        )
    return df


def infer_group_ids_from_targets(targets: np.ndarray) -> np.ndarray | None:
    """Derive group ids from exact matches across a dataset's *full* target array.

    **Regression only** -- do not call this for classification. Two rows are
    treated as the same physical sample/measurement when every one of their
    target columns has an identical value (zero entries excluded from the
    key, on the convention that 0 means "not measured" for that analyte in
    this domain). This is deliberately the same signal the retired
    ``RamanBenchmark._grouped_train_test_split`` relied on -- a coincidental
    exact match across several independent continuous-valued targets is
    vanishingly unlikely, so a real match is strong evidence of a shared
    physical sample -- just computed explicitly once per dataset (over *all*
    of its targets together) rather than inline, per split, with the old
    hash-order reproducibility bug.

    Uses the dataset's full target matrix, not just the one target currently
    being benchmarked -- a sample's identity is defined by everything
    measured about it, and the same grouping must then be reused for every
    individual target's benchmark task on that dataset (so a replicate can
    never leak across train/test on any of them).

    Returns ``None`` when no row's key is shared by any other row -- i.e.
    the data shows no real replicate structure. Returning ``None`` rather
    than an all-unique array is intentional: it signals "no grouping needed"
    distinctly from "grouped, and every group happens to have size 1".
    """
    targets_2d = targets.reshape(-1, 1) if targets.ndim == 1 else targets
    n = len(targets_2d)
    group_ids = np.arange(n)

    key_to_rows: dict[tuple, list[int]] = {}
    for i, row in enumerate(targets_2d):
        nonzero = tuple(v for v in row if v != 0)
        if not nonzero:
            continue  # all-zero row: no signal, leave it as its own unique group
        key_to_rows.setdefault(nonzero, []).append(i)

    found_real_group = False
    for rows in key_to_rows.values():
        if len(rows) > 1:
            found_real_group = True
            shared_id = rows[0]
            for i in rows[1:]:
                group_ids[i] = shared_id

    return group_ids if found_real_group else None


class RamanBenchTaskWrapper(OpenMLTaskWrapper):
    """An :class:`OpenMLTaskWrapper` that drops the group-id column from X.

    TabArena's ``group_on`` mechanism (grouped splitting) does not exclude the
    group-id column from the feature matrix handed to models -- confirmed by
    direct testing, not merely reading the source. Without this fix, a model
    would be fit with the group id as an ordinary feature.
    """

    def __init__(self, task, **kwargs):
        super().__init__(task, **kwargs)
        group_cols = self._group_cols()
        if group_cols and not self.lazy_load_data:
            self.X = self.X.drop(columns=group_cols)

    def _group_cols(self) -> list[str]:
        group_on = getattr(self.task, "group_on", None)
        if group_on is None:
            return []
        return group_on if isinstance(group_on, list) else [group_on]

    def get_train_test_split(self, *args, **kwargs):
        X_train, y_train, X_test, y_test = super().get_train_test_split(*args, **kwargs)
        group_cols = self._group_cols()
        if group_cols:
            X_train = X_train.drop(columns=[c for c in group_cols if c in X_train.columns])
            X_test = X_test.drop(columns=[c for c in group_cols if c in X_test.columns])
        return X_train, y_train, X_test, y_test


def _split_indices_per_seed(
    df: pd.DataFrame,
    *,
    label_col: str,
    problem_type: Literal["classification", "regression"],
    seeds: list[int],
    test_size: float,
    group_col: str | None = GROUP_COL,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build one (train_idx, test_idx) split per seed.

    Uses ``GroupShuffleSplit`` keyed on an explicit group-id column when
    present -- for classification *and* regression alike -- so replicate rows
    (same physical sample/measurement) never land on both sides of a split.
    Falls back to ``StratifiedShuffleSplit`` (classification) or
    ``ShuffleSplit`` (regression) when no group-id column is available.
    Note: sklearn has no stratified *and* grouped single-split splitter, so a
    grouped classification split is not additionally class-balanced -- the
    same trade-off the previous (regression-only) grouped split already made.
    """
    has_groups = group_col is not None and group_col in df.columns and df[group_col].notna().any()
    splits = []
    for seed in seeds:
        if has_groups:
            groups = df[group_col].values
            splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
            train_idx, test_idx = next(splitter.split(df, groups=groups))
        elif problem_type == "classification":
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
            train_idx, test_idx = next(splitter.split(df, df[label_col]))
        else:
            splitter = ShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
            train_idx, test_idx = next(splitter.split(df))
        splits.append((train_idx, test_idx))
    return splits


def build_user_task(
    *,
    task_name: str,
    df: pd.DataFrame,
    label_col: str,
    problem_type: Literal["classification", "regression"],
    seeds: list[int],
    test_size: float = 0.2,
    group_col: str | None = GROUP_COL,
    task_cache_path=None,
):
    """Build a TabArena ``OpenMLSupervisedTask`` for one (dataset, target) pair.

    One repeat per seed (``seeds[i]`` -> repeat index ``i``), fold always 0.
    Returns the raw task object; wrap it with :class:`RamanBenchTaskWrapper`
    (not plain :class:`OpenMLTaskWrapper`) before fitting.
    """
    has_groups = group_col is not None and group_col in df.columns and df[group_col].notna().any()

    sklearn_splits = _split_indices_per_seed(
        df,
        label_col=label_col,
        problem_type=problem_type,
        seeds=seeds,
        test_size=test_size,
        group_col=group_col,
    )
    splits = from_sklearn_splits_to_user_task_splits(sklearn_splits, n_splits=1)

    user_task = UserTask(task_name=task_name, task_cache_path=task_cache_path)
    task_obj = user_task.create_local_openml_task(
        target_feature=label_col,
        problem_type=problem_type,
        dataset=df,
        splits=splits,
        group_on=group_col if has_groups else None,
        group_labels=GroupLabelTypes.PER_SAMPLE if has_groups else None,
        stratify_on=label_col if problem_type == "classification" else None,
        dataset_name=task_name,
    )
    return user_task, task_obj
