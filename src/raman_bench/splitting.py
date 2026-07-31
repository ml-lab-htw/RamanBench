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
:class:`~tabarena.benchmark.task.user_task.UserTask` covering real repeated
k-fold cross-validation — matching TabArena's own documented convention for
custom/local datasets (see ``examples/benchmarking/
run_quickstart_tabarena_custom_datasets.py`` in the ``tabarena`` repo:
``RepeatedStratifiedKFold(n_repeats=10, n_splits=3)``), not RamanBench's
earlier "3 independent holdout splits, no folds" scheme. A cluster job for
repeat ``r`` / fold ``f`` calls
``experiment.run(task=wrapped_task, fold=f, repeat=r, ...)``.

sklearn has no "repeated" wrapper for the *grouped* splitters
(``GroupKFold``/``StratifiedGroupKFold``), so the repeated-group case below
loops ``n_repeats`` times, reseeding the (shuffled) splitter each time —
this also lifts the previous single-split scheme's known limitation that a
grouped classification split couldn't additionally be class-balanced:
``StratifiedGroupKFold`` does both at once.

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
from sklearn.model_selection import (
    GroupKFold,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    StratifiedGroupKFold,
)
from tabarena.benchmark.task.metadata import GroupLabelTypes
from tabarena.benchmark.task.openml.task_wrapper import OpenMLTaskWrapper
from tabarena.benchmark.task.user_task import (
    UserTask,
    from_sklearn_splits_to_user_task_splits,
)

GROUP_COL = "_group_id"
DEFAULT_N_REPEATS = 10
DEFAULT_N_SPLITS = 3


def get_n_repeats(n_instances: int, tabarena_lite: bool = False) -> int:
    """Dataset-size-adaptive repeat count, matching TabArena's own real protocol exactly.

    Ported from ``tabarena.nips2025_utils.fetch_metadata._get_n_repeats`` (the function
    behind the actual per-dataset ``tabarena_num_repeats`` values used for their real
    51-dataset benchmark -- confirmed directly against
    ``curated_tabarena_dataset_metadata.csv``: crosses from 10 to 3 repeats exactly at the
    2,500-instance boundary, num_folds is a fixed 3 for every dataset). A uniform
    ``n_repeats`` for every dataset regardless of size is *not* what the real protocol
    does -- smaller/noisier datasets get more repeats for statistical power; large ones
    need fewer since a single evaluation is already stable (and, practically, large
    datasets are also the ones most likely to hit a job's time limit).
    """
    if tabarena_lite:
        return 1
    if n_instances < 2_500:
        return 10
    if n_instances > 250_000:
        return 1
    return 3


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


def _repeated_kfold_splits_labeled(
    df: pd.DataFrame,
    *,
    label_col: str,
    problem_type: Literal["classification", "regression"],
    n_repeats: int,
    n_splits: int,
    group_col: str | None = GROUP_COL,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build ``n_repeats * n_splits`` (train_idx, test_idx) folds, repeat-major,
    over a fully-labeled ``df`` (every row has a non-null ``label_col``).

    Non-grouped: ``RepeatedStratifiedKFold`` (classification) or
    ``RepeatedKFold`` (regression) -- these already iterate repeat-major,
    fold-minor natively, matching what
    :func:`~tabarena.benchmark.task.user_task.from_sklearn_splits_to_user_task_splits`
    expects.

    Grouped: sklearn has no repeated wrapper for ``GroupKFold``/
    ``StratifiedGroupKFold``, so this loops ``n_repeats`` times, reseeding a
    freshly-shuffled splitter (``random_state=repeat_idx``) each time --
    ``StratifiedGroupKFold`` for classification balances classes *and*
    respects groups in the same split, unlike the old single-split scheme's
    ``GroupShuffleSplit``-only approach.
    """
    has_groups = group_col is not None and group_col in df.columns and df[group_col].notna().any()

    if has_groups:
        groups = df[group_col].values
        splits = []
        for repeat_idx in range(n_repeats):
            if problem_type == "classification":
                splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=repeat_idx)
                repeat_splits = splitter.split(df, df[label_col], groups=groups)
            else:
                splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=repeat_idx)
                repeat_splits = splitter.split(df, groups=groups)
            splits.extend(repeat_splits)
        return splits

    if problem_type == "classification":
        splitter = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=0)
        return list(splitter.split(df, df[label_col]))
    splitter = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=0)
    return list(splitter.split(df))


def _repeated_kfold_splits(
    df: pd.DataFrame,
    *,
    label_col: str,
    problem_type: Literal["classification", "regression"],
    n_repeats: int,
    n_splits: int,
    group_col: str | None = GROUP_COL,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build ``n_repeats * n_splits`` (train_idx, test_idx) folds, repeat-major.

    Rows with a missing (NaN) ``label_col`` -- present when a caller opted
    into semi-supervised benchmarking by *not* dropping them first (see
    ``run_experiment.py``'s ``filter_unlabeled`` flag) -- are never eligible
    for a test fold (there is no ground truth to score against); they are
    computed over the labeled subset only, then unioned into each fold's
    train indices. If every row is labeled this is a no-op passthrough to
    :func:`_repeated_kfold_splits_labeled`.

    For grouped datasets, an unlabeled row is only added to a given fold's
    train indices if its group doesn't also appear in that fold's *test*
    indices -- an unlabeled row can share a physical-replicate group with a
    labeled row, and blindly adding every unlabeled row to every fold would
    silently violate the "a group never spans train/test" guarantee for
    whichever folds hold that group's labeled replicate in test.
    """
    labeled_mask = df[label_col].notna().to_numpy()
    if labeled_mask.all():
        return _repeated_kfold_splits_labeled(
            df, label_col=label_col, problem_type=problem_type,
            n_repeats=n_repeats, n_splits=n_splits, group_col=group_col,
        )

    labeled_positions = np.flatnonzero(labeled_mask)
    unlabeled_positions = np.flatnonzero(~labeled_mask)
    df_labeled = df.iloc[labeled_positions].reset_index(drop=True)

    labeled_splits = _repeated_kfold_splits_labeled(
        df_labeled, label_col=label_col, problem_type=problem_type,
        n_repeats=n_repeats, n_splits=n_splits, group_col=group_col,
    )

    has_groups = group_col is not None and group_col in df.columns and df[group_col].notna().any()
    if not has_groups:
        return [
            (np.concatenate([labeled_positions[train_idx], unlabeled_positions]), labeled_positions[test_idx])
            for train_idx, test_idx in labeled_splits
        ]

    groups = df[group_col].to_numpy()
    unlabeled_groups = groups[unlabeled_positions]
    result = []
    for train_idx, test_idx in labeled_splits:
        mapped_test = labeled_positions[test_idx]
        test_groups = set(groups[mapped_test].tolist())
        safe_unlabeled = unlabeled_positions[[g not in test_groups for g in unlabeled_groups]]
        result.append((np.concatenate([labeled_positions[train_idx], safe_unlabeled]), mapped_test))
    return result


def build_user_task(
    *,
    task_name: str,
    df: pd.DataFrame,
    label_col: str,
    problem_type: Literal["classification", "regression"],
    n_repeats: int = DEFAULT_N_REPEATS,
    n_splits: int = DEFAULT_N_SPLITS,
    group_col: str | None = GROUP_COL,
    task_cache_path=None,
):
    """Build a TabArena ``OpenMLSupervisedTask`` for one (dataset, target) pair.

    Real repeated k-fold CV (``n_repeats`` repeats of ``n_splits``-fold CV
    each), matching TabArena's own documented convention for custom/local
    datasets. Returns the raw task object; wrap it with
    :class:`RamanBenchTaskWrapper` (not plain :class:`OpenMLTaskWrapper`)
    before fitting.

    ``df`` may contain rows with a missing (NaN) ``label_col`` -- semi-supervised
    benchmarking support: a caller that chose not to drop them (see
    ``run_experiment.py``'s ``filter_unlabeled`` flag) gets them included in
    every fold's train split (never test, since there's no ground truth to
    score against) rather than silently breaking the split.
    """
    has_groups = group_col is not None and group_col in df.columns and df[group_col].notna().any()

    sklearn_splits = _repeated_kfold_splits(
        df,
        label_col=label_col,
        problem_type=problem_type,
        n_repeats=n_repeats,
        n_splits=n_splits,
        group_col=group_col,
    )
    splits = from_sklearn_splits_to_user_task_splits(sklearn_splits, n_splits=n_splits)

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
