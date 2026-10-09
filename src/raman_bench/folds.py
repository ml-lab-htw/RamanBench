"""The v1 outer cross-validation folds as plain row indices.

Every v1 leaderboard result was produced on these folds. :func:`export_folds` writes
them as one table, so another benchmarking framework can train and test on exactly the
same rows without TabArena or AutoGluon. A model scored on these folds with each task's
metric (``protocol.json``: ROC AUC for binary, log loss for multiclass, RMSE for
regression) is comparable with the leaderboard models' per-fold results.

One row of the table per (task, repeat, spectrum):

- ``spectrum_id``: the spectrum's position in the dataset as ``raman_data`` loads it
  (``dataset.to_dataframe(target_idx)``'s index), which is also its row in the
  Hugging Face mirror's parquet file.
- ``fold``: the outer fold whose test set holds the spectrum. For fold ``f`` the test
  set is the task's rows with ``fold == f`` (in that repeat), the training set all its
  other rows.

Spectra a task doesn't use are not listed: a NaN in the spectrum, no label for the
target, a class with fewer than ``min_samples_per_class`` spectra, or (on the datasets
in ``max_train_samples_overrides``) not drawn into the row cap.

How the folds are made: :func:`raman_bench.splitting.build_user_task` (stratified for
classification, grouped where a dataset has replicate groups, fixed seeds).
:func:`task_folds` builds them with the same code as ``scripts/run_experiment.py``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from raman_bench.compare import load_protocol, protocol_tasks
from raman_bench.experiment_utils import load_dataframe, prepare_task_dataframe
from raman_bench.splitting import GROUP_COL, _repeated_kfold_splits

logger = logging.getLogger(__name__)

FOLD_COLUMNS = ["task", "dataset", "target_idx", "problem_type", "repeat", "spectrum_id", "fold"]


def _task_frame(task: str, *, cache_dir: str, use_mirror: bool, mirror_repo: str):
    """``(spec, df, problem_type)``: the protocol entry of *task* and its rows as the runner splits them.

    ``df``'s row order is the order the folds' positional indices refer to (the
    results' ``y_test_idx``); its index is the ``spectrum_id``.
    """
    protocol = load_protocol()
    (spec,) = [t for t in protocol_tasks(tasks=[task]) if t["task"] == task]
    dataset_name, target_idx = spec["dataset"], spec["target_idx"]

    dataset, df, sample_idx, problem_type = load_dataframe(
        dataset_name,
        target_idx,
        max_train_samples=protocol["max_train_samples_overrides"].get(dataset_name),
        cache_dir=cache_dir,
        use_mirror=use_mirror,
        mirror_repo=mirror_repo,
    )
    df = prepare_task_dataframe(
        dataset_name=dataset_name,
        target_idx=target_idx,
        df=df,
        raw_targets=dataset.targets,
        problem_type=problem_type,
        sample_idx=sample_idx,
        min_samples_per_class=protocol["min_samples_per_class"],
    )
    if df is None:
        raise RuntimeError(f"{task} has no usable rows; it should not be in the protocol")
    return spec, df, problem_type


def task_folds(
    task: str,
    *,
    cache_dir: str = ".cache_v1",
    use_mirror: bool = True,
    mirror_repo: str = "HTW-KI-Werkstatt/RamanBench",
) -> pd.DataFrame:
    """The outer folds of one protocol task (``"<dataset>__<target_idx>"``), columns :data:`FOLD_COLUMNS`."""
    spec, df, problem_type = _task_frame(
        task, cache_dir=cache_dir, use_mirror=use_mirror, mirror_repo=mirror_repo
    )
    dataset_name, target_idx = spec["dataset"], spec["target_idx"]
    n_splits = spec["n_folds"]
    splits = _repeated_kfold_splits(
        df,
        label_col=df.columns[-1],
        problem_type=problem_type,
        n_repeats=spec["n_repeats"],
        n_splits=n_splits,
        group_col=GROUP_COL if GROUP_COL in df.columns else None,
    )
    spectrum_ids = df.index.to_numpy()
    parts = []
    for i, (_, test_pos) in enumerate(splits):
        parts.append(
            pd.DataFrame(
                {
                    "repeat": i // n_splits,
                    "spectrum_id": spectrum_ids[test_pos],
                    "fold": i % n_splits,
                }
            )
        )
    out = pd.concat(parts, ignore_index=True)
    # Every used spectrum is in exactly one test fold per repeat.
    per_repeat = out.groupby("repeat")["spectrum_id"]
    if not (per_repeat.nunique() == len(df)).all() or not (per_repeat.size() == len(df)).all():
        raise AssertionError(f"{task}: the test folds don't partition the task's rows")

    out.insert(0, "problem_type", problem_type)
    out.insert(0, "target_idx", target_idx)
    out.insert(0, "dataset", dataset_name)
    out.insert(0, "task", task)
    return out.sort_values(["repeat", "spectrum_id"], ignore_index=True)[FOLD_COLUMNS]


def export_folds(
    path: str | Path | None = None,
    *,
    task_type: str | None = None,
    tasks: list[str] | None = None,
    cache_dir: str = ".cache_v1",
    use_mirror: bool = True,
    mirror_repo: str = "HTW-KI-Werkstatt/RamanBench",
) -> pd.DataFrame:
    """The outer folds of every protocol task (or of *task_type* / *tasks*, as in
    :func:`~raman_bench.compare.protocol_tasks`), written to *path* if given.

    *path* ending in ``.csv`` writes CSV, anything else parquet. Downloads each dataset
    from the mirror on first use (cached in *cache_dir*).
    """
    selected = protocol_tasks(task_type=task_type, tasks=tasks)
    frames = []
    for i, spec in enumerate(selected, 1):
        logger.info("[%d/%d] %s", i, len(selected), spec["task"])
        frames.append(
            task_folds(
                spec["task"], cache_dir=cache_dir, use_mirror=use_mirror, mirror_repo=mirror_repo
            )
        )
    folds = pd.concat(frames, ignore_index=True)
    folds["fold"] = folds["fold"].astype(np.int8)
    folds["repeat"] = folds["repeat"].astype(np.int8)
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".csv":
            folds.to_csv(path, index=False)
        else:
            folds.to_parquet(path, index=False)
    return folds


def train_test_ids(
    folds: pd.DataFrame, task: str, fold: int, repeat: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """``(train, test)`` spectrum ids of one task's *fold* from an :func:`export_folds` table."""
    rows = folds[(folds["task"] == task) & (folds["repeat"] == repeat)]
    if rows.empty:
        raise KeyError(f"No folds for task {task!r} (repeat {repeat})")
    in_test = rows["fold"] == fold
    return rows.loc[~in_test, "spectrum_id"].to_numpy(), rows.loc[in_test, "spectrum_id"].to_numpy()
