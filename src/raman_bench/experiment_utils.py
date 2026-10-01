"""Shared dataset-cleaning/splitting glue for cluster experiment runners.

Extracted from ``scripts/run_experiment.py``'s ``run_one`` (the per-model,
``ConfigGenerator``-based runner) so ``scripts/run_autogluon_baseline.py`` (the
whole-predictor AutoGluon preset runner -- no per-model ``ConfigGenerator``, no
``model_key``-keyed compatibility checks) can build the exact same TabArena task
object without duplicating this logic. Both callers need byte-identical splits for
the same (dataset, target_idx, repeat, fold, n_repeats, n_splits) -- a second,
independently-written copy of this sequence would risk silently drifting (different
NaN-handling order, different group-id inference) and producing results that aren't
actually comparable.

Takes an already-loaded dataframe/dataset rather than loading one itself: loading
(and any ``max_train_samples`` subsampling) stays with each caller, since
``run_one`` needs the raw dataframe's shape *before* splitting to run its own
model-specific compatibility checks (``MAX_FEATURES_MODELS``/``VRAM_CAPPED_MODELS``,
which don't apply to the model-agnostic AutoGluon baseline runner at all) --
threading those checks through this helper would couple it back to
``model_key``-specific concerns it's deliberately free of.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from raman_bench.splitting import (
    GROUP_COL,
    RamanBenchTaskWrapper,
    TooFewClassesError,
    build_user_task,
    filter_rare_classes,
    infer_group_ids_from_targets,
)

logger = logging.getLogger(__name__)


def build_task(
    *,
    dataset_name: str,
    target_idx: int,
    df: pd.DataFrame,
    raw_targets: np.ndarray,
    problem_type: str,
    n_repeats: int,
    n_splits: int,
    sample_idx: np.ndarray | None = None,
    min_samples_per_class: int = 9,
    filter_unlabeled: bool = True,
) -> tuple[str, RamanBenchTaskWrapper] | tuple[None, None]:
    """Clean and split an already-loaded (dataset, target) dataframe into a TabArena task.

    ``df`` is the caller's own ``dataset.to_dataframe(target_idx)`` result (after any
    ``max_train_samples`` subsampling the caller chose to apply). ``raw_targets`` is
    the dataset's full, un-subsampled target matrix (``dataset.targets``), used for
    regression group-id inference; ``sample_idx`` is the row-index array a caller's
    subsampling produced (``df.sample(...).index.to_numpy()``), or ``None`` if ``df``
    wasn't subsampled -- mirrors ``run_one``'s own alignment between the (possibly
    subsampled) dataframe and the full target matrix.

    Returns ``(task_name, task_wrapper)``, or ``(None, None)`` if this target should
    be cleanly skipped -- every row has a missing label, or (classification only)
    rare-class filtering drops below 2 classes. Matches ``run_one``'s own skip
    semantics: log a message and return a sentinel, not raise -- the caller should
    treat this the same way (log, exit 0, no ``results.pkl`` written).
    """
    label_col = df.columns[-1]

    if GROUP_COL not in df.columns and problem_type == "regression":
        targets_for_grouping = raw_targets[sample_idx] if sample_idx is not None else raw_targets
        inferred = infer_group_ids_from_targets(targets_for_grouping)
        if inferred is not None:
            df[GROUP_COL] = inferred
            logger.info(
                "%s: no explicit group_ids -- inferred %d group(s) from matching target values",
                dataset_name,
                len(set(inferred.tolist())),
            )

    feature_cols = [c for c in df.columns if c not in (label_col, GROUP_COL)]
    nan_feature_mask = df[feature_cols].isna().any(axis=1)
    n_nan_feature_rows = int(nan_feature_mask.sum())
    if n_nan_feature_rows:
        logger.info(
            "%s target %d: dropping %d/%d rows with a NaN value in their feature "
            "(spectral) columns before any split or preprocessing",
            dataset_name,
            target_idx,
            n_nan_feature_rows,
            len(df),
        )
        df = df[~nan_feature_mask]

    n_before = len(df)
    if filter_unlabeled:
        df = df[df[label_col].notna()]
        if len(df) < n_before:
            logger.info(
                "%s target %d: dropped %d/%d rows with a missing (NaN) label",
                dataset_name,
                target_idx,
                n_before - len(df),
                n_before,
            )
    else:
        n_unlabeled = int(df[label_col].isna().sum())
        if n_unlabeled:
            logger.info(
                "%s target %d: keeping %d/%d unlabeled (NaN-label) rows for "
                "semi-supervised splitting (filter_unlabeled=False)",
                dataset_name,
                target_idx,
                n_unlabeled,
                n_before,
            )
    if df.empty or df[label_col].notna().sum() == 0:
        logger.info(
            "Skipping %s target %d: every row has a missing label", dataset_name, target_idx
        )
        return None, None

    if problem_type == "classification":
        try:
            df = filter_rare_classes(
                df, label_col=label_col, min_samples_per_class=min_samples_per_class
            )
        except TooFewClassesError as e:
            logger.info("Skipping %s target %d: %s", dataset_name, target_idx, e)
            return None, None

    task_name = f"{dataset_name}__{target_idx}"
    _, task_obj = build_user_task(
        task_name=task_name,
        df=df,
        label_col=label_col,
        problem_type=problem_type,
        n_repeats=n_repeats,
        n_splits=n_splits,
        group_col=GROUP_COL if GROUP_COL in df.columns else None,
    )
    task_wrapper = RamanBenchTaskWrapper(task=task_obj)
    return task_name, task_wrapper
