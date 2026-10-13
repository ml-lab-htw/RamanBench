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

``build_task`` takes an already-loaded dataframe/dataset rather than loading one itself:
``run_one`` needs the raw dataframe's shape *before* splitting to run its own
model-specific compatibility checks (``MAX_FEATURES_MODELS``/``VRAM_CAPPED_MODELS``,
which don't apply to the model-agnostic AutoGluon baseline runner at all) --
threading those checks through this helper would couple it back to
``model_key``-specific concerns it's deliberately free of. Loading itself is the
separate :func:`load_dataframe`.

:func:`load_dataframe`, :func:`bag_experiment_kwargs` and :func:`run_cached` are the rest
of a v1 run (load, configure the bagged TabArena experiment, run and cache one fold),
shared by ``run_one`` and :func:`raman_bench.evaluate.evaluate_estimator` so a
scikit-learn estimator evaluated for comparison runs exactly like a leaderboard model.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

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


def write_hardware_info(cache_path: str) -> None:
    """Write ``gpu.json`` next to ``results.pkl`` recording which device actually ran
    this task's fit -- called by both runner scripts immediately after a real
    ``experiment.run(...)`` (never on a cache-hit early-return, since no compute
    happened there, so there is nothing new to attribute to hardware).

    Reads the device directly from torch, so the hardware is recorded next to the
    result itself and never has to be reconstructed from an external tracker later.
    """
    try:
        import torch

        if torch.cuda.is_available():
            info = {"gpu": torch.cuda.get_device_name(0), "gpu_count": torch.cuda.device_count()}
        elif torch.backends.mps.is_available():
            info = {"gpu": "Apple MPS", "gpu_count": 1}
        else:
            info = {"gpu": None, "gpu_count": 0}
    except Exception:
        logger.warning("Could not determine compute device for hardware info", exc_info=True)
        info = {"gpu": None, "gpu_count": None, "note": "device detection failed"}

    (Path(cache_path) / "gpu.json").write_text(json.dumps(info, indent=2))


def prepare_task_dataframe(
    *,
    dataset_name: str,
    target_idx: int,
    df: pd.DataFrame,
    raw_targets: np.ndarray,
    problem_type: str,
    sample_idx: np.ndarray | None = None,
    min_samples_per_class: int = 9,
    filter_unlabeled: bool = True,
) -> pd.DataFrame | None:
    """The rows of one (dataset, target) that enter the cross-validation, as :func:`build_task` uses them.

    Adds inferred regression group ids, then drops rows with a NaN feature, (by default)
    rows without a label and, for classification, classes with fewer than
    *min_samples_per_class* rows. The index is kept, so it still names the
    ``dataset.to_dataframe(target_idx)`` row. Returns ``None`` when the target is
    skipped (no labelled rows, or fewer than 2 classes left). Arguments as for
    :func:`build_task`.
    """
    label_col = df.columns[-1]

    if GROUP_COL not in df.columns and problem_type == "regression":
        targets_for_grouping = raw_targets[sample_idx] if sample_idx is not None else raw_targets
        inferred = infer_group_ids_from_targets(targets_for_grouping)
        if inferred is not None:
            # Before the label: callers take the last column as the label.
            df.insert(len(df.columns) - 1, GROUP_COL, inferred)
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
        return None

    if problem_type == "classification":
        try:
            df = filter_rare_classes(
                df, label_col=label_col, min_samples_per_class=min_samples_per_class
            )
        except TooFewClassesError as e:
            logger.info("Skipping %s target %d: %s", dataset_name, target_idx, e)
            return None

    return df


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
    df = prepare_task_dataframe(
        dataset_name=dataset_name,
        target_idx=target_idx,
        df=df,
        raw_targets=raw_targets,
        problem_type=problem_type,
        sample_idx=sample_idx,
        min_samples_per_class=min_samples_per_class,
        filter_unlabeled=filter_unlabeled,
    )
    if df is None:
        return None, None
    label_col = df.columns[-1]

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


def load_dataframe(
    dataset_name: str,
    target_idx: int,
    *,
    max_train_samples: int | None = None,
    cache_dir: str = ".cache_v1",
    use_mirror: bool = True,
    mirror_repo: str = "HTW-KI-Werkstatt/RamanBench",
):
    """Load one (dataset, target) as a dataframe, the way every v1 runner does.

    Returns ``(dataset, df, sample_idx, problem_type)``: the ``raman_data`` dataset,
    ``dataset.to_dataframe(target_idx)`` (randomly subsampled to *max_train_samples*
    rows with ``random_state=0`` when it has more), the kept row index (``None`` when
    not subsampled) and ``"classification"``/``"regression"``. Pass all of it on to
    :func:`build_task`.

    Uses ``RamanBenchmark``'s mirror-first loading (with fallback to the original
    source); ``use_mirror=False`` forces the original source.
    """
    from raman_data import TASK_TYPE

    from raman_bench.benchmark import RamanBenchmark

    bench = RamanBenchmark(
        dataset_names_classification=[],
        dataset_names_regression=[],
        cache_dir=cache_dir,
        use_mirror=use_mirror,
        mirror_repo=mirror_repo,
    )
    dataset = bench._load_raman_dataset(dataset_name)
    if dataset is None:
        raise RuntimeError(f"Failed to load dataset {dataset_name!r}")

    df = dataset.to_dataframe(target_idx)
    sample_idx = None
    if max_train_samples is not None and len(df) > max_train_samples:
        n_before = len(df)
        df = df.sample(n=max_train_samples, random_state=0)
        sample_idx = df.index.to_numpy()
        logger.info(
            "Subsampled %s: %d -> %d rows (max_train_samples=%d); this is a real change to "
            "what is measured for this dataset, not a performance-neutral optimisation.",
            dataset_name, n_before, len(df), max_train_samples,
        )
    problem_type = "classification" if dataset.task_type == TASK_TYPE.Classification else "regression"
    return dataset, df, sample_idx, problem_type


def bag_experiment_kwargs(
    *,
    num_random_configs: int,
    time_limit: float,
    num_bag_folds: int,
    scratch_dir: str | None = None,
    verbosity: int | None = None,
) -> dict:
    """Keyword arguments for ``ConfigGenerator.generate_all_bag_experiments``.

    The settings every v1 result was produced with, apart from the numbers passed in:

    - TabArena's ``AGModelBagExperiment`` takes the bagging count from a
      ``ValidationProtocol``; ``num_bag_folds`` stays at the configured value
      regardless of dataset size (2026-09-25 decision, no small-dataset scaling).
    - ``require_warmup=False``: TabArena's pre-flight "dummy fit" fails for every
      RamanBench custom model (they are not in the model registry TabArena's warm-up
      was built against), so it is disabled.
    - ``add_seed="fold-config-wise"``: TabArena's production default, so each internal
      bag fold and each HPO config gets its own random seed.

    ``scratch_dir`` pins AutoGluon's predictor path (default: a timestamped directory
    under the working directory), so a cluster wrapper can clean it up. ``verbosity`` is
    AutoGluon's (0-4, default 2).
    """
    from pathlib import Path

    from tabarena.benchmark.validation_protocol import ValidationProtocol

    kwargs = dict(
        num_random_configs=num_random_configs,
        time_limit=time_limit,
        validation_protocol=ValidationProtocol(num_bag_folds=num_bag_folds),
        fold_fitting_strategy="sequential_local",
        experiment_kwargs={"require_warmup": False},
        add_seed="fold-config-wise",
    )
    init_kwargs = {}
    if scratch_dir is not None:
        Path(scratch_dir).mkdir(parents=True, exist_ok=True)
        init_kwargs["path"] = scratch_dir
    if verbosity is not None:
        init_kwargs["verbosity"] = verbosity
    if init_kwargs:
        kwargs["method_kwargs"] = {"init_kwargs": init_kwargs}
    return kwargs


def run_cached(
    experiment,
    *,
    task_name: str,
    task_wrapper: RamanBenchTaskWrapper,
    repeat: int,
    fold: int,
    results_dir: str,
    experiment_dir_name: str | None = None,
    force_recompute: bool = False,
) -> dict:
    """Run one TabArena experiment on one fold, caching the result.

    The result lands in ``results_dir/<experiment_dir_name>/<task_name>/<repeat>_<fold>/results.pkl``
    (``experiment_dir_name`` defaults to ``experiment.name``, e.g. ``PLS_c1_BAG_L1``), which is
    the layout :func:`raman_bench.aggregation.aggregate` reads, with a ``gpu.json`` next to
    it (:func:`write_hardware_info`) after a real run.

    A result already cached there is returned as-is, even if it was fit under a different
    ``num_bag_folds``/``time_limit`` (the path does not encode those): an existing result
    stands as final, and TabArena would otherwise refuse the cached run's mismatching
    validation protocol. ``force_recompute`` overwrites it instead, which is meant for a
    dataset-definition fix that makes the cached result meaningless.
    """
    import os
    from pathlib import Path

    from tabarena.utils.cache import CacheFunctionPickle

    cache_path = os.path.join(results_dir, experiment_dir_name or experiment.name, task_name, f"{repeat}_{fold}")
    Path(cache_path).mkdir(parents=True, exist_ok=True)
    cacher = CacheFunctionPickle(cache_name="results", cache_path=cache_path, include_self_in_call=True)
    if cacher.exists and not force_recompute:
        out = cacher.load_cache()
        logger.info(
            "%s on %s repeat=%d fold=%d: using existing cached result at %s "
            "(possibly fit under a different num_bag_folds/time_limit -- not refit)",
            experiment.name, task_name, repeat, fold, cache_path,
        )
        logger.info("Done: metric_error=%s", out.get("metric_error"))
        return out

    logger.info("Running %s on %s repeat=%d fold=%d -> %s", experiment.name, task_name, repeat, fold, cache_path)
    out = experiment.run(
        task=task_wrapper,
        fold=fold,
        repeat=repeat,
        task_name=task_name,
        # The task's canonical cache identifier; TabArena uses it to scope its
        # text-embedding cache. Where results land is decided by cache_path above.
        cache_task_key=task_name,
        cacher=cacher,
        ignore_cache=force_recompute,
    )
    write_hardware_info(cache_path)
    logger.info("Done: metric_error=%s", out.get("metric_error"))
    return out
