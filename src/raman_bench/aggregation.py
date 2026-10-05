"""Turn cached v1 ``results.pkl`` files into TabArena result tables.

Every v1 runner (``scripts/run_experiment.py``, :func:`raman_bench.evaluate.evaluate_estimator`)
caches one ``results.pkl`` per (model config, task, repeat, fold) under
``{results_dir}/{experiment_name}/{task_name}/{repeat}_{fold}/``. :func:`aggregate` loads
them all and calls TabArena's own ``EndToEnd.from_raw`` once to recycle each model's raw
per-config runs into default/tuned/tuned+ensemble rows, so the scoring matches TabArena's.
``scripts/aggregate_results.py`` is the command-line wrapper.
"""

from __future__ import annotations

import glob
import logging
import os

import pandas as pd
from tabarena.utils.pickle_utils import load_pickle

logger = logging.getLogger(__name__)


def scan_cached_results(results_dir: str) -> list[dict]:
    """Load every cached results.pkl under results_dir.

    Uses ``tabarena``'s own ``load_pickle`` (rather than a raw ``pickle.load``)
    since ``CacheFunctionPickle`` gzip-compresses its cache writes by default as
    of a later ``tabarena`` version than when this was first written;
    ``load_pickle`` transparently handles both compressed and uncompressed
    files (matching ``CacheFunctionPickle.load_cache``'s own read path).
    """
    pattern = os.path.join(results_dir, "*", "*", "*", "results.pkl")
    paths = sorted(glob.glob(pattern))
    results_lst = []
    for path in paths:
        try:
            results_lst.append(load_pickle(path))
        except Exception as e:
            logger.warning("Failed to load %s: %s", path, e)
    return results_lst


def build_task_metadata(results_lst: list[dict]) -> pd.DataFrame:
    """One row per distinct task (dataset,target key) seen across results_lst.

    ``TaskMetadataCollection.from_legacy_df`` (a later tabarena version than
    when this was first written) now requires a fuller set of columns than
    the tid/dataset/task_type/name this used to build. ``n_folds``/``n_repeats``
    are derived accurately from what's actually been run per task (max
    fold/repeat index + 1); the per-dataset stats (``n_features``, ``n_classes``,
    ``NumberOfInstances``, ``n_samples_train_per_fold``, ``n_samples_test_per_fold``)
    aren't tracked anywhere in the results cache, so are filled with a clearly-
    marked placeholder (0) -- ``from_legacy_df`` is documented as a lossy shim
    tolerating exactly this kind of sparse data; only ``model_results``/
    ``hpo_results`` (this script's actual output) are needed, not dataset stats.
    """
    tasks: dict[str, dict] = {}
    for out in results_lst:
        tm = out["task_metadata"]
        name = tm["name"]
        entry = tasks.setdefault(
            name,
            {"tid": tm["tid"], "problem_type": out["problem_type"], "max_fold": -1, "max_repeat": -1},
        )
        entry["max_fold"] = max(entry["max_fold"], tm["fold"])
        entry["max_repeat"] = max(entry["max_repeat"], tm["repeat"])
    rows = [
        {
            "tid": entry["tid"],
            "dataset": name,
            "name": name,
            "problem_type": entry["problem_type"],
            "task_type": (
                "Supervised Regression"
                if entry["problem_type"] == "regression"
                else "Supervised Classification"
            ),
            "n_folds": entry["max_fold"] + 1,
            "n_repeats": entry["max_repeat"] + 1,
            "n_features": 0,
            "n_classes": 0 if entry["problem_type"] == "regression" else 2,
            "NumberOfInstances": 0,
            "n_samples_train_per_fold": 0,
            "n_samples_test_per_fold": 0,
        }
        for name, entry in tasks.items()
    ]
    return pd.DataFrame(rows)


def aggregate(results_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    from tabarena.benchmark.task.metadata.collection import TaskMetadataCollection
    from tabarena.end_to_end.end_to_end import EndToEnd

    results_lst = scan_cached_results(results_dir)
    if not results_lst:
        logger.warning("No cached results found under %s", results_dir)
        return pd.DataFrame(), pd.DataFrame()

    task_metadata = build_task_metadata(results_lst)
    logger.info(
        "Found %d cached result(s) across %d task(s): %s",
        len(results_lst), len(task_metadata), sorted(task_metadata["dataset"].tolist()),
    )
    # EndToEnd.from_raw no longer accepts a legacy task_metadata DataFrame directly
    # (a later tabarena version than when this was first written) -- it needs a
    # TaskMetadataCollection now. from_legacy_df is a lossy shim (this repo's own
    # task_metadata frame never had the richer native fields anyway) but sufficient:
    # what's actually consumed downstream is just the (tid, dataset, task_type, name)
    # identity each cached result is keyed by.
    task_metadata = TaskMetadataCollection.from_legacy_df(task_metadata)

    # from_raw now returns EndToEndResults directly (a later tabarena version than when
    # this was first written; there used to be a separate EndToEnd.to_results() step and
    # .model_results/.hpo_results properties). get_results(use_model_results=...) is the
    # unified replacement: True -> raw per-config rows (the old .model_results), False ->
    # the recycled default/tuned/tuned+ensemble rows (the old .hpo_results).
    end_to_end_results = EndToEnd.from_raw(
        results_lst=results_lst, task_metadata=task_metadata, cache=False, cache_raw=False
    )
    model_results = end_to_end_results.get_results(use_model_results=True)
    hpo_results = end_to_end_results.get_results(use_model_results=False)
    return model_results, hpo_results


# Switches outside mixin._PREP_STEP_DEFINITIONS.
_EXTRA_PREP_SWITCHES = {"prep_ensemble_enabled": "ensemble", "prep_scaling_enabled": "scaling"}


def _prep_labels() -> dict[str, str]:
    """``prep_*_enabled`` hyperparameter -> step name (e.g. ``prep_bl_enabled`` -> ``baseline_correction``)."""
    from raman_bench.preprocessing.mixin import _PREP_STEP_DEFINITIONS

    labels = {
        p: step for step, d in _PREP_STEP_DEFINITIONS.items() for p in d["defaults"] if p.endswith("_enabled")
    }
    return {**labels, **_EXTRA_PREP_SWITCHES}


def enabled_preprocessing(result: dict) -> str | None:
    """RamanBench preprocessing steps a cached result was fit with, e.g. ``"baseline_correction, snv"``.

    Read from the first bagged child's hyperparameters; ``"none"`` when every step was off,
    ``None`` when the result carries no per-model hyperparameters (e.g. a whole AutoGluon
    predictor).
    """
    children = result.get("method_metadata", {}).get("info", {}).get("children_info") or {}
    if not children:
        return None
    hp = next(iter(children.values())).get("hyperparameters") or {}
    labels = _prep_labels()
    steps = [labels.get(k, k[len("prep_") : -len("_enabled")]) for k, v in hp.items()
             if k.startswith("prep_") and k.endswith("_enabled") and v]  # fmt: skip
    return ", ".join(steps) if steps else "none"


def preprocessing_by_model(results_dir: str) -> dict[str, str]:
    """``{model key: enabled preprocessing}`` from one cached result per experiment directory.

    The model key is TabArena's ``model_type`` (the ``config_type`` the leaderboard ranks
    by). Only default-config directories (``*_c1_BAG_L1``) are read.
    """
    out: dict[str, str] = {}
    for exp_dir in sorted(glob.glob(os.path.join(results_dir, "*_c1_BAG_L1"))):
        path = next(iter(sorted(glob.glob(os.path.join(exp_dir, "*", "*", "results.pkl")))), None)
        if path is None:
            continue
        try:
            result = load_pickle(path)
        except Exception as e:
            logger.warning("Failed to load %s: %s", path, e)
            continue
        key = result.get("method_metadata", {}).get("model_type")
        prep = enabled_preprocessing(result)
        if key and prep is not None:
            out[key] = prep
    return out
