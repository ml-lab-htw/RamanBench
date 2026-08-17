#!/usr/bin/env python
"""Aggregate cached v1 per-config results into a combined leaderboard-ready table.

Scans ``{results_dir}/{experiment_name}/{task_name}/{repeat}_{fold}/results.pkl``
caches (as written by ``scripts/run_experiment.py``) and calls TabArena's own
``EndToEnd.from_raw`` once, across every cached result, to recycle each
model's raw per-config runs into default/tuned/tuned+ensemble rows -- reusing
TabArena's machinery directly rather than re-deriving it (see the v1 refactor
plan's Phase 3).

Writes two CSVs:
  - ``model_results.csv`` -- one row per (task, fold, raw config), e.g.
    ``PLS_c1_BAG_L1``, ``PLS_r7_BAG_L1``, ...
  - ``hpo_results.csv``   -- one row per (task, model, {default,tuned,tuned+ensemble}),
    the recycled result. "tuned"/"tuned+ensemble" only differ from "default"
    when more than the default (``_c1``) config was actually run for that
    model on that task.

Optionally also applies the TabArena-derived trivial-dataset filter
(``raman_bench.filters``, off by default) with ``--trivial-filter``: flags
(dataset, target) keys whose results carry no discriminative signal (a model
scores perfect on every fold, or 2+ models tie for the top score on every
fold) and, when any are found, additionally writes
``model_results_nontrivial.csv``/``hpo_results_nontrivial.csv`` (the same
tables with flagged keys dropped) alongside a ``trivial_keys.csv``
(key, reason) listing what was excluded and why. The unfiltered
``model_results.csv``/``hpo_results.csv`` are always written regardless, so
this is purely additive.

Usage:
    python scripts/aggregate_results.py --results-dir results/v1/data --output-dir results/v1/aggregated

    # Also flag and drop trivial (dataset, target) keys:
    python scripts/aggregate_results.py --trivial-filter --trivial-filter-min-tie-models 2
"""

from __future__ import annotations

import argparse
import glob
import logging
import os

import pandas as pd
from tabarena.utils.pickle_utils import load_pickle

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


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


def apply_trivial_filter(
    model_results: pd.DataFrame,
    hpo_results: pd.DataFrame,
    output_dir: str,
    args: argparse.Namespace,
) -> None:
    """Compute and write the trivial-dataset filter's output (see module docstring).

    No-op (nothing computed, nothing written) unless ``args.trivial_filter`` is set --
    keeps the default ``aggregate_results.py`` run byte-identical to before this was
    added.
    """
    from raman_bench.filters import compute_trivial_keys, filter_trivial_keys

    if not args.trivial_filter:
        return

    flagged = compute_trivial_keys(
        hpo_results,
        perfect_clf=args.trivial_filter_perfect_clf,
        perfect_reg=args.trivial_filter_perfect_reg,
        min_tie_models=args.trivial_filter_min_tie_models,
        tie_decimals=args.trivial_filter_tie_decimals,
        exclude_models=args.trivial_filter_exclude_model,
    )
    trivial_keys = set(flagged.keys())
    if flagged:
        logger.info("[trivial-filter] excluding %d dataset key(s):", len(flagged))
        for key, reason in sorted(flagged.items()):
            logger.info("  - %s  (%s)", key, reason)

    trivial_keys_path = os.path.join(output_dir, "trivial_keys.csv")
    # Always write trivial_keys.csv when the filter is enabled, even if empty,
    # so a downstream consumer can distinguish "filter ran, found nothing" from
    # "filter never ran".
    trivial_df = pd.DataFrame(sorted(flagged.items()), columns=["dataset", "reason"])
    trivial_df.to_csv(trivial_keys_path, index=False)
    logger.info("Wrote %d trivial key(s) to %s", len(trivial_keys), trivial_keys_path)

    if not trivial_keys:
        return

    model_nontrivial = filter_trivial_keys(model_results, trivial_keys)
    hpo_nontrivial = filter_trivial_keys(hpo_results, trivial_keys)
    model_nontrivial_path = os.path.join(output_dir, "model_results_nontrivial.csv")
    hpo_nontrivial_path = os.path.join(output_dir, "hpo_results_nontrivial.csv")
    model_nontrivial.to_csv(model_nontrivial_path, index=False)
    hpo_nontrivial.to_csv(hpo_nontrivial_path, index=False)
    logger.info("Wrote %d row(s) to %s", len(model_nontrivial), model_nontrivial_path)
    logger.info("Wrote %d row(s) to %s", len(hpo_nontrivial), hpo_nontrivial_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default="results/v1/data")
    parser.add_argument("--output-dir", default="results/v1/aggregated")
    parser.add_argument(
        "--trivial-filter",
        action="store_true",
        help=(
            "Also flag TabArena-trivial (dataset, target) keys (raman_bench.filters, "
            "off by default) and write *_nontrivial.csv / trivial_keys.csv."
        ),
    )
    parser.add_argument(
        "--trivial-filter-perfect-clf",
        type=float,
        default=0.0,
        help="Criterion-1 threshold for classification keys: metric_error <= this counts as perfect (default 0.0).",
    )
    parser.add_argument(
        "--trivial-filter-perfect-reg",
        type=float,
        default=0.0,
        help="Criterion-1 threshold for regression keys: metric_error <= this counts as perfect (default 0.0).",
    )
    parser.add_argument(
        "--trivial-filter-min-tie-models",
        type=int,
        default=2,
        help="Criterion-2: how many models must tie for the top score on every fold (default 2).",
    )
    parser.add_argument(
        "--trivial-filter-tie-decimals",
        type=int,
        default=4,
        help="Round metric_error to this many decimals before tie comparison (default 4).",
    )
    parser.add_argument(
        "--trivial-filter-exclude-model",
        action="append",
        default=[],
        help="Model (ta_name) to exclude before evaluating either criterion; repeatable.",
    )
    args = parser.parse_args()

    model_results, hpo_results = aggregate(args.results_dir)

    os.makedirs(args.output_dir, exist_ok=True)
    model_results_path = os.path.join(args.output_dir, "model_results.csv")
    hpo_results_path = os.path.join(args.output_dir, "hpo_results.csv")
    model_results.to_csv(model_results_path, index=False)
    hpo_results.to_csv(hpo_results_path, index=False)
    logger.info("Wrote %d row(s) to %s", len(model_results), model_results_path)
    logger.info("Wrote %d row(s) to %s", len(hpo_results), hpo_results_path)

    if not hpo_results.empty:
        with pd.option_context("display.max_columns", None, "display.width", 200):
            logger.info("\n%s", hpo_results.to_string(index=False))

    apply_trivial_filter(model_results, hpo_results, args.output_dir, args)


if __name__ == "__main__":
    main()
