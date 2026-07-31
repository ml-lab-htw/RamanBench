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

Usage:
    python scripts/aggregate_results.py --results-dir results/v1/data --output-dir results/v1/aggregated
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import pickle

import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


def scan_cached_results(results_dir: str) -> list[dict]:
    """Load every cached results.pkl under results_dir."""
    pattern = os.path.join(results_dir, "*", "*", "*", "results.pkl")
    paths = sorted(glob.glob(pattern))
    results_lst = []
    for path in paths:
        try:
            with open(path, "rb") as f:
                results_lst.append(pickle.load(f))
        except Exception as e:
            logger.warning("Failed to load %s: %s", path, e)
    return results_lst


def build_task_metadata(results_lst: list[dict]) -> pd.DataFrame:
    """One row per distinct task (dataset,target key) seen across results_lst."""
    tasks: dict[str, tuple[int, str]] = {}
    for out in results_lst:
        tm = out["task_metadata"]
        tasks[tm["name"]] = (tm["tid"], out["problem_type"])
    rows = [
        {
            "tid": tid,
            "dataset": name,
            "task_type": "Supervised Regression" if problem_type == "regression" else "Supervised Classification",
            "name": name,
        }
        for name, (tid, problem_type) in tasks.items()
    ]
    return pd.DataFrame(rows)


def aggregate(results_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    from tabarena.nips2025_utils.end_to_end import EndToEnd

    results_lst = scan_cached_results(results_dir)
    if not results_lst:
        logger.warning("No cached results found under %s", results_dir)
        return pd.DataFrame(), pd.DataFrame()

    task_metadata = build_task_metadata(results_lst)
    logger.info(
        "Found %d cached result(s) across %d task(s): %s",
        len(results_lst), len(task_metadata), sorted(task_metadata["dataset"].tolist()),
    )

    end_to_end = EndToEnd.from_raw(results_lst=results_lst, task_metadata=task_metadata, cache=False, cache_raw=False)
    results = end_to_end.to_results()
    model_results = results.model_results if results.model_results is not None else pd.DataFrame()
    hpo_results = results.hpo_results if results.hpo_results is not None else pd.DataFrame()
    return model_results, hpo_results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default="results/v1/data")
    parser.add_argument("--output-dir", default="results/v1/aggregated")
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


if __name__ == "__main__":
    main()
