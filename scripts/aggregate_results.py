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

Optionally also applies the "not learnable" filter (``raman_bench.filters``,
off by default) with ``--learnability-filter``: flags (dataset, target) keys
where no in-scope model meaningfully beats the ``Dummy`` baseline, and writes
``model_results_learnable.csv``/``hpo_results_learnable.csv`` +
``unlearnable_keys.csv``. This is the periodic "learnability sweep" re-check
described in ``configs/v1/EXCLUDED_TARGETS.md`` -- intended to be run now and
then against a curated top-model results set, not on every routine sweep, to
see whether a key previously excluded as not-learnable has since been beaten
by a newer/better model.

Before aggregating, every result's stored test indices are checked against its task's
canonical outer fold (``raman_bench.folds.check_result_folds``; the datasets are
loaded from the mirror for that). A result tested on other rows (from before a change
to the row cleaning, the grouping or the splitter) is left out, since the scoring
compares fold k across models and assumes it is the same spectra; a fold on a per-model
row sample (``model_max_train_samples_overrides``) counts as matching. Every result's
status goes to ``fold_check.csv``. ``--keep-mismatched-folds`` keeps the mismatches in
(still listed), ``--no-fold-check`` skips the check.

Usage:
    python scripts/aggregate_results.py --results-dir results/v1/data --output-dir results/v1/aggregated

    # Also flag and drop trivial (dataset, target) keys:
    python scripts/aggregate_results.py --trivial-filter --trivial-filter-min-tie-models 2

    # Periodic learnability re-check (see configs/v1/EXCLUDED_TARGETS.md):
    python scripts/aggregate_results.py --learnability-filter
"""

from __future__ import annotations

import argparse
import logging
import os

import pandas as pd

from raman_bench.aggregation import aggregate_results, scan_cached_results

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


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


def apply_learnability_filter(
    model_results: pd.DataFrame,
    hpo_results: pd.DataFrame,
    output_dir: str,
    args: argparse.Namespace,
) -> None:
    """Compute and write the "not learnable" filter's output (see
    ``raman_bench.filters.compute_unlearnable_keys`` and
    ``configs/v1/EXCLUDED_TARGETS.md``).

    No-op (nothing computed, nothing written) unless ``args.learnability_filter``
    is set -- keeps the default run byte-identical to before this was added.
    This is the "learnability sweep" re-check mechanism: run periodically (not
    every routine sweep) against a curated top-model results set to see if a
    key previously flagged "not learnable" has since been beaten.
    """
    from raman_bench.filters import compute_unlearnable_keys, filter_trivial_keys

    if not args.learnability_filter:
        return

    flagged = compute_unlearnable_keys(
        hpo_results,
        min_dummy_margin=args.learnability_filter_min_dummy_margin,
        dummy_model=args.learnability_filter_dummy_model,
    )
    unlearnable_keys = set(flagged.keys())
    if flagged:
        logger.info("[learnability-filter] excluding %d dataset key(s):", len(flagged))
        for key, reason in sorted(flagged.items()):
            logger.info("  - %s  (%s)", key, reason)

    unlearnable_keys_path = os.path.join(output_dir, "unlearnable_keys.csv")
    unlearnable_df = pd.DataFrame(sorted(flagged.items()), columns=["dataset", "reason"])
    unlearnable_df.to_csv(unlearnable_keys_path, index=False)
    logger.info("Wrote %d unlearnable key(s) to %s", len(unlearnable_keys), unlearnable_keys_path)

    if not unlearnable_keys:
        return

    model_learnable = filter_trivial_keys(model_results, unlearnable_keys)
    hpo_learnable = filter_trivial_keys(hpo_results, unlearnable_keys)
    model_learnable_path = os.path.join(output_dir, "model_results_learnable.csv")
    hpo_learnable_path = os.path.join(output_dir, "hpo_results_learnable.csv")
    model_learnable.to_csv(model_learnable_path, index=False)
    hpo_learnable.to_csv(hpo_learnable_path, index=False)
    logger.info("Wrote %d row(s) to %s", len(model_learnable), model_learnable_path)
    logger.info("Wrote %d row(s) to %s", len(hpo_learnable), hpo_learnable_path)


def check_folds(results_lst: list[dict], args: argparse.Namespace) -> list[dict]:
    """Write ``fold_check.csv`` and return *results_lst* without mismatched folds (see module docstring)."""
    from raman_bench.folds import check_result_folds

    check = check_result_folds(results_lst, cache_dir=args.cache_dir)
    path = os.path.join(args.output_dir, "fold_check.csv")
    check.to_csv(path, index=False)
    logger.info("Fold check (%s): %s", path, check["status"].value_counts().to_dict())
    bad = (check["status"] == "mismatch").to_numpy()
    if not bad.any():
        return results_lst
    per_model = check[bad].groupby("framework").size().sort_values(ascending=False)
    logger.warning(
        "%d result(s) were tested on other rows than their canonical fold%s: %s",
        int(bad.sum()),
        "" if args.keep_mismatched_folds else " and are left out",
        per_model.to_dict(),
    )
    if args.keep_mismatched_folds:
        return results_lst
    return [r for r, b in zip(results_lst, bad) if not b]


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
    parser.add_argument(
        "--learnability-filter",
        action="store_true",
        help=(
            "Also flag 'not learnable' (dataset, target) keys -- no in-scope model "
            "meaningfully beats Dummy (raman_bench.filters, off by default) -- and "
            "write *_learnable.csv / unlearnable_keys.csv. This is the periodic "
            "'learnability sweep' re-check, see configs/v1/EXCLUDED_TARGETS.md."
        ),
    )
    parser.add_argument(
        "--learnability-filter-min-dummy-margin",
        type=float,
        default=0.05,
        help="How much lower (better) the best model's mean metric_error must be than "
             "Dummy's for a key to count as learnable (default 0.05).",
    )
    parser.add_argument(
        "--learnability-filter-dummy-model",
        default="DUMMY",
        help="Model (ta_name) treated as the baseline (default 'DUMMY').",
    )
    parser.add_argument(
        "--no-fold-check",
        action="store_true",
        help="Don't check the results' test rows against the canonical folds.",
    )
    parser.add_argument(
        "--keep-mismatched-folds",
        action="store_true",
        help="Aggregate results tested on other rows than their canonical fold too (still listed).",
    )
    parser.add_argument(
        "--cache-dir", default=".cache_v1", help="Dataset cache for the fold check (default .cache_v1)."
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    results_lst = scan_cached_results(args.results_dir)
    if not results_lst:
        logger.warning("No cached results found under %s", args.results_dir)
    elif not args.no_fold_check:
        results_lst = check_folds(results_lst, args)
    model_results, hpo_results = aggregate_results(results_lst) if results_lst else (pd.DataFrame(), pd.DataFrame())

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
    apply_learnability_filter(model_results, hpo_results, args.output_dir, args)


if __name__ == "__main__":
    main()
