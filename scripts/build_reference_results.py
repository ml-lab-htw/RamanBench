#!/usr/bin/env python
"""Bundle the v1 leaderboard results with the package, so new models can be compared.

Reads an aggregated ``hpo_results.csv`` (``scripts/aggregate_results.py``), keeps the
models of ``--scope`` and the targets of ``--target-list`` exactly as the leaderboard
does (:func:`raman_bench.plotting.results.load_results`), and writes into
``--output-dir``:

- ``reference_results.parquet``: one row per (task, fold, model) with ``metric_error``,
  ``metric``, ``task``, ``time_train_s``, ``time_infer_s``, ``reference``,
  ``num_instances``, ``ta_name`` and, with ``--fold-metrics``
  (``scripts/compute_fold_metrics.py``), one column per metric of
  :data:`raman_bench.fold_metrics.METRICS`. ``DUMMY`` is kept;
  :func:`raman_bench.compare.load_reference` leaves it out by default.
- ``protocol.json``: the evaluation protocol those results were produced with (folds,
  bagging, time budget, row caps), the list of tasks and each model's enabled RamanBench
  preprocessing (``preprocessing``). The preprocessing is read from the raw cached results
  (``--results-dir``, :func:`raman_bench.aggregation.preprocessing_by_model`) or from a
  ``{model: steps}`` JSON file written that way elsewhere (``--preprocessing-json``).
  :func:`raman_bench.evaluate.evaluate_estimator` runs exactly this protocol.

Usage:
    python scripts/build_reference_results.py \\
        --input results/v1/aggregated/hpo_results.csv \\
        --results-dir results/v1/data \\
        --fold-metrics results/v1/aggregated/fold_metrics.parquet \\
        --output-dir src/raman_bench/data/precomputed/v1
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
from pathlib import Path

from raman_bench.plotting import models as model_info
from raman_bench.plotting.results import DEFAULT_SCOPE, DEFAULT_TARGET_LIST, load_results

logger = logging.getLogger(__name__)

# Settings of every v1 run that the scope file doesn't carry.
MIN_SAMPLES_PER_CLASS = 9
REFERENCE_MODEL = "RF"


def build_protocol(scope: dict, targets: list[dict], results, preprocessing: dict[str, str]) -> dict:
    """The protocol a new model has to follow to be comparable with *results*."""
    task_type = results.groupby("dataset")["task"].first()
    metric = results.groupby("dataset")["metric"].first()
    n_folds = results.groupby("dataset")["fold"].nunique()
    tasks = []
    for t in targets:
        key = f"{t['dataset']}__{t['target_idx']}"
        if t["excluded"] or key not in task_type:
            continue
        tasks.append(
            {
                "task": key,
                "dataset": t["dataset"],
                "target_idx": t["target_idx"],
                "target_name": t.get("target_name"),
                "problem_type": task_type[key],
                "metric": metric[key],
                "num_instances": t["num_instances"],
                "n_repeats": t["n_repeats"],
                "n_folds": int(n_folds[key]),
            }
        )
    return {
        "version": "v1",
        "created": datetime.date.today().isoformat(),
        "n_splits": scope["n_splits"],
        "num_bag_folds": scope["num_bag_folds"],
        "time_limit": scope["time_limit"],
        "num_random_configs": scope["num_random_configs"],
        "time_limit_overrides": scope.get("time_limit_overrides", {}),
        "max_train_samples_overrides": scope.get("max_train_samples_overrides", {}),
        "model_max_train_samples_overrides": scope.get("model_max_train_samples_overrides", {}),
        "min_samples_per_class": MIN_SAMPLES_PER_CLASS,
        "reference_model": REFERENCE_MODEL,
        "models": sorted(results["model"].unique()),
        "preprocessing": preprocessing,
        "tasks": tasks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default="results/v1/aggregated/hpo_results.csv")
    parser.add_argument("--output-dir", default="src/raman_bench/data/precomputed/v1")
    parser.add_argument("--scope", default=str(DEFAULT_SCOPE), help="Its model list is what gets bundled")
    parser.add_argument("--target-list", default=str(DEFAULT_TARGET_LIST))
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--results-dir", help="Raw results (results.pkl) to read each model's preprocessing from")
    source.add_argument("--preprocessing-json", help="{model: enabled preprocessing} instead of --results-dir")
    parser.add_argument(
        "--exclude-models", nargs="*", default=[], help="Model keys left out (default: none, DUMMY is kept)"
    )
    parser.add_argument(
        "--fold-metrics", help="Per-fold metrics (scripts/compute_fold_metrics.py) to add as columns"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    results = load_results(
        args.input, scope=args.scope, target_list=args.target_list, exclude_models=tuple(args.exclude_models)
    )
    scope = json.loads(Path(args.scope).read_text())
    targets = json.loads(Path(args.target_list).read_text())
    if args.results_dir:
        from raman_bench.aggregation import preprocessing_by_model

        found = preprocessing_by_model(args.results_dir)
    else:
        found = json.loads(Path(args.preprocessing_json).read_text())
    preprocessing = {}
    for model in sorted(results["model"].unique()):
        if model in model_info.REFERENCE_MODELS:
            preprocessing[model] = "AutoGluon presets"
        elif model in found:
            preprocessing[model] = found[model]
        else:
            logger.warning("No preprocessing found for %s", model)
    protocol = build_protocol(scope, targets, results, preprocessing)

    if args.fold_metrics:
        import pandas as pd

        from raman_bench.compare import attach_fold_metrics
        from raman_bench.fold_metrics import METRIC_COLUMNS, to_error

        per_fold = pd.read_parquet(args.fold_metrics)
        results = attach_fold_metrics(results, per_fold)
        no_metrics = results[results[METRIC_COLUMNS].isna().all(axis=1)]
        if len(no_metrics):
            logger.warning(
                "%d row(s) got no per-fold metrics: %s", len(no_metrics), no_metrics["model"].value_counts().to_dict()
            )
        # The protocol metric computed from the predictions must equal the stored error.
        for column in ("roc_auc", "log_loss", "rmse"):
            rows = (results["metric"] == column) & results[column].notna()
            gap = (to_error(results.loc[rows, column], column) - results.loc[rows, "metric_error"]).abs()
            if (gap > 1e-6).any():
                logger.warning("%d %s value(s) differ from metric_error", int((gap > 1e-6).sum()), column)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = results.sort_values(["dataset", "fold", "model"]).reset_index(drop=True)
    results.to_parquet(out / "reference_results.parquet", index=False, compression="zstd")
    (out / "protocol.json").write_text(json.dumps(protocol, indent=1) + "\n")
    logger.info(
        "Wrote %d rows (%d models, %d tasks) and the protocol to %s",
        len(results), results["model"].nunique(), len(protocol["tasks"]), out,
    )


if __name__ == "__main__":
    main()
