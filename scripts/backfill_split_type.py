#!/usr/bin/env python
"""Backfill split_type metadata for an already-completed run (see issue #6).

For every key that already has predictions in a run, derive its split regime
(``"grouped"``/``"iid"``/``"stratified"``) and ``n_groups``/``largest_group_size``
without recomputing any predictions. The split itself is a pure function of
the data and config (same dataset, target, ``test_size``, ``random_state``,
``group_regression_splits``), so re-deriving it here reproduces the identical
split the run actually used -- see ``tests/test_grouped_split.py::
test_split_is_stable_across_hash_seeds``. No model is fit; this only reads
(or, for a key predating this feature, re-derives and re-caches) the
train/test split itself.

Writes ``{key}_split_info.json`` next to each seed's predictions, matching
what ``compute_predictions`` now writes going forward for new runs.

Usage
-----
    python scripts/backfill_split_type.py --config configs/v1_default.json
"""

from __future__ import annotations

import argparse
import glob
import os

from raman_bench.benchmark import configure_benchmark
from raman_bench.config import load_config
from raman_bench.predictions import _atomic_write_json
from raman_bench.seeds import get_seeds


def backfill(config: dict) -> dict:
    output_dir = config["output_dir"]
    seeds = get_seeds(config)
    stats = {"written": 0, "already_had_it": 0, "no_predictions": 0, "could_not_derive": 0}

    for seed in seeds:
        config["random_state"] = seed
        benchmark = configure_benchmark(config)

        predictions_dir = os.path.join(output_dir, f"seed_{seed}", "predictions")
        if not os.path.isdir(predictions_dir):
            continue

        for key in benchmark._key_list:
            dest = os.path.join(predictions_dir, f"{key}_split_info.json")
            if os.path.exists(dest):
                stats["already_had_it"] += 1
                continue

            # Only backfill keys real work was actually done for in this run
            # -- a key the benchmark knows about but that was never predicted
            # (e.g. skipped, or simply not reached yet) has nothing to backfill.
            if not glob.glob(os.path.join(predictions_dir, f"{key}_*_predictions.csv")):
                stats["no_predictions"] += 1
                continue

            split_info = benchmark.get_split_info(key)
            if split_info is None:
                # Predates this feature even in the benchmark's own cache --
                # re-derive (deterministic, no model fit) and cache it there
                # too, so a second backfill run (or a fresh compute_predictions
                # call against the same cache) doesn't re-derive it again.
                try:
                    data_train, data_test, split_info = benchmark._load_dataset_from_key(key)
                except Exception as e:
                    print(f"  seed {seed} / {key}: could not derive split_info ({e})")
                    stats["could_not_derive"] += 1
                    continue
                if split_info is None:
                    stats["could_not_derive"] += 1
                    continue
                benchmark._save_dataset(key, data_train, data_test, split_info)

            _atomic_write_json(dest, split_info)
            stats["written"] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Path to the run's benchmark config JSON")
    args = parser.parse_args()

    config = load_config(args.config)
    stats = backfill(config)

    print(
        f"Backfilled {stats['written']} key(s). "
        f"{stats['already_had_it']} already had split_info, "
        f"{stats['no_predictions']} had no predictions to backfill, "
        f"{stats['could_not_derive']} could not be derived."
    )


if __name__ == "__main__":
    main()
