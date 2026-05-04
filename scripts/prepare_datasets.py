#!/usr/bin/env python
"""Pre-cache all benchmark dataset splits to disk.

Running this script before the main benchmark pipeline avoids re-loading and
re-splitting datasets on each model run, significantly speeding up parallel
execution on compute clusters.

All splits are cached in ``<cache_dir>/datasets_processed/seed_N/`` as pickle
files.  The benchmark pipeline loads them automatically on subsequent runs.

Usage
-----
::

    python scripts/prepare_datasets.py
    python scripts/prepare_datasets.py --config configs/benchmark_v0.1.json
    python scripts/prepare_datasets.py --cache-dir /scratch/raman_bench_cache

Links
-----
- raman-data: https://github.com/ml-lab-htw/raman_data | pip install raman-data
- raman-bench: https://github.com/ml-lab-htw/RamanBench | pip install raman-bench
- Leaderboard: https://huggingface.co/spaces/HTW-KI-Werkstatt/RamanBench
- Paper (NeurIPS 2026): https://arxiv.org/abs/TBD
"""

import argparse
import logging
import os

from raman_bench.benchmark import RamanBenchmark
from raman_bench.config import load_config
from raman_bench.logging_utils import LOG_FORMAT
from raman_bench.seeds import get_seeds

logger = logging.getLogger(__name__)


def _default_config() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "..", "configs", "benchmark_v0.1.json")


def main():
    parser = argparse.ArgumentParser(description="Pre-cache RamanBench dataset splits")
    parser.add_argument("--config", default=_default_config())
    parser.add_argument("--cache-dir", default=None, help="Override cache directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")

    config = load_config(args.config)
    if args.cache_dir:
        config["cache_dir"] = args.cache_dir

    seeds = get_seeds(config)
    dataset_names_clf = config.get("dataset_names_classification")
    dataset_names_reg = config.get("dataset_names_regression")
    cache_dir = config.get("cache_dir") or ".cache"

    logger.info("Pre-caching %d seeds to %s", len(seeds), cache_dir)

    for seed in seeds:
        logger.info("--- Seed %d ---", seed)
        bench = RamanBenchmark(
            dataset_names_classification=dataset_names_clf,
            dataset_names_regression=dataset_names_reg,
            test_size=config["test_size"],
            random_state=seed,
            cache_dir=cache_dir,
            min_samples_per_class=config.get("min_samples_per_class", 9),
            group_regression_splits=config.get("group_regression_splits", True),
        )
        bench.init_datasets()
        logger.info("Seed %d: %d dataset splits cached.", seed, len(bench))

    logger.info("Done. All splits cached.")


if __name__ == "__main__":
    main()
