#!/usr/bin/env python
"""Run the RamanBench benchmark pipeline.

This is a thin wrapper around the ``raman-bench run`` CLI command.
It is provided for users who prefer running a script directly over the CLI.

Pipeline steps
--------------
1. **predictions** — train all models on all datasets and save prediction CSVs
2. **metrics**     — compute evaluation metrics from the prediction CSVs
3. **plots**       — generate figures (requires optional plotting dependencies)

Usage
-----
::

    # Full pipeline (default config)
    python scripts/run_benchmark.py

    # Custom config
    python scripts/run_benchmark.py --config configs/benchmark_v0.1.json

    # Single step
    python scripts/run_benchmark.py --step predictions
    python scripts/run_benchmark.py --step metrics

    # Single model (useful for parallel execution)
    python scripts/run_benchmark.py --model GBM

    # Parallel seeds
    python scripts/run_benchmark.py --seed-index 0  # seed 0
    python scripts/run_benchmark.py --seed-index 1  # seed 1
    python scripts/run_benchmark.py --seed-index 2  # seed 2

Links
-----
- raman-data: https://github.com/ml-lab-htw/raman_data | pip install raman-data
- raman-bench: https://github.com/ml-lab-htw/RamanBench | pip install raman-bench
- Leaderboard: https://huggingface.co/spaces/HTW-KI-Werkstatt/RamanBench
- Paper (under review): https://arxiv.org/abs/2605.02003
"""

import argparse
import logging
import os
import sys
import warnings

warnings.filterwarnings("ignore", message="'force_all_finite' was renamed")

from raman_bench.config import load_config
from raman_bench.evaluation import compute_metrics_from_predictions
from raman_bench.logging_utils import LOG_FORMAT
from raman_bench.predictions import compute_predictions

logger = logging.getLogger(__name__)


def _default_config() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "..", "configs", "benchmark_v0.1.json")


def main():
    parser = argparse.ArgumentParser(
        description="Run the RamanBench benchmark pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        default=_default_config(),
        help="Path to config JSON (default: configs/benchmark_v0.1.json)",
    )
    parser.add_argument(
        "--step",
        choices=["all", "predictions", "metrics", "plots"],
        default="all",
        help="Pipeline step to run (default: all)",
    )
    parser.add_argument("--output", default=None, help="Override output directory")
    parser.add_argument(
        "--seed-index",
        type=int,
        default=None,
        help="Run only this zero-based seed index (for parallel seed execution)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Run only this model (overrides the model list in config)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing prediction files",
    )
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="Iterate datasets in reverse order (useful for parallel forward+reverse jobs)",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")

    config = load_config(args.config)

    if args.output:
        config["output_dir"] = args.output
    if args.model:
        config["models"] = [args.model]

    if args.step in ("all", "predictions"):
        compute_predictions(
            config,
            seed_index=args.seed_index,
            overwrite=args.overwrite,
            reverse=args.reverse,
        )

    if args.step in ("all", "metrics"):
        compute_metrics_from_predictions(config)

    if args.step in ("all", "plots"):
        try:
            from raman_bench.plotting import PlotPipeline
            PlotPipeline(config).run_all()
        except (ImportError, AttributeError) as e:
            logger.warning("Plotting unavailable: %s. Install raman-bench[full].", e)


if __name__ == "__main__":
    main()
