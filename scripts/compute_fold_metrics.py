#!/usr/bin/env python
"""Compute every metric of ``raman_bench.fold_metrics`` per fold for a results directory.

Reads each cached ``results.pkl`` (its stored test predictions) and writes one row per
result: ``ta_name``, ``dataset``, ``fold`` and one column per metric.
``scripts/build_reference_results.py --fold-metrics`` merges the table into the bundled
reference results.

Usage:
    python scripts/compute_fold_metrics.py --results-dir results/v1/data \\
        --output results/v1/aggregated/fold_metrics.parquet
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from raman_bench.fold_metrics import fold_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output", required=True, help=".parquet or .csv")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    table = fold_metrics(args.results_dir)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".csv":
        table.to_csv(out, index=False)
    else:
        table.to_parquet(out, index=False)
    logging.info("Wrote %d rows (%d result directories) to %s", len(table), table["ta_name"].nunique(), out)


if __name__ == "__main__":
    main()
