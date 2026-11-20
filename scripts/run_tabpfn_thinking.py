#!/usr/bin/env python
"""Run the hosted TabPFN thinking model on the v1 protocol, via ``evaluate_estimator``.

The fits run on Prior Labs' servers (``tabpfn-client``), so every task's training and test data
is uploaded and each thinking fit draws from the account's token budget. Needs ``TABPFN_TOKEN``
(read from the environment or a ``.env`` file).

On the grouped tasks the group ids are sent to the API (``group_col``, ``use_groups=True``), which
the other v1 models do not get: those results are a different variant (grouped), not the plain protocol.
Thinking mode bins spectra wider than 2000 columns (``THINKING_MAX_COLUMNS``), see
``raman_bench.models.api.tabpfn_api``.

    python scripts/run_tabpfn_thinking.py --effort mini --tasks acetic_acid_species
"""

from __future__ import annotations

import argparse

from raman_bench.env import load_env
from raman_bench.evaluate import evaluate_estimator
from raman_bench.models.api.tabpfn_api import TabPFNAPIClassifier, TabPFNAPIRegressor

NAME = "TABPFN-V3.5-THINKING"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--effort", required=True,
        choices=["micro", "mini", "low", "medium", "high", "xhigh", "max"],
        help="Thinking effort, from least to most compute (the API's own default is medium)",
    )
    ap.add_argument("--timeout-s", type=float, default=None, help="Optional thinking timeout, max 2400")
    ap.add_argument("--max-columns", type=int, default=20000, help="Bin spectra wider than this (v3.5 server ceiling: 20000)")
    ap.add_argument("--tasks", nargs="*", default=None, help="Task keys or dataset names; default all")
    ap.add_argument("--results-dir", default="results/v1/user")
    ap.add_argument("--cache-dir", default=".cache_v1")
    ap.add_argument("--verbosity", type=int, default=1)
    args = ap.parse_args()

    load_env()
    result = evaluate_estimator(
        NAME,
        classifier=TabPFNAPIClassifier(
            thinking=True, thinking_effort=args.effort, thinking_timeout_s=args.timeout_s,
            use_groups=True, max_columns=args.max_columns,
        ),
        regressor=TabPFNAPIRegressor(
            thinking=True, thinking_effort=args.effort, thinking_timeout_s=args.timeout_s,
            use_groups=True, max_columns=args.max_columns,
        ),
        tasks=args.tasks,
        results_dir=args.results_dir,
        cache_dir=args.cache_dir,
        verbosity=args.verbosity,
    )
    print(f"{NAME} effort={args.effort}: {len(result)} folds", flush=True)


if __name__ == "__main__":
    main()
