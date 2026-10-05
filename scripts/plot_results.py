#!/usr/bin/env python
"""Generate the leaderboard figures (static PNG/PDF + interactive HTML) from v1 results.

Reads ``hpo_results.csv`` as written by ``scripts/aggregate_results.py`` and
writes, under ``--output-dir``: per-task-group leaderboard CSVs, the static
figures (``static/``), their interactive twins (``interactive/``) and an
``index.html`` linking all of them. See :mod:`raman_bench.plotting`.

Usage:
    python scripts/plot_results.py --input results/v1/aggregated/hpo_results.csv \\
        --output-dir results/v1/figures

    # Colour every model instead of the top 2 per category:
    python scripts/plot_results.py --focus-top-k 0

    # Tuned + ensembled variant where a model has one, default otherwise:
    python scripts/plot_results.py --variant tuned_ensemble
"""

from __future__ import annotations

import argparse
import logging

from raman_bench.plotting import generate_all
from raman_bench.plotting.results import DEFAULT_SCOPE, DEFAULT_TARGET_LIST, VARIANTS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default="results/v1/aggregated/hpo_results.csv")
    parser.add_argument("--output-dir", default="results/v1/figures")
    parser.add_argument("--variant", choices=VARIANTS, default="default")
    parser.add_argument(
        "--focus-top-k", type=int, default=2,
        help="Models per category drawn in colour (by Elo); the rest are grey. 0 colours all.",
    )
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"], help="Static figure formats")
    parser.add_argument("--no-interactive", action="store_true", help="Skip the HTML figures")
    parser.add_argument(
        "--inline-plotlyjs", action="store_true",
        help="Embed plotly.js in every HTML file (works offline, ~4.5 MB each) instead of loading it from a CDN",
    )
    parser.add_argument("--scope", default=str(DEFAULT_SCOPE), help="Scope config; its model list is plotted")
    parser.add_argument("--target-list", default=str(DEFAULT_TARGET_LIST), help="Excluded targets are dropped")
    parser.add_argument("--reference-model", default="RF", help="Elo anchor (= 1000) and imputation source")
    parser.add_argument(
        "--max-imputed-pct", type=float, default=50.0,
        help="Drop models whose results are more than this percent imputed",
    )
    parser.add_argument("--bootstrap-rounds", type=int, default=200, help="Elo bootstrap rounds for the CIs")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    logging.getLogger("raman_bench").setLevel(logging.INFO)
    written = generate_all(
        args.input,
        args.output_dir,
        variant=args.variant,
        focus_top_k=args.focus_top_k or None,
        formats=tuple(args.formats),
        make_interactive=not args.no_interactive,
        include_plotlyjs=True if args.inline_plotlyjs else "cdn",
        scope=args.scope,
        target_list=args.target_list,
        reference_model=args.reference_model,
        max_imputed_pct=args.max_imputed_pct,
        bootstrap_rounds=args.bootstrap_rounds,
    )
    print(f"Open {written['index'][0]}")


if __name__ == "__main__":
    main()
