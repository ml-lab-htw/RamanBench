#!/usr/bin/env python
"""Find the cached results that ran under other settings than a protocol and move them out of the way.

Reads every ``results.pkl`` under ``--results-dir``, compares the row cap and time limit it ran under (``--old-scope``,
the scope file of that run) with ``--protocol``, and writes ``stale_results.csv`` (every result with its reason, empty
for a valid one) under ``--output-dir``. Without ``--quarantine-dir`` nothing else happens. With it, the fold directories
of the stale results are moved there (same relative paths), so the next sweep recomputes them; a move is undone by
moving them back. See :mod:`raman_bench.stale_results` for what counts as stale.
"""

import argparse
import json
import logging
import os
import time

from raman_bench.aggregation import scan_cached_results
from raman_bench.stale_results import THRESHOLD, classify, quarantine

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--old-scope", required=True, help="Scope JSON the results ran under")
    parser.add_argument("--protocol", default="configs/v2/protocol.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--quarantine-dir", default=None)
    parser.add_argument("--threshold", type=float, default=THRESHOLD, help="Share of the old time limit that marks a fit as budget-bound")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    old_scope = json.load(open(args.old_scope))
    protocol = json.load(open(args.protocol))
    start = time.time()
    results = scan_cached_results(args.results_dir, workers=args.workers)
    logger.info("Loaded %d result(s) in %.0f s", len(results), time.time() - start)
    table = classify(results, old_scope, protocol, args.threshold)
    os.makedirs(args.output_dir, exist_ok=True)
    path = os.path.join(args.output_dir, "stale_results.csv")
    table.to_csv(path, index=False)
    stale = table[table["reason"] != ""]
    logger.info("Stale: %d of %d result(s) (%s); table in %s", len(stale), len(table), stale["reason"].value_counts().to_dict(), path)
    per_model = stale.groupby("model", dropna=False).size().sort_values(ascending=False)
    logger.info("Stale results per model:\n%s", per_model.to_string())
    if args.quarantine_dir:
        moved = quarantine(args.results_dir, stale, args.quarantine_dir)
        logger.info("Moved %d fold directorie(s) to %s", moved, args.quarantine_dir)
    logger.info("DONE: %d stale of %d, quarantined=%s", len(stale), len(table), bool(args.quarantine_dir))


if __name__ == "__main__":
    main()
