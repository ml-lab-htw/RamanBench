#!/usr/bin/env python
"""Build HIVE-COTE 2 from the cached results of its four components.

Run after STC, DRCIF, ARSENAL and TDE have finished (``scripts/run_experiment.py``);
writes ``HIVECOTEV2-ASSEMBLED_c1_BAG_L1/<task>/<repeat>_<fold>/results.pkl`` under
``--results-dir`` for every split all four have. Each component is weighted by its
out-of-fold accuracy to the power 4 (HC2's CAWPE weighting); see
:mod:`raman_bench.models.custom.aeon.assemble`.

Usage:
    python scripts/assemble_hivecote.py --results-dir results/v1/data
"""

from __future__ import annotations

import argparse
import logging

from raman_bench.models.custom.aeon.assemble import ALPHA, assemble


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--results-dir", default="results/v1/data")
    parser.add_argument("--alpha", type=float, default=ALPHA, help="CAWPE exponent (HC2: 4)")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild splits already assembled")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    written = assemble(args.results_dir, alpha=args.alpha, overwrite=args.overwrite)
    print(f"Wrote {len(written)} assembled result(s)")


if __name__ == "__main__":
    main()
