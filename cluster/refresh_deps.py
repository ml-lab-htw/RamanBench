#!/usr/bin/env python
"""Refresh released dependencies before submitting a job batch.

This is a deliberate, one-time step cluster-agent (or a human) runs against
the target environment *before* calling ``submit_job.py``/``submit_v1.sh`` --
not something individual SLURM jobs do for themselves anymore.
``run_experiment.sbatch`` used to ``git pull`` sibling repo checkouts inside
every single job; that meant concurrent array tasks could race each other and
each job could silently see different code, and it fought against "rely on a
released version" -- a checkout that's live-pulled mid-batch isn't really
pinned to anything. Now: refresh once, here, then every job in the batch runs
against the exact same, already-resolved versions.

``raman-data`` is always upgraded to the latest PyPI release -- this is what
determines which datasets/DatasetInfo entries (and their ``is_grouped``
metadata) are visible, so skipping this step is exactly how a newly-released
dataset gets silently missed by a stale environment.

``raman-bench`` itself doesn't have a released v1 yet (mid-refactor, see the
v1 plan) -- pass ``--workspace`` to also ``git pull`` its checkout in place
(the one that supplies ``scripts/``/``cluster/``, which are tooling, not
shipped in the wheel, so a checkout is required regardless of release status).
Once a real v1 release exists, add ``raman-bench[full]`` to the upgrade list
here the same way ``raman-data`` is handled today.

Usage
-----
    # On the cluster, in the target conda env, before submitting anything:
    python cluster/refresh_deps.py --workspace ~/workspace/RamanBench_v1
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version


def _installed_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "(not installed)"


def refresh_pip_package(package: str) -> None:
    before = _installed_version(package)
    print(f"{package}: currently {before}, checking for a newer release...")
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", package], check=True)
    after = _installed_version(package)
    if before == after:
        print(f"{package}: already up to date ({after}).")
    else:
        print(f"{package}: upgraded {before} -> {after}.")


def refresh_workspace_checkout(workspace: str) -> None:
    print(f"Pulling {workspace} ...")
    subprocess.run(["git", "-C", workspace, "pull"], check=True)
    result = subprocess.run(
        ["git", "-C", workspace, "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    )
    print(f"{workspace} now at {result.stdout.strip()}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--workspace", default=None,
        help="Path to the RamanBench checkout supplying scripts/cluster tooling -- "
             "pulled in place if given (not released to PyPI yet). Omit to skip.",
    )
    args = parser.parse_args()

    refresh_pip_package("raman-data")

    if args.workspace:
        refresh_workspace_checkout(args.workspace)
    else:
        print("No --workspace given -- skipping the RamanBench checkout pull "
              "(remember to update it separately if scripts/cluster/ changed).")


if __name__ == "__main__":
    main()
