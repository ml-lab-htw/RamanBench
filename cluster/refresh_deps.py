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
    """Pull ``workspace`` in place if it's a git checkout. Some real deployments
    (e.g. HTW's ``RamanBench_v1``) are deliberately rsync'd, not git-cloned --
    to avoid disturbing another in-progress checkout at deploy time, and because
    raman-bench itself has no released v1 yet for a plain ``pip install`` to
    pull from. A hard ``git pull`` there always fails. Confirmed in practice:
    this crashed unconditionally (``set -e`` in the caller) on every single
    hourly opportunistic-scheduler cron tick for over a week with nothing
    submitted, because the workspace it was pointed at is exactly this kind of
    rsync deployment -- silently, since cron discards stdout unless redirected.
    Skip cleanly instead of crashing; fixes are deployed there via rsync, not
    a pull, so there is nothing this step could do for it anyway.

    For real git checkouts, the pull always passes ``--no-rebase`` explicitly
    rather than relying on a bare ``git pull``. Confirmed in practice again,
    separately: on hosts where ``pull.rebase``/``pull.ff`` are unset in git
    config (true of at least one real cluster login node, git >= 2.27ish),
    a bare ``git pull`` is fatal ("Need to specify how to reconcile divergent
    branches") the moment the checkout has *any* local-only commit -- which
    it always will here after the very first pull, since a non-fast-forward
    pull with no configured strategy still creates a local merge commit that
    is never pushed anywhere. That leaves the checkout permanently one commit
    "ahead" of origin, so the next pull that also has anything new to bring in
    is simultaneously ahead and behind -- i.e. diverged -- and fails the same
    way again. This recurred every single hourly tick once new commits landed
    upstream, submitting nothing each time, until reconciled by hand. Passing
    the strategy on the command line makes this deterministic regardless of
    git version/config on whatever host runs this, instead of depending on
    a config default nobody deliberately set. Using merge (not rebase) here
    deliberately, matching every prior successful pull on record for this
    checkout, since rebase would rewrite local-only commits some deployments
    may have relied on already having a fixed hash."""
    is_git = subprocess.run(
        ["git", "-C", workspace, "rev-parse", "--is-inside-work-tree"],
        capture_output=True, text=True,
    ).returncode == 0
    if not is_git:
        print(f"{workspace} is not a git checkout (deployed via rsync) -- skipping pull.")
        return
    print(f"Pulling {workspace} ...")
    subprocess.run(["git", "-C", workspace, "pull", "--no-rebase"], check=True)
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
