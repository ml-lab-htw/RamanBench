#!/usr/bin/env python
"""Detect which SLURM cluster (if any) this process is running on.

Used by ``submit_job.py``/the model-adding agent to decide which resource
profile to use, without hardcoding any institution-specific hostnames beyond
the generic patterns below (real account/partition/mail details live in
whichever *profile* YAML is selected -- see ``cluster/profiles/``, and for
HTW/TU specifically, the private ``raman_bench_paper/cluster/profiles/``).

Returns one of:
    "htw"      -- an HTW-Berlin "kiwi*" login/compute node
    "tu"       -- a TU-Berlin cluster node (hostname or $SLURM_CLUSTER_NAME hints)
    "unknown"  -- SLURM (sbatch) is available, but the hostname doesn't match
                  any known pattern. Callers should ask the user which profile
                  to use rather than guess.
    "none"     -- no SLURM available at all (sbatch not on PATH). Callers
                  should offer running locally, or ask whether to request
                  cluster access.
"""

from __future__ import annotations

import os
import re
import shutil
import socket
from dataclasses import dataclass

# Generic hostname patterns for the two clusters this project has historically
# run on. Add new patterns here as new clusters are used -- keep any real
# account/partition/credential details out of this file, they belong in a
# profile YAML instead.
_HOSTNAME_PATTERNS = {
    "htw": re.compile(r"^kiwi(head|node)\d*", re.IGNORECASE),
    "tu": re.compile(r"^(frontend|tu-)\d*", re.IGNORECASE),
}


@dataclass
class ClusterDetection:
    cluster: str  # "htw" | "tu" | "unknown" | "none"
    hostname: str
    has_slurm: bool
    reason: str


def detect_cluster() -> ClusterDetection:
    hostname = socket.gethostname()
    has_slurm = shutil.which("sbatch") is not None

    if not has_slurm:
        return ClusterDetection(
            cluster="none", hostname=hostname, has_slurm=False,
            reason="`sbatch` not found on PATH -- no SLURM available here.",
        )

    for cluster, pattern in _HOSTNAME_PATTERNS.items():
        if pattern.match(hostname):
            return ClusterDetection(
                cluster=cluster, hostname=hostname, has_slurm=True,
                reason=f"hostname {hostname!r} matches the {cluster!r} pattern.",
            )

    # SLURM's own cluster-name env var, when the scheduler sets one, is a
    # second, more authoritative signal than hostname guessing.
    slurm_cluster_name = os.environ.get("SLURM_CLUSTER_NAME", "")
    for cluster in _HOSTNAME_PATTERNS:
        if cluster in slurm_cluster_name.lower():
            return ClusterDetection(
                cluster=cluster, hostname=hostname, has_slurm=True,
                reason=f"$SLURM_CLUSTER_NAME={slurm_cluster_name!r} matches {cluster!r}.",
            )

    return ClusterDetection(
        cluster="unknown", hostname=hostname, has_slurm=True,
        reason=(
            f"SLURM is available (sbatch found) but hostname {hostname!r} matches "
            "no known cluster pattern -- ask which profile to use rather than guess."
        ),
    )


def main():
    d = detect_cluster()
    print(f"cluster={d.cluster} hostname={d.hostname} has_slurm={d.has_slurm}")
    print(f"reason: {d.reason}")


if __name__ == "__main__":
    main()
