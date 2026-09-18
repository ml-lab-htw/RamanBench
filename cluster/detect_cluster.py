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

SLURM detection above is about *where this process is running* (a login/compute
node reached via ssh). A Kubernetes cluster is different: jobs are submitted
FROM a workstation/login node that merely has ``kubectl`` configured to talk to
a remote k8s API server, not by ssh-ing onto the cluster itself. ``detect_k8s()``
below answers that separate question ("can this machine submit k8s jobs right
now") and is checked independently of ``detect_cluster()`` -- a machine can be
both a SLURM login node AND have kubectl configured for a k8s cluster at the
same time, so ``submit_job.py`` treats k8s as an explicit ``--cluster``/
``--profile`` choice rather than folding it into the SLURM auto-detection above.
"""

from __future__ import annotations

import os
import re
import shutil
import socket
import subprocess
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


@dataclass
class K8sDetection:
    available: bool
    context: str | None
    reason: str


def detect_k8s() -> K8sDetection:
    """Check whether this machine can submit jobs to a Kubernetes cluster
    right now -- ``kubectl`` on PATH AND a current-context configured. Does
    NOT verify the API server is actually reachable (that's left to the real
    ``kubectl apply``/``kubectl create`` call in ``submit_job.py``, which
    surfaces a clearer error at the point of use than a preflight ping would)."""
    if shutil.which("kubectl") is None:
        return K8sDetection(available=False, context=None, reason="`kubectl` not found on PATH.")
    try:
        result = subprocess.run(
            ["kubectl", "config", "current-context"],
            capture_output=True, text=True, timeout=10,
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        return K8sDetection(available=False, context=None, reason=f"`kubectl config current-context` failed: {e}")
    if result.returncode != 0:
        return K8sDetection(
            available=False, context=None,
            reason=f"No current kubectl context configured ({result.stderr.strip()}).",
        )
    context = result.stdout.strip()
    return K8sDetection(available=True, context=context, reason=f"kubectl context {context!r} is configured.")


def main():
    d = detect_cluster()
    print(f"cluster={d.cluster} hostname={d.hostname} has_slurm={d.has_slurm}")
    print(f"reason: {d.reason}")
    k = detect_k8s()
    print(f"k8s_available={k.available} context={k.context}")
    print(f"reason: {k.reason}")


if __name__ == "__main__":
    main()
