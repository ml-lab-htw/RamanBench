#!/usr/bin/env python
"""Submit AutoGluon-extreme's 1h budget with a broad node_affinity (a100 OR
h100 OR h200 OR b200) -- lets the scheduler place each pod on whichever of
these has free capacity, instead of being stuck Pending on a100 specifically
(9/21 pods of the original rb-autogluon-extreme-1h job were stuck this way).

A running Job's pod template (including node affinity) is immutable -- the
already-pending a100-only pods can't be "moved"; this is a separate,
additional job covering the same task list. Same cache-skip reasoning as
rb-autogluon-extreme-1h-b200: whichever job reaches a given task first does
the real work, the other(s) just load the cached result.

throttle=6: generous since this can land on any of 4 node types, but still
capped to avoid monopolizing whatever capacity other tenants need.
"""

from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import yaml

REPO = Path("/Users/koddenbrock/Repository/RamanBench")
sys.path.insert(0, str(REPO / "cluster"))

from submit_job import build_k8s_job_manifest_autogluon, resolve_profile, write_jobspec_autogluon  # noqa: E402

PROFILE_PATH = "/Users/koddenbrock/Repository/raman_bench_paper/cluster/profiles/k8s.yaml"
EXTREME_TIME_BUDGETS = {"5m": 300, "1h": 3600, "4h": 14400}
GPU_TYPES = ["a100", "h100", "h200", "b200"]


def main() -> None:
    dry_run = "--dry-run" in sys.argv
    throttle = 6

    with open(REPO / "configs/v1/target_list.json") as f:
        targets = json.load(f)
    targets = [t for t in targets if not t.get("excluded")]
    print(f"{len(targets)} non-excluded target(s)")

    budget_label = "1h"
    time_limit = EXTREME_TIME_BUDGETS[budget_label]
    n_splits = 3
    jobs = []
    for t in targets:
        n_repeats = t.get("n_repeats") or 1
        for repeat in range(n_repeats):
            for fold in range(n_splits):
                jobs.append((t["dataset"], t["target_idx"], repeat, fold, time_limit, budget_label, n_repeats, n_splits))
    print(f"{len(jobs)} task(s)")

    profile = resolve_profile(PROFILE_PATH, None)
    anygpu_profile = deepcopy(profile)
    anygpu_profile["node_selector"] = {}
    anygpu_profile["node_affinity_match_expressions"] = [
        {"key": "gpu", "operator": "In", "values": GPU_TYPES},
    ]

    slug = "extreme-1h-anygpu"
    jobspec_path = write_jobspec_autogluon(jobs, slug)
    job_name, configmap_name, n_pods, job_manifest = build_k8s_job_manifest_autogluon(
        slug=slug, n_tasks=len(jobs), results_dir="results/v1/data", cache_dir=".cache_v1",
        mirror_repo="HTW-KI-Werkstatt/RamanBench", profile=anygpu_profile, use_gpu=True,
        tasks_per_pod=20, throttle=throttle,
    )
    namespace = anygpu_profile.get("namespace", "default")
    print(f"  jobspec: {jobspec_path}  ({len(jobs)} task(s))")
    print(f"  configmap: {configmap_name}  job: {job_name}  pods={n_pods} (parallelism={min(throttle, n_pods) or 1})")
    print(f"  node affinity: gpu in {GPU_TYPES}")

    if dry_run:
        return

    cm_yaml = subprocess.run(
        ["kubectl", "create", "configmap", configmap_name, "-n", namespace,
         "--from-file", f"jobspec.txt={jobspec_path}", "--dry-run=client", "-o", "yaml"],
        check=True, capture_output=True, text=True,
    ).stdout
    subprocess.run(["kubectl", "apply", "--validate=false", "-f", "-"], input=cm_yaml, text=True, check=True)
    subprocess.run(
        ["kubectl", "apply", "--validate=false", "-f", "-"],
        input=yaml.safe_dump(job_manifest), text=True, check=True,
    )
    print(f"Done: k8s Job {job_name!r} created ({n_pods} pod(s), parallelism={min(throttle, n_pods) or 1}).")


if __name__ == "__main__":
    main()
