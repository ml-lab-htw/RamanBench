#!/usr/bin/env python
"""Submit AutoGluon-extreme's 1h budget to the real b200 node (cl-worker36),
alongside the already-running a100 job (rb-autogluon-extreme-1h), to use free
B200 capacity and relieve a100 contention (8/21 pods stuck Pending there).

Same 405-task target list as the existing a100 job -- intentional, not a
hand-picked subset: run_autogluon_baseline.py's cache (keyed by (experiment
name, dataset, target_idx, repeat, fold), independent of which hardware ran
it) means any task the a100 job already finished is just a fast disk-read
here, so running the same full list on both doesn't meaningfully waste B200
compute, it just lets whichever job reaches a given task first do the real
work.

Distinct slug ("extreme-1h-b200") from the existing "extreme-1h" job so the
k8s Job/ConfigMap names don't collide with the still-running a100 submission.

throttle=4 per user's own "start with 4 pods" -- conservative given 6 free
B200 GPU slots (2 of 8 already used by the TABPFN-WIDE b200 resubmission).
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


def main() -> None:
    dry_run = "--dry-run" in sys.argv
    throttle = 4

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
    b200_profile = deepcopy(profile)
    b200_profile["node_selector"] = {}
    b200_profile["node_affinity_match_expressions"] = [
        {"key": "gpu", "operator": "In", "values": ["b200"]},
    ]

    slug = "extreme-1h-b200"
    jobspec_path = write_jobspec_autogluon(jobs, slug)
    job_name, configmap_name, n_pods, job_manifest = build_k8s_job_manifest_autogluon(
        slug=slug, n_tasks=len(jobs), results_dir="results/v1/data", cache_dir=".cache_v1",
        mirror_repo="HTW-KI-Werkstatt/RamanBench", profile=b200_profile, use_gpu=True,
        tasks_per_pod=20, throttle=throttle,
    )
    namespace = b200_profile.get("namespace", "default")
    print(f"  jobspec: {jobspec_path}  ({len(jobs)} task(s))")
    print(f"  configmap: {configmap_name}  job: {job_name}  pods={n_pods} (parallelism={min(throttle, n_pods) or 1})")

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
