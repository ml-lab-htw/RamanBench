#!/usr/bin/env python
"""Resubmit TABPFN-WIDE's full target list, routed entirely to the cluster's
real b200 node (cl-worker36, confirmed via `kubectl describe node` -- 8x
NVIDIA-B200, 183GB VRAM each, no taints) instead of the profile default (a100).

Both prior TABPFN-WIDE jobs (rb-tabpfn-wide-full/large) actually ran their
entire batch to completion (303/303 and 102/102 tasks attempted) and only
showed k8s `Failed` because of the (since-fixed) job-wide backoffLimit:0 bug
-- real failures were a small number of genuine per-dataset issues
(cancer_cell_cooh/nh2, diabetes_skin_vein in "full"; mlrod,
bacteria_identification in "large"). Resubmitting the FULL target list here
(not a hand-picked straggler list) is intentional and cheap: every
already-succeeded task is cache-hit (an instant disk read, no GPU needed,
see run_experiment.py's ignore_cache/force_recompute logic) -- only the
genuine stragglers do real new compute, now on a much bigger GPU.

Mirrors scripts/submit_full_benchmark.py's own main/large target split
(scope_default.json's large_datasets) and resource/HPO defaults exactly, just
with the whole model (not only the "large" partition) pinned to b200 via a
direct profile patch -- submit_full_benchmark.py's own --big-gpu-types flag
only reroutes the "large" pod, not "full".
"""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

REPO = Path("/Users/koddenbrock/Repository/RamanBench")
sys.path.insert(0, str(REPO / "cluster"))

from submit_job import resolve_profile, submit_jobs  # noqa: E402

PROFILE_PATH = "/Users/koddenbrock/Repository/raman_bench_paper/cluster/profiles/k8s.yaml"
MODEL = "TABPFN-WIDE"
N_SPLITS = 3
NUM_BAG_FOLDS = 8
TIME_LIMIT = 3600
RESULTS_DIR = "results/v1/data"
CACHE_DIR = ".cache_v1"
MIRROR_REPO = "HTW-KI-Werkstatt/RamanBench"


def main() -> None:
    dry_run = "--dry-run" in sys.argv

    with open(REPO / "configs/v1/target_list.json") as f:
        targets = json.load(f)
    targets = [t for t in targets if not t.get("excluded")]

    with open(REPO / "configs/v1/scope_default.json") as f:
        scope = json.load(f)
    large_datasets = set(scope.get("large_datasets", []))
    max_train_samples_overrides = scope.get("max_train_samples_overrides", {})
    model_max_train_samples_override = scope.get("model_max_train_samples_overrides", {}).get(MODEL)

    def _jobs_for(target_subset):
        return [
            (t["dataset"], t["target_idx"], repeat, fold, 0, t.get("n_repeats", 10))
            for t in target_subset
            for repeat in range(t.get("n_repeats", 10))
            for fold in range(N_SPLITS)
        ]

    large_targets = [t for t in targets if t["dataset"] in large_datasets]
    main_targets = [t for t in targets if t["dataset"] not in large_datasets]

    profile = resolve_profile(PROFILE_PATH, None)
    # Route to the real b200 node instead of the profile default (a100): clear
    # node_selector (would otherwise AND with the affinity below and make the
    # pod unschedulable) in favor of node_affinity_match_expressions.
    b200_profile = deepcopy(profile)
    b200_profile["node_selector"] = {}
    b200_profile["node_affinity_match_expressions"] = [
        {"key": "gpu", "operator": "In", "values": ["b200"]},
    ]

    print(f"Resubmitting {MODEL} on b200: {len(main_targets)} main target(s), "
          f"{len(large_targets)} large target(s)")

    main_jobs = _jobs_for(main_targets)
    print(f"  full: {len(main_jobs)} task(s)")
    submit_jobs(
        model=MODEL, jobs=main_jobs, slug="full", n_splits=N_SPLITS,
        num_random_configs=0, num_bag_folds=NUM_BAG_FOLDS, time_limit=TIME_LIMIT,
        results_dir=RESULTS_DIR, cache_dir=CACHE_DIR, mirror_repo=MIRROR_REPO,
        profile=b200_profile, throttle=8, dry_run=dry_run,
        max_train_samples_overrides=max_train_samples_overrides,
        model_max_train_samples_overrides=model_max_train_samples_override,
    )

    large_jobs = _jobs_for(large_targets)
    print(f"  large: {len(large_jobs)} task(s)")
    submit_jobs(
        model=MODEL, jobs=large_jobs, slug="large", n_splits=N_SPLITS,
        num_random_configs=0, num_bag_folds=NUM_BAG_FOLDS, time_limit=TIME_LIMIT,
        results_dir=RESULTS_DIR, cache_dir=CACHE_DIR, mirror_repo=MIRROR_REPO,
        profile=b200_profile, throttle=8, dry_run=dry_run,
        max_train_samples_overrides=max_train_samples_overrides,
        model_max_train_samples_overrides=model_max_train_samples_override,
    )


if __name__ == "__main__":
    main()
