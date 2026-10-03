#!/usr/bin/env python
"""Resubmit LIMIX2's full target list, routed to the cluster's real V100 nodes
(cl-worker20/22/23/25, Tesla-V100S-PCIE-32GB) instead of the profile default
(a100) -- explicit choice to trade speed for availability: V100 capacity is
far less contended right now than a100/h100/h200/b200, and the user has
explicitly accepted a multi-day runtime in exchange for not competing for the
cluster's scarce, high-demand GPU tiers.

Both prior LIMIX2 jobs (rb-limix2-full/large) were cancelled to free up a100
capacity for other work -- this resubmits the SAME full target list (not a
hand-picked subset): every already-succeeded task is a cache-hit (instant
disk read, no GPU needed), only genuinely new/incomplete tasks do real new
compute. Same time_limit=3600 as the original submission. num_bag_folds now
comes from scope_default.json (3): this script originally hardcoded 8, copied
from that submission, which left 133 LimiX2 results on 8 bag folds instead of
the benchmark's 3. Those are kept as they are (2026-10-03 decision: the
inconsistency is acceptable vs. the compute to regenerate them); only tasks
without any result were computed afterwards, with 3 folds.

resolve_k8s_image automatically picks the py312 image for LIMIX2 via the
profile's own image_overrides -- no need to set that here.
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
MODEL = "LIMIX2"
N_SPLITS = 3
TIME_LIMIT = 3600
RESULTS_DIR = "results/v1/data"
CACHE_DIR = ".cache_v1"
MIRROR_REPO = "HTW-KI-Werkstatt/RamanBench"


def main() -> None:
    dry_run = "--dry-run" in sys.argv
    smoke_test = "--smoke-test" in sys.argv

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

    if smoke_test:
        # Single small, known-good dataset (already smoke-tested successfully
        # on a100 earlier this session) -- just confirm LIMIX2 actually runs on
        # V100 hardware (CUDA compute-capability compatibility, driver support)
        # before committing the full multi-day sweep.
        main_targets = [t for t in main_targets if t["dataset"] == "alzheimer"]
        large_targets = []

    profile = resolve_profile(PROFILE_PATH, None)
    # Route to the real V100 nodes instead of the profile default (a100): clear
    # node_selector (would otherwise AND with the affinity below and make the
    # pod unschedulable) in favor of node_affinity_match_expressions.
    v100_profile = deepcopy(profile)
    v100_profile["node_selector"] = {}
    v100_profile["node_affinity_match_expressions"] = [
        {"key": "gpu", "operator": "In", "values": ["v100"]},
    ]

    slug_suffix = "-v100-smoke" if smoke_test else ""
    print(f"Resubmitting {MODEL} on v100{' (SMOKE TEST)' if smoke_test else ''}: "
          f"{len(main_targets)} main target(s), {len(large_targets)} large target(s)")

    main_jobs = _jobs_for(main_targets)
    print(f"  full: {len(main_jobs)} task(s)")
    submit_jobs(
        model=MODEL, jobs=main_jobs, slug=f"full{slug_suffix}", n_splits=N_SPLITS,
        num_random_configs=0, num_bag_folds=scope["num_bag_folds"], time_limit=TIME_LIMIT,
        results_dir=RESULTS_DIR, cache_dir=CACHE_DIR, mirror_repo=MIRROR_REPO,
        profile=v100_profile, throttle=4, dry_run=dry_run,
        max_train_samples_overrides=max_train_samples_overrides,
        model_max_train_samples_overrides=model_max_train_samples_override,
    )

    if large_targets:
        large_jobs = _jobs_for(large_targets)
        print(f"  large: {len(large_jobs)} task(s)")
        submit_jobs(
            model=MODEL, jobs=large_jobs, slug=f"large{slug_suffix}", n_splits=N_SPLITS,
            num_random_configs=0, num_bag_folds=scope["num_bag_folds"], time_limit=TIME_LIMIT,
            results_dir=RESULTS_DIR, cache_dir=CACHE_DIR, mirror_repo=MIRROR_REPO,
            profile=v100_profile, throttle=4, dry_run=dry_run,
            max_train_samples_overrides=max_train_samples_overrides,
            model_max_train_samples_overrides=model_max_train_samples_override,
        )


if __name__ == "__main__":
    main()
