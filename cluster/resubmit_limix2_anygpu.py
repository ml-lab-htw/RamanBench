#!/usr/bin/env python
"""Rerun LIMIX2's not-yet-completed tasks on any free a100/h100/h200 node, with the
rebuilt py312 image, LIMIX2's model-wide time override and the benchmark's
num_bag_folds from scope_default.json (earlier LIMIX2 submissions hardcoded 8).

Replaces rb-limix2-full/-large (cancelled 2026-10-03). Their failures:
- 25 of the first 28 large tasks: TimeLimitExceeded at 3600s
- cancer_cell_cooh/nh2 and rruff_mineral_raw: no tabpfn_extensions in the image
- fuel_benchtop: single-row 0-d prediction (Prep_LIMIX2 now keeps the row axis)

Only tasks WITHOUT a results.pkl on the PVC are submitted -- an explicit list, not
the full target list. A cache hit loads the pickle, and while BHT's CephFS stalls on
reads, a cache-hit task could hang the pod in uninterruptible IO.

The done list is read from the PVC beforehand, metadata only (``find -name
results.pkl``), into the JSON passed as the first argument: a list of
[source, dataset, target_idx, repeat, fold, n_repeats] rows.

Usage::

    python cluster/resubmit_limix2_anygpu.py <todo.json> [--dry-run]
"""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "cluster"))

from submit_job import resolve_profile, submit_jobs  # noqa: E402

PROFILE_PATH = REPO.parent / "raman_bench_paper/cluster/profiles/k8s.yaml"
MODEL = "LIMIX2"
IMAGE = "registry.datexis.com/mkoddenbrock-ext/ramanbench:py312-20261003"
N_SPLITS = 3
TIME_LIMIT = 3600
RESULTS_DIR = "results/v1/data"
CACHE_DIR = ".cache_v1"
MIRROR_REPO = "HTW-KI-Werkstatt/RamanBench"
THROTTLE = 4
# Profile default batches up to 5000 tasks into one pod (one sequential pod per Job);
# smaller batches let THROTTLE pods run in parallel.
TASKS_PER_POD = 10


def main() -> None:
    todo_path = Path(sys.argv[1])
    dry_run = "--dry-run" in sys.argv
    todo = json.loads(todo_path.read_text())

    with open(REPO / "configs/v1/scope_default.json") as f:
        scope = json.load(f)

    profile = deepcopy(resolve_profile(str(PROFILE_PATH), None))
    profile["image_overrides"] = {**profile.get("image_overrides", {}), MODEL: IMAGE}
    # node_selector would AND with the affinity below (profile default gpu=a100).
    profile["tasks_per_pod"] = TASKS_PER_POD
    profile["node_selector"] = {}
    profile["node_affinity_match_expressions"] = [
        {"key": "gpu", "operator": "In", "values": ["a100", "h100", "h200"]},
    ]

    for source in ("full", "large"):
        jobs = [(ds, t, r, f, 0, n) for src, ds, t, r, f, n in todo if src == source]
        if not jobs:
            continue
        print(f"{MODEL} anygpu-{source}: {len(jobs)} task(s)")
        submit_jobs(
            model=MODEL, jobs=jobs, slug=f"anygpu-{source}", n_splits=N_SPLITS,
            num_random_configs=0, num_bag_folds=scope["num_bag_folds"], time_limit=TIME_LIMIT,
            results_dir=RESULTS_DIR, cache_dir=CACHE_DIR, mirror_repo=MIRROR_REPO,
            profile=profile, throttle=THROTTLE, dry_run=dry_run,
            dataset_time_limit_overrides=scope.get("time_limit_overrides"),
            model_time_limit_overrides=scope.get("model_time_limit_overrides", {}).get(MODEL),
            max_train_samples_overrides=scope.get("max_train_samples_overrides", {}),
            model_max_train_samples_overrides=scope.get("model_max_train_samples_overrides", {}).get(MODEL),
        )


if __name__ == "__main__":
    main()
