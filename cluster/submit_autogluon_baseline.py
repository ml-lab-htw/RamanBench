#!/usr/bin/env python
"""Submit AutoGluon-extreme preset baseline jobs for a target list (k8s only).

Reads a JSON target list (same shape as submit_full_benchmark.py's own --targets-file:
list of {"dataset": ..., "target_idx": ..., "excluded": bool, "n_repeats": int, ...}),
builds one (dataset, target_idx, repeat, fold, time_limit, budget_label) task per
non-excluded target per requested budget, and submits them as k8s Indexed Job(s) via
cluster/k8s_entrypoint_autogluon.sh + scripts/run_autogluon_baseline.py (NOT
run_experiment.py -- see that script's own module docstring for why this needs a
separate runner/jobspec format entirely).

Defaults to EXTREME_TIME_BUDGETS' three budgets (5m/1h/4h, matching TabArena's own
tabarena.models.automl.generate_autogluon_extreme_experiments) and a small
--tasks-per-pod -- unlike the per-model routine sweep's "one pod per model's entire
run" convention (profiles' tasks_per_pod=5000), AutoGluon-extreme's per-task budgets
run up to 4h, so batching thousands of tasks sequentially into one pod would make a
real sweep take impractically long. A small tasks_per_pod here means many
shorter-lived, genuinely parallel pods instead (k8s's own Indexed Job `parallelism`,
capped by --throttle).

Usage:
    python cluster/submit_autogluon_baseline.py \\
        --targets-file ../RamanBench/configs/v1/core_target_list.json \\
        --profile cluster/profiles/k8s.yaml --cluster k8s \\
        --budgets 5m 1h --tasks-per-pod 4 --throttle 16
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

CLUSTER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CLUSTER_DIR))

from submit_job import (  # noqa: E402
    build_k8s_job_manifest_autogluon,
    resolve_profile,
    write_jobspec_autogluon,
)

# TabArena's own three "extreme" preset budgets (tabarena.models.automl.
# generate_autogluon_extreme_experiments) -- kept in sync with
# scripts/run_autogluon_baseline.py's own EXTREME_TIME_BUDGETS.
EXTREME_TIME_BUDGETS = {"5m": 300, "1h": 3600, "4h": 14400}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--targets-file",
        required=True,
        help="JSON list of {dataset, target_idx, excluded, n_repeats}",
    )
    parser.add_argument(
        "--budgets",
        nargs="+",
        default=list(EXTREME_TIME_BUDGETS.keys()),
        choices=list(EXTREME_TIME_BUDGETS.keys()),
        help="Which of EXTREME_TIME_BUDGETS' labels to submit. Default: all three.",
    )
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=None,
        help="Fallback only, used for a target lacking its own 'n_repeats' -- the "
        "per-target value is used whenever present. Defaults to 1 if neither is available "
        "(AutoGluon-extreme is a baseline comparison, not matched to the routine sweep's "
        "own n_repeats convention).",
    )
    parser.add_argument("--results-dir", default="results/v1/data")
    parser.add_argument("--cache-dir", default=".cache_v1")
    parser.add_argument("--mirror-repo", default="HTW-KI-Werkstatt/RamanBench")
    parser.add_argument("--profile", default=None)
    parser.add_argument("--cluster", default=None, choices=["htw", "tu", "local", "k8s"])
    parser.add_argument(
        "--tasks-per-pod",
        type=int,
        default=4,
        help="Deliberately small (unlike the per-model path's profile default of 5000) -- "
        "see this script's own module docstring for why.",
    )
    parser.add_argument("--throttle", type=int, default=16, help="Max concurrent pods")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    profile = resolve_profile(args.profile, args.cluster)
    if profile.get("backend") != "k8s":
        raise SystemExit(
            "AutoGluon-baseline submission is k8s-only (no SLURM path -- new tooling, not a "
            "port of an existing SLURM mechanism). Pass a k8s profile/--cluster k8s."
        )

    with open(args.targets_file) as f:
        targets = json.load(f)
    targets = [t for t in targets if not t.get("excluded")]
    print(f"{len(targets)} non-excluded target(s) from {args.targets_file}")

    jobs = []
    for t in targets:
        n_repeats = t.get("n_repeats") or args.n_repeats or 1
        for budget_label in args.budgets:
            time_limit = EXTREME_TIME_BUDGETS[budget_label]
            for repeat in range(n_repeats):
                for fold in range(args.n_splits):
                    jobs.append(
                        (
                            t["dataset"],
                            t["target_idx"],
                            repeat,
                            fold,
                            time_limit,
                            budget_label,
                            n_repeats,
                            args.n_splits,
                        )
                    )

    print(
        f"{len(jobs)} task(s) across {len(targets)} target(s) x {len(args.budgets)} "
        f"budget(s) ({', '.join(args.budgets)})"
    )

    # Budget-specific, not a bare "extreme" -- otherwise submitting a different
    # --budgets value (e.g. 1h after an earlier 5m run) collides on the same job
    # name/jobspec file as the previous submission (confirmed live: a second
    # submission with --budgets 1h would have overwritten the still-running "5m"
    # job's jobspec and tried to recreate the exact same k8s Job name).
    slug = "extreme-" + "-".join(args.budgets)
    jobspec_path = write_jobspec_autogluon(jobs, slug)
    job_name, configmap_name, n_pods, job_manifest = build_k8s_job_manifest_autogluon(
        slug=slug,
        n_tasks=len(jobs),
        results_dir=args.results_dir,
        cache_dir=args.cache_dir,
        mirror_repo=args.mirror_repo,
        profile=profile,
        use_gpu=True,
        tasks_per_pod=args.tasks_per_pod,
        throttle=args.throttle,
    )
    namespace = profile.get("namespace", "default")
    print(f"  jobspec: {jobspec_path}  ({len(jobs)} task(s))")
    print(
        f"  configmap: {configmap_name}  job: {job_name}  pods={n_pods} (<= {args.tasks_per_pod} task(s)/pod, "
        f"parallelism={min(args.throttle, n_pods) or 1})"
    )

    if args.dry_run:
        return

    cm_yaml = subprocess.run(
        [
            "kubectl",
            "create",
            "configmap",
            configmap_name,
            "-n",
            namespace,
            "--from-file",
            f"jobspec.txt={jobspec_path}",
            "--dry-run=client",
            "-o",
            "yaml",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    subprocess.run(
        ["kubectl", "apply", "--validate=false", "-f", "-"],
        input=cm_yaml,
        text=True,
        check=True,
    )
    subprocess.run(
        ["kubectl", "apply", "--validate=false", "-f", "-"],
        input=yaml.safe_dump(job_manifest),
        text=True,
        check=True,
    )
    print(
        f"Done: k8s Job {job_name!r} created ({n_pods} pod(s), parallelism={min(args.throttle, n_pods) or 1})"
    )


if __name__ == "__main__":
    main()
