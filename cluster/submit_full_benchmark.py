#!/usr/bin/env python
"""Submit one array job per (dataset, target) for a given model.

Reads a JSON target list (as written by ``scripts/build_target_list.py``: list of
{"dataset": ..., "target_idx": ..., "excluded": bool, "n_repeats": int, ...}) and calls
cluster/submit_job.py once per non-excluded entry -- i.e. one job array per (dataset,
target), each array covering every (repeat, fold) x config-index combination for that
target (real repeated k-fold CV, matching TabArena's own documented protocol -- see
raman_bench.splitting.get_n_repeats). This is the "run this model across the whole
benchmark" primitive model-agent's workflow describes.

Each target's own ``n_repeats`` (dataset-size-adaptive, per TabArena's real protocol) is
used if the target list provides one; ``--n-repeats`` is only a fallback for target lists
that don't (e.g. hand-written ones), not an override -- don't pass it to force a uniform
value across every dataset, that's exactly what the per-target value exists to avoid.

Usage:
    python scripts/build_target_list.py --dataset-list classification_all.json \\
        --dataset-list regression_all.json --output targets.json
    python cluster/submit_full_benchmark.py --model PLS --targets-file targets.json \\
        --profile ~/workspace/htw_v1_profile.yaml --n-splits 3 --config-indices 0
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

CLUSTER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CLUSTER_DIR))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--targets-file", required=True, help="JSON list of {dataset, target_idx, excluded}")
    parser.add_argument("--profile", default=None)
    parser.add_argument("--cluster", default=None, choices=["htw", "tu", "local", "k8s"])
    parser.add_argument(
        "--n-repeats", type=int, default=None,
        help="Fallback only, used for a target lacking its own 'n_repeats' -- the "
             "per-target value (dataset-size-adaptive, from build_target_list.py) is "
             "used whenever present. Defaults to 10 if neither is available.",
    )
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--config-indices", type=int, nargs="+", default=[0])
    parser.add_argument(
        "--num-random-configs", type=int, default=0,
        help="Only matters if --config-indices includes indices beyond 0 (an HPO sweep) -- "
             "config_index 0 is always the static default config regardless of this value. "
             "Defaults to 0 (no random configs generated at all) since the routine "
             "config-0-only sweep never uses them; TabArena's generate_all_configs_lst "
             "unconditionally tries to sample this many random configs when >0, which raises "
             "ExhaustedSearchSpaceError and fails EVERY task for any model whose real search "
             "space is smaller than this value (confirmed for real: HYDRA has 39 possible "
             "configs, PERPETUAL_BOOSTER has 5) -- confirmed happening in production on "
             "2026-09-17 (a shared k8s cluster job silently failed every single task), "
             "no pod crash. Only raise this above 0 when actually opting into an HPO sweep "
             "(passing config-indices beyond just [0]).",
    )
    parser.add_argument("--num-bag-folds", type=int, default=8)
    parser.add_argument("--time-limit", type=float, default=3600)
    parser.add_argument("--results-dir", default="results/v1/data")
    parser.add_argument("--cache-dir", default=".cache_v1")
    parser.add_argument("--mirror-repo", default="HTW-KI-Werkstatt/RamanBench")
    parser.add_argument("--throttle", type=int, default=8)
    parser.add_argument(
        "--scope", default=None,
        help="Path to a scope JSON (e.g. configs/v1/scope_default.json) to read "
             "'large_datasets' (k8s only, routes those datasets to a separate pod on a "
             "bigger GPU) and 'max_train_samples_overrides' (both backends, a dataset-keyed "
             "row-subsampling cap, e.g. {'mlrod': 10000}) from. Defaults to "
             "configs/v1/scope_default.json next to this script's repo root if present; "
             "pass an empty scope (or a file with neither key) to disable both.",
    )
    parser.add_argument(
        "--big-gpu-types", nargs="+", default=["h100", "h200"],
        help="k8s only: node 'gpu' label values eligible for the large-dataset pod.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    with open(args.targets_file) as f:
        targets = json.load(f)
    targets = [t for t in targets if not t.get("excluded")]
    n_repeats_counts = {}
    for t in targets:
        n_repeats_counts[t.get("n_repeats", args.n_repeats or 10)] = (
            n_repeats_counts.get(t.get("n_repeats", args.n_repeats or 10), 0) + 1
        )
    print(f"Submitting {len(targets)} target(s) for model {args.model!r} "
          f"(n_repeats distribution: {n_repeats_counts}, {args.n_splits} fold(s) x "
          f"{len(args.config_indices)} config(s) each)")

    from copy import deepcopy

    from submit_job import resolve_profile, submit_jobs

    profile = resolve_profile(args.profile, args.cluster)

    # Global across both backends: a dataset-keyed row-subsampling cap (e.g.
    # {"mlrod": 10000}), see submit_job.resolve_max_train_samples's docstring
    # for why (REZERONET's real TimeLimitExceeded on mlrod under the reduced
    # 600s/3-bag-fold compute-scaling settings). large_datasets (k8s-only,
    # big-GPU pod routing) is loaded from the same scope file.
    large_datasets: set[str] = set()
    max_train_samples_overrides: dict[str, int] = {}
    scope_path = Path(args.scope) if args.scope else CLUSTER_DIR.parent / "configs" / "v1" / "scope_default.json"
    if scope_path.exists():
        with open(scope_path) as f:
            scope_data = json.load(f)
        large_datasets = set(scope_data.get("large_datasets", []))
        max_train_samples_overrides = scope_data.get("max_train_samples_overrides", {})

    if profile.get("backend") == "k8s":
        # One pod for THIS MODEL'S ENTIRE sweep across every target -- unlike
        # the SLURM path below (one array per target, proven/unchanged), a
        # per-target subprocess loop here would mean one k8s Job (and pod) per
        # target, i.e. up to 160 pods for one model. Building the full combined
        # task list in-process and calling submit_jobs (-> submit_jobs_k8s)
        # keeps it to the single pod the profile's tasks_per_pod/max_array_size
        # are now sized for (see k8s.yaml's own comment) -- TWO pods, not one,
        # when large_datasets splits off a big-GPU pod (see below).

        def _jobs_for(target_subset):
            return [
                (t["dataset"], t["target_idx"], repeat, fold, cfg, t.get("n_repeats", args.n_repeats or 10))
                for t in target_subset
                for repeat in range(t.get("n_repeats", args.n_repeats or 10))
                for fold in range(args.n_splits)
                for cfg in args.config_indices
            ]

        large_targets = [t for t in targets if t["dataset"] in large_datasets]
        main_targets = [t for t in targets if t["dataset"] not in large_datasets]

        all_job_ids = []
        if main_targets:
            main_jobs = _jobs_for(main_targets)
            print(f"k8s backend: submitting {len(main_jobs)} task(s) across {len(main_targets)} "
                  f"target(s) as the main pod for model {args.model!r}")
            all_job_ids += submit_jobs(
                model=args.model, jobs=main_jobs, slug="full",
                n_splits=args.n_splits, num_random_configs=args.num_random_configs,
                num_bag_folds=args.num_bag_folds, time_limit=args.time_limit,
                results_dir=args.results_dir, cache_dir=args.cache_dir, mirror_repo=args.mirror_repo,
                profile=profile, throttle=args.throttle, dry_run=args.dry_run,
                max_train_samples_overrides=max_train_samples_overrides,
            )
        if large_targets:
            large_jobs = _jobs_for(large_targets)
            # Separate pod on a bigger GPU (H100/H200 by default) for the outlier
            # large/wide datasets -- see configs/v1/scope_default.json's
            # "_comment_large_datasets" for why these specific targets. Clears the
            # plain node_selector (would otherwise AND with the affinity below and
            # make the pod unschedulable -- e.g. "gpu=a100" AND "gpu in [h100,h200]"
            # is never satisfiable) in favor of node_affinity_match_expressions.
            big_gpu_profile = deepcopy(profile)
            big_gpu_profile["node_selector"] = {}
            big_gpu_profile["node_affinity_match_expressions"] = [
                {"key": "gpu", "operator": "In", "values": args.big_gpu_types},
            ]
            print(f"k8s backend: submitting {len(large_jobs)} task(s) across {len(large_targets)} "
                  f"large/wide target(s) as a separate {'/'.join(args.big_gpu_types)} pod for model {args.model!r}: "
                  f"{sorted(t['dataset'] for t in large_targets)}")
            all_job_ids += submit_jobs(
                model=args.model, jobs=large_jobs, slug="large",
                n_splits=args.n_splits, num_random_configs=args.num_random_configs,
                num_bag_folds=args.num_bag_folds, time_limit=args.time_limit,
                results_dir=args.results_dir, cache_dir=args.cache_dir, mirror_repo=args.mirror_repo,
                profile=big_gpu_profile, throttle=args.throttle, dry_run=args.dry_run,
                max_train_samples_overrides=max_train_samples_overrides,
            )
        print(f"Done: {len(all_job_ids)} k8s Job(s) created: {all_job_ids}")
        return

    ok, failed = 0, []
    for t in targets:
        n_repeats = t.get("n_repeats", args.n_repeats or 10)
        cmd = [
            sys.executable, str(CLUSTER_DIR / "submit_job.py"),
            "--dataset", t["dataset"], "--target-idx", str(t["target_idx"]),
            "--model", args.model,
            "--n-repeats", str(n_repeats), "--n-splits", str(args.n_splits),
            "--config-indices", *[str(c) for c in args.config_indices],
            "--num-random-configs", str(args.num_random_configs),
            "--num-bag-folds", str(args.num_bag_folds),
            "--time-limit", str(args.time_limit),
            "--results-dir", args.results_dir, "--cache-dir", args.cache_dir,
            "--throttle", str(args.throttle),
        ]
        if t["dataset"] in max_train_samples_overrides:
            cmd += ["--max-train-samples", str(max_train_samples_overrides[t["dataset"]])]
        if args.profile:
            cmd += ["--profile", args.profile]
        if args.cluster:
            cmd += ["--cluster", args.cluster]
        if args.dry_run:
            cmd.append("--dry-run")

        result = subprocess.run(cmd, capture_output=True, text=True)
        key = f"{t['dataset']}__{t['target_idx']}"
        if result.returncode != 0:
            failed.append((key, result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "unknown error"))
            print(f"  [{key}] FAILED: {failed[-1][1]}")
        else:
            last_line = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
            print(f"  [{key}] {last_line}")
            ok += 1

    print(f"\nDone: {ok} submitted, {len(failed)} failed")
    for key, err in failed:
        print(f"  FAILED {key}: {err}")


if __name__ == "__main__":
    main()
