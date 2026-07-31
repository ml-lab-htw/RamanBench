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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--targets-file", required=True, help="JSON list of {dataset, target_idx, excluded}")
    parser.add_argument("--profile", default=None)
    parser.add_argument("--cluster", default=None, choices=["htw", "tu", "local"])
    parser.add_argument(
        "--n-repeats", type=int, default=None,
        help="Fallback only, used for a target lacking its own 'n_repeats' -- the "
             "per-target value (dataset-size-adaptive, from build_target_list.py) is "
             "used whenever present. Defaults to 10 if neither is available.",
    )
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--config-indices", type=int, nargs="+", default=[0])
    parser.add_argument("--num-random-configs", type=int, default=50)
    parser.add_argument("--num-bag-folds", type=int, default=8)
    parser.add_argument("--time-limit", type=float, default=3600)
    parser.add_argument("--results-dir", default="results/v1/data")
    parser.add_argument("--cache-dir", default=".cache_v1")
    parser.add_argument("--throttle", type=int, default=8)
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
