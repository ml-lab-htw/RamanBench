#!/usr/bin/env python
"""Submit one array job per (dataset, target) for a given model.

Reads a JSON target list (list of {"dataset": ..., "target_idx": ...,
"excluded": bool, ...}) and calls cluster/submit_job.py once per
non-excluded entry -- i.e. one job array per (dataset, target), each array
covering every requested seed x config-index combination for that target.
This is the "run this model across the whole benchmark" primitive
model-agent's workflow describes.

Usage:
    python cluster/submit_full_benchmark.py --model PLS --targets-file targets.json \\
        --profile ~/workspace/htw_v1_profile.yaml --seeds 0 1 2 --config-indices 0
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
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
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
    print(f"Submitting {len(targets)} target(s) for model {args.model!r} "
          f"({len(args.seeds)} seed(s) x {len(args.config_indices)} config(s) each)")

    ok, failed = 0, []
    for t in targets:
        cmd = [
            sys.executable, str(CLUSTER_DIR / "submit_job.py"),
            "--dataset", t["dataset"], "--target-idx", str(t["target_idx"]),
            "--model", args.model,
            "--seeds", *[str(s) for s in args.seeds],
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
