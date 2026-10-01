#!/usr/bin/env python
"""Submit one or a few DATASETS across the whole model roster -- the inverse
of submit_full_benchmark.py (one MODEL across every dataset).

Why this exists: adding a new dataset, or recomputing an existing one after a
dataset-definition fix (see docs/internal/invalidating-results.md), used to
mean one pod PER MODEL -- ~50 pods to touch one dataset. This submits ONE pod
per (Python-version image, CPU/GPU tier) GROUP instead -- in practice today,
up to 3: (main image, CPU), (main image, GPU), (py312 image, GPU) -- each pod
iterating every model in its group, sequentially, against every non-excluded
target of the requested dataset(s) (every target_idx, not just 0 -- a
multi-target dataset gets all of its targets in the same sweep).

k8s only (no SLURM path -- this is new tooling, not a port of an existing
SLURM mechanism). Reuses submit_job.py's resolve_time_limit/
resolve_max_train_samples/resolve_k8s_image/GPU_MODELS -- same resolution
logic as the per-model path, just grouped and invoked differently. See
cluster/k8s_entrypoint_by_dataset.sh for the per-task command this drives.

Usage:
    python cluster/submit_dataset_sweep.py --dataset synthetic_organic_pigments_raw \\
        --profile ~/workspace/k8s_profile.yaml --scope configs/v1/scope_default.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

CLUSTER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CLUSTER_DIR))
REPO_ROOT = CLUSTER_DIR.parent

from submit_job import (  # noqa: E402
    GPU_MODELS,
    _chunk,
    _k8s_name,
    _max_memory_tier,
    build_k8s_job_manifest_multimodel,
    resolve_k8s_image,
    resolve_profile,
    write_jobspec_multimodel,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", action="append", required=True, dest="datasets",
                         help="Dataset name(s) to sweep across every model. Repeatable.")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--scope", default=str(REPO_ROOT / "configs" / "v1" / "scope_default.json"))
    parser.add_argument("--targets-file", default=None,
                         help="Defaults to --scope's own targets_file (resolved relative to --scope).")
    parser.add_argument("--n-splits", type=int, default=None, help="Defaults to --scope's n_splits.")
    parser.add_argument("--num-random-configs", type=int, default=None)
    parser.add_argument("--num-bag-folds", type=int, default=None)
    parser.add_argument("--time-limit", type=float, default=None)
    parser.add_argument("--config-indices", type=int, nargs="+", default=[0])
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--mirror-repo", default=None)
    parser.add_argument("--throttle", type=int, default=None)
    parser.add_argument("--tasks-per-pod", type=int, default=5000)
    parser.add_argument(
        "--force-recompute", action="store_true",
        help="Recompute and overwrite already-cached results instead of skipping them -- "
             "the normal reason to use this script for a dataset that already has some "
             "results, e.g. after a dataset-definition fix. See "
             "docs/internal/invalidating-results.md.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    scope = json.loads(Path(args.scope).read_text())
    models: list[str] = scope["models"]
    n_splits = args.n_splits if args.n_splits is not None else scope["n_splits"]
    num_random_configs = args.num_random_configs if args.num_random_configs is not None else scope["num_random_configs"]
    num_bag_folds = args.num_bag_folds if args.num_bag_folds is not None else scope["num_bag_folds"]
    time_limit = args.time_limit if args.time_limit is not None else scope["time_limit"]
    results_dir = args.results_dir or scope.get("results_dir", "results/v1/data")
    cache_dir = args.cache_dir or scope.get("cache_dir", ".cache_v1")
    mirror_repo = args.mirror_repo or scope.get("mirror_repo", "HTW-KI-Werkstatt/RamanBench")
    throttle = args.throttle if args.throttle is not None else scope.get("throttle", 8)
    dataset_time_limit_overrides = scope.get("time_limit_overrides", {})
    max_train_samples_overrides = scope.get("max_train_samples_overrides", {})
    model_time_limit_overrides = scope.get("model_time_limit_overrides", {})
    model_max_train_samples_overrides = scope.get("model_max_train_samples_overrides", {})

    targets_file = args.targets_file
    if targets_file is None:
        targets_file = str((Path(args.scope).parent / scope["targets_file"]).resolve())
    targets = json.loads(Path(targets_file).read_text())

    requested = set(args.datasets)
    matched = [t for t in targets if t["dataset"] in requested and not t.get("excluded")]
    if not matched:
        print(f"No non-excluded targets found for dataset(s) {sorted(requested)} in {targets_file}", file=sys.stderr)
        sys.exit(1)
    found_datasets = {t["dataset"] for t in matched}
    missing = requested - found_datasets
    if missing:
        print(f"WARNING: no targets found for {sorted(missing)} (typo, or fully excluded)", file=sys.stderr)
    print(f"{len(matched)} target(s) across {len(found_datasets)} dataset(s): {sorted(found_datasets)}")

    profile = resolve_profile(args.profile, "k8s")
    if profile.get("backend") != "k8s":
        print("submit_dataset_sweep.py is k8s-only.", file=sys.stderr)
        sys.exit(1)

    # Group models by (image, is_gpu) -- the only two axes that actually force
    # a separate pod (different container/Python version, or GPU vs CPU-only
    # resource shape). Every model in a group runs sequentially in the same
    # pod(s), against every matched target.
    groups: dict[tuple[str, bool], list[str]] = defaultdict(list)
    for model in models:
        image = resolve_k8s_image(profile, model)
        use_gpu = model in GPU_MODELS
        groups[(image, use_gpu)].append(model)

    dataset_slug = "-".join(sorted(found_datasets))[:30]
    created_jobs = []
    for (image, use_gpu), group_models in groups.items():
        # repeat is always 0, matching every existing submission this project
        # has made so far (target_list.json's own n_repeats=1 convention);
        # folds 0..n_splits-1, one line per requested config_index.
        expanded = [
            (model, t["dataset"], t["target_idx"], 0, fold, config_index, t.get("n_repeats", 1))
            for model in group_models
            for t in matched
            for fold in range(n_splits)
            for config_index in args.config_indices
        ]

        tier = "gpu" if use_gpu else "cpu"
        image_slug = image.rsplit(":", 1)[-1].rsplit("/", 1)[-1]
        group_slug = f"dataset-{dataset_slug}-{image_slug}-{tier}"
        mem_tier = _max_memory_tier(
            [profile.get("mem_tiers", {}).get(m, profile.get("default_memory", "64G")) for m in group_models]
        ) if group_models else profile.get("default_memory", "64G")

        chunks = _chunk(expanded, profile.get("max_array_size", 5000))
        multi_part = len(chunks) > 1
        for part, chunk_tasks in enumerate(chunks):
            part_slug = f"p{part}" if multi_part else "main"
            jobspec_path = write_jobspec_multimodel(
                chunk_tasks, f"{group_slug}_{part_slug}",
                default_time_limit=time_limit,
                dataset_time_limit_overrides=dataset_time_limit_overrides,
                model_time_limit_overrides=model_time_limit_overrides,
                max_train_samples_overrides=max_train_samples_overrides,
                model_max_train_samples_overrides=model_max_train_samples_overrides,
            )
            job_name, configmap_name, n_pods, job_manifest = build_k8s_job_manifest_multimodel(
                group_slug=group_slug, part_slug=part_slug, n_tasks=len(chunk_tasks),
                n_splits=n_splits, num_random_configs=num_random_configs, num_bag_folds=num_bag_folds,
                time_limit=time_limit, results_dir=results_dir, cache_dir=cache_dir, mirror_repo=mirror_repo,
                profile=profile, image=image, mem_tier=mem_tier, use_gpu=use_gpu,
                tasks_per_pod=args.tasks_per_pod, throttle=throttle,
                force_recompute=args.force_recompute,
            )
            print(f"\n=== group ({image}, {'gpu' if use_gpu else 'cpu'}): {len(group_models)} model(s) "
                  f"x {len(matched)} target(s) x {n_splits} fold(s) x {len(args.config_indices)} config(s) "
                  f"= {len(chunk_tasks)} task(s) ===")
            print(f"  models: {sorted(group_models)}")
            print(f"  jobspec: {jobspec_path} ({len(chunk_tasks)} task(s))")
            print(f"  configmap: {configmap_name}  job: {job_name}  pods={n_pods} mem={mem_tier}")

            if args.dry_run:
                continue

            import subprocess

            import yaml

            namespace = profile.get("namespace", "default")
            # Same two-call pattern (and --validate=false reasoning) as
            # submit_job.py's submit_jobs_k8s -- this cluster's API server has
            # repeatedly timed out on client-side schema validation
            # ("failed to download openapi: ... TLS handshake timeout"); these
            # manifests are generated by this script from a fixed,
            # already-correct shape, not hand-written, so skipping that
            # validation trades no real safety for reliability.
            cm_yaml = subprocess.run(
                ["kubectl", "create", "configmap", configmap_name, "-n", namespace,
                 "--from-file", f"jobspec.txt={jobspec_path}", "--dry-run=client", "-o", "yaml"],
                check=True, capture_output=True, text=True,
            ).stdout
            subprocess.run(
                ["kubectl", "apply", "--validate=false", "-f", "-"], input=cm_yaml, text=True, check=True,
            )
            subprocess.run(
                ["kubectl", "apply", "--validate=false", "-f", "-"],
                input=yaml.safe_dump(job_manifest), text=True, check=True,
            )
            created_jobs.append(job_name)

    if not args.dry_run:
        print(f"\nDone: {len(created_jobs)} k8s Job(s) created: {created_jobs}")


if __name__ == "__main__":
    main()
