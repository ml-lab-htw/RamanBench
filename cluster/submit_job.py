#!/usr/bin/env python
"""Generalized cluster submitter for RamanBench v1.

One job array task = one (model, dataset, target, seed, config-index)
experiment -- "every single evaluation gets a completely fresh job" (see the
v1 refactor plan). Generalizes the institution-specific logic previously
hardcoded in ``raman_bench_paper/cluster/submit_per_model.sh`` (per-model
memory tiers, GPU-vs-CPU node selection, TU-vs-HTW sbatch dialect) into a
profile-driven, no-secrets-in-public-repo design:

    - This script and ``run_experiment.sbatch`` are cluster-agnostic.
    - A profile YAML (``cluster/profiles/*.yaml``) supplies the actual
      sbatch dialect/resource values. ``cluster/profiles/example.yaml`` is a
      template with no real values; HTW/TU's real profiles (account,
      partition, mail, workspace) live in the private
      ``raman_bench_paper/cluster/profiles/{htw,tu}.yaml``.
    - ``--profile`` accepts either an explicit path, or (with no --profile
      and no --cluster) auto-detection via ``detect_cluster.py`` -- which
      falls back to asking the user rather than guessing when ambiguous.

Usage
-----
    # Auto-detect cluster, submit one array covering 3 seeds x 1 config (default only)
    python cluster/submit_job.py --dataset wheat_lines --target-idx 0 --model PLS \\
        --seeds 0 1 2 --config-indices 0 --results-dir results/v1/data

    # Explicit profile, HPO sweep (default + 5 random configs) for one seed
    python cluster/submit_job.py --profile cluster/profiles/htw.yaml \\
        --dataset wheat_lines --target-idx 0 --model REZERONET \\
        --seeds 0 --config-indices 0 1 2 3 4 5 --num-random-configs 5

    # No cluster available -- run locally instead (prompts unless --yes)
    python cluster/submit_job.py --profile cluster/profiles/local.yaml \\
        --dataset wheat_lines --target-idx 0 --model PLS --seeds 0
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from detect_cluster import detect_cluster

REPO_ROOT = Path(__file__).resolve().parent.parent
CLUSTER_DIR = Path(__file__).resolve().parent
JOBSPEC_DIR = REPO_ROOT / ".slurm_jobspecs"

with open(CLUSTER_DIR / "gpu_models.json") as f:
    GPU_MODELS = set(json.load(f))


def load_profile(path: str | Path) -> dict:
    with open(path) as f:
        profile = yaml.safe_load(f)
    return profile


def resolve_profile(profile_arg: str | None, cluster_arg: str | None) -> dict:
    if profile_arg:
        return load_profile(profile_arg)
    if cluster_arg:
        candidate = CLUSTER_DIR / "profiles" / f"{cluster_arg}.yaml"
        if not candidate.exists():
            raise FileNotFoundError(
                f"No profile for --cluster {cluster_arg!r} at {candidate}. "
                "Pass --profile explicitly (e.g. a private HTW/TU profile)."
            )
        return load_profile(candidate)

    detection = detect_cluster()
    if detection.cluster in ("htw", "tu"):
        candidate = CLUSTER_DIR / "profiles" / f"{detection.cluster}.yaml"
        if candidate.exists():
            print(f"Auto-detected cluster: {detection.cluster} ({detection.reason})")
            return load_profile(candidate)
        print(
            f"Auto-detected cluster {detection.cluster!r} but no bundled profile exists "
            f"in the public repo at {candidate} -- this cluster's real profile lives in "
            "the private raman_bench_paper repo. Pass --profile explicitly."
        )
    if detection.cluster == "none":
        print(f"No SLURM cluster detected ({detection.reason}).")
        answer = input("Run locally instead? [y/N] (or Ctrl-C to request cluster access): ").strip().lower()
        if answer == "y":
            return load_profile(CLUSTER_DIR / "profiles" / "local.yaml")
        raise SystemExit("Aborted -- request cluster access, or pass --profile/--cluster explicitly.")

    raise SystemExit(
        f"Cluster detection was ambiguous ({detection.reason}). "
        "Pass --profile <path> or --cluster {htw,tu,local} explicitly."
    )


def resolve_mem_flags(profile: dict, model: str) -> list[str]:
    if profile.get("mem_flag_style") == "per_cpu":
        return [f"--mem-per-cpu={profile.get('per_cpu_mem', '25GB')}"]
    mem = profile.get("mem_tiers", {}).get(model, profile.get("default_mem", "64G"))
    return [f"--mem={mem}"]


def resolve_gpu_flags(profile: dict, use_gpu: bool) -> list[str]:
    if not use_gpu:
        return []
    if profile.get("gpu_flag_style") == "gpus_per_task":
        return ["--gpus-per-task=1"]
    return ["--gres=gpu:1"]


def write_jobspec(jobs: list[tuple[int, int]], slug: str) -> Path:
    """One line per (seed, config_index) task; array task N reads line N+1."""
    JOBSPEC_DIR.mkdir(exist_ok=True)
    path = JOBSPEC_DIR / f"{slug}.txt"
    with open(path, "w") as f:
        for seed, config_index in jobs:
            f.write(f"{seed} {config_index}\n")
    return path


def submit(
    *,
    dataset: str,
    target_idx: int,
    model: str,
    seeds: list[int],
    config_indices: list[int],
    num_random_configs: int,
    num_bag_folds: int,
    time_limit: float,
    results_dir: str,
    cache_dir: str,
    mirror_repo: str,
    profile: dict,
    throttle: int,
    dry_run: bool,
) -> None:
    use_gpu = model in GPU_MODELS
    jobs = [(seed, cfg) for seed in seeds for cfg in config_indices]

    if not profile.get("slurm", True):
        # No SLURM -- run every (seed, config) job as a local subprocess.
        print(f"Profile {profile['name']!r} has no SLURM -- running {len(jobs)} job(s) locally.")
        for seed, cfg in jobs:
            slug = f"{dataset}_{target_idx}_{model}".replace("/", "_")
            scratch_dir = str(REPO_ROOT / ".scratch_v1" / f"local_{slug}_{seed}_{cfg}")
            cmd = [
                sys.executable, str(REPO_ROOT / "scripts" / "run_experiment.py"),
                "--dataset", dataset, "--target-idx", str(target_idx), "--model", model,
                "--seed", str(seed), "--seeds", *[str(s) for s in seeds],
                "--config-index", str(cfg), "--num-random-configs", str(num_random_configs),
                "--num-bag-folds", str(num_bag_folds), "--time-limit", str(time_limit),
                "--results-dir", results_dir, "--cache-dir", cache_dir, "--mirror-repo", mirror_repo,
                "--scratch-dir", scratch_dir,
            ]
            if use_gpu:
                cmd.append("--use-gpu")
            print(f"  seed={seed} config_index={cfg}: {' '.join(cmd)}")
            if not dry_run:
                try:
                    subprocess.run(cmd, check=True)
                finally:
                    shutil.rmtree(scratch_dir, ignore_errors=True)
        return

    slug = f"{dataset}_{target_idx}_{model}".replace("/", "_")
    jobspec_path = write_jobspec(jobs, slug)

    sbatch_args = [
        "sbatch",
        f"--array=0-{len(jobs) - 1}%{throttle}",
        f"--job-name=RB_{model}_{slug}",
        "--cpus-per-task", str(profile.get("default_cpus_per_task", 16)),
        "--time", profile.get("default_time", "10-00:00:00"),
    ]
    sbatch_args += resolve_mem_flags(profile, model)
    sbatch_args += resolve_gpu_flags(profile, use_gpu)
    if profile.get("account"):
        sbatch_args.append(f"--account={profile['account']}")
    if profile.get("partition"):
        sbatch_args.append(f"--partition={profile['partition']}")
    if profile.get("mail_user"):
        sbatch_args += [f"--mail-user={profile['mail_user']}", f"--mail-type={profile.get('mail_type', 'FAIL')}"]
    sbatch_args += profile.get("extra_sbatch_args", [])

    export_vars = ",".join([
        f"DATASET={dataset}",
        f"TARGET_IDX={target_idx}",
        f"MODEL={model}",
        f"SEEDS={' '.join(str(s) for s in seeds)}",
        f"NUM_RANDOM_CONFIGS={num_random_configs}",
        f"NUM_BAG_FOLDS={num_bag_folds}",
        f"TIME_LIMIT={time_limit}",
        f"RESULTS_DIR={results_dir}",
        f"CACHE_DIR={cache_dir}",
        f"MIRROR_REPO={mirror_repo}",
        f"USE_GPU={1 if use_gpu else 0}",
        f"JOBSPEC={jobspec_path}",
        f"ACTIVATION={profile.get('activation') or 'conda'}",
        f"CONDA_ENV={profile.get('conda_env') or ''}",
        f"VENV_PATH={profile.get('venv_path') or ''}",
        f"WORKSPACE={profile.get('workspace') or ''}",
    ])
    sbatch_args += ["--export", export_vars, str(CLUSTER_DIR / "run_experiment.sbatch")]

    print(f"{len(jobs)} job(s) ({len(seeds)} seed(s) x {len(config_indices)} config(s)) as array: {slug}")
    print(f"  jobspec: {jobspec_path}")
    print(f"  {' '.join(sbatch_args)}")
    if dry_run:
        return
    result = subprocess.run(sbatch_args, capture_output=True, text=True, check=True)
    print(result.stdout.strip())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--target-idx", type=int, default=0)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument(
        "--config-indices", type=int, nargs="+", default=[0],
        help="0 = default config only (routine sweep); 1..N = HPO pool configs (opt-in per model)",
    )
    parser.add_argument("--num-random-configs", type=int, default=50)
    parser.add_argument("--num-bag-folds", type=int, default=8)
    parser.add_argument("--time-limit", type=float, default=3600)
    parser.add_argument("--results-dir", default="results/v1/data")
    parser.add_argument("--cache-dir", default=".cache_v1")
    parser.add_argument("--mirror-repo", default="HTW-KI-Werkstatt/RamanBench")
    parser.add_argument("--profile", default=None, help="Path to a cluster profile YAML")
    parser.add_argument("--cluster", default=None, choices=["htw", "tu", "local"])
    parser.add_argument("--throttle", type=int, default=8, help="Max concurrent array tasks")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    profile = resolve_profile(args.profile, args.cluster)
    submit(
        dataset=args.dataset, target_idx=args.target_idx, model=args.model,
        seeds=args.seeds, config_indices=args.config_indices,
        num_random_configs=args.num_random_configs, num_bag_folds=args.num_bag_folds,
        time_limit=args.time_limit, results_dir=args.results_dir, cache_dir=args.cache_dir,
        mirror_repo=args.mirror_repo, profile=profile, throttle=args.throttle, dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
