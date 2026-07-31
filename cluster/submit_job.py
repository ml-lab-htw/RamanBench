#!/usr/bin/env python
"""Generalized cluster submitter for RamanBench v1.

One job array task = one (model, dataset, target, repeat, fold, config-index)
experiment -- "every single evaluation gets a completely fresh job" (see the
v1 refactor plan). Splitting is real repeated k-fold CV (see
``raman_bench.splitting``, matching TabArena's own documented convention for
custom datasets), so the full sweep for one (dataset, target, model) fans out
over every (repeat, fold) pair, not just a list of seeds. Generalizes the
institution-specific logic previously hardcoded in
``raman_bench_paper/cluster/submit_per_model.sh`` (per-model memory tiers,
GPU-vs-CPU node selection, TU-vs-HTW sbatch dialect) into a profile-driven,
no-secrets-in-public-repo design:

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
    # Auto-detect cluster, submit one array covering 10 repeats x 3 folds x 1 config (default only)
    python cluster/submit_job.py --dataset wheat_lines --target-idx 0 --model PLS \\
        --n-repeats 10 --n-splits 3 --config-indices 0 --results-dir results/v1/data

    # Explicit profile, HPO sweep (default + 5 random configs)
    python cluster/submit_job.py --profile cluster/profiles/htw.yaml \\
        --dataset wheat_lines --target-idx 0 --model REZERONET \\
        --n-repeats 10 --n-splits 3 --config-indices 0 1 2 3 4 5 --num-random-configs 5

    # No cluster available -- run locally instead (prompts unless --yes)
    python cluster/submit_job.py --profile cluster/profiles/local.yaml \\
        --dataset wheat_lines --target-idx 0 --model PLS --n-repeats 10 --n-splits 3
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

# SLURM's MaxArraySize commonly defaults to 1001 (array indices 0..1000); a profile
# can override via max_array_size if its cluster is configured differently. Confirmed
# via a real failed submission: "sbatch: error: ... Invalid job array specification"
# for a 1530-task array (10 repeats x 3 folds x 51 configs) on a cluster with the
# common default.
DEFAULT_MAX_ARRAY_SIZE = 1000

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


# A single array task's full identity: (dataset, target_idx, repeat, fold,
# config_index, n_repeats). dataset/target_idx/n_repeats are carried per-line
# (rather than fixed --export env vars for the whole array) so one array can
# span multiple (dataset, target) pairs for the same model -- not just
# multiple (repeat, fold, config) tuples for a single fixed target. The
# single-target CLI path below just repeats the same three values on every
# line; the opportunistic scheduler (``opportunistic_scheduler.py``) is what
# actually varies them within one array.
Job = tuple[str, int, int, int, int, int]


def write_jobspec(jobs: list[Job], slug: str) -> Path:
    """One line per (dataset, target_idx, repeat, fold, config_index, n_repeats)
    task; array task N reads line N+1."""
    JOBSPEC_DIR.mkdir(exist_ok=True)
    path = JOBSPEC_DIR / f"{slug}.txt"
    with open(path, "w") as f:
        for dataset, target_idx, repeat, fold, config_index, n_repeats in jobs:
            f.write(f"{dataset} {target_idx} {repeat} {fold} {config_index} {n_repeats}\n")
    return path


def _chunk(jobs: list[Job], size: int) -> list[list[Job]]:
    """Split into groups of at most ``size`` -- SLURM's MaxArraySize (commonly 1001)
    rejects an array bigger than that with "Invalid job array specification", confirmed
    by a real failed submission (a 10-repeat x 3-fold x 51-config target needs 1530
    tasks). Each chunk becomes its own array-task-index-0-based jobspec/sbatch call, so
    a chunk boundary never needs to line up with any (dataset, target, repeat, fold,
    config) boundary."""
    return [jobs[i : i + size] for i in range(0, len(jobs), size)] or [[]]


def submit_jobs(
    *,
    model: str,
    jobs: list[Job],
    slug: str,
    n_splits: int,
    num_random_configs: int,
    num_bag_folds: int,
    time_limit: float,
    results_dir: str,
    cache_dir: str,
    mirror_repo: str,
    profile: dict,
    throttle: int,
    dry_run: bool,
) -> list[str]:
    """Submit ``jobs`` (all for one ``model`` -- resource flags are resolved once per
    call, so every task in an array must share the same GPU/CPU and memory tier) as
    one or more SLURM array jobs, chunked at the profile's MaxArraySize. Returns the
    list of submitted SLURM job IDs (empty on a dry run or a non-SLURM profile).

    Shared by the single-(dataset,target) CLI path (``submit()`` below) and
    ``opportunistic_scheduler.py``'s multi-(dataset,target) backlog submission --
    the only difference between them is how ``jobs`` was built.
    """
    use_gpu = model in GPU_MODELS
    job_ids: list[str] = []

    if not profile.get("slurm", True):
        # No SLURM -- run every job as a local subprocess.
        print(f"Profile {profile['name']!r} has no SLURM -- running {len(jobs)} job(s) locally.")
        for dataset, target_idx, repeat, fold, cfg, n_repeats in jobs:
            task_slug = f"{dataset}_{target_idx}_{model}".replace("/", "_")
            scratch_dir = str(REPO_ROOT / ".scratch_v1" / f"local_{task_slug}_{repeat}_{fold}_{cfg}")
            cmd = [
                sys.executable, str(REPO_ROOT / "scripts" / "run_experiment.py"),
                "--dataset", dataset, "--target-idx", str(target_idx), "--model", model,
                "--repeat", str(repeat), "--fold", str(fold),
                "--n-repeats", str(n_repeats), "--n-splits", str(n_splits),
                "--config-index", str(cfg), "--num-random-configs", str(num_random_configs),
                "--num-bag-folds", str(num_bag_folds), "--time-limit", str(time_limit),
                "--results-dir", results_dir, "--cache-dir", cache_dir, "--mirror-repo", mirror_repo,
                "--scratch-dir", scratch_dir,
            ]
            if use_gpu:
                cmd.append("--use-gpu")
            print(f"  {dataset}[{target_idx}] repeat={repeat} fold={fold} config_index={cfg}: {' '.join(cmd)}")
            if not dry_run:
                try:
                    subprocess.run(cmd, check=True)
                finally:
                    shutil.rmtree(scratch_dir, ignore_errors=True)
        return job_ids

    max_array_size = profile.get("max_array_size", DEFAULT_MAX_ARRAY_SIZE)
    chunks = _chunk(jobs, max_array_size)
    multi_part = len(chunks) > 1

    print(
        f"{len(jobs)} job(s) as {slug}"
        + (f" -- split into {len(chunks)} array(s) of <={max_array_size} (SLURM MaxArraySize)" if multi_part else "")
    )

    for part, chunk_jobs in enumerate(chunks):
        part_slug = f"{slug}_p{part}" if multi_part else slug
        jobspec_path = write_jobspec(chunk_jobs, part_slug)

        sbatch_args = [
            "sbatch",
            f"--array=0-{len(chunk_jobs) - 1}%{throttle}",
            f"--job-name=RB_{model}_{part_slug}",
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
            sbatch_args += [
                f"--mail-user={profile['mail_user']}", f"--mail-type={profile.get('mail_type', 'FAIL')}"
            ]
        sbatch_args += profile.get("extra_sbatch_args", [])

        export_vars = ",".join([
            f"MODEL={model}",
            f"N_SPLITS={n_splits}",
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

        print(f"  jobspec: {jobspec_path}")
        print(f"  {' '.join(sbatch_args)}")
        if dry_run:
            continue
        result = subprocess.run(sbatch_args, capture_output=True, text=True, check=True)
        print(result.stdout.strip())
        # sbatch's stdout is "Submitted batch job <id>"
        job_ids.append(result.stdout.strip().rsplit(" ", 1)[-1])

    return job_ids


def submit(
    *,
    dataset: str,
    target_idx: int,
    model: str,
    n_repeats: int,
    n_splits: int,
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
    jobs: list[Job] = [
        (dataset, target_idx, repeat, fold, cfg, n_repeats)
        for repeat in range(n_repeats)
        for fold in range(n_splits)
        for cfg in config_indices
    ]
    slug = f"{dataset}_{target_idx}_{model}".replace("/", "_")
    submit_jobs(
        model=model, jobs=jobs, slug=slug, n_splits=n_splits,
        num_random_configs=num_random_configs, num_bag_folds=num_bag_folds, time_limit=time_limit,
        results_dir=results_dir, cache_dir=cache_dir, mirror_repo=mirror_repo,
        profile=profile, throttle=throttle, dry_run=dry_run,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--target-idx", type=int, default=0)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--n-repeats", type=int, default=10,
        help="Repeats of the repeated-kfold split (matches TabArena's own custom-dataset convention)",
    )
    parser.add_argument("--n-splits", type=int, default=3, help="Folds per repeat")
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
        n_repeats=args.n_repeats, n_splits=args.n_splits, config_indices=args.config_indices,
        num_random_configs=args.num_random_configs, num_bag_folds=args.num_bag_folds,
        time_limit=args.time_limit, results_dir=args.results_dir, cache_dir=args.cache_dir,
        mirror_repo=args.mirror_repo, profile=profile, throttle=args.throttle, dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
