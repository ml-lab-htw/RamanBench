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

    - This script, ``run_experiment.sbatch``, and ``k8s_entrypoint.sh`` are
      all cluster-agnostic.
    - A profile YAML (``cluster/profiles/*.yaml``) supplies the actual
      sbatch dialect/resource values, OR (when the profile sets
      ``backend: k8s``) the Kubernetes image/PVC/namespace/resource values.
      ``cluster/profiles/example.yaml`` and ``k8s_example.yaml`` are
      templates with no real values; HTW/TU/k8s's real profiles (account,
      partition, mail, workspace, image, PVC) live in the private
      ``raman_bench_paper/cluster/profiles/{htw,tu,k8s}.yaml``.
    - ``--profile`` accepts either an explicit path, or (with no --profile
      and no --cluster) auto-detection via ``detect_cluster.py`` for the
      SLURM clusters -- which falls back to asking the user rather than
      guessing when ambiguous. k8s is never auto-detected (a machine can be
      a SLURM login node AND have kubectl configured at the same time), so
      it's always an explicit ``--cluster k8s``/``--profile ...k8s.yaml``.

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
import time
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

    if cluster_arg == "k8s":
        candidate = CLUSTER_DIR / "profiles" / "k8s.yaml"
        if not candidate.exists():
            raise FileNotFoundError(
                f"No profile at {candidate}. The real k8s profile lives in the private "
                "raman_bench_paper repo -- pass --profile explicitly (or copy/edit "
                "cluster/profiles/k8s_example.yaml)."
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


def resolve_k8s_resources(profile: dict, model: str, use_gpu: bool) -> dict:
    """Build a k8s container `resources` block, mirroring resolve_mem_flags/
    resolve_gpu_flags's per-model memory tiers and GPU-vs-CPU switch for the
    SLURM path."""
    mem = profile.get("mem_tiers", {}).get(model, profile.get("default_memory", "64G"))
    cpu = str(profile.get("default_cpu", "16"))
    limits = {"cpu": cpu, "memory": mem}
    if use_gpu:
        limits[profile.get("gpu_resource_key", "nvidia.com/gpu")] = str(profile.get("gpu_count", 1))
    return {"requests": {"cpu": cpu, "memory": mem}, "limits": limits}


# A single array task's full identity: (dataset, target_idx, repeat, fold,
# config_index, n_repeats). dataset/target_idx/n_repeats are carried per-line
# (rather than fixed --export env vars for the whole array) so one array can
# span multiple (dataset, target) pairs for the same model -- not just
# multiple (repeat, fold, config) tuples for a single fixed target. The
# single-target CLI path below just repeats the same three values on every
# line; the opportunistic scheduler (``opportunistic_scheduler.py``) is what
# actually varies them within one array.
Job = tuple[str, int, int, int, int, int]


def resolve_time_limit(
    default: float,
    dataset: str,
    dataset_time_limit_overrides: dict[str, float] | None = None,
    model_time_limit_overrides: dict[str, float] | float | None = None,
) -> float:
    """The time_limit for ONE task, resolved from its own dataset -- the
    per-task counterpart to ``opportunistic_scheduler.effective_time_limit``,
    which computes a single value for an entire chunk (the max needed by ANY
    dataset present in it). That whole-chunk max is still used as the
    array-wide ``--export TIME_LIMIT=...`` fallback (see ``submit_jobs``
    below), but applying it to every task regardless of that task's own
    dataset means a fast task sharing a chunk with e.g. an ``mlrod`` task
    inherits mlrod's inflated budget for no reason -- confirmed live on the
    cluster (job 36545: an `alzheimer` task sat at the full ~1350s/fold slice
    implied by mlrod's 10800s override despite alzheimer needing nowhere near
    that). This function is what lets ``write_jobspec`` give each line its
    own, dataset-appropriate value instead.

    Same override semantics as ``effective_time_limit`` (bump the default up,
    never down; dataset-keyed and model-keyed overrides both apply and the
    larger wins), just resolved for one dataset instead of maxed over a whole
    chunk's worth of datasets.

    ``model_time_limit_overrides`` is normally a dataset-keyed dict (EBM/
    ORIONMSP: "this model needs more time, but only on these specific wide
    datasets"). It may instead be a bare number, applied regardless of
    ``dataset`` -- for a model that's fast/deterministic enough that no
    single-fit HPO search ever runs against it under the routine
    config_index=0 backlog, a time cap doesn't bound anything meaningful; it
    only risks AutoGluon's bagged-fold-fitting strategy extrapolating from an
    early fold and aborting with ``TimeLimitExceeded`` before the (finite,
    just slower-than-guessed) remaining folds get a chance to finish --
    confirmed in practice for LR on wide/large datasets (wheat_lines,
    bacteria_identification: ~530-560s/fold x 8 sequential folds, well past
    the 3600s default, same failure mode already seen for PLS on mlrod). The
    real backstop in that case is the SLURM array's own ``--time`` (the
    profile's ``default_time``, e.g. 10 days) -- entirely independent of this
    value.
    """
    candidates = [default]
    if dataset_time_limit_overrides and dataset in dataset_time_limit_overrides:
        candidates.append(dataset_time_limit_overrides[dataset])
    if isinstance(model_time_limit_overrides, (int, float)):
        candidates.append(model_time_limit_overrides)
    elif model_time_limit_overrides and dataset in model_time_limit_overrides:
        candidates.append(model_time_limit_overrides[dataset])
    return max(candidates)


def resolve_max_train_samples(
    dataset: str,
    max_train_samples_overrides: dict[str, int] | None = None,
) -> int | None:
    """The ``--max-train-samples`` row-subsampling cap for ONE task, resolved
    from its own dataset -- ``None`` (no subsampling) unless ``dataset`` has an
    entry in ``max_train_samples_overrides`` (``configs/v1/scope_default.json``'s
    ``max_train_samples_overrides``, dataset-keyed, e.g. ``{"mlrod": 10000}``).

    Global, applies to every model (not a per-model override like
    ``model_time_limit_overrides`` -- there's no known need for a
    model-specific version of this yet, unlike time_limit's EBM/ORIONMSP/LR
    cases). Added 2026-09-25 after REZERONET's real ``TimeLimitExceeded`` on
    ``mlrod`` (130,061 rows) under the 600s/3-bag-fold compute-scaling
    settings -- subsampling the row count directly addresses the actual cost
    driver (more rows -> proportionally more per-epoch compute), rather than
    raising the time budget (which ``time_limit_overrides``/
    ``model_time_limit_overrides`` already do, but only through the SLURM
    path -- ``submit_full_benchmark.py``'s k8s branch doesn't read them).
    """
    if max_train_samples_overrides and dataset in max_train_samples_overrides:
        return max_train_samples_overrides[dataset]
    return None


def write_jobspec(
    jobs: list[Job],
    slug: str,
    *,
    default_time_limit: float,
    dataset_time_limit_overrides: dict[str, float] | None = None,
    model_time_limit_overrides: dict[str, float] | float | None = None,
    max_train_samples_overrides: dict[str, int] | None = None,
) -> Path:
    """One line per (dataset, target_idx, repeat, fold, config_index,
    n_repeats, time_limit, max_train_samples) task; array task N reads line
    N+1. ``time_limit`` (the 7th field) is resolved PER TASK from its own
    dataset via ``resolve_time_limit`` -- not one flat value for the whole
    array -- so a chunk mixing an oversized dataset (e.g. ``mlrod``) in with
    ordinary ones no longer forces every task in that chunk onto the
    oversized dataset's budget. ``run_experiment.sbatch``/``k8s_entrypoint.sh``
    read this field directly; the array-wide ``--export TIME_LIMIT=...`` env
    var (still the whole-chunk max -- see ``submit_jobs``) is kept only as a
    fallback for jobspec lines without this field (backward compat with any
    already-queued array whose jobspec predates this field).

    ``max_train_samples`` (the 8th field, added 2026-09-25) is resolved PER
    TASK from its own dataset via ``resolve_max_train_samples`` -- empty
    (blank field) unless the dataset has a ``max_train_samples_overrides``
    entry, in which case ``run_experiment.sbatch``/``k8s_entrypoint.sh`` add
    ``--max-train-samples <value>`` to the task's command; empty otherwise
    (the flag is simply omitted, matching ``run_experiment.py``'s own
    no-subsampling default). Global across every model, unlike
    ``model_time_limit_overrides`` -- no known need yet for a model-specific
    version.
    """
    JOBSPEC_DIR.mkdir(exist_ok=True)
    path = JOBSPEC_DIR / f"{slug}.txt"
    with open(path, "w") as f:
        for dataset, target_idx, repeat, fold, config_index, n_repeats in jobs:
            task_time_limit = resolve_time_limit(
                default_time_limit, dataset, dataset_time_limit_overrides, model_time_limit_overrides,
            )
            task_max_train_samples = resolve_max_train_samples(dataset, max_train_samples_overrides)
            max_train_samples_field = "" if task_max_train_samples is None else str(task_max_train_samples)
            f.write(
                f"{dataset} {target_idx} {repeat} {fold} {config_index} {n_repeats} "
                f"{task_time_limit} {max_train_samples_field}\n"
            )
    return path


def _chunk(jobs: list[Job], size: int) -> list[list[Job]]:
    """Split into groups of at most ``size`` -- SLURM's MaxArraySize (commonly 1001)
    rejects an array bigger than that with "Invalid job array specification", confirmed
    by a real failed submission (a 10-repeat x 3-fold x 51-config target needs 1530
    tasks). Each chunk becomes its own array-task-index-0-based jobspec/sbatch call, so
    a chunk boundary never needs to line up with any (dataset, target, repeat, fold,
    config) boundary."""
    return [jobs[i : i + size] for i in range(0, len(jobs), size)] or [[]]


_SBATCH_MAX_ATTEMPTS = 8
_SBATCH_RETRY_BACKOFF_SECONDS = 15


def _sbatch_with_retry(sbatch_args: list[str]) -> subprocess.CompletedProcess:
    """Run ``sbatch``, retrying a few times on a transient controller failure.

    Confirmed as a real, recurring production failure (not a one-off): a
    real multi-array submission run hit "sbatch: error: Slurm temporarily
    unable to accept job, sleeping and retrying" on 2/11 calls in one batch
    -- sbatch's OWN internal retry sometimes gives up and exits 1 instead of
    eventually succeeding (confirmed the very same sbatch_args succeeded
    immediately when just re-run by hand seconds later). A bare `check=True`
    call previously let one transient failure abort an entire multi-chunk
    submission loop, leaving later chunks/models never attempted."""
    last_result = None
    for attempt in range(1, _SBATCH_MAX_ATTEMPTS + 1):
        result = subprocess.run(sbatch_args, capture_output=True, text=True)
        if result.returncode == 0:
            return result
        last_result = result
        if attempt < _SBATCH_MAX_ATTEMPTS:
            print(
                f"  sbatch failed (attempt {attempt}/{_SBATCH_MAX_ATTEMPTS}): "
                f"{result.stderr.strip()} -- retrying in {_SBATCH_RETRY_BACKOFF_SECONDS}s",
                file=sys.stderr,
            )
            time.sleep(_SBATCH_RETRY_BACKOFF_SECONDS)
    raise subprocess.CalledProcessError(
        last_result.returncode, sbatch_args, output=last_result.stdout, stderr=last_result.stderr,
    )


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
    dataset_time_limit_overrides: dict[str, float] | None = None,
    model_time_limit_overrides: dict[str, float] | float | None = None,
    default_time_limit: float | None = None,
    max_train_samples_overrides: dict[str, int] | None = None,
) -> list[str]:
    """Submit ``jobs`` (all for one ``model`` -- resource flags are resolved once per
    call, so every task in an array must share the same GPU/CPU and memory tier) as
    one or more SLURM array jobs, chunked at the profile's MaxArraySize. Returns the
    list of submitted SLURM job IDs (empty on a dry run or a non-SLURM profile).

    ``time_limit`` is used ONLY for the array-wide ``--export TIME_LIMIT=...``
    fallback (read by ``run_experiment.sbatch`` only for a jobspec line that
    somehow lacks the per-task 7th field -- e.g. an older-format jobspec).
    Callers that pre-compute a whole-chunk ceiling (e.g.
    ``opportunistic_scheduler.effective_time_limit``, which folds in the max
    override needed by ANY dataset in the chunk) should pass that ceiling
    here.

    ``default_time_limit`` is the FLAT baseline each task's own
    ``resolve_time_limit`` call bumps up from -- deliberately a separate
    parameter from ``time_limit`` above: if the whole-chunk ceiling were used
    as this baseline instead, every task would inherit at least that ceiling
    (``max(ceiling, ...)`` can never go below ``ceiling``), silently
    reproducing the exact per-chunk-not-per-task bug this parameter exists to
    fix. Defaults to ``time_limit`` when omitted, which is correct for the
    single-(dataset,target) CLI path below (no chunk-wide inflation to worry
    about there -- every task already shares one dataset, so there's only
    ever one flat value in play, and ``dataset_time_limit_overrides``/
    ``model_time_limit_overrides`` are both None there too).

    ``dataset_time_limit_overrides``/``model_time_limit_overrides`` (both
    optional) let each task's own dataset bump ``default_time_limit`` up via
    ``resolve_time_limit``, so one array/chunk spanning multiple datasets
    doesn't force a fast dataset onto a slow dataset's budget.

    Shared by the single-(dataset,target) CLI path (``submit()`` below) and
    ``opportunistic_scheduler.py``'s multi-(dataset,target) backlog submission --
    the only difference between them is how ``jobs`` was built.
    """
    use_gpu = model in GPU_MODELS
    job_ids: list[str] = []
    # See the docstring above for why this must NOT be `time_limit` itself
    # when a caller passes a pre-inflated whole-chunk ceiling there.
    base_time_limit = time_limit if default_time_limit is None else default_time_limit

    if profile.get("backend") == "k8s":
        return submit_jobs_k8s(
            model=model, jobs=jobs, slug=slug, n_splits=n_splits,
            num_random_configs=num_random_configs, num_bag_folds=num_bag_folds, time_limit=time_limit,
            results_dir=results_dir, cache_dir=cache_dir, mirror_repo=mirror_repo,
            profile=profile, throttle=throttle, dry_run=dry_run,
            dataset_time_limit_overrides=dataset_time_limit_overrides,
            model_time_limit_overrides=model_time_limit_overrides,
            default_time_limit=base_time_limit,
            max_train_samples_overrides=max_train_samples_overrides,
        )

    if not profile.get("slurm", True):
        # No SLURM -- run every job as a local subprocess.
        print(f"Profile {profile['name']!r} has no SLURM -- running {len(jobs)} job(s) locally.")
        for dataset, target_idx, repeat, fold, cfg, n_repeats in jobs:
            task_time_limit = resolve_time_limit(
                base_time_limit, dataset, dataset_time_limit_overrides, model_time_limit_overrides,
            )
            task_max_train_samples = resolve_max_train_samples(dataset, max_train_samples_overrides)
            task_slug = f"{dataset}_{target_idx}_{model}".replace("/", "_")
            scratch_dir = str(REPO_ROOT / ".scratch_v1" / f"local_{task_slug}_{repeat}_{fold}_{cfg}")
            cmd = [
                sys.executable, str(REPO_ROOT / "scripts" / "run_experiment.py"),
                "--dataset", dataset, "--target-idx", str(target_idx), "--model", model,
                "--repeat", str(repeat), "--fold", str(fold),
                "--n-repeats", str(n_repeats), "--n-splits", str(n_splits),
                "--config-index", str(cfg), "--num-random-configs", str(num_random_configs),
                "--num-bag-folds", str(num_bag_folds), "--time-limit", str(task_time_limit),
                "--results-dir", results_dir, "--cache-dir", cache_dir, "--mirror-repo", mirror_repo,
                "--scratch-dir", scratch_dir,
            ]
            if task_max_train_samples is not None:
                cmd += ["--max-train-samples", str(task_max_train_samples)]
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
        jobspec_path = write_jobspec(
            chunk_jobs, part_slug,
            default_time_limit=base_time_limit,
            dataset_time_limit_overrides=dataset_time_limit_overrides,
            model_time_limit_overrides=model_time_limit_overrides,
            max_train_samples_overrides=max_train_samples_overrides,
        )

        sbatch_args = [
            "sbatch",
            f"--array=0-{len(chunk_jobs) - 1}%{throttle}",
            f"--job-name=RB_{model}_{part_slug}",
            "--cpus-per-task", str(profile.get("default_cpus_per_task", 16)),
            "--time", profile.get("default_time", "10-00:00:00"),
        ]
        if profile.get("workspace"):
            # run_experiment.sbatch's #SBATCH --output/--error paths are relative
            # (.logs/...) and SLURM resolves those against the job's working
            # directory at submission time -- NOT wherever the script later `cd`s
            # to. Without an explicit --chdir, a caller invoked from anywhere other
            # than the workspace itself (a cron job's default cwd is $HOME, not the
            # repo) submits a job whose output file can't be created, which SLURM
            # reports as an immediate FAILED with no log ever written. Confirmed by
            # a real failure: the opportunistic scheduler's first live submission
            # (invoked from a login shell's home directory) failed every task this
            # way before this fix.
            sbatch_args += ["--chdir", str(Path(profile["workspace"]).expanduser())]
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
            # Array-wide fallback only -- run_experiment.sbatch prefers the
            # per-task 7th jobspec field (see write_jobspec/resolve_time_limit
            # above) and falls back to this env var just for jobspec lines
            # that don't have it (older-format jobspecs already queued).
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
        result = _sbatch_with_retry(sbatch_args)
        print(result.stdout.strip())
        # sbatch's stdout is "Submitted batch job <id>"
        job_ids.append(result.stdout.strip().rsplit(" ", 1)[-1])

    return job_ids


def _k8s_name(*parts: str) -> str:
    """A DNS-1123-safe k8s object name: lowercase, alnum + '-' only, <=63 chars.
    Collisions from truncation are astronomically unlikely for this project's
    (model, dataset, part) name space and aren't guarded against.

    Truncated to 57, not 63: an Indexed Job's own pods are named
    ``<job-name>-<index>``, so a job name using the full 63-char DNS-1123
    budget produces an invalid (>63-char) pod name the moment the index
    suffix is appended -- confirmed failing for real
    (``rb-perpetual-booster-marine-pathogens-binary-0-perpetual-booste``,
    64 chars, rejected by the API server with "will not able to create pod
    with invalid DNS label"). 57 leaves room for "-" plus up to a 5-digit
    index (this project's own ``tasks_per_pod`` ceiling), i.e. jobs with up
    to 99999 pods.
    """
    raw = "-".join(parts).lower().replace("_", "-").replace(".", "-").replace("/", "-")
    raw = "".join(c for c in raw if c.isalnum() or c == "-").strip("-")
    return raw[:57].rstrip("-")


def _abs_under_workspace(path: str, workspace: str) -> str:
    """Resolve a relative RESULTS_DIR/CACHE_DIR path to an absolute path under
    the PVC-mounted workspace. The k8s image bakes code into /app (the
    Dockerfile's WORKDIR) -- profile["workspace"] on the PVC holds no code at
    all, only persistent data. So results_dir/cache_dir must be made absolute
    under the PVC mount HERE, not resolved by a `cd $WORKSPACE` in the
    entrypoint (that used to `cd` into the PVC path before running `python
    scripts/run_experiment.py`, a path that only exists in /app -- confirmed
    failing every task with "No such file or directory" once a fail-fast
    check caught the silent-fallback bug this replaced). Absolute paths make
    cwd irrelevant either way. An already-absolute path (or no workspace
    configured) passes through unchanged."""
    if workspace and not path.startswith("/"):
        return f"{workspace.rstrip('/')}/{path}"
    return path


def _build_k8s_job_manifest(
    *,
    model: str,
    part_slug: str,
    n_tasks: int,
    n_splits: int,
    num_random_configs: int,
    num_bag_folds: int,
    time_limit: float,
    results_dir: str,
    cache_dir: str,
    mirror_repo: str,
    profile: dict,
    use_gpu: bool,
    tasks_per_pod: int,
    throttle: int,
) -> tuple[str, str, int, dict]:
    """Build the (job_name, configmap_name, n_pods, job_manifest) for one k8s
    Indexed Job -- pure construction, no cluster I/O, so it's directly
    testable without mocking kubectl. Split out of ``submit_jobs_k8s`` so the
    manifest-building logic (env vars, secrets, image pull policy, node
    affinity, priority class) can be unit-tested on its own."""
    job_name = _k8s_name("rb", model, part_slug)
    configmap_name = _k8s_name("rb-jobspec", model, part_slug)
    n_pods = -(-n_tasks // tasks_per_pod)  # ceil division, no float rounding surprises
    namespace = profile.get("namespace", "default")

    workspace = profile.get("workspace") or ""

    env = [
        {"name": "MODEL", "value": model},
        {"name": "N_SPLITS", "value": str(n_splits)},
        {"name": "NUM_RANDOM_CONFIGS", "value": str(num_random_configs)},
        {"name": "NUM_BAG_FOLDS", "value": str(num_bag_folds)},
        {"name": "TIME_LIMIT", "value": str(time_limit)},
        {"name": "RESULTS_DIR", "value": _abs_under_workspace(results_dir, workspace)},
        {"name": "CACHE_DIR", "value": _abs_under_workspace(cache_dir, workspace)},
        {"name": "MIRROR_REPO", "value": mirror_repo},
        {"name": "USE_GPU", "value": "1" if use_gpu else "0"},
        {"name": "JOBSPEC", "value": "/jobspec/jobspec.txt"},
        {"name": "TASKS_PER_POD", "value": str(tasks_per_pod)},
    ]
    if profile.get("hf_secret"):
        env += [
            {"name": "HF_TOKEN", "valueFrom": {"secretKeyRef": {"name": profile["hf_secret"], "key": "HF_TOKEN"}}},
            {"name": "HUGGING_FACE_HUB_TOKEN",
             "valueFrom": {"secretKeyRef": {"name": profile["hf_secret"], "key": "HUGGING_FACE_HUB_TOKEN"}}},
        ]
    if profile.get("tabpfn_secret"):
        # One-time TabPFN license acceptance token -- required non-interactively
        # by any model with a TabPFN backbone (RAMANPFN, TABPFN-V3, TABPFN-WIDE,
        # REALTABPFN-*, ...) since tabpfn>=9.0 gates weight downloads on this.
        # Confirmed failing without it: tabpfn.errors.TabPFNLicenseError on a
        # real RAMANPFN smoke test.
        env.append({
            "name": "TABPFN_TOKEN",
            "valueFrom": {"secretKeyRef": {"name": profile["tabpfn_secret"], "key": "TABPFN_TOKEN"}},
        })
    if profile.get("wandb_secret"):
        # Presence of WANDB_API_KEY is what turns on per-task tracking (see
        # scripts/run_experiment.py's _wandb_enabled) -- no separate flag needed.
        env.append({
            "name": "WANDB_API_KEY",
            "valueFrom": {"secretKeyRef": {"name": profile["wandb_secret"], "key": "WANDB_API_KEY"}},
        })
        if profile.get("wandb_project"):
            env.append({"name": "WANDB_PROJECT", "value": profile["wandb_project"]})
        if profile.get("wandb_entity"):
            env.append({"name": "WANDB_ENTITY", "value": profile["wandb_entity"]})

    volume_mounts = [{"name": "workspace", "mountPath": profile.get("pvc_mount_path", "/data")},
                      {"name": "jobspec", "mountPath": "/jobspec"}]
    volumes = [
        {"name": "workspace", "persistentVolumeClaim": {"claimName": profile["pvc_claim_name"]}},
        {"name": "jobspec", "configMap": {"name": configmap_name}},
    ]
    if profile.get("kaggle_secret"):
        volume_mounts.append({"name": "kaggle", "mountPath": "/root/.kaggle"})
        volumes.append({"name": "kaggle", "secret": {"secretName": profile["kaggle_secret"]}})

    pod_spec: dict = {
        "restartPolicy": "Never",
        "containers": [{
            "name": "run-experiment",
            "image": profile["image"],
            # Without this, a node that already cached this tag from an earlier
            # submission silently keeps running the stale image after a rebuild+
            # push of the same mutable tag -- confirmed in practice: a real fix
            # pushed under the same `:v1` tag was ignored by an already-warm node.
            "imagePullPolicy": "Always",
            "env": env,
            "volumeMounts": volume_mounts,
            "resources": resolve_k8s_resources(profile, model, use_gpu),
        }],
        "volumes": volumes,
    }
    if profile.get("image_pull_secret"):
        pod_spec["imagePullSecrets"] = [{"name": profile["image_pull_secret"]}]
    if profile.get("node_selector"):
        pod_spec["nodeSelector"] = profile["node_selector"]
    if profile.get("node_affinity_match_expressions"):
        pod_spec["affinity"] = {"nodeAffinity": {"requiredDuringSchedulingIgnoredDuringExecution": {
            "nodeSelectorTerms": [{"matchExpressions": profile["node_affinity_match_expressions"]}]
        }}}
    if profile.get("priority_class_name"):
        # Cluster-endorsed shared-tenancy courtesy for routine sweep jobs (e.g.
        # an "unimportant" priority class some k8s clusters document HPO/sweep
        # jobs under) -- higher-priority pods preempt these rather than queuing
        # behind them, so this is stronger than tuning tasks_per_pod/throttle
        # alone. See your cluster's own priority-class documentation and set
        # `priority_class_name` in your private profile accordingly.
        pod_spec["priorityClassName"] = profile["priority_class_name"]

    job_manifest = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": job_name, "namespace": namespace,
                     "labels": {"app": "raman-bench", **profile.get("extra_pod_labels", {})}},
        "spec": {
            "completions": n_pods,
            "parallelism": min(throttle, n_pods) or 1,
            "completionMode": "Indexed",
            "backoffLimit": 0,
            "template": {"metadata": {"labels": {"app": "raman-bench"}}, "spec": pod_spec},
        },
    }
    return job_name, configmap_name, n_pods, job_manifest


def submit_jobs_k8s(
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
    dataset_time_limit_overrides: dict[str, float] | None = None,
    model_time_limit_overrides: dict[str, float] | float | None = None,
    default_time_limit: float | None = None,
    max_train_samples_overrides: dict[str, int] | None = None,
) -> list[str]:
    """k8s analogue of the SLURM branch of ``submit_jobs`` above: one
    Kubernetes Indexed Job per chunk (k8s's array-job equivalent --
    ``JOB_COMPLETION_INDEX`` plays the role of ``SLURM_ARRAY_TASK_ID``, read
    by ``cluster/k8s_entrypoint.sh``). The jobspec text is delivered via a
    ConfigMap (created alongside the Job, one per chunk) rather than a file on
    a shared filesystem -- unlike the SLURM clusters, a k8s job's submitting
    machine has no guaranteed filesystem path in common with the pods it
    schedules, but it does always have kubectl API access, and ConfigMaps
    comfortably fit a jobspec chunk (<=1000 short lines, well under the 1MiB
    ConfigMap size cap).

    Returns the list of created Job names (empty on a dry run)."""
    use_gpu = model in GPU_MODELS
    job_names: list[str] = []
    base_time_limit = time_limit if default_time_limit is None else default_time_limit
    namespace = profile.get("namespace", "default")

    max_array_size = profile.get("max_array_size", DEFAULT_MAX_ARRAY_SIZE)
    # Batch multiple jobspec lines into each pod -- courtesy default for a
    # shared, multi-tenant cluster, see k8s_entrypoint.sh's module comment for
    # the full reasoning (fewer, longer-running pods instead of one pod per
    # individual task). tasks_per_pod=1 (the profile default) reproduces the
    # original one-task-per-pod behavior exactly.
    tasks_per_pod = max(1, int(profile.get("tasks_per_pod", 1)))
    chunks = _chunk(jobs, max_array_size)
    multi_part = len(chunks) > 1

    print(
        f"{len(jobs)} job(s) as {slug} (k8s backend, namespace={namespace}, tasks_per_pod={tasks_per_pod})"
        + (f" -- split into {len(chunks)} Job(s) of <={max_array_size} task(s) each" if multi_part else "")
    )

    for part, chunk_jobs in enumerate(chunks):
        part_slug = f"{slug}_p{part}" if multi_part else slug
        jobspec_path = write_jobspec(
            chunk_jobs, part_slug,
            default_time_limit=base_time_limit,
            dataset_time_limit_overrides=dataset_time_limit_overrides,
            model_time_limit_overrides=model_time_limit_overrides,
            max_train_samples_overrides=max_train_samples_overrides,
        )
        job_name, configmap_name, n_pods, job_manifest = _build_k8s_job_manifest(
            model=model, part_slug=part_slug, n_tasks=len(chunk_jobs), n_splits=n_splits,
            num_random_configs=num_random_configs, num_bag_folds=num_bag_folds,
            time_limit=time_limit, results_dir=results_dir, cache_dir=cache_dir,
            mirror_repo=mirror_repo, profile=profile, use_gpu=use_gpu,
            tasks_per_pod=tasks_per_pod, throttle=throttle,
        )

        print(f"  jobspec: {jobspec_path}  ({len(chunk_jobs)} task(s))")
        print(f"  configmap: {configmap_name}  job: {job_name}  pods={n_pods} (<= {tasks_per_pod} task(s)/pod)")
        if dry_run:
            continue

        cm_yaml = subprocess.run(
            ["kubectl", "create", "configmap", configmap_name, "-n", namespace,
             "--from-file", f"jobspec.txt={jobspec_path}", "--dry-run=client", "-o", "yaml"],
            check=True, capture_output=True, text=True,
        ).stdout
        # --validate=false: client-side `kubectl apply` downloads the full OpenAPI
        # schema on every single call to validate against it -- this cluster's API
        # server has repeatedly timed out on that specific request ("failed to
        # download openapi: ... TLS handshake timeout") while a plain `kubectl get`
        # against the same server succeeds immediately, confirmed as a real,
        # recurring submission failure (not a one-off) across multiple models'
        # submissions in one session. The manifests here are generated by this
        # script itself from a fixed, already-correct shape, not hand-written, so
        # skipping client-side schema validation trades no real safety for
        # reliability.
        subprocess.run(
            ["kubectl", "apply", "--validate=false", "-f", "-"], input=cm_yaml, text=True, check=True,
        )
        subprocess.run(
            ["kubectl", "apply", "--validate=false", "-f", "-"],
            input=yaml.safe_dump(job_manifest), text=True, check=True,
        )
        job_names.append(job_name)

    return job_names


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
    max_train_samples: int | None = None,
) -> None:
    jobs: list[Job] = [
        (dataset, target_idx, repeat, fold, cfg, n_repeats)
        for repeat in range(n_repeats)
        for fold in range(n_splits)
        for cfg in config_indices
    ]
    slug = f"{dataset}_{target_idx}_{model}".replace("/", "_")
    max_train_samples_overrides = {dataset: max_train_samples} if max_train_samples is not None else None
    submit_jobs(
        model=model, jobs=jobs, slug=slug, n_splits=n_splits,
        num_random_configs=num_random_configs, num_bag_folds=num_bag_folds, time_limit=time_limit,
        results_dir=results_dir, cache_dir=cache_dir, mirror_repo=mirror_repo,
        profile=profile, throttle=throttle, dry_run=dry_run,
        max_train_samples_overrides=max_train_samples_overrides,
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
    parser.add_argument("--cluster", default=None, choices=["htw", "tu", "local", "k8s"])
    parser.add_argument("--throttle", type=int, default=8, help="Max concurrent array tasks")
    parser.add_argument(
        "--max-train-samples", type=int, default=None,
        help="Row-subsampling cap for this dataset (see resolve_max_train_samples); "
             "omitted means no subsampling.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    profile = resolve_profile(args.profile, args.cluster)
    submit(
        dataset=args.dataset, target_idx=args.target_idx, model=args.model,
        n_repeats=args.n_repeats, n_splits=args.n_splits, config_indices=args.config_indices,
        num_random_configs=args.num_random_configs, num_bag_folds=args.num_bag_folds,
        time_limit=args.time_limit, results_dir=args.results_dir, cache_dir=args.cache_dir,
        mirror_repo=args.mirror_repo, profile=profile, throttle=args.throttle, dry_run=args.dry_run,
        max_train_samples=args.max_train_samples,
    )


if __name__ == "__main__":
    main()
