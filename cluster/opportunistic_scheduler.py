#!/usr/bin/env python
"""One tick of the opportunistic, capacity-aware benchmark scheduler.

Design (see the v1 refactor plan's Phase 4b): the full curated benchmark is a lot
of real, sustained compute demand against a shared cluster, but running it is
explicitly *not urgent* -- it should fill idle capacity when present and back off
(by simply not submitting more, never by cancelling anything already queued) when
the cluster is busy with other users' work.

Each invocation of this script does exactly ONE tick:
    1. Compute the backlog: for every model in ``--scope``, every non-excluded
       (dataset, target) row in its target list, every (repeat, fold) in its
       repeated-k-fold scheme, check whether ``results.pkl`` already exists under
       ``results_dir`` (the same cache convention ``run_experiment.py`` /
       ``submit_job.py`` already use) AND whether it already has a queued/running
       SLURM task (via squeue/sacct + the submitted jobspec files -- see
       ``_in_flight_targets``). Recomputed fresh every tick -- nothing is tracked
       in a separate state file, so a completed task just stops appearing in the
       backlog next time around (self-healing: nothing can get out of sync).
    2. Check capacity: idle CPUs on the target partition (``sinfo``) and this
       user's own current resident task count (``squeue``).
    3. If there's room, submit exactly one modest-sized chunk from the backlog
       (combining as many (dataset, target) pairs for one model into one SLURM
       array as fits, chunked at the profile's MaxArraySize) and stop. If not,
       skip this tick entirely. Either way, log the decision.

Meant to be driven by a cron entry (see ``raman_bench_paper/cluster/`` for the
real per-institution scope + crontab wrapper) -- this script itself is generic
and carries no institution-specific values.

Usage
-----
    python cluster/opportunistic_scheduler.py --scope cluster/scope.example.json \\
        --profile cluster/profiles/htw.yaml --log cluster/.scheduler_log/htw.jsonl

    # See what it WOULD do without submitting or writing a log entry:
    python cluster/opportunistic_scheduler.py --scope ... --profile ... --dry-run
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import subprocess
import sys
from pathlib import Path

CLUSTER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CLUSTER_DIR))

from submit_job import JOBSPEC_DIR, REPO_ROOT, Job, resolve_profile, submit_jobs  # noqa: E402

DEFAULT_CHUNK_SIZE = 300
DEFAULT_COURTESY_CEILING = 200
DEFAULT_MIN_IDLE_CPUS = 32
DEFAULT_THROTTLE = 8
DEFAULT_TIME_LIMIT = 3600


def load_scope(path: str | Path) -> dict:
    """Load a scope JSON. ``targets_file`` is resolved relative to the scope file's
    own directory (a static config asset checked into the repo, e.g.
    ``configs/v1/target_list.json``) if not already absolute -- so the scope file
    works regardless of the caller's cwd. ``results_dir``/``cache_dir`` are runtime
    data directories that live under the RamanBench workspace root -- resolved
    relative to ``submit_job.REPO_ROOT`` (the same anchor ``submit_job.py`` already
    uses for ``JOBSPEC_DIR`` and the ``--chdir`` it passes every SLURM submission)
    if not already absolute. This matters because the routine cron tick invokes
    this script from an arbitrary cwd (cron's default is ``$HOME``, not the
    workspace) -- left cwd-relative, every on-disk results.pkl check silently saw
    "nothing is ever done" whenever a tick fired from cron, so the backlog never
    shrank there even once real completions existed (confirmed in practice: this
    is why the in-flight dedup fix alone didn't stop duplicate submissions on the
    real cron path -- the cache-based half of the exclusion was cwd-blind)."""
    path = Path(path)
    with open(path) as f:
        scope = json.load(f)
    targets_file = Path(scope["targets_file"])
    if not targets_file.is_absolute():
        scope["targets_file"] = str((path.parent / targets_file).resolve())
    for key in ("results_dir", "cache_dir"):
        if key in scope and not Path(scope[key]).is_absolute():
            scope[key] = str((REPO_ROOT / scope[key]).resolve())
    return scope


def _ag_name(model_key: str) -> str:
    """The name TabArena's ConfigGenerator gives the default-config experiment
    (e.g. Prep_RIDGE's registry key is "RIDGE" but its ag_name is "Ridge") -- this
    is the directory name under ``results_dir`` the cache actually lives in, NOT
    the CLI/registry model key. Confirmed empirically: config_index 0's experiment
    is always named ``f"{ag_name}_c1_BAG_L1"`` regardless of model or num_bag_folds.
    """
    from raman_bench.models.registry import infer_model_cls

    return infer_model_cls(model_key).ag_name


def _in_flight_targets(model: str, user: str) -> set[tuple[str, int, int, int]]:
    """(dataset, target_idx, repeat, fold) tuples for ``model`` (config_index 0
    only -- that's all the routine backlog tracks) that already have a queued or
    running SLURM task, so ``compute_backlog`` doesn't re-count them.

    Every array task's identity lives in a jobspec file under
    ``submit_job.JOBSPEC_DIR`` (one line per task), and ``submit_jobs`` always
    names the SLURM job ``RB_{model}_{part_slug}`` where ``part_slug`` is exactly
    that jobspec file's stem -- so a live job name tells us which file to read
    back, with no separate state tracking needed. Checked via squeue first;
    sacct is also checked (restricted to the last hour) as a fallback in case a
    job submitted moments ago hasn't propagated to squeue yet on this cluster.

    Confirmed necessary in practice: without this, three duplicate 300-task PLS
    arrays got submitted in one afternoon because every tick recomputed the same
    "not yet cached" backlog for tasks that were already queued/running from the
    previous tick.
    """
    name_prefix = f"RB_{model}_"
    part_slugs: set[str] = set()

    try:
        out = subprocess.run(
            ["squeue", "-h", "-u", user, "-t", "pending,running", "-o", "%.200j"],
            capture_output=True, text=True, check=True,
        ).stdout
        for name in out.splitlines():
            name = name.strip()
            if name.startswith(name_prefix):
                part_slugs.add(name[len(name_prefix):])
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass  # no SLURM here -- fall through to cache-only backlog, as before

    try:
        cutoff = (datetime.datetime.now() - datetime.timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S")
        out = subprocess.run(
            ["sacct", "-h", "-n", "-X", "-u", user, "--starttime", cutoff,
             "--state", "PENDING,RUNNING,REQUEUED", "-o", "JobName%200"],
            capture_output=True, text=True, check=True,
        ).stdout
        for name in out.splitlines():
            name = name.strip()
            if name.startswith(name_prefix):
                part_slugs.add(name[len(name_prefix):])
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    targets: set[tuple[str, int, int, int]] = set()
    for part_slug in part_slugs:
        jobspec_path = JOBSPEC_DIR / f"{part_slug}.txt"
        if not jobspec_path.exists():
            continue
        with open(jobspec_path) as f:
            for line in f:
                parts = line.split()
                if len(parts) != 6:
                    continue
                dataset, target_idx, repeat, fold, config_index, _n_repeats = parts
                if int(config_index) != 0:
                    continue
                targets.add((dataset, int(target_idx), int(repeat), int(fold)))
    return targets


def compute_backlog(scope: dict, profile: dict) -> dict[str, list[Job]]:
    """Return ``{model: [(dataset, target_idx, repeat, fold, config_index=0, n_repeats), ...]}``
    for every not-yet-cached AND not-already-queued/running task in scope. Only
    the routine default config (config_index 0) is considered -- per the plan's
    Decision 3, HPO sweeps are opted into per model separately, not part of the
    routine opportunistic backlog.
    """
    results_dir = Path(scope["results_dir"])
    n_splits = scope["n_splits"]
    user = profile.get("cron_user") or _current_user()

    with open(scope["targets_file"]) as f:
        targets = json.load(f)

    backlog: dict[str, list[Job]] = {}
    for model in scope["models"]:
        ag_name = _ag_name(model)
        experiment_name = f"{ag_name}_c1_BAG_L1"
        in_flight = _in_flight_targets(model, user)
        model_backlog: list[Job] = []
        for row in targets:
            if row.get("excluded"):
                continue
            dataset = row["dataset"]
            target_idx = row["target_idx"]
            n_repeats = row["n_repeats"]
            task_name = f"{dataset}__{target_idx}"
            for repeat in range(n_repeats):
                for fold in range(n_splits):
                    if (dataset, target_idx, repeat, fold) in in_flight:
                        continue
                    cache_file = results_dir / experiment_name / task_name / f"{repeat}_{fold}" / "results.pkl"
                    if not cache_file.exists():
                        model_backlog.append((dataset, target_idx, repeat, fold, 0, n_repeats))
        backlog[model] = model_backlog
    return backlog


def check_capacity(profile: dict, min_idle_cpus: int, courtesy_ceiling: int) -> tuple[bool, str]:
    """Return (has_room, reason). ``has_room`` requires BOTH idle cluster capacity
    (this partition isn't busy with other users' work) AND this user's own
    resident (pending+running) task count staying under a courtesy ceiling (so
    the scheduler itself never grows into "occupying the whole cluster" even if
    the partition looks idle to everyone else too)."""
    partition = profile.get("partition")
    sinfo_cmd = ["sinfo", "-h", "-o", "%C"]
    if partition:
        sinfo_cmd += ["-p", partition]
    try:
        out = subprocess.run(sinfo_cmd, capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        return False, f"sinfo failed ({e}) -- treating as no capacity, conservatively"

    # sinfo -o '%C' -> "allocated/idle/other/total" CPUs, one line per matching
    # partition state; sum idle across lines (a partition can appear more than
    # once, e.g. mixed node states).
    idle_total = 0
    for line in out.splitlines():
        m = re.match(r"(\d+)/(\d+)/(\d+)/(\d+)", line.strip())
        if m:
            idle_total += int(m.group(2))

    try:
        squeue_out = subprocess.run(
            ["squeue", "-h", "-u", profile.get("cron_user") or _current_user(), "-t", "pending,running"],
            capture_output=True, text=True, check=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        return False, f"squeue failed ({e}) -- treating as at ceiling, conservatively"
    my_resident = len([line for line in squeue_out.splitlines() if line.strip()])

    if idle_total < min_idle_cpus:
        return False, f"only {idle_total} idle CPU(s) on partition {partition!r} (need >={min_idle_cpus})"
    if my_resident >= courtesy_ceiling:
        return False, f"already have {my_resident} resident task(s) (courtesy ceiling {courtesy_ceiling})"
    return True, f"{idle_total} idle CPU(s), {my_resident} resident task(s) -- room to submit"


def _current_user() -> str:
    import getpass

    return getpass.getuser()


def pick_chunk(backlog: dict[str, list[Job]], chunk_size: int) -> tuple[str | None, list[Job]]:
    """Pick the first model (in scope order) with a nonempty backlog and take up
    to ``chunk_size`` tasks from it. One chunk never spans more than one model --
    resource flags (mem/GPU) are resolved once per SLURM array submission."""
    for model, jobs in backlog.items():
        if jobs:
            return model, jobs[:chunk_size]
    return None, []


def effective_time_limit(scope: dict, model: str, chunk: list[Job]) -> float:
    """The flat ``scope["time_limit"]`` (default 3600s), bumped up to the largest
    applicable entry from two override sources among the datasets present in
    this chunk:

    - ``scope["time_limit_overrides"]``: a dataset-name-keyed dict, same shape
      as a cluster profile's ``mem_tiers``, applied regardless of which model
      is running (a dataset property, e.g. row count, that makes ANY model
      slower -- see ``mlrod`` below).
    - ``scope["model_time_limit_overrides"]``: a model-name -> dataset-name ->
      seconds dict, applied only when ``model`` matches (a MODEL-specific
      slowness on specific datasets, not a dataset property every model
      shares -- see EBM/ORIONMSP below). A chunk never spans more than one
      model (``pick_chunk`` guarantees this -- resource flags are resolved
      once per array submission), so looking up just ``model``'s own
      sub-dict is enough; no cross-model leakage is possible.

    One SLURM array submission gets one time_limit -- if a chunk mixes an
    oversized dataset in with ordinary ones, everyone in that chunk gets the
    larger budget, which is harmless (a task that finishes early just exits
    early; this only raises the ceiling, it doesn't force anyone to run
    longer).

    Confirmed necessary in practice (dataset-keyed): ``mlrod`` has 130,061
    rows -- the largest dataset in the target list by a wide margin (next is
    78,500; the median across all 158 targets is 179). AutoGluon's own
    bagged-fold-fitting strategy extrapolates from how long the first of 8
    sequential PLS folds takes and aborts early (TimeLimitExceeded) if it
    predicts the remaining folds won't fit in what's left of the budget --
    observed taking ~950s for fold 1 alone, i.e. needing roughly 950*8 =~
    7600s total against the default 3600s, well before actually running out
    of wall-clock time.

    Confirmed necessary in practice (model-keyed): a flat dataset-keyed
    override can't express "EBM needs more time on wide datasets but other
    models on the same dataset are fine" -- PLS/RF/etc. finish
    microgel_synthesis (11,084 features) in seconds, while EBM's own
    interaction-detection pre-scan (see
    ``preprocessing.wrapped_models.Prep_EBM._fit``) needs much longer just on
    that one model. Forcing every model sharing a wide dataset onto EBM's
    budget would be wasteful (nothing about e.g. PLS gets slower there), and
    forcing a single global time_limit bump big enough for EBM would apply to
    every OTHER model in scope too, everywhere -- hence the extra,
    model-scoped override layer instead of stretching the dataset-only one to
    do a model-specific job."""
    default = scope.get("time_limit", DEFAULT_TIME_LIMIT)
    dataset_overrides = scope.get("time_limit_overrides", {})
    model_overrides = scope.get("model_time_limit_overrides", {}).get(model, {})
    if not dataset_overrides and not model_overrides:
        return default
    datasets_in_chunk = {dataset for dataset, *_rest in chunk}
    applicable = [dataset_overrides[ds] for ds in datasets_in_chunk if ds in dataset_overrides]
    applicable += [model_overrides[ds] for ds in datasets_in_chunk if ds in model_overrides]
    return max([default, *applicable])


def log_tick(log_path: str | Path | None, entry: dict) -> None:
    if log_path is None:
        return
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def run_tick(scope: dict, profile: dict, log_path: str | Path | None, dry_run: bool) -> dict:
    timestamp = datetime.datetime.now(datetime.UTC).isoformat()
    backlog = compute_backlog(scope, profile)
    backlog_sizes = {model: len(jobs) for model, jobs in backlog.items()}
    total_backlog = sum(backlog_sizes.values())

    if total_backlog == 0:
        entry = {"timestamp": timestamp, "action": "skip", "reason": "backlog empty", "backlog_sizes": backlog_sizes}
        print(json.dumps(entry, indent=2))
        log_tick(log_path, entry)
        return entry

    if not profile.get("slurm", True):
        entry = {
            "timestamp": timestamp, "action": "skip",
            "reason": f"profile {profile.get('name')!r} has no SLURM -- opportunistic scheduling needs a real cluster",
            "backlog_sizes": backlog_sizes,
        }
        print(json.dumps(entry, indent=2))
        log_tick(log_path, entry)
        return entry

    min_idle_cpus = scope.get("min_idle_cpus", DEFAULT_MIN_IDLE_CPUS)
    courtesy_ceiling = scope.get("courtesy_ceiling", DEFAULT_COURTESY_CEILING)
    has_room, reason = check_capacity(profile, min_idle_cpus, courtesy_ceiling)

    if not has_room:
        entry = {
            "timestamp": timestamp, "action": "skip", "reason": reason,
            "backlog_sizes": backlog_sizes, "total_backlog": total_backlog,
        }
        print(json.dumps(entry, indent=2))
        log_tick(log_path, entry)
        return entry

    chunk_size = scope.get("chunk_size", DEFAULT_CHUNK_SIZE)
    model, chunk = pick_chunk(backlog, chunk_size)
    time_limit = effective_time_limit(scope, model, chunk)

    slug = f"{scope['name']}_{model}_{timestamp.replace(':', '').replace('-', '').split('.')[0]}"
    job_ids = submit_jobs(
        model=model, jobs=chunk, slug=slug,
        n_splits=scope["n_splits"], num_random_configs=scope.get("num_random_configs", 50),
        num_bag_folds=scope.get("num_bag_folds", 8), time_limit=time_limit,
        results_dir=scope["results_dir"], cache_dir=scope.get("cache_dir", ".cache_v1"),
        mirror_repo=scope.get("mirror_repo", "HTW-KI-Werkstatt/RamanBench"),
        profile=profile, throttle=scope.get("throttle", DEFAULT_THROTTLE), dry_run=dry_run,
    )

    entry = {
        "timestamp": timestamp, "action": "dry_run_submit" if dry_run else "submit",
        "reason": reason, "model": model, "n_submitted": len(chunk), "time_limit": time_limit,
        "job_ids": job_ids, "backlog_sizes": backlog_sizes, "total_backlog": total_backlog,
    }
    print(json.dumps(entry, indent=2))
    log_tick(log_path, entry)
    return entry


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scope", required=True, help="Path to a scope JSON file (see scope.example.json)")
    parser.add_argument("--profile", required=True, help="Path to a cluster profile YAML")
    parser.add_argument("--log", default=None, help="Append a JSON line per tick to this file")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    scope = load_scope(args.scope)
    profile = resolve_profile(args.profile, None)
    run_tick(scope, profile, args.log, args.dry_run)


if __name__ == "__main__":
    main()
