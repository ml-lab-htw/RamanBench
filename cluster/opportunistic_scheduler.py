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
       ``submit_job.py`` already use). Recomputed fresh every tick -- nothing is
       tracked in a separate state file, so a completed task just stops appearing
       in the backlog next time around (self-healing: nothing can get out of sync).
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

from submit_job import Job, resolve_profile, submit_jobs  # noqa: E402

DEFAULT_CHUNK_SIZE = 300
DEFAULT_COURTESY_CEILING = 200
DEFAULT_MIN_IDLE_CPUS = 32
DEFAULT_THROTTLE = 8


def load_scope(path: str | Path) -> dict:
    """Load a scope JSON. ``targets_file`` is resolved relative to the scope file's
    own directory (a static config asset checked into the repo, e.g.
    ``configs/v1/target_list.json``) if not already absolute -- so the scope file
    works regardless of the caller's cwd. ``results_dir``/``cache_dir`` are left as
    given: they're runtime data directories (typically resolved relative to the
    cluster workspace cwd), not repo-relative assets."""
    path = Path(path)
    with open(path) as f:
        scope = json.load(f)
    targets_file = Path(scope["targets_file"])
    if not targets_file.is_absolute():
        scope["targets_file"] = str((path.parent / targets_file).resolve())
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


def compute_backlog(scope: dict) -> dict[str, list[Job]]:
    """Return ``{model: [(dataset, target_idx, repeat, fold, config_index=0, n_repeats), ...]}``
    for every not-yet-cached task in scope. Only the routine default config
    (config_index 0) is considered -- per the plan's Decision 3, HPO sweeps are
    opted into per model separately, not part of the routine opportunistic backlog.
    """
    results_dir = Path(scope["results_dir"])
    n_splits = scope["n_splits"]

    with open(scope["targets_file"]) as f:
        targets = json.load(f)

    backlog: dict[str, list[Job]] = {}
    for model in scope["models"]:
        ag_name = _ag_name(model)
        experiment_name = f"{ag_name}_c1_BAG_L1"
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


def log_tick(log_path: str | Path | None, entry: dict) -> None:
    if log_path is None:
        return
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def run_tick(scope: dict, profile: dict, log_path: str | Path | None, dry_run: bool) -> dict:
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    backlog = compute_backlog(scope)
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

    slug = f"{scope['name']}_{model}_{timestamp.replace(':', '').replace('-', '').split('.')[0]}"
    job_ids = submit_jobs(
        model=model, jobs=chunk, slug=slug,
        n_splits=scope["n_splits"], num_random_configs=scope.get("num_random_configs", 50),
        num_bag_folds=scope.get("num_bag_folds", 8), time_limit=scope.get("time_limit", 3600),
        results_dir=scope["results_dir"], cache_dir=scope.get("cache_dir", ".cache_v1"),
        mirror_repo=scope.get("mirror_repo", "HTW-KI-Werkstatt/RamanBench"),
        profile=profile, throttle=scope.get("throttle", DEFAULT_THROTTLE), dry_run=dry_run,
    )

    entry = {
        "timestamp": timestamp, "action": "dry_run_submit" if dry_run else "submit",
        "reason": reason, "model": model, "n_submitted": len(chunk),
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
