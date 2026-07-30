#!/usr/bin/env python
"""Safety-net sweep for orphaned per-job scratch directories.

``run_experiment.sbatch``'s own ``trap cleanup EXIT`` handles the vast
majority of exits (normal completion, errors, and the graceful pre-timeout
``USR1`` signal) -- but SIGKILL (a hard OOM-kill by the kernel, `scancel -9`,
or a node failure) cannot be trapped by any shell script. This is the actual
root cause diagnosed for the ~27TB of orphaned AutoGluon predictor artifacts
on the HTW/TU clusters: SLURM does **not** auto-requeue a cancelled/killed
array task, so its scratch dir (and everything AutoGluon wrote under it)
would otherwise never be cleaned up.

This script cross-references every scratch dir under
``{cache_dir}/scratch_v1/`` (named ``{job_id}_{array_task_id}`` -- see
``run_experiment.sbatch``) against ``squeue``/``sacct``, and removes any
whose owning job is no longer running/pending AND has passed a grace period
(to avoid racing a job that only just started, before it's visible in
``squeue`` yet, or whose own EXIT trap is about to fire).

Usage
-----
    python cluster/janitor.py --cache-dir .cache_v1                 # dry-run report
    python cluster/janitor.py --cache-dir .cache_v1 --delete        # actually remove orphans
    python cluster/janitor.py --cache-dir .cache_v1 --grace-minutes 30 --delete
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ScratchDirStatus:
    path: Path
    job_id: str
    array_task_id: str
    age_minutes: float
    job_active: bool
    orphaned: bool


def _active_job_ids() -> set[str]:
    """Job ids currently in squeue (running, pending, or otherwise not yet finished)."""
    try:
        out = subprocess.run(
            ["squeue", "-h", "-o", "%A"], capture_output=True, text=True, timeout=30, check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return set()
    ids = set()
    for line in out.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        # Array tasks show as "<job_id>_<array_task_id>" or just "<job_id>".
        ids.add(line.split("_")[0])
        ids.add(line)
    return ids


def scan(cache_dir: str, grace_minutes: float = 15.0) -> list[ScratchDirStatus]:
    scratch_root = Path(cache_dir) / "scratch_v1"
    if not scratch_root.exists():
        return []

    active_ids = _active_job_ids()
    now = time.time()
    statuses = []
    for d in sorted(scratch_root.iterdir()):
        if not d.is_dir():
            continue
        job_json = d / "job.json"
        job_id = array_task_id = "unknown"
        if job_json.exists():
            try:
                meta = json.loads(job_json.read_text())
                job_id = str(meta.get("job_id", "unknown"))
                array_task_id = str(meta.get("array_task_id", "unknown"))
            except (json.JSONDecodeError, OSError):
                pass
        else:
            # Fall back to parsing the directory name convention itself.
            parts = d.name.rsplit("_", 1)
            if len(parts) == 2:
                job_id, array_task_id = parts

        age_minutes = (now - d.stat().st_mtime) / 60.0
        job_active = job_id in active_ids or f"{job_id}_{array_task_id}" in active_ids
        orphaned = (not job_active) and (age_minutes >= grace_minutes)
        statuses.append(ScratchDirStatus(
            path=d, job_id=job_id, array_task_id=array_task_id,
            age_minutes=age_minutes, job_active=job_active, orphaned=orphaned,
        ))
    return statuses


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache-dir", default=".cache_v1")
    parser.add_argument(
        "--grace-minutes", type=float, default=15.0,
        help="Minimum age before an inactive-job scratch dir is considered orphaned",
    )
    parser.add_argument("--delete", action="store_true", help="Actually remove orphaned dirs (default: dry-run report)")
    args = parser.parse_args()

    statuses = scan(args.cache_dir, grace_minutes=args.grace_minutes)
    if not statuses:
        print(f"No scratch dirs found under {args.cache_dir}/scratch_v1/")
        return

    total_size = 0
    orphan_count = 0
    for s in statuses:
        size_mb = sum(f.stat().st_size for f in s.path.rglob("*") if f.is_file()) / 1e6
        total_size += size_mb
        flag = "ORPHAN" if s.orphaned else ("active" if s.job_active else "grace-period")
        print(f"  [{flag:12s}] {s.path.name:24s} job={s.job_id:>10s} age={s.age_minutes:7.1f}min size={size_mb:8.1f}MB")
        if s.orphaned:
            orphan_count += 1
            if args.delete:
                shutil.rmtree(s.path, ignore_errors=True)
                print(f"               -> deleted")

    print(f"\n{len(statuses)} scratch dir(s), {total_size:.1f}MB total, {orphan_count} orphaned"
          + (" (deleted)" if args.delete else " (dry-run -- pass --delete to remove)"))


if __name__ == "__main__":
    main()
