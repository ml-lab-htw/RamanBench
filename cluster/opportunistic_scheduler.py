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

from submit_job import (  # noqa: E402
    GPU_MODELS,
    JOBSPEC_DIR,
    REPO_ROOT,
    Job,
    resolve_profile,
    resolve_time_limit,
    submit_jobs,
)

DEFAULT_CHUNK_SIZE = 300
# GPU-tier models are real training loops (deep learning, foundation-model
# inference), not the near-instant classical fits CPU-tier models are -- the
# same flat 300-task chunk size that drains in minutes for LR/RF/etc. can keep
# a GPU array running for many hours under throttle-limited concurrency.
# Confirmed live: an NN_TORCH array (throttle=8, 300 tasks) needed over 2
# hours just to reach 168/300 (roughly 5-6 min/task at 8-way concurrency),
# implying ~3.5+ hours to fully drain one array alone -- directly against the
# scheduler's own "opportunistic: short bounded bursts, back off between
# ticks" design goal. 32 tasks at throttle=8 is 4 waves of concurrent
# execution -- sized to finish within roughly an hour even for a model this
# slow; faster GPU models just drain it well under that.
DEFAULT_GPU_CHUNK_SIZE = 32
# courtesy_ceiling (200) alone still lets several gpu_chunk_size-sized arrays
# (32 each) queue up back to back before tripping -- confirmed live: 4
# separate NN_TORCH arrays (32 tasks each) piled up again after the
# courtesy_ceiling/gpu_chunk_size fix, since 4*32=128 is still well under
# 200. By the time the backlog is dominated by GPU-tier/slow models (the
# fast CPU baselines finish first), every remaining array is expected to run
# past an hour regardless of chunk size -- at that point "one array
# resident at a time" is a simpler and more robust invariant than tuning a
# ceiling number, and doesn't cost meaningful CPU throughput either (a CPU
# array drains in minutes, so it only holds the single slot briefly).
DEFAULT_MAX_CONCURRENT_ARRAYS = 1
DEFAULT_COURTESY_CEILING = 200
DEFAULT_MIN_IDLE_CPUS = 32
DEFAULT_THROTTLE = 8
# CPU-only models aren't bounded by the cluster's 4-GPU ceiling the way GPU-tier
# models are, so a flat throttle=8 for everything leaves real idle CPU capacity
# unused once a CPU-only model's array is running -- e.g. RF at 16 cpus-per-task
# only ever occupies 128 of a node's 256 CPUs under throttle=8. Doubled, still
# well under check_capacity()'s independent min_idle_cpus/courtesy_ceiling gates
# (which govern whether/how much to submit at all, not concurrency within an
# already-submitted array), so this doesn't erode those protections.
DEFAULT_CPU_THROTTLE = 16
DEFAULT_TIME_LIMIT = 3600
# check_capacity()'s idle-CPU check only reflects what `sinfo` reports at that
# instant -- it says nothing about whether SLURM will actually schedule OUR
# jobs against it. Confirmed as a real, live production failure: `sinfo`
# reported 128 idle CPUs on every tick for 18+ hours straight while every one
# of the scheduler's own submitted array-jobs sat 100% PENDING the entire
# time (reason: "Priority" -- some cluster-side scheduling factor, e.g.
# fairshare decay from this account's own sustained prior usage, was holding
# them back regardless of raw idle capacity). Because idle_total alone still
# looked fine, the scheduler kept submitting a fresh 300-task chunk every
# hour on top of an already-completely-stalled queue -- 22 array-jobs
# accumulated, cancelled by hand once discovered. DEFAULT_MAX_PENDING caps how
# many of the user's OWN array-job entries are allowed to sit fully PENDING
# before the scheduler treats that as "no real room" and backs off, matching
# the tool's own stated design goal (back off by not submitting more, never
# by cancelling) instead of blindly trusting sinfo.
DEFAULT_MAX_PENDING = 5


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
                # 6 fields = pre-per-task-time_limit jobspec (still valid --
                # any array submitted before this field existed may still be
                # queued/running); 7 = current format with the trailing
                # per-task time_limit (see write_jobspec). Only the first 6
                # are needed here either way.
                parts = line.split()
                if len(parts) not in (6, 7):
                    continue
                dataset, target_idx, repeat, fold, config_index, _n_repeats = parts[:6]
                if int(config_index) != 0:
                    continue
                targets.add((dataset, int(target_idx), int(repeat), int(fold)))
    return targets


# --- Persistent failure tracking -- lets compute_backlog() notice a task that
# keeps failing (not just currently in flight) and stop resubmitting it, so one
# permanently-broken (model, dataset) pair can't silently starve every other
# model behind it in pick_chunk()'s strict priority order forever. Confirmed as
# a real, live incident: LR's entire remaining backlog (18 tasks, wheat_lines +
# bacteria_identification) hit AutoGluon's TimeLimitExceeded on every single
# hourly resubmission for 19+ consecutive ticks -- since LR sits ahead of
# NN_TORCH/FASTAI/DUMMY/... (each with a ~4,200-task backlog) in
# configs/v1/scope_default.json's model list, pick_chunk() never once advanced
# past LR in that entire window, despite a fully idle second node. That
# specific case is also fixed at the root (LR's own time_limit, see
# submit_job.resolve_time_limit's scalar-override handling) -- this mechanism
# is the general defense: ANY model could wedge the same way for an unrelated
# reason (a real code bug, a bad dataset, a cluster-side issue) with nothing
# else in the scheduler noticing or logging it.
STUCK_FAILURE_THRESHOLD = 3
STUCK_FAILURE_WINDOW_HOURS = 24
FAILURE_STATE_RETENTION_DAYS = 14


def _sacct_failed_tasks(user: str, since_days: int) -> dict[str, str]:
    """``{'{job_name}_{array_idx}': 'FAILED'|'TIMEOUT'|'OUT_OF_MEMORY'|
    'NODE_FAIL'|'CANCELLED'}`` for this user's SLURM accounting history within
    the lookback window, one entry per individual array task. ``-X`` restricts
    to job (not job-step, i.e. no .batch/.extern noise) records -- for an
    array job each array index is still its own top-level record, so this
    alone gives one line per array task."""
    cutoff = (datetime.datetime.now() - datetime.timedelta(days=since_days)).strftime("%Y-%m-%dT%H:%M:%S")
    try:
        out = subprocess.run(
            ["sacct", "-h", "-n", "-X", "-u", user, "--starttime", cutoff,
             "--state", "FAILED,TIMEOUT,OUT_OF_MEMORY,NODE_FAIL,CANCELLED",
             "-o", "JobID,JobName%200,State", "-P"],
            capture_output=True, text=True, check=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {}
    states: dict[str, str] = {}
    for line in out.splitlines():
        parts = line.strip().split("|")
        if len(parts) != 3:
            continue
        job_id, name, state = parts
        if "_" not in job_id:
            continue  # not an array-task record (e.g. a lone non-array job)
        states[f"{name}_{job_id.rsplit('_', 1)[1]}"] = state
    return states


def _resolve_array_task(model: str, key: str) -> tuple[str, int, int, int] | None:
    """Reverse a ``{job_name}_{array_idx}`` key (as returned by
    ``_sacct_failed_tasks``) back to (dataset, target_idx, repeat, fold) via
    the jobspec file, using the same ``LINE_NO = array_idx + 1`` convention
    ``run_experiment.sbatch`` itself uses."""
    prefix = f"RB_{model}_"
    if not key.startswith(prefix):
        return None
    rest = key[len(prefix):]
    part_slug, _, array_idx_str = rest.rpartition("_")
    if not part_slug or not array_idx_str.isdigit():
        return None
    jobspec_path = JOBSPEC_DIR / f"{part_slug}.txt"
    if not jobspec_path.exists():
        return None
    lines = jobspec_path.read_text().splitlines()
    array_idx = int(array_idx_str)
    if array_idx >= len(lines):
        return None
    parts = lines[array_idx].split()
    if len(parts) not in (6, 7):
        return None
    dataset, target_idx, repeat, fold = parts[0], int(parts[1]), int(parts[2]), int(parts[3])
    return (dataset, target_idx, repeat, fold)


def _load_failure_state(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _save_failure_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(state, f, indent=2, sort_keys=True)


def update_failure_state(state_path: Path, model: str, user: str) -> dict[str, dict[str, str]]:
    """Query recent SLURM accounting for ``model``'s FAILED/TIMEOUT/
    OUT_OF_MEMORY/NODE_FAIL/CANCELLED tasks, record each newly-seen failed
    SLURM task attempt against its own ``"dataset|target_idx|repeat|fold"``
    key in a persistent per-profile JSON state file (so failures accumulate
    across ticks -- a single tick only sees whatever's in ``sacct``'s own
    lookback window, not this task's full history), and return the updated
    per-task attempt log for ``model`` (``{task_key: {job_ref: observed_at_iso}}``).

    Purely additive record-keeping -- ``stuck_tasks`` below decides what
    actually counts as "stuck" from this. Entries older than
    ``FAILURE_STATE_RETENTION_DAYS`` are pruned on every call so the file
    can't grow unbounded; a task that stops failing (succeeds, or simply
    isn't attempted again) ages out of this file on its own, nothing else
    needs to explicitly clear it.
    """
    state = _load_failure_state(state_path)
    model_state = state.setdefault(model, {})
    now_iso = datetime.datetime.now().isoformat()

    for job_ref in _sacct_failed_tasks(user, since_days=FAILURE_STATE_RETENTION_DAYS):
        task = _resolve_array_task(model, job_ref)
        if task is None:
            continue
        task_key = "|".join(str(part) for part in task)
        attempts = model_state.setdefault(task_key, {})
        # job_ref ("{job_name}_{array_idx}") uniquely identifies one SLURM
        # task attempt -- re-observing the same one on a later tick (it's
        # still within sacct's own lookback window) must not inflate the
        # count, only genuinely new attempts should.
        attempts.setdefault(job_ref, now_iso)

    cutoff = datetime.datetime.now() - datetime.timedelta(days=FAILURE_STATE_RETENTION_DAYS)
    for task_key in list(model_state.keys()):
        attempts = model_state[task_key]
        for job_ref in list(attempts.keys()):
            if datetime.datetime.fromisoformat(attempts[job_ref]) < cutoff:
                del attempts[job_ref]
        if not attempts:
            del model_state[task_key]

    _save_failure_state(state_path, state)
    return model_state


def stuck_tasks(
    model_failure_state: dict[str, dict[str, str]],
    threshold: int = STUCK_FAILURE_THRESHOLD,
    window_hours: int = STUCK_FAILURE_WINDOW_HOURS,
) -> set[tuple[str, int, int, int]]:
    """Tasks with at least ``threshold`` distinct failed attempts within the
    last ``window_hours`` -- i.e. persistently broken, not a one-off transient
    failure (a real cluster hiccup, a single bad node) that doesn't deserve to
    be permanently excluded after just one occurrence."""
    cutoff = datetime.datetime.now() - datetime.timedelta(hours=window_hours)
    stuck: set[tuple[str, int, int, int]] = set()
    for task_key, attempts in model_failure_state.items():
        recent = [ts for ts in attempts.values() if datetime.datetime.fromisoformat(ts) >= cutoff]
        if len(recent) >= threshold:
            dataset, target_idx, repeat, fold = task_key.split("|")
            stuck.add((dataset, int(target_idx), int(repeat), int(fold)))
    return stuck


def compute_backlog(scope: dict, profile: dict, failure_state_path: Path | str | None = None) -> dict[str, list[Job]]:
    """Return ``{model: [(dataset, target_idx, repeat, fold, config_index=0, n_repeats), ...]}``
    for every not-yet-cached AND not-already-queued/running task in scope. Only
    the routine default config (config_index 0) is considered -- per the plan's
    Decision 3, HPO sweeps are opted into per model separately, not part of the
    routine opportunistic backlog.

    ``failure_state_path``, if given, additionally excludes tasks flagged
    "stuck" by ``stuck_tasks`` (>= ``STUCK_FAILURE_THRESHOLD`` distinct failed
    attempts within ``STUCK_FAILURE_WINDOW_HOURS``) -- see the module-level
    comment above ``STUCK_FAILURE_THRESHOLD`` for why this exists: without it,
    a persistently-failing task keeps reappearing in the backlog every tick
    forever, and since ``pick_chunk`` is strict-priority (not round-robin), it
    can starve every model listed after it in ``scope["models"]`` indefinitely.
    ``None`` (the default) disables this check entirely -- e.g. for callers
    without real SLURM accounting to query (tests, ``--dry-run`` probes
    against a scope with no matching cluster).
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
        stuck: set[tuple[str, int, int, int]] = set()
        if failure_state_path is not None:
            model_failure_state = update_failure_state(Path(failure_state_path), model, user)
            stuck = stuck_tasks(model_failure_state)
        model_backlog: list[Job] = []
        skipped_stuck: list[tuple[str, int, int, int]] = []
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
                    if cache_file.exists():
                        continue
                    if (dataset, target_idx, repeat, fold) in stuck:
                        skipped_stuck.append((dataset, target_idx, repeat, fold))
                        continue
                    model_backlog.append((dataset, target_idx, repeat, fold, 0, n_repeats))
        if skipped_stuck:
            preview = skipped_stuck[:5]
            print(
                f"WARNING: {model} has {len(skipped_stuck)} task(s) excluded from its backlog as "
                f"'stuck' (>= {STUCK_FAILURE_THRESHOLD} failed attempts within "
                f"{STUCK_FAILURE_WINDOW_HOURS}h, still no results.pkl, not currently queued/running): "
                f"{preview}{' ...' if len(skipped_stuck) > len(preview) else ''}. These will keep "
                f"failing silently forever otherwise -- investigate the root cause (see .logs/), then "
                f"either fix it or clear the corresponding entries from {failure_state_path} to retry.",
                file=sys.stderr,
            )
        backlog[model] = model_backlog
    return backlog


def check_capacity(
    profile: dict, min_idle_cpus: int, courtesy_ceiling: int, max_pending: int = DEFAULT_MAX_PENDING,
    max_concurrent_arrays: int = DEFAULT_MAX_CONCURRENT_ARRAYS,
) -> tuple[bool, str]:
    """Return (has_room, reason). ``has_room`` requires ALL FOUR:
    (1) idle cluster capacity (this partition isn't busy with other users' work),
    (2) this user's own resident (pending+running) task count staying under a
    courtesy ceiling (so the scheduler itself never grows into "occupying the
    whole cluster" even if the partition looks idle to everyone else too),
    (3) this user's own PENDING count staying under ``max_pending`` -- idle CPU
    capacity as reported by ``sinfo`` does not guarantee SLURM will actually
    schedule OUR jobs against it (confirmed in practice: 128 idle CPUs reported
    every tick for 18+ hours while every one of the scheduler's own submitted
    array-jobs sat 100% PENDING the whole time, reason "Priority" -- see
    DEFAULT_MAX_PENDING's comment for the full incident). Check (2) alone
    doesn't catch this, since ``courtesy_ceiling`` (default 200) is sized to
    cap total occupancy, not to notice that already-submitted work isn't
    actually starting -- a queue can be miles under that ceiling and still be
    completely stalled. If a substantial number of the user's own array-jobs
    are already sitting PENDING with zero of them progressing to RUNNING,
    that's a direct signal something (fairshare, a reservation, cluster
    policy) is blocking real scheduling regardless of what ``sinfo`` claims,
    and submitting more just piles additional stuck work onto the same
    problem instead of backing off the way this tool is designed to, AND
    (4) this user's own number of DISTINCT resident RamanBench array-jobs
    staying under ``max_concurrent_arrays`` (default 1) -- courtesy_ceiling
    alone still lets several ``gpu_chunk_size``-sized arrays (e.g. 4 x 32 =
    128) queue up back to back before tripping, confirmed live as a real,
    repeat incident even after the courtesy_ceiling accuracy fix and
    gpu_chunk_size existed. Counts DISTINCT job names among resident tasks
    (every task in one array shares the same ``RB_{model}_{part_slug}`` job
    name), not raw task count -- unlike (2)/(3), this is about how many
    SEPARATE array submissions are outstanding at once, regardless of how
    big any one of them is."""
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

    user = profile.get("cron_user") or _current_user()

    # Two separate squeue views for two genuinely different signals -- do NOT
    # collapse them into one query:
    #
    # (a) courtesy_ceiling needs the TRUE resident (pending+running) task
    #     count. squeue's DEFAULT view collapses an entire still-pending array
    #     range onto ONE line (e.g. a fresh 300-task array shows as a single
    #     "62167_[0-299%8]" row until SLURM starts scheduling individual tasks
    #     out of it) -- confirmed live as a real incident: this collapsing
    #     meant 5 separate 300-task GPU (NN_TORCH) arrays piled up back to
    #     back over 5 hourly ticks, each tick's check seeing only ~1 line per
    #     array (~14 total) while the true pending+running count was over
    #     1,000 -- courtesy_ceiling (200) never tripped because it was never
    #     given an accurate count to compare against. ``-r`` expands pending
    #     ranges into one line per task, same technique already used in
    #     raman_bench_paper/cluster/dashboard_snapshot.py's _live_task_states.
    #
    # (b) max_pending's fairshare-stall detection is a DIFFERENT thing and
    #     must NOT use the accurate (b) count: its whole premise (see the
    #     docstring above) is "a small number of PENDING array-JOBS with none
    #     of them progressing to RUNNING at all" -- e.g. every one of my
    #     arrays sitting 100% pending for hours despite reported idle
    #     capacity. Once (a) is fixed, a single freshly-submitted array under
    #     normal throttled concurrency (e.g. 300 tasks, throttle=8) always has
    #     close to 300 individual PENDING lines the instant it's submitted --
    #     that's completely normal, expected backlog, not a stall. Using the
    #     accurate per-task count for this check would make it misfire on
    #     literally every tick after the first ever submission (since
    #     max_pending's default, 5, is far smaller than any real chunk size),
    #     permanently freezing the scheduler. squeue's own default collapsed
    #     view -- one line per distinct pending ARRAY plus one per actively
    #     running/attempting task -- is what makes this check work as
    #     designed: a healthy array quickly grows individually-listed RUNNING
    #     lines as SLURM schedules it, while a genuinely stalled one stays
    #     collapsed to a small, unmoving handful of lines indefinitely.
    try:
        # %j (job name) added so the same query also yields the distinct
        # array count for (4) -- no extra squeue round-trip needed.
        squeue_resident_out = subprocess.run(
            ["squeue", "-r", "-h", "-u", user, "-t", "pending,running", "-o", "%T|%j"],
            capture_output=True, text=True, check=True,
        ).stdout
        squeue_stall_out = subprocess.run(
            ["squeue", "-h", "-u", user, "-t", "pending,running", "-o", "%T"],
            capture_output=True, text=True, check=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        return False, f"squeue failed ({e}) -- treating as at ceiling, conservatively"
    resident_lines = [line.strip() for line in squeue_resident_out.splitlines() if line.strip()]
    my_resident = len(resident_lines)
    resident_array_names = {
        name for _state, _sep, name in (line.partition("|") for line in resident_lines) if name.startswith("RB_")
    }
    my_array_count = len(resident_array_names)
    stall_states = [line.strip() for line in squeue_stall_out.splitlines() if line.strip()]
    my_pending = sum(1 for s in stall_states if s == "PENDING")

    if idle_total < min_idle_cpus:
        return False, f"only {idle_total} idle CPU(s) on partition {partition!r} (need >={min_idle_cpus})"
    if my_resident >= courtesy_ceiling:
        return False, f"already have {my_resident} resident task(s) (courtesy ceiling {courtesy_ceiling})"
    if my_pending >= max_pending:
        return False, (
            f"already have {my_pending} of my own PENDING array-job(s) not yet RUNNING "
            f"(max_pending {max_pending}) -- sinfo reports idle capacity, but my own "
            f"queued work isn't actually starting, so backing off instead of piling on more"
        )
    if my_array_count >= max_concurrent_arrays:
        return False, (
            f"already have {my_array_count} of my own RamanBench array-job(s) resident "
            f"(max_concurrent_arrays {max_concurrent_arrays}) -- waiting for it to drain "
            f"before submitting another, regardless of total task count"
        )
    return True, f"{idle_total} idle CPU(s), {my_resident} resident task(s) -- room to submit"


def _current_user() -> str:
    import getpass

    return getpass.getuser()


def pick_chunk(
    backlog: dict[str, list[Job]], chunk_size: int, gpu_chunk_size: int | None = None,
) -> tuple[str | None, list[Job]]:
    """Pick the first model (in scope order) with a nonempty backlog and take up
    to its own chunk size worth of tasks from it -- ``gpu_chunk_size`` if the
    model is GPU-tier and one is given, else the flat ``chunk_size``. One chunk
    never spans more than one model -- resource flags (mem/GPU) are resolved
    once per SLURM array submission.

    GPU models get a separate, much smaller chunk size by default (see
    run_tick's DEFAULT_GPU_CHUNK_SIZE) -- see that constant's own comment for
    why: a chunk size appropriately sized for fast CPU baselines can leave a
    GPU array running for many hours under throttle-limited concurrency,
    against the scheduler's own "short bounded bursts, back off between
    ticks" design goal."""
    for model, jobs in backlog.items():
        if jobs:
            size = gpu_chunk_size if (gpu_chunk_size is not None and model in GPU_MODELS) else chunk_size
            return model, jobs[:size]
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
      sub-dict is enough; no cross-model leakage is possible. A model's own
      entry may instead be a bare number rather than a dataset-keyed dict --
      applied regardless of which datasets are in the chunk (see LR below:
      a model with no real per-dataset variation in whether it needs more
      time, just a blanket "don't cap this one").

    This whole-chunk max is no longer what each task actually runs with --
    ``submit_jobs``/``write_jobspec`` resolve a PER-TASK time_limit from each
    task's own dataset (``submit_job.resolve_time_limit``), so a chunk mixing
    an oversized dataset (e.g. mlrod) in with ordinary ones no longer forces
    every task in that chunk onto mlrod's budget -- confirmed as a real,
    wasted-throughput problem in practice (job 36545: a small/fast
    `alzheimer` task sat at the full ~1350s/fold slice implied by mlrod's
    10800s override despite never needing anywhere near that). This
    function's return value is still meaningful, though, as the array-wide
    ``--export TIME_LIMIT=...`` fallback (used by ``run_experiment.sbatch``
    only for jobspec lines that predate the per-task field) -- i.e. still a
    ceiling, just no longer the value every task actually runs with.

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
    if isinstance(model_overrides, (int, float)):
        return max(default, model_overrides)
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


def run_tick(
    scope: dict, profile: dict, log_path: str | Path | None, dry_run: bool,
    failure_state_path: str | Path | None = None,
) -> dict:
    timestamp = datetime.datetime.now(datetime.UTC).isoformat()
    backlog = compute_backlog(scope, profile, failure_state_path=failure_state_path)
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
    max_pending = scope.get("max_pending", DEFAULT_MAX_PENDING)
    max_concurrent_arrays = scope.get("max_concurrent_arrays", DEFAULT_MAX_CONCURRENT_ARRAYS)
    has_room, reason = check_capacity(profile, min_idle_cpus, courtesy_ceiling, max_pending, max_concurrent_arrays)

    if not has_room:
        entry = {
            "timestamp": timestamp, "action": "skip", "reason": reason,
            "backlog_sizes": backlog_sizes, "total_backlog": total_backlog,
        }
        print(json.dumps(entry, indent=2))
        log_tick(log_path, entry)
        return entry

    chunk_size = scope.get("chunk_size", DEFAULT_CHUNK_SIZE)
    gpu_chunk_size = scope.get("gpu_chunk_size", DEFAULT_GPU_CHUNK_SIZE)
    model, chunk = pick_chunk(backlog, chunk_size, gpu_chunk_size)
    time_limit = effective_time_limit(scope, model, chunk)  # whole-chunk ceiling/export-fallback -- see its docstring
    # Deliberately NOT `time_limit` above -- that's already maxed up over
    # every dataset in the chunk (e.g. inflated to 10800 by a single mlrod
    # task), and using it as the per-task baseline would mean every task's
    # own resolve_time_limit floors at that ceiling regardless of its own
    # dataset, silently reproducing the exact bug this fix removes. This is
    # the FLAT scope default each task's own override lookup bumps up from.
    flat_default_time_limit = scope.get("time_limit", DEFAULT_TIME_LIMIT)
    dataset_time_limit_overrides = scope.get("time_limit_overrides", {})
    model_time_limit_overrides = scope.get("model_time_limit_overrides", {}).get(model, {})
    throttle = (
        scope.get("throttle", DEFAULT_THROTTLE)
        if model in GPU_MODELS
        else scope.get("cpu_throttle", DEFAULT_CPU_THROTTLE)
    )

    slug = f"{scope['name']}_{model}_{timestamp.replace(':', '').replace('-', '').split('.')[0]}"
    job_ids = submit_jobs(
        model=model, jobs=chunk, slug=slug,
        n_splits=scope["n_splits"], num_random_configs=scope.get("num_random_configs", 50),
        num_bag_folds=scope.get("num_bag_folds", 8), time_limit=time_limit,
        results_dir=scope["results_dir"], cache_dir=scope.get("cache_dir", ".cache_v1"),
        mirror_repo=scope.get("mirror_repo", "HTW-KI-Werkstatt/RamanBench"),
        profile=profile, throttle=throttle, dry_run=dry_run,
        dataset_time_limit_overrides=dataset_time_limit_overrides,
        model_time_limit_overrides=model_time_limit_overrides,
        default_time_limit=flat_default_time_limit,
    )

    # Per-task values actually written to the jobspec (see write_jobspec) --
    # surfaced here, distinct from the single whole-chunk `time_limit` above,
    # so a tick log entry makes it directly observable that a mixed chunk got
    # more than one budget instead of one inflated value for everyone.
    per_task_time_limits = sorted({
        resolve_time_limit(
            flat_default_time_limit, dataset, dataset_time_limit_overrides, model_time_limit_overrides,
        )
        for dataset, *_rest in chunk
    })

    entry = {
        "timestamp": timestamp, "action": "dry_run_submit" if dry_run else "submit",
        "reason": reason, "model": model, "n_submitted": len(chunk),
        "time_limit": time_limit, "per_task_time_limits": per_task_time_limits,
        "throttle": throttle, "job_ids": job_ids, "backlog_sizes": backlog_sizes,
        "total_backlog": total_backlog,
    }
    print(json.dumps(entry, indent=2))
    log_tick(log_path, entry)
    return entry


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scope", required=True, help="Path to a scope JSON file (see scope.example.json)")
    parser.add_argument("--profile", required=True, help="Path to a cluster profile YAML")
    parser.add_argument("--log", default=None, help="Append a JSON line per tick to this file")
    parser.add_argument(
        "--failure-state", default=None,
        help="Persistent per-profile JSON file tracking failed task attempts (see "
             "update_failure_state/stuck_tasks) -- a task with >= STUCK_FAILURE_THRESHOLD "
             "failures within STUCK_FAILURE_WINDOW_HOURS is excluded from the backlog "
             "instead of being resubmitted forever. Defaults to "
             "cluster/.scheduler_state/<profile-name>_failures.json; pass an explicit path "
             "to change it, or an empty string to disable the mechanism entirely.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    scope = load_scope(args.scope)
    profile = resolve_profile(args.profile, None)
    failure_state_path = args.failure_state
    if failure_state_path is None:
        failure_state_path = CLUSTER_DIR / ".scheduler_state" / f"{profile.get('name', 'default')}_failures.json"
    elif failure_state_path == "":
        failure_state_path = None
    run_tick(scope, profile, args.log, args.dry_run, failure_state_path=failure_state_path)


if __name__ == "__main__":
    main()
