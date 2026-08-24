"""Regression test for a real production incident: ``check_capacity()`` trusted
``sinfo``'s idle-CPU count as the only signal for "is there room to submit",
but idle CPU capacity does not guarantee SLURM will actually schedule the
scheduler's own jobs against it.

Observed live on the HTW cluster: `sinfo` reported 128 idle CPUs on this
partition on every hourly tick for 18+ hours straight, while every single one
of the opportunistic scheduler's own previously-submitted array-jobs sat
100% PENDING (SLURM reason: "Priority" -- some cluster-side scheduling factor,
e.g. fairshare decay from this account's own sustained prior usage, held them
back regardless of raw idle capacity). Because `idle_total` alone still looked
healthy every tick, the scheduler kept submitting a fresh 300-task chunk on
top of an already-completely-stalled queue -- 22 array-jobs accumulated
before this was noticed and cancelled by hand.

The existing `courtesy_ceiling` gate (default 200) didn't catch this either:
it caps total resident (pending+running) occupancy, not whether resident work
is actually progressing -- the queue was nowhere near 200 and still fully
stalled the entire time.

Fix: `check_capacity()` now also tracks how many of the user's OWN array-jobs
are sitting PENDING specifically (not just resident total), and backs off
once that exceeds `max_pending`, regardless of what `sinfo` reports -- the
scheduler's own stated design ("back off by not submitting more, never by
cancelling") applies here too.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CLUSTER_DIR = REPO_ROOT / "cluster"


def _load_opportunistic_scheduler():
    if str(CLUSTER_DIR) not in sys.path:
        sys.path.insert(0, str(CLUSTER_DIR))
    spec = importlib.util.spec_from_file_location(
        "_opportunistic_scheduler_under_test_capacity_pending", CLUSTER_DIR / "opportunistic_scheduler.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def scheduler():
    return _load_opportunistic_scheduler()


def _fake_run(sinfo_line: str, squeue_states: list[str]):
    def fake_run(cmd, capture_output, text, check):
        class _Result:
            pass

        result = _Result()
        if cmd[0] == "sinfo":
            result.stdout = sinfo_line
        else:  # squeue
            result.stdout = "\n".join(squeue_states)
        return result

    return fake_run


def _fake_run_distinguishing_r(sinfo_line: str, expanded_states: list[str], collapsed_states: list[str]):
    """Like ``_fake_run``, but returns DIFFERENT squeue output depending on
    whether ``-r`` is in the command -- models the real behavior squeue
    itself has (an array with a large still-pending remainder collapses to
    ~1 line without ``-r``, but expands to one line per task with it)."""

    def fake_run(cmd, capture_output, text, check):
        class _Result:
            pass

        result = _Result()
        if cmd[0] == "sinfo":
            result.stdout = sinfo_line
        elif "-r" in cmd:
            result.stdout = "\n".join(expanded_states)
        else:
            result.stdout = "\n".join(collapsed_states)
        return result

    return fake_run


def test_healthy_queue_has_room(scheduler):
    # 128 idle, well under courtesy_ceiling, and only 2 of my own jobs pending
    # (below max_pending) -- this is what a genuinely healthy tick looks like.
    fake_run = _fake_run(
        "128/128/0/256",
        ["RUNNING"] * 3 + ["PENDING"] * 2,
    )
    with patch("subprocess.run", side_effect=fake_run):
        has_room, reason = scheduler.check_capacity(
            {"partition": "Debug_node"}, min_idle_cpus=32, courtesy_ceiling=200, max_pending=5
        )
    assert has_room is True
    assert "room to submit" in reason


def test_low_idle_cpu_blocks(scheduler):
    fake_run = _fake_run("240/16/0/256", [])
    with patch("subprocess.run", side_effect=fake_run):
        has_room, reason = scheduler.check_capacity(
            {"partition": "Debug_node"}, min_idle_cpus=32, courtesy_ceiling=200, max_pending=5
        )
    assert has_room is False
    assert "idle CPU" in reason


def test_courtesy_ceiling_blocks(scheduler):
    fake_run = _fake_run("0/256/0/256", ["RUNNING"] * 200)
    with patch("subprocess.run", side_effect=fake_run):
        has_room, reason = scheduler.check_capacity(
            {"partition": "Debug_node"}, min_idle_cpus=32, courtesy_ceiling=200, max_pending=5
        )
    assert has_room is False
    assert "courtesy ceiling" in reason


def test_stalled_pending_queue_blocks_even_with_idle_cpus(scheduler):
    """The actual incident: sinfo reports plenty of idle CPUs, resident count
    is nowhere near the courtesy ceiling, but the user's own jobs are 100%
    PENDING -- this must block, not submit."""
    fake_run = _fake_run(
        "128/128/0/256",
        ["PENDING"] * 22,  # matches the real incident's accumulated count
    )
    with patch("subprocess.run", side_effect=fake_run):
        has_room, reason = scheduler.check_capacity(
            {"partition": "Debug_node"}, min_idle_cpus=32, courtesy_ceiling=200, max_pending=5
        )
    assert has_room is False
    assert "PENDING" in reason
    assert "max_pending" in reason


def test_a_few_pending_is_normal_and_does_not_block(scheduler):
    # Some transient PENDING is expected/healthy (e.g. a chunk just submitted
    # this tick, briefly waiting for a throttle slot) -- only a sustained pile
    # past max_pending should back off.
    fake_run = _fake_run(
        "128/128/0/256",
        ["RUNNING"] * 5 + ["PENDING"] * 3,
    )
    with patch("subprocess.run", side_effect=fake_run):
        has_room, reason = scheduler.check_capacity(
            {"partition": "Debug_node"}, min_idle_cpus=32, courtesy_ceiling=200, max_pending=5
        )
    assert has_room is True


def test_courtesy_ceiling_catches_collapsed_pending_array(scheduler):
    """The actual second incident: squeue's default view collapses a large
    still-pending array range onto ~1 line, so a naive single-query count
    stayed far under courtesy_ceiling while the TRUE resident count (visible
    only via ``-r``) was already over it. This must now block."""
    fake_run = _fake_run_distinguishing_r(
        "128/128/0/256",
        expanded_states=["RUNNING"] * 8 + ["PENDING"] * 292,  # -r: true count, 300 total
        collapsed_states=["RUNNING"] * 8 + ["PENDING"] * 1,  # no -r: 1 line for the whole range
    )
    with patch("subprocess.run", side_effect=fake_run):
        has_room, reason = scheduler.check_capacity(
            {"partition": "Debug_node"}, min_idle_cpus=32, courtesy_ceiling=200, max_pending=5
        )
    assert has_room is False
    assert "courtesy ceiling" in reason


def test_max_pending_does_not_misfire_on_normal_throttled_backlog(scheduler):
    """Once courtesy_ceiling uses the accurate (-r) count, max_pending must
    NOT also switch to it -- a single freshly-submitted, perfectly healthy
    array under normal throttle-limited concurrency has hundreds of
    individual PENDING lines under -r, which would misfire max_pending
    (default 5) on literally every tick if it used that count instead of the
    collapsed view it was designed around."""
    fake_run = _fake_run_distinguishing_r(
        "128/128/0/256",
        expanded_states=["RUNNING"] * 8 + ["PENDING"] * 32,  # -r: true count, 40 total (under courtesy_ceiling)
        collapsed_states=["RUNNING"] * 8 + ["PENDING"] * 1,  # no -r: 1 collapsed line for the pending remainder
    )
    with patch("subprocess.run", side_effect=fake_run):
        has_room, reason = scheduler.check_capacity(
            {"partition": "Debug_node"}, min_idle_cpus=32, courtesy_ceiling=200, max_pending=5
        )
    assert has_room is True


def _fake_run_with_job_names(sinfo_line: str, resident_rows: list[tuple[str, str]]):
    """``resident_rows`` is a list of (state, job_name) tuples -- models the
    real ``-o "%T|%j"`` query format check_capacity now uses to derive both
    the resident task count and the distinct-array count from one query."""

    def fake_run(cmd, capture_output, text, check):
        class _Result:
            pass

        result = _Result()
        if cmd[0] == "sinfo":
            result.stdout = sinfo_line
        elif "-r" in cmd:
            result.stdout = "\n".join(f"{state}|{name}" for state, name in resident_rows)
        else:
            result.stdout = "\n".join(state for state, _name in resident_rows)
        return result

    return fake_run


def test_max_concurrent_arrays_blocks_a_second_array(scheduler):
    """The actual repeat incident: courtesy_ceiling (200) alone still lets
    several gpu_chunk_size-sized arrays (32 each) queue up. A second,
    differently-named array must be blocked once one is already resident,
    even though total task count is nowhere near courtesy_ceiling."""
    fake_run = _fake_run_with_job_names(
        "128/128/0/256",
        [("RUNNING", "RB_NN_TORCH_v1_default_NN_TORCH_20260823T100000")] * 6
        + [("PENDING", "RB_NN_TORCH_v1_default_NN_TORCH_20260823T100000")] * 26,
    )
    with patch("subprocess.run", side_effect=fake_run):
        has_room, reason = scheduler.check_capacity(
            {"partition": "Debug_node"}, min_idle_cpus=32, courtesy_ceiling=200,
            max_pending=50, max_concurrent_arrays=1,
        )
    assert has_room is False
    assert "max_concurrent_arrays" in reason


def test_max_concurrent_arrays_allows_room_when_none_resident(scheduler):
    fake_run = _fake_run_with_job_names("128/128/0/256", [])
    with patch("subprocess.run", side_effect=fake_run):
        has_room, reason = scheduler.check_capacity(
            {"partition": "Debug_node"}, min_idle_cpus=32, courtesy_ceiling=200,
            max_pending=50, max_concurrent_arrays=1,
        )
    assert has_room is True


def test_max_concurrent_arrays_counts_distinct_names_not_tasks(scheduler):
    """A single array with many resident tasks (all sharing one job name)
    counts as ONE array, not one-per-task -- only a genuinely SECOND,
    differently-named array should push the count to 2."""
    fake_run = _fake_run_with_job_names(
        "128/128/0/256",
        [("RUNNING", "RB_LR_v1_default_LR_20260823T090000")] * 200,
    )
    with patch("subprocess.run", side_effect=fake_run):
        has_room, reason = scheduler.check_capacity(
            {"partition": "Debug_node"}, min_idle_cpus=32, courtesy_ceiling=500,
            max_pending=250, max_concurrent_arrays=1,
        )
    assert has_room is False  # courtesy_ceiling(500) not hit at 200, but max_concurrent_arrays(1) should be
    assert "max_concurrent_arrays" in reason


def test_max_concurrent_arrays_ignores_non_ramanbench_jobs(scheduler):
    """A resident job that isn't one of the scheduler's own (no RB_ prefix,
    e.g. an unrelated ablation-study job) must not count toward
    max_concurrent_arrays."""
    fake_run = _fake_run_with_job_names(
        "128/128/0/256",
        [("RUNNING", "RPP_prep"), ("RUNNING", "IVC_deep")],
    )
    with patch("subprocess.run", side_effect=fake_run):
        has_room, reason = scheduler.check_capacity(
            {"partition": "Debug_node"}, min_idle_cpus=32, courtesy_ceiling=200,
            max_pending=50, max_concurrent_arrays=1,
        )
    assert has_room is True


def test_run_tick_threads_max_concurrent_arrays_from_scope(scheduler, monkeypatch):
    captured = {}

    def fake_check_capacity(
        profile, min_idle_cpus, courtesy_ceiling,
        max_pending=scheduler.DEFAULT_MAX_PENDING,
        max_concurrent_arrays=scheduler.DEFAULT_MAX_CONCURRENT_ARRAYS,
    ):
        captured["max_concurrent_arrays"] = max_concurrent_arrays
        return False, "stopped for test"

    monkeypatch.setattr(
        scheduler, "compute_backlog",
        lambda scope, profile, **kwargs: {"PLS": [("wheat_lines", 0, 0, 0, 0, 10)]},
    )
    monkeypatch.setattr(scheduler, "check_capacity", fake_check_capacity)

    scope = {
        "name": "test", "results_dir": "results", "n_splits": 3,
        "max_concurrent_arrays": 3,
    }
    scheduler.run_tick(scope, {"slurm": True, "partition": "Debug_node"}, log_path=None, dry_run=True)

    assert captured["max_concurrent_arrays"] == 3


def test_run_tick_threads_max_pending_from_scope(scheduler, monkeypatch):
    """run_tick() must read scope['max_pending'] (falling back to
    DEFAULT_MAX_PENDING) and actually pass it into check_capacity -- not just
    have the parameter exist unused."""
    captured = {}

    def fake_check_capacity(
        profile, min_idle_cpus, courtesy_ceiling,
        max_pending=scheduler.DEFAULT_MAX_PENDING,
        max_concurrent_arrays=scheduler.DEFAULT_MAX_CONCURRENT_ARRAYS,
    ):
        captured["max_pending"] = max_pending
        return False, "stopped for test"

    monkeypatch.setattr(
        scheduler, "compute_backlog",
        lambda scope, profile, **kwargs: {"PLS": [("wheat_lines", 0, 0, 0, 0, 10)]},
    )
    monkeypatch.setattr(scheduler, "check_capacity", fake_check_capacity)

    scope = {
        "name": "test", "results_dir": "results", "n_splits": 3,
        "max_pending": 7,
    }
    scheduler.run_tick(scope, {"slurm": True, "partition": "Debug_node"}, log_path=None, dry_run=True)

    assert captured["max_pending"] == 7
