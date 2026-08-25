"""Regression tests for persistent-failure tracking in the opportunistic
scheduler.

The problem, observed live on the cluster: ``pick_chunk`` picks models in
strict priority order (see ``configs/v1/scope_default.json``'s own comment on
why PLS is listed last) and only advances past a model once its backlog hits
zero. LR's entire remaining backlog (18 tasks: wheat_lines +
bacteria_identification) hit ``autogluon.core.utils.exceptions.TimeLimitExceeded``
on every single hourly resubmission for 19+ consecutive ticks -- since
``compute_backlog`` only ever excluded tasks that were already DONE
(``results.pkl`` exists) or currently in flight (queued/running), a task that
keeps crashing shortly after being resubmitted reappears in the backlog every
tick forever, silently starving every model listed after it (NN_TORCH,
FASTAI, DUMMY, ... each with a ~4,200-task backlog of their own) despite a
fully idle second node the whole time.

The fix: ``update_failure_state``/``stuck_tasks`` track each task's recent
failed SLURM attempts (via ``sacct``) in a persistent per-profile JSON file,
and ``compute_backlog`` excludes any task with >= ``STUCK_FAILURE_THRESHOLD``
distinct failures within ``STUCK_FAILURE_WINDOW_HOURS`` -- logging a clear
warning instead of resubmitting it again. A one-off transient failure (a real
cluster hiccup) does NOT get excluded after just one occurrence -- only
persistent, repeated failure does.
"""

from __future__ import annotations

import datetime
import importlib.util
import json
import sys
import uuid
from pathlib import Path
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CLUSTER_DIR = REPO_ROOT / "cluster"


def _load_opportunistic_scheduler():
    if str(CLUSTER_DIR) not in sys.path:
        sys.path.insert(0, str(CLUSTER_DIR))
    spec = importlib.util.spec_from_file_location(
        "_opportunistic_scheduler_under_test_stuck", CLUSTER_DIR / "opportunistic_scheduler.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def osched():
    return _load_opportunistic_scheduler()


def _fake_sacct_run(job_ids: list[str]):
    """Build a subprocess.run stand-in whose sacct call returns FAILED records
    for exactly the given ``job_id`` strings (e.g. "61945_0"), one per line,
    pipe-delimited to match ``-P``."""

    def fake_run(cmd, capture_output, text, check):
        class _Result:
            pass

        result = _Result()
        if cmd[0] == "sacct":
            result.stdout = "\n".join(
                f"{job_id}|RB_LR_v1_default_LR_20260822T160705|FAILED" for job_id in job_ids
            )
        else:
            result.stdout = ""
        return result

    return fake_run


# --- _resolve_array_task: reverse a "{job_name}_{array_idx}" key back to
#     (dataset, target_idx, repeat, fold) via the jobspec file. ---


def test_resolve_array_task_reads_correct_line(osched, tmp_path, monkeypatch):
    monkeypatch.setattr(osched, "JOBSPEC_DIR", tmp_path)
    (tmp_path / "v1_default_LR_20260822T160705.txt").write_text(
        "wheat_lines 0 0 0 0 3 3600\nwheat_lines 0 0 1 0 3 3600\nbacteria_identification 0 0 0 0 3 3600\n"
    )
    assert osched._resolve_array_task("LR", "RB_LR_v1_default_LR_20260822T160705_0") == (
        "wheat_lines",
        0,
        0,
        0,
    )
    assert osched._resolve_array_task("LR", "RB_LR_v1_default_LR_20260822T160705_1") == (
        "wheat_lines",
        0,
        0,
        1,
    )
    assert osched._resolve_array_task("LR", "RB_LR_v1_default_LR_20260822T160705_2") == (
        "bacteria_identification",
        0,
        0,
        0,
    )


def test_resolve_array_task_wrong_model_prefix_returns_none(osched, tmp_path, monkeypatch):
    monkeypatch.setattr(osched, "JOBSPEC_DIR", tmp_path)
    (tmp_path / "v1_default_LR_20260822T160705.txt").write_text("wheat_lines 0 0 0 0 3 3600\n")
    assert osched._resolve_array_task("PLS", "RB_LR_v1_default_LR_20260822T160705_0") is None


def test_resolve_array_task_missing_jobspec_returns_none(osched, tmp_path, monkeypatch):
    monkeypatch.setattr(osched, "JOBSPEC_DIR", tmp_path)
    assert osched._resolve_array_task("LR", "RB_LR_nonexistent_0") is None


def test_resolve_array_task_out_of_range_index_returns_none(osched, tmp_path, monkeypatch):
    monkeypatch.setattr(osched, "JOBSPEC_DIR", tmp_path)
    (tmp_path / "v1_default_LR_20260822T160705.txt").write_text("wheat_lines 0 0 0 0 3 3600\n")
    assert osched._resolve_array_task("LR", "RB_LR_v1_default_LR_20260822T160705_5") is None


# --- update_failure_state: persistent, deduplicated, pruned failure log. ---


def test_update_failure_state_records_new_failure(osched, tmp_path, monkeypatch):
    monkeypatch.setattr(osched, "JOBSPEC_DIR", tmp_path)
    (tmp_path / "v1_default_LR_20260822T160705.txt").write_text("wheat_lines 0 0 0 0 3 3600\n")
    state_path = tmp_path / "failures.json"

    with patch("subprocess.run", side_effect=_fake_sacct_run(["61945_0"])):
        model_state = osched.update_failure_state(state_path, "LR", "testuser")

    assert "wheat_lines|0|0|0" in model_state
    assert "RB_LR_v1_default_LR_20260822T160705_0" in model_state["wheat_lines|0|0|0"]
    assert state_path.exists()


def test_update_failure_state_does_not_double_count_same_attempt(osched, tmp_path, monkeypatch):
    """Re-observing the exact same SLURM job/array-index on a later tick (still
    within sacct's own lookback window) must not inflate the failure count --
    only genuinely distinct attempts should."""
    monkeypatch.setattr(osched, "JOBSPEC_DIR", tmp_path)
    (tmp_path / "v1_default_LR_20260822T160705.txt").write_text("wheat_lines 0 0 0 0 3 3600\n")
    state_path = tmp_path / "failures.json"

    with patch("subprocess.run", side_effect=_fake_sacct_run(["61945_0"])):
        osched.update_failure_state(state_path, "LR", "testuser")
        model_state = osched.update_failure_state(state_path, "LR", "testuser")

    assert len(model_state["wheat_lines|0|0|0"]) == 1


def test_update_failure_state_accumulates_distinct_attempts(osched, tmp_path, monkeypatch):
    monkeypatch.setattr(osched, "JOBSPEC_DIR", tmp_path)
    (tmp_path / "v1_default_LR_20260822T160705.txt").write_text("wheat_lines 0 0 0 0 3 3600\n")
    (tmp_path / "v1_default_LR_20260822T170705.txt").write_text("wheat_lines 0 0 0 0 3 3600\n")
    state_path = tmp_path / "failures.json"

    def fake_run(cmd, capture_output, text, check):
        class _Result:
            pass

        result = _Result()
        if cmd[0] == "sacct":
            result.stdout = (
                "61945_0|RB_LR_v1_default_LR_20260822T160705|FAILED\n"
                "61963_0|RB_LR_v1_default_LR_20260822T170705|FAILED"
            )
        else:
            result.stdout = ""
        return result

    with patch("subprocess.run", side_effect=fake_run):
        model_state = osched.update_failure_state(state_path, "LR", "testuser")

    assert len(model_state["wheat_lines|0|0|0"]) == 2


def test_update_failure_state_prunes_entries_older_than_retention(osched, tmp_path, monkeypatch):
    monkeypatch.setattr(osched, "JOBSPEC_DIR", tmp_path)
    state_path = tmp_path / "failures.json"
    stale = (
        datetime.datetime.now() - datetime.timedelta(days=osched.FAILURE_STATE_RETENTION_DAYS + 1)
    ).isoformat()
    state_path.write_text(json.dumps({"LR": {"wheat_lines|0|0|0": {"old_job_0": stale}}}))

    with patch("subprocess.run", side_effect=_fake_sacct_run([])):
        model_state = osched.update_failure_state(state_path, "LR", "testuser")

    assert "wheat_lines|0|0|0" not in model_state


def test_update_failure_state_ignores_other_models(osched, tmp_path, monkeypatch):
    """A failure record under one model's job-name prefix must never be
    attributed to a different model being queried."""
    monkeypatch.setattr(osched, "JOBSPEC_DIR", tmp_path)
    (tmp_path / "v1_default_LR_20260822T160705.txt").write_text("wheat_lines 0 0 0 0 3 3600\n")
    state_path = tmp_path / "failures.json"

    with patch("subprocess.run", side_effect=_fake_sacct_run(["61945_0"])):
        pls_state = osched.update_failure_state(state_path, "PLS", "testuser")

    assert pls_state == {}


# --- stuck_tasks: threshold + recency window. ---


def test_stuck_tasks_below_threshold_not_flagged(osched):
    now = datetime.datetime.now().isoformat()
    model_state = {"wheat_lines|0|0|0": {"a": now, "b": now}}  # 2 attempts, threshold is 3
    assert osched.stuck_tasks(model_state) == set()


def test_stuck_tasks_at_threshold_flagged(osched):
    now = datetime.datetime.now().isoformat()
    model_state = {"wheat_lines|0|0|0": {"a": now, "b": now, "c": now}}
    assert osched.stuck_tasks(model_state) == {("wheat_lines", 0, 0, 0)}


def test_stuck_tasks_ignores_attempts_outside_window(osched):
    stale = (datetime.datetime.now() - datetime.timedelta(hours=48)).isoformat()
    recent = datetime.datetime.now().isoformat()
    model_state = {"wheat_lines|0|0|0": {"a": stale, "b": stale, "c": recent}}
    # Only 1 of 3 attempts is within the default 24h window -- not stuck.
    assert osched.stuck_tasks(model_state) == set()


def test_stuck_tasks_one_off_transient_failure_not_flagged(osched):
    """A single recent failure alone must not permanently exclude a task --
    only persistent, repeated failure should."""
    now = datetime.datetime.now().isoformat()
    model_state = {"wheat_lines|0|0|0": {"a": now}}
    assert osched.stuck_tasks(model_state) == set()


# --- compute_backlog: end-to-end wiring -- a stuck task is excluded from the
#     backlog, unblocking every model behind it in priority order. ---


def test_compute_backlog_excludes_stuck_tasks(osched, tmp_path, monkeypatch):
    monkeypatch.setattr(osched, "JOBSPEC_DIR", tmp_path)
    monkeypatch.setattr(osched, "_ag_name", lambda model: model)
    monkeypatch.setattr(osched, "_in_flight_targets", lambda model, user: set())
    monkeypatch.setattr(osched, "_current_user", lambda: "testuser")

    results_dir = tmp_path / "results"
    targets_file = tmp_path / "target_list.json"
    targets_file.write_text(
        json.dumps(
            [
                {"dataset": "wheat_lines", "target_idx": 0, "n_repeats": 1, "excluded": False},
            ]
        )
    )
    scope = {
        "results_dir": str(results_dir),
        "n_splits": 1,
        "targets_file": str(targets_file),
        "models": ["LR"],
    }
    profile = {}

    jobspec_path = tmp_path / "v1_default_LR_20260822T160705.txt"
    jobspec_path.write_text("wheat_lines 0 0 0 0 1 3600\n")
    state_path = tmp_path / "failures.json"
    now = datetime.datetime.now().isoformat()
    state_path.write_text(
        json.dumps(
            {
                "LR": {"wheat_lines|0|0|0": {"a": now, "b": now, "c": now}},
            }
        )
    )

    with patch("subprocess.run", side_effect=_fake_sacct_run([])):
        backlog = osched.compute_backlog(scope, profile, failure_state_path=state_path)

    assert backlog["LR"] == []


def test_compute_backlog_without_failure_state_path_ignores_stuck_mechanism(
    osched, tmp_path, monkeypatch
):
    """Passing no failure_state_path (the default) must behave exactly as
    before this feature existed -- a persistently-failing task still shows up
    in the backlog, since nothing was told to track failures at all."""
    monkeypatch.setattr(osched, "JOBSPEC_DIR", tmp_path)
    monkeypatch.setattr(osched, "_ag_name", lambda model: model)
    monkeypatch.setattr(osched, "_in_flight_targets", lambda model, user: set())
    monkeypatch.setattr(osched, "_current_user", lambda: "testuser")

    results_dir = tmp_path / "results"
    targets_file = tmp_path / "target_list.json"
    targets_file.write_text(
        json.dumps(
            [
                {"dataset": "wheat_lines", "target_idx": 0, "n_repeats": 1, "excluded": False},
            ]
        )
    )
    scope = {
        "results_dir": str(results_dir),
        "n_splits": 1,
        "targets_file": str(targets_file),
        "models": ["LR"],
    }
    profile = {}

    backlog = osched.compute_backlog(scope, profile)  # no failure_state_path
    assert backlog["LR"] == [("wheat_lines", 0, 0, 0, 0, 1)]
