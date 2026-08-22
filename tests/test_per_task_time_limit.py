"""Regression tests for per-task (rather than per-chunk) SLURM time_limit
resolution.

The problem, observed live on the cluster: ``opportunistic_scheduler.py``'s
``effective_time_limit(scope, model, chunk)`` computes ONE ``time_limit`` for
an entire chunk (up to 300 tasks), taking the max needed by ANY dataset
present in that chunk. That single value used to be the only thing threaded
through to every task in the array (via one array-wide
``--export TIME_LIMIT=...``), so a fast/small dataset sharing a chunk with a
slow one (e.g. ``mlrod``, which has a blanket 10800s override) inherited the
slow dataset's inflated budget for no reason -- confirmed real: job 36545 (a
300-task CAT array) contained 9 ``mlrod`` tasks mixed with 291 others; task 0
(``alzheimer``, a small/fast dataset) ran at the full ~1350s/fold slice
implied by the 10800s budget instead of a budget appropriate to its own size.

The fix: ``submit_job.resolve_time_limit`` resolves a time_limit for ONE
dataset (rather than maxing over a whole chunk), and ``write_jobspec`` calls
it per line, so the per-task jobspec format (already one line per
(dataset, target_idx, repeat, fold, config_index, n_repeats) task -- see
``run_experiment.sbatch``'s ``LINE_NO``/``sed -n`` read) gains a 7th field:
the task's own resolved time_limit. ``run_experiment.sbatch`` reads that
field directly and only falls back to the array-wide ``TIME_LIMIT`` env var
for older-format (6-field) jobspec lines.

This file intentionally does NOT touch ``tests/test_resource_tuning_fixes.py``
(which already has its own ``effective_time_limit``/override tests, and had
real uncommitted in-progress changes from another agent's ORIONMSP fix at the
time this file was written) -- kept fully separate to avoid any risk of
colliding edits in a shared working tree.
"""

from __future__ import annotations

import importlib.util
import sys
import uuid
from pathlib import Path
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CLUSTER_DIR = REPO_ROOT / "cluster"


def _load_submit_job():
    # cluster/ isn't a package, and cluster/submit_job.py itself does
    # `from detect_cluster import ...` assuming CLUSTER_DIR is on sys.path
    # (see submit_job.py's own top-of-file sys.path handling via its
    # sibling opportunistic_scheduler.py) -- mirror that here.
    if str(CLUSTER_DIR) not in sys.path:
        sys.path.insert(0, str(CLUSTER_DIR))
    spec = importlib.util.spec_from_file_location(
        "_submit_job_under_test_time_limit", CLUSTER_DIR / "submit_job.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_opportunistic_scheduler():
    if str(CLUSTER_DIR) not in sys.path:
        sys.path.insert(0, str(CLUSTER_DIR))
    spec = importlib.util.spec_from_file_location(
        "_opportunistic_scheduler_under_test_time_limit", CLUSTER_DIR / "opportunistic_scheduler.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def submit_job():
    return _load_submit_job()


@pytest.fixture(scope="module")
def opportunistic_scheduler():
    return _load_opportunistic_scheduler()


# --- resolve_time_limit: the per-dataset counterpart to effective_time_limit's
#     whole-chunk max. Same override semantics, evaluated for one dataset. ---


def test_resolve_time_limit_no_overrides(submit_job):
    assert submit_job.resolve_time_limit(3600, "wheat_lines") == 3600


def test_resolve_time_limit_dataset_override_applies(submit_job):
    assert submit_job.resolve_time_limit(3600, "mlrod", {"mlrod": 10800}) == 10800


def test_resolve_time_limit_dataset_override_does_not_leak_to_other_datasets(submit_job):
    """The exact bug being fixed: a dataset-keyed override for one dataset
    (mlrod) must not affect the resolved value for a different dataset
    (alzheimer) even when both are considered against the same overrides
    dict, e.g. within the same chunk."""
    overrides = {"mlrod": 10800}
    assert submit_job.resolve_time_limit(3600, "mlrod", overrides) == 10800
    assert submit_job.resolve_time_limit(3600, "alzheimer", overrides) == 3600


def test_resolve_time_limit_model_override_applies(submit_job):
    assert (
        submit_job.resolve_time_limit(3600, "microgel_synthesis", None, {"microgel_synthesis": 10800})
        == 10800
    )


def test_resolve_time_limit_combines_both_sources(submit_job):
    value = submit_job.resolve_time_limit(
        3600, "mlrod", {"mlrod": 10800}, {"mlrod": 7200},
    )
    assert value == 10800  # larger of the two applicable overrides wins


def test_resolve_time_limit_never_lowers_below_default(submit_job):
    # An override smaller than the default must not lower the effective value
    # (max(), not a straight substitution).
    assert submit_job.resolve_time_limit(3600, "mlrod", {"mlrod": 100}) == 3600


def test_resolve_time_limit_model_override_scalar_applies_regardless_of_dataset(submit_job):
    """A model_time_limit_overrides entry may be a bare number instead of a
    dataset-keyed dict -- a blanket override applying no matter which dataset
    is asked about (e.g. LR: no per-dataset variation, just "don't cap this
    one" everywhere)."""
    assert submit_job.resolve_time_limit(3600, "wheat_lines", None, 800000) == 800000
    assert submit_job.resolve_time_limit(3600, "alzheimer", None, 800000) == 800000
    assert submit_job.resolve_time_limit(3600, "mlrod", None, 800000) == 800000


def test_resolve_time_limit_model_override_scalar_combines_with_dataset_override(submit_job):
    value = submit_job.resolve_time_limit(3600, "mlrod", {"mlrod": 10800}, 800000)
    assert value == 800000  # scalar model override still wins if larger


# --- write_jobspec: the 7th field must be resolved PER LINE, from that
#     line's own dataset, not one flat value for the whole file. ---


def _read_jobspec_lines(path: Path) -> list[list[str]]:
    with open(path) as f:
        return [line.split() for line in f if line.strip()]


def test_write_jobspec_per_task_time_limit_differs_within_one_chunk(submit_job, tmp_path, monkeypatch):
    """Direct regression test for the reported live bug: a chunk mixing an
    mlrod-class task with a fast-dataset task must produce DIFFERENT
    time_limit fields for each -- not one inflated value shared by both."""
    monkeypatch.setattr(submit_job, "JOBSPEC_DIR", tmp_path)
    jobs = [
        ("mlrod", 0, 0, 0, 0, 10),
        ("alzheimer", 0, 0, 0, 0, 10),
        ("alzheimer", 0, 0, 1, 0, 10),
    ]
    path = submit_job.write_jobspec(
        jobs, f"test_{uuid.uuid4().hex}",
        default_time_limit=3600,
        dataset_time_limit_overrides={"mlrod": 10800},
    )
    lines = _read_jobspec_lines(path)
    assert len(lines) == 3
    for parts in lines:
        assert len(parts) == 7, f"expected 7 fields (incl. time_limit), got {parts!r}"

    by_dataset = {parts[0]: parts[6] for parts in lines}
    assert by_dataset["mlrod"] == "10800"
    assert by_dataset["alzheimer"] == "3600"


def test_write_jobspec_no_overrides_uses_flat_default_everywhere(submit_job, tmp_path, monkeypatch):
    monkeypatch.setattr(submit_job, "JOBSPEC_DIR", tmp_path)
    jobs = [("wheat_lines", 0, 0, 0, 0, 10), ("alzheimer", 0, 0, 0, 0, 10)]
    path = submit_job.write_jobspec(jobs, f"test_{uuid.uuid4().hex}", default_time_limit=3600)
    lines = _read_jobspec_lines(path)
    assert all(parts[6] == "3600" for parts in lines)


# --- submit_jobs: the full path opportunistic_scheduler.run_tick drives --
#     guards specifically against passing the already-chunk-maxed ceiling as
#     the per-task baseline (which would silently reproduce the bug, since
#     max(ceiling, ...) can never fall back below the ceiling). ---


def test_submit_jobs_chunk_ceiling_does_not_leak_into_per_task_baseline(submit_job, tmp_path, monkeypatch):
    """Mirrors exactly how opportunistic_scheduler.run_tick calls submit_jobs:
    `time_limit` is the whole-chunk ceiling (as effective_time_limit would
    return for a chunk containing mlrod -- 10800), while `default_time_limit`
    is the flat scope default (3600) each task's own dataset should bump up
    from. If `default_time_limit` were ever accidentally dropped/ignored and
    the chunk ceiling used as the per-task baseline instead, every task
    (including the fast alzheimer one) would incorrectly resolve to >=10800."""
    monkeypatch.setattr(submit_job, "JOBSPEC_DIR", tmp_path)
    chunk = [
        ("mlrod", 0, 0, 0, 0, 10),
        ("alzheimer", 0, 0, 0, 0, 10),
    ]
    profile = {"name": "test_profile", "slurm": True}
    slug = f"test_{uuid.uuid4().hex}"

    job_ids = submit_job.submit_jobs(
        model="CAT", jobs=chunk, slug=slug,
        n_splits=3, num_random_configs=50, num_bag_folds=8,
        time_limit=10800,  # whole-chunk ceiling (effective_time_limit's output)
        default_time_limit=3600,  # flat scope default (the fix under test)
        dataset_time_limit_overrides={"mlrod": 10800},
        results_dir="results/v1/data", cache_dir=".cache_v1",
        mirror_repo="HTW-KI-Werkstatt/RamanBench",
        profile=profile, throttle=8, dry_run=True,
    )
    assert job_ids == []  # dry run -- nothing actually submitted

    jobspec_path = tmp_path / f"{slug}.txt"
    lines = _read_jobspec_lines(jobspec_path)
    by_dataset = {parts[0]: parts[6] for parts in lines}
    assert by_dataset["mlrod"] == "10800"
    assert by_dataset["alzheimer"] == "3600", (
        "alzheimer must NOT inherit mlrod's chunk-wide ceiling -- "
        f"got {by_dataset['alzheimer']!r}"
    )


def test_submit_jobs_default_time_limit_falls_back_to_time_limit(submit_job, tmp_path, monkeypatch):
    """When default_time_limit is omitted (the single-(dataset,target) CLI
    path's usage -- see submit()), it falls back to `time_limit` itself,
    which is correct there since that path never has a chunk-wide-inflated
    ceiling to worry about (one dataset per array)."""
    monkeypatch.setattr(submit_job, "JOBSPEC_DIR", tmp_path)
    jobs = [("wheat_lines", 0, 0, 0, 0, 10), ("wheat_lines", 0, 0, 1, 0, 10)]
    profile = {"name": "test_profile", "slurm": True}
    slug = f"test_{uuid.uuid4().hex}"

    submit_job.submit_jobs(
        model="PLS", jobs=jobs, slug=slug,
        n_splits=3, num_random_configs=50, num_bag_folds=8,
        time_limit=3600,
        results_dir="results/v1/data", cache_dir=".cache_v1",
        mirror_repo="HTW-KI-Werkstatt/RamanBench",
        profile=profile, throttle=8, dry_run=True,
    )
    lines = _read_jobspec_lines(tmp_path / f"{slug}.txt")
    assert all(parts[6] == "3600" for parts in lines)


# --- _in_flight_targets: must keep parsing OLDER-format (6-field) jobspec
#     files -- an already-queued array submitted before this fix has a
#     6-field jobspec on disk that this dedup logic still needs to read. ---


def test_in_flight_targets_parses_both_6_and_7_field_jobspecs(opportunistic_scheduler, tmp_path, monkeypatch):
    monkeypatch.setattr(opportunistic_scheduler, "JOBSPEC_DIR", tmp_path)

    old_format_path = tmp_path / "old_part.txt"
    old_format_path.write_text("wheat_lines 0 0 0 0 10\n")

    new_format_path = tmp_path / "new_part.txt"
    new_format_path.write_text("mlrod 0 0 0 0 10 10800\nalzheimer 1 0 0 0 10 3600\n")

    def fake_run(cmd, capture_output, text, check):
        class _Result:
            pass

        result = _Result()
        if cmd[0] == "squeue":
            result.stdout = "RB_PLS_old_part\nRB_PLS_new_part\n"
        else:  # sacct
            result.stdout = ""
        return result

    with patch("subprocess.run", side_effect=fake_run):
        targets = opportunistic_scheduler._in_flight_targets("PLS", "testuser")

    assert ("wheat_lines", 0, 0, 0) in targets
    assert ("mlrod", 0, 0, 0) in targets
    assert ("alzheimer", 1, 0, 0) in targets
