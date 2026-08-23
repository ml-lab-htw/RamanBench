"""Regression test for a real production incident: GPU-tier model arrays used
the same flat ``chunk_size`` (300) as fast CPU baselines, but GPU-tier models
are real training loops -- confirmed live: an NN_TORCH array (throttle=8, 300
tasks) needed over 2 hours just to reach 168/300, implying 3.5+ hours to
fully drain ONE array, directly against the scheduler's own "opportunistic:
short bounded bursts, back off between ticks" design goal. Combined with a
separate courtesy_ceiling accuracy bug (see test_capacity_pending_gate.py),
5 separate 300-task NN_TORCH arrays piled up back to back before this was
noticed and cancelled by hand.

Fix: ``pick_chunk`` takes a much smaller ``gpu_chunk_size`` (default 32,
DEFAULT_GPU_CHUNK_SIZE) for any model in GPU_MODELS, leaving the flat
``chunk_size`` for CPU-tier models unchanged.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CLUSTER_DIR = REPO_ROOT / "cluster"


def _load_opportunistic_scheduler():
    if str(CLUSTER_DIR) not in sys.path:
        sys.path.insert(0, str(CLUSTER_DIR))
    spec = importlib.util.spec_from_file_location(
        "_opportunistic_scheduler_under_test_gpu_chunk", CLUSTER_DIR / "opportunistic_scheduler.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def scheduler():
    return _load_opportunistic_scheduler()


def _jobs(n: int) -> list[tuple]:
    return [("wheat_lines", 0, 0, i, 0, 10) for i in range(n)]


def test_gpu_model_uses_gpu_chunk_size(scheduler):
    """NN_TORCH is in GPU_MODELS -- a 300-task backlog must be chunked down to
    gpu_chunk_size, not the flat (CPU-sized) chunk_size."""
    backlog = {"NN_TORCH": _jobs(300)}
    model, chunk = scheduler.pick_chunk(backlog, chunk_size=300, gpu_chunk_size=32)
    assert model == "NN_TORCH"
    assert len(chunk) == 32


def test_cpu_model_unaffected_by_gpu_chunk_size(scheduler):
    """A CPU-tier model (not in GPU_MODELS) must keep using the flat
    chunk_size regardless of gpu_chunk_size being set."""
    backlog = {"LR": _jobs(300)}
    model, chunk = scheduler.pick_chunk(backlog, chunk_size=300, gpu_chunk_size=32)
    assert model == "LR"
    assert len(chunk) == 300


def test_gpu_chunk_size_none_falls_back_to_flat_chunk_size(scheduler):
    """Omitting gpu_chunk_size (the default) must reproduce the exact
    pre-fix behavior -- one flat chunk_size for every model, GPU or not."""
    backlog = {"NN_TORCH": _jobs(300)}
    model, chunk = scheduler.pick_chunk(backlog, chunk_size=300)
    assert model == "NN_TORCH"
    assert len(chunk) == 300


def test_gpu_chunk_size_smaller_backlog_not_padded(scheduler):
    """A GPU model with fewer tasks than gpu_chunk_size just takes what's
    there -- slicing past the end of a list is a no-op, not an error."""
    backlog = {"NN_TORCH": _jobs(10)}
    model, chunk = scheduler.pick_chunk(backlog, chunk_size=300, gpu_chunk_size=32)
    assert model == "NN_TORCH"
    assert len(chunk) == 10


def test_run_tick_reads_gpu_chunk_size_from_scope(scheduler, monkeypatch):
    """run_tick() must read scope['gpu_chunk_size'] (falling back to
    DEFAULT_GPU_CHUNK_SIZE) and actually pass it into pick_chunk."""
    captured = {}
    real_pick_chunk = scheduler.pick_chunk

    def spy_pick_chunk(backlog, chunk_size, gpu_chunk_size=None):
        captured["chunk_size"] = chunk_size
        captured["gpu_chunk_size"] = gpu_chunk_size
        return real_pick_chunk(backlog, chunk_size, gpu_chunk_size)

    monkeypatch.setattr(
        scheduler, "compute_backlog",
        lambda scope, profile, **kwargs: {"NN_TORCH": _jobs(300)},
    )
    monkeypatch.setattr(scheduler, "check_capacity", lambda *a, **k: (True, "room"))
    monkeypatch.setattr(scheduler, "submit_jobs", lambda **kwargs: [])
    monkeypatch.setattr(scheduler, "pick_chunk", spy_pick_chunk)

    scope = {
        "name": "test", "results_dir": "results", "n_splits": 3,
        "models": ["NN_TORCH"], "gpu_chunk_size": 16,
    }
    scheduler.run_tick(scope, {"slurm": True, "partition": "Debug_node"}, log_path=None, dry_run=True)

    assert captured["gpu_chunk_size"] == 16


def test_run_tick_gpu_chunk_size_defaults_when_omitted(scheduler, monkeypatch):
    captured = {}
    real_pick_chunk = scheduler.pick_chunk

    def spy_pick_chunk(backlog, chunk_size, gpu_chunk_size=None):
        captured["gpu_chunk_size"] = gpu_chunk_size
        return real_pick_chunk(backlog, chunk_size, gpu_chunk_size)

    monkeypatch.setattr(
        scheduler, "compute_backlog",
        lambda scope, profile, **kwargs: {"NN_TORCH": _jobs(300)},
    )
    monkeypatch.setattr(scheduler, "check_capacity", lambda *a, **k: (True, "room"))
    monkeypatch.setattr(scheduler, "submit_jobs", lambda **kwargs: [])
    monkeypatch.setattr(scheduler, "pick_chunk", spy_pick_chunk)

    scope = {"name": "test", "results_dir": "results", "n_splits": 3, "models": ["NN_TORCH"]}
    scheduler.run_tick(scope, {"slurm": True, "partition": "Debug_node"}, log_path=None, dry_run=True)

    assert captured["gpu_chunk_size"] == scheduler.DEFAULT_GPU_CHUNK_SIZE
