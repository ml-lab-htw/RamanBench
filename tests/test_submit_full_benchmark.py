"""cluster/submit_full_benchmark.py applies the scope's time budgets on both backends."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

_CLUSTER_DIR = Path(__file__).resolve().parent.parent / "cluster"
if str(_CLUSTER_DIR) not in sys.path:
    sys.path.insert(0, str(_CLUSTER_DIR))

_spec = importlib.util.spec_from_file_location("submit_full_benchmark", _CLUSTER_DIR / "submit_full_benchmark.py")
submit_full_benchmark = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(submit_full_benchmark)
import submit_job  # noqa: E402  (the module the script imports at run time)

SCOPE = {
    "time_limit": 600,
    "num_bag_folds": 3,
    "time_limit_overrides": {"big": 10800},
    "model_time_limit_overrides": {"M": {"wide": 5400}},
    "large_datasets": [],
}


@pytest.fixture
def run(tmp_path, monkeypatch):
    targets = tmp_path / "targets.json"
    targets.write_text(json.dumps([
        {"dataset": d, "target_idx": 0, "excluded": False, "n_repeats": 1} for d in ("small", "big", "wide")
    ]))
    scope = tmp_path / "scope.json"
    scope.write_text(json.dumps(SCOPE))

    def run(backend, *extra):
        monkeypatch.setattr(submit_job, "resolve_profile", lambda *a: {"backend": backend})
        monkeypatch.setattr(
            sys, "argv",
            ["submit_full_benchmark.py", "--model", "M", "--targets-file", str(targets),
             "--scope", str(scope), "--dry-run", *extra],
        )
        submit_full_benchmark.main()

    return run


def test_k8s_path_resolves_each_tasks_time_limit(run, monkeypatch):
    calls = []
    monkeypatch.setattr(submit_job, "submit_jobs", lambda **kw: calls.append(kw) or [])
    run("k8s")
    (kw,) = calls
    assert kw["default_time_limit"] == 600 and kw["num_bag_folds"] == 3
    limits = {
        d: submit_job.resolve_time_limit(
            kw["default_time_limit"], d, kw["dataset_time_limit_overrides"], kw["model_time_limit_overrides"]
        )
        for d in ("small", "big", "wide")
    }
    assert limits == {"small": 600, "big": 10800, "wide": 5400}
    assert kw["time_limit"] == 10800  # the array-wide fallback covers every task


def test_slurm_path_passes_each_tasks_time_limit(run, monkeypatch):
    cmds = []

    def fake_run(cmd, **_):
        cmds.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="ok\n", stderr="")

    monkeypatch.setattr(submit_full_benchmark.subprocess, "run", fake_run)
    run("slurm", "--time-limit", "900")
    got = {c[c.index("--dataset") + 1]: float(c[c.index("--time-limit") + 1]) for c in cmds}
    assert got == {"small": 900, "big": 10800, "wide": 5400}
