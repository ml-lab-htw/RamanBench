"""scripts/run_system.py: every fold runs with its own OpenML cache, so parallel workers cannot corrupt each other's."""

import importlib.util
import os
from pathlib import Path

import pytest

pytest.importorskip("tabarena")
openml = pytest.importorskip("openml")

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_system.py"


def test_each_fold_gets_its_own_openml_cache(monkeypatch):
    spec = importlib.util.spec_from_file_location("run_system_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    seen = []

    def fake_run_one(**kwargs):
        seen.append(str(openml.config.get_cache_directory()))
        return {"metric_error": 0.0}

    monkeypatch.setattr(module, "run_one", fake_run_one)
    original = openml.config.get_cache_directory()
    for fold in range(3):
        task, f, status = module._run_job({"task": "t", "fold": fold})
        assert status.startswith("metric_error")

    assert len(set(seen)) == 3 and all("openml_cache_" in d for d in seen)
    assert not any(os.path.exists(d) for d in seen)  # removed after each fold
    openml.config.set_root_cache_directory(original)
