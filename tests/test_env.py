"""Tests for .env loading (raman_bench.env)."""

import os
import subprocess
import sys

from raman_bench.env import load_env

_VAR = "RAMAN_BENCH_TEST_DOTENV"


def test_import_does_not_load_dotenv(tmp_path):
    (tmp_path / ".env").write_text(f"{_VAR}=loaded\n")
    env = {k: v for k, v in os.environ.items() if k != _VAR}
    out = subprocess.run(
        [sys.executable, "-c", f"import os, raman_bench; print(os.environ.get('{_VAR}'))"],
        cwd=tmp_path, env=env, capture_output=True, text=True, check=True,
    )
    assert out.stdout.strip() == "None"


def test_load_env_finds_dotenv_above_cwd(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text(f"{_VAR}=loaded\n")
    (tmp_path / "sub").mkdir()
    monkeypatch.chdir(tmp_path / "sub")
    monkeypatch.delenv(_VAR, raising=False)
    load_env()
    assert os.environ.pop(_VAR) == "loaded"


def test_load_env_keeps_existing_variables(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text(f"{_VAR}=loaded\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(_VAR, "set")
    load_env()
    assert os.environ[_VAR] == "set"
