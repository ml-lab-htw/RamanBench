"""Unit tests for scripts/run_experiment_batch.py (no dataset downloads)."""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_experiment_batch as rb  # noqa: E402

JOB = {"dataset_name": "ds", "target_idx": 2, "model_key": "PLS", "repeat": 0, "fold": 1,
       "recipe_config": "configs/snv.json"}


def test_claim_is_exclusive_and_named_by_job(tmp_path):
    first = rb.try_claim(tmp_path, JOB)
    assert first is not None and first.name == "snv__ds__2__0_1__c0" and first.parent.name == "PLS"
    assert (first / "owner").exists()
    assert rb.try_claim(tmp_path, JOB) is None
    assert rb.try_claim(tmp_path, {**JOB, "fold": 2}) is not None


def test_read_jobspec_rejects_unknown_and_missing_keys(tmp_path):
    p = tmp_path / "jobs.jsonl"
    p.write_text(json.dumps(JOB) + "\n\n")
    assert rb.read_jobspec(p) == [JOB]
    p.write_text(json.dumps({**JOB, "dataset": "x"}) + "\n")
    with pytest.raises(AssertionError, match="unknown keys"):
        rb.read_jobspec(p)
    p.write_text(json.dumps({k: v for k, v in JOB.items() if k != "fold"}) + "\n")
    with pytest.raises(AssertionError, match="missing keys"):
        rb.read_jobspec(p)


def test_stage_timer_counts_nested_calls_once():
    t = rb.StageTimer()

    def inner():
        return 1

    inner_w = t.wrap("prep", inner)

    def outer():
        return inner_w() + 1

    assert t.wrap("prep", outer)() == 2
    assert t.calls["prep"] == 1
    t.reset()
    assert not t.seconds and not t.calls


def test_memoized_loader_loads_once_and_returns_copies(monkeypatch):
    from raman_bench import experiment_utils as eu

    calls = []

    def fake_load(dataset_name, target_idx, **kwargs):
        calls.append((dataset_name, target_idx))
        return "dataset", pd.DataFrame({"a": [1.0, 2.0]}), None, "regression"

    monkeypatch.setattr(eu, "load_dataframe", fake_load)
    rb.memoize_load_dataframe()
    _, df1, _, _ = eu.load_dataframe("ds", 0, cache_dir="c")
    df1.loc[0, "a"] = 99.0
    _, df2, _, _ = eu.load_dataframe("ds", 0, cache_dir="c")
    eu.load_dataframe("ds", 1, cache_dir="c")
    assert calls == [("ds", 0), ("ds", 1)]
    assert df2.loc[0, "a"] == 1.0


def test_memoized_torch_load_reads_once_and_copies(tmp_path, monkeypatch):
    import torch

    path = tmp_path / "ckpt.pt"
    torch.save({"w": torch.arange(4.0)}, path)
    reads = []
    real = torch.load

    def counting(f, *a, **k):
        reads.append(str(f))
        return real(f, *a, **k)

    monkeypatch.setattr(torch, "load", counting)
    rb.memoize_torch_load()
    a = torch.load(str(path), map_location="cpu")
    a["w"][0] = 99.0  # a caller modifying its copy must not affect the next one
    b = torch.load(path, map_location="cpu")
    assert len(reads) == 1 and b["w"][0] == 0.0
    torch.load(str(path), map_location="cpu", weights_only=True)  # other kwargs: own entry
    assert len(reads) == 2
    torch.save({"w": torch.ones(2)}, path)  # changed file is read again
    import os
    os.utime(path, ns=(1, 1))
    assert torch.load(str(path), map_location="cpu")["w"].shape == (2,) and len(reads) == 3
