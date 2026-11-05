"""Mirror loading: the pinned mirror revision and the fallback to the original source."""

import json

import pandas as pd
import pytest

import raman_bench.benchmark as benchmark_module
from raman_bench.benchmark import RamanBenchmark


@pytest.fixture
def mirror_files(tmp_path):
    """One dataset as the mirror stores it: a parquet file and its metadata.json."""
    parquet = tmp_path / "train.parquet"
    pd.DataFrame({"100.0": [1.0, 2.0], "101.0": [3.0, 4.0], "label": ["a", "b"]}).to_parquet(
        parquet
    )
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps({"target_names": ["label"], "task_type": "classification"}))
    return {"parquet": str(parquet), "metadata.json": str(metadata)}


def _bench(tmp_path, **kwargs):
    return RamanBenchmark(
        dataset_names_classification=[],
        dataset_names_regression=[],
        cache_dir=str(tmp_path / "cache"),
        **kwargs,
    )


def test_mirror_revision_is_passed_to_every_download(tmp_path, monkeypatch, mirror_files):
    calls = []

    def fake_download(*, repo_id, filename, repo_type, revision, cache_dir):
        calls.append((filename, revision))
        return mirror_files["metadata.json" if filename.endswith("metadata.json") else "parquet"]

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    bench = _bench(tmp_path, mirror_revision="0123abcd")

    dataset = bench._load_raman_dataset("ds")
    assert bench.get_target_names("ds") == ["label"]

    assert dataset.target_names == ["label"]
    assert calls == [
        ("ds/data/train-00000-of-00001.parquet", "0123abcd"),
        ("ds/metadata.json", "0123abcd"),
        ("ds/metadata.json", "0123abcd"),
    ]


@pytest.mark.parametrize("fallback_to_source", [True, False])
def test_failed_mirror_load_falls_back_only_when_allowed(tmp_path, monkeypatch, fallback_to_source):
    source_loads = []

    def fake_download(**kwargs):
        raise OSError("mirror unreachable")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    monkeypatch.setattr(
        benchmark_module, "raman_data", lambda name, **kw: source_loads.append(name) or "source"
    )
    bench = _bench(tmp_path, fallback_to_source=fallback_to_source)

    if fallback_to_source:
        assert bench._load_raman_dataset("ds") == "source"
        assert source_loads == ["ds"]
    else:
        with pytest.raises(OSError, match="mirror unreachable"):
            bench._load_raman_dataset("ds")
        assert source_loads == []


def test_pinned_commit_loads_from_prefilled_cache_without_the_hub(tmp_path, monkeypatch):
    commit = "0123456789abcdef0123456789abcdef01234567"
    snapshot = (
        tmp_path
        / "cache"
        / "datasets_mirror"
        / "datasets--HTW-KI-Werkstatt--RamanBench"
        / "snapshots"
        / commit
        / "ds"
    )
    (snapshot / "data").mkdir(parents=True)
    pd.DataFrame({"100.0": [1.0, 2.0], "label": ["a", "b"]}).to_parquet(
        snapshot / "data" / "train-00000-of-00001.parquet"
    )
    (snapshot / "metadata.json").write_text(json.dumps({"target_names": ["label"]}))

    def no_hub(*args, **kwargs):
        raise AssertionError("contacted the Hub")

    monkeypatch.setattr("huggingface_hub.file_download.get_hf_file_metadata", no_hub)
    bench = _bench(tmp_path, mirror_revision=commit, fallback_to_source=False)

    assert bench._load_raman_dataset("ds").target_names == ["label"]
