"""Tests that the HF mirror's group-id column survives loading.

``raman_data``'s mirror export writes replicate structure into a ``_group_id``
column. ``_load_from_mirror`` classified every non-float-parseable column as a
target, so ``_group_id`` was silently treated as one: it never reached
``RamanDataset``, and on a single-target dataset it also made ``targets``
two-dimensional.
"""

import pandas as pd
import pytest

from raman_bench import benchmark as bench_mod
from raman_bench.benchmark import GROUP_ID_COL, RamanBenchmark, _is_float_col


def _mirror_df(with_group_id: bool):
    """Two wavenumber columns, one target, optionally a group id."""
    data = {
        "400.0": [1.0, 2.0, 3.0, 4.0],
        "401.5": [5.0, 6.0, 7.0, 8.0],
    }
    if with_group_id:
        data[GROUP_ID_COL] = [0, 0, 1, 1]
    data["target"] = [0, 0, 1, 1]
    return pd.DataFrame(data)


def _bench():
    """A RamanBenchmark stub carrying only what the mirror loader reads."""
    b = RamanBenchmark.__new__(RamanBenchmark)
    b.mirror_repo = "dummy/repo"
    b.cache_dir_raw = ".cache"
    return b


@pytest.fixture
def stub_mirror(monkeypatch):
    """Serve a fabricated parquet; make metadata.json unavailable."""

    def _apply(df):
        def fake_download(repo_id, filename, repo_type, cache_dir):
            if filename.endswith("metadata.json"):
                raise FileNotFoundError("no metadata in this stub")
            return "unused.parquet"

        monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download, raising=False)
        monkeypatch.setattr(bench_mod.pd, "read_parquet", lambda _: df)

    return _apply


def test_group_id_is_not_treated_as_a_target(stub_mirror):
    stub_mirror(_mirror_df(with_group_id=True))
    ds = _bench()._load_from_mirror("stub_dataset")

    assert ds is not None
    assert ds.target_names == ["target"]
    assert ds.targets.ndim == 1


def test_group_ids_reach_the_dataset(stub_mirror):
    stub_mirror(_mirror_df(with_group_id=True))
    ds = _bench()._load_from_mirror("stub_dataset")

    assert ds.group_ids is not None
    assert ds.group_ids.tolist() == [0, 0, 1, 1]
    assert ds.spectra.shape == (4, 2)


def test_absent_group_id_column_is_fine(stub_mirror):
    stub_mirror(_mirror_df(with_group_id=False))
    ds = _bench()._load_from_mirror("stub_dataset")

    assert ds.group_ids is None
    assert ds.target_names == ["target"]


def test_group_id_col_is_not_wavenumber_parseable():
    """Guards the assumption the filter rests on."""
    assert not _is_float_col(GROUP_ID_COL)
