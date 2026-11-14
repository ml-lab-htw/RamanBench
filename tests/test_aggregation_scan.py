"""``scan_cached_results``: threaded reading and the per-model cache."""

from __future__ import annotations

import os
import pickle

from raman_bench.aggregation import scan_cached_results


def _write(root, model, task, fold, value):
    d = root / model / task / f"0_{fold}"
    d.mkdir(parents=True)
    with open(d / "results.pkl", "wb") as f:
        pickle.dump({"model": model, "task": task, "fold": fold, "value": value}, f)
    return d / "results.pkl"


def _tree(root):
    for model in ("A_c1_BAG_L1", "B_c1_BAG_L1"):
        for task in ("t1__0", "t2__0"):
            for fold in range(3):
                _write(root, model, task, fold, 0)


def _key(results):
    return [(r["model"], r["task"], r["fold"], r["value"]) for r in results]


def test_threads_keep_the_sorted_path_order(tmp_path):
    _tree(tmp_path)
    serial = scan_cached_results(str(tmp_path), workers=1)
    threaded = scan_cached_results(str(tmp_path), workers=8)
    assert len(serial) == 12
    assert _key(serial) == _key(threaded)
    assert _key(serial) == sorted(_key(serial))


def test_cache_gives_the_same_results_and_skips_reading(tmp_path, monkeypatch):
    data, cache = tmp_path / "data", tmp_path / "cache"
    _tree(data)
    first = scan_cached_results(str(data), cache_dir=str(cache))
    assert sorted(p.name.split(".")[0] for p in cache.iterdir()) == ["A_c1_BAG_L1", "B_c1_BAG_L1"]

    import raman_bench.aggregation as agg

    monkeypatch.setattr(agg, "load_pickle", lambda path: (_ for _ in ()).throw(AssertionError(path)))
    second = scan_cached_results(str(data), cache_dir=str(cache))
    assert _key(second) == _key(first)


def test_cache_rereads_only_the_changed_model(tmp_path, monkeypatch):
    data, cache = tmp_path / "data", tmp_path / "cache"
    _tree(data)
    scan_cached_results(str(data), cache_dir=str(cache))
    before = {p.name for p in cache.iterdir()}

    path = _write_over(data / "B_c1_BAG_L1" / "t1__0" / "0_1" / "results.pkl", 7)
    import raman_bench.aggregation as agg

    read = []
    real = agg.load_pickle
    monkeypatch.setattr(agg, "load_pickle", lambda p: (read.append(p), real(p))[1])
    out = scan_cached_results(str(data), cache_dir=str(cache))

    assert {os.path.relpath(p, data).split(os.sep)[0] for p in read} == {"B_c1_BAG_L1"}
    assert [r["value"] for r in out if r["model"] == "B_c1_BAG_L1" and r["task"] == "t1__0" and r["fold"] == 1] == [7]
    after = {p.name for p in cache.iterdir()}
    assert len(after) == 2, "the stale cache file of the changed model must be removed"
    assert {n for n in after if n.startswith("A_")} == {n for n in before if n.startswith("A_")}
    assert path.exists()


def test_cache_notices_a_new_result(tmp_path):
    data, cache = tmp_path / "data", tmp_path / "cache"
    _tree(data)
    assert len(scan_cached_results(str(data), cache_dir=str(cache))) == 12
    _write(data, "A_c1_BAG_L1", "t3__0", 0, 0)
    assert len(scan_cached_results(str(data), cache_dir=str(cache))) == 13


def test_unreadable_file_is_skipped_with_a_warning(tmp_path, caplog):
    _tree(tmp_path)
    (tmp_path / "A_c1_BAG_L1" / "t1__0" / "0_0" / "results.pkl").write_bytes(b"not a pickle")
    out = scan_cached_results(str(tmp_path))
    assert len(out) == 11
    assert "Failed to load" in caplog.text


def _write_over(path, value):
    d = path.parent
    with open(path, "wb") as f:
        pickle.dump({"model": d.parent.parent.name, "task": d.parent.name, "fold": int(d.name.split("_")[1]), "value": value}, f)
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
    return path
