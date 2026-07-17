"""Concurrent writers must never produce a torn prediction file.

Two jobs computing the same (key, model, seed) is not supposed to happen, but it did:
overlapping array submissions raced on the same paths. With a plain
``DataFrame.to_csv(path)`` — mode 'w', truncate then stream — the two writes interleaved
and left a single file holding one writer's rows, a torn line where the other's flush
landed mid-row, and then the tail of the longer write. It still parsed as CSV, so nothing
raised; it simply carried duplicate rows and shredded index values, and only surfaced
much later as a silently absent roc_auc.

These tests pin the write down to an atomic publish, so a duplicated job is merely
wasteful instead of corrupting.
"""

import ast
import multiprocessing as mp
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "src", "raman_bench", "predictions.py",
)

BIG, SMALL = 120_000, 90_000


def _load_atomic_to_csv():
    """Pull _atomic_to_csv out of predictions.py without importing the module.

    Importing raman_bench.predictions drags in autogluon; this test is about file IO
    only, so parse the one function out and exec it. It is the real source, not a copy.
    """
    tree = ast.parse(open(SRC).read())
    fn = next(
        n for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_atomic_to_csv"
    )
    ns = {"os": os, "tempfile": tempfile, "pd": pd}
    exec(compile(ast.Module([fn], []), SRC, "exec"), ns)
    return ns["_atomic_to_csv"]


def _make(n, seed):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.random((n, 3)), index=np.arange(n), columns=list("012"))


def _worker(path, n, seed, barrier, atomic):
    df = _make(n, seed)  # build before the barrier so only the write races
    barrier.wait()
    if atomic:
        _load_atomic_to_csv()(df, path)
    else:
        df.to_csv(path, index=True)


def _race(tmpdir, atomic):
    """Run two writers at once on one path; return the file as pandas reads it."""
    path = os.path.join(tmpdir, "key_MODEL_proba.csv")
    barrier = mp.Barrier(2)
    procs = [
        mp.Process(target=_worker, args=(path, BIG, 1, barrier, atomic)),
        mp.Process(target=_worker, args=(path, SMALL, 2, barrier, atomic)),
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    return path


def test_concurrent_writes_do_not_tear(tmp_path):
    """The published file is always exactly one writer's output, never a blend."""
    path = _race(str(tmp_path), atomic=True)
    df = pd.read_csv(path, index_col=0)

    assert len(df) in (BIG, SMALL), f"row count {len(df)} belongs to neither writer"
    assert df.index.nunique() == len(df), "duplicate index rows: writes interleaved"
    assert not df.isna().any().any(), "NaN cells: a row was torn mid-write"
    assert list(df.index) == list(range(len(df))), "index is not the contiguous original"


def test_no_temp_files_left_behind(tmp_path):
    """The atomic publish must not litter the predictions dir with .tmp files."""
    _race(str(tmp_path), atomic=True)
    leftovers = [f for f in os.listdir(tmp_path) if f.endswith(".tmp")]
    assert leftovers == [], f"temp files survived: {leftovers}"


def test_temp_file_is_removed_when_writing_fails(tmp_path):
    """A failed write leaves neither a temp file nor a half-written target."""
    atomic = _load_atomic_to_csv()
    path = os.path.join(str(tmp_path), "key_MODEL_predictions.csv")

    class Boom(pd.DataFrame):
        def to_csv(self, *a, **k):
            raise RuntimeError("disk full")

    with pytest.raises(RuntimeError):
        atomic(Boom(_make(10, 0)), path)

    assert not os.path.exists(path), "target created despite a failed write"
    assert [f for f in os.listdir(tmp_path) if f.endswith(".tmp")] == []


def test_plain_to_csv_is_what_tore(tmp_path):
    """The bug this guards against is real: the plain write corrupts under the race.

    Not a test of our code — a demonstration that the race genuinely produces the
    duplicate rows and shredded ids seen in the results, so the atomic write above is
    load-bearing rather than defensive decoration.
    """
    path = _race(str(tmp_path), atomic=False)
    try:
        df = pd.read_csv(path, index_col=0)
    except pd.errors.ParserError:
        return  # torn badly enough that the parser rejects it: bug reproduced

    corrupt = (
        len(df) not in (BIG, SMALL)
        or df.index.nunique() != len(df)
        or bool(df.isna().any().any())
    )
    if not corrupt:
        pytest.skip("the writers did not overlap this run; the race is timing-dependent")
