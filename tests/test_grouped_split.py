"""Tests for the group-aware regression split.

The split must be a pure function of (data, test_size, random_state). It once was
not: group labels were built with ``str(frozenset(...))``, which renders items in
hash order, and Python randomises string hashes per process. Because
``GroupShuffleSplit`` sorts the labels via ``np.unique``, a different spelling
permuted the group order and produced a different test set from identical inputs —
so two runs of the same benchmark disagreed on the split.
"""

import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from raman_bench.benchmark import RamanBenchmark


def _bench(test_size=0.34, random_state=0):
    """A RamanBenchmark stub carrying only what the split method reads."""
    b = RamanBenchmark.__new__(RamanBenchmark)
    b.test_size = test_size
    b.random_state = random_state
    return b


def _replicated_df():
    """12 rows = 6 replicate pairs; rows i and i+6 share a target vector."""
    rng = np.random.RandomState(0)
    base = rng.rand(6, 5).round(3)
    return pd.DataFrame([dict(zip(list("abcde"), base[i % 6])) for i in range(12)])


def test_split_is_stable_across_hash_seeds():
    """Same data + same random_state must give the same split in any process.

    Runs in subprocesses because PYTHONHASHSEED is fixed at interpreter start.
    """
    prog = textwrap.dedent("""
        import numpy as np, pandas as pd
        from raman_bench.benchmark import RamanBenchmark
        rng = np.random.RandomState(0)
        base = rng.rand(6, 5).round(3)
        df = pd.DataFrame([dict(zip(list("abcde"), base[i % 6])) for i in range(12)])
        b = RamanBenchmark.__new__(RamanBenchmark)
        b.test_size, b.random_state = 0.34, 0
        _, te = b._grouped_train_test_split(df, group_by_df=df)
        print(sorted(te.index.tolist()))
        """)
    outs = set()
    for seed in ("1", "2", "3", "4", "5"):
        r = subprocess.run(
            [sys.executable, "-c", prog],
            capture_output=True,
            text=True,
            env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"},
        )
        assert r.returncode == 0, r.stderr
        outs.add(r.stdout.strip())
    assert len(outs) == 1, f"split varies with PYTHONHASHSEED: {outs}"


def test_replicates_never_span_train_and_test():
    """The whole point of grouping: no replicate pair may straddle the split."""
    df = _replicated_df()
    pairs = [(i, i + 6) for i in range(6)]
    for random_state in range(5):
        _, test = _bench(random_state=random_state)._grouped_train_test_split(df, group_by_df=df)
        in_test = set(test.index)
        for lo, hi in pairs:
            assert (lo in in_test) == (
                hi in in_test
            ), f"replicate pair ({lo},{hi}) split across partitions at seed {random_state}"


def test_random_state_still_varies_the_split():
    """Determinism must not collapse into a constant split."""
    df = _replicated_df()
    outs = {
        tuple(sorted(_bench(random_state=s)._grouped_train_test_split(df, group_by_df=df)[1].index))
        for s in range(8)
    }
    assert len(outs) > 1


def test_all_unique_groups_fall_back_to_plain_split():
    """With no replicates there is nothing to group; must match train_test_split."""
    rng = np.random.RandomState(1)
    df = pd.DataFrame(rng.rand(10, 3).round(4))
    _, test = _bench(test_size=0.2)._grouped_train_test_split(df, group_by_df=df)
    _, expected = train_test_split(df, test_size=0.2, random_state=0)
    assert sorted(test.index) == sorted(expected.index)
