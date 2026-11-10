"""Spectral binning for models with a feature cap (TabSTAR-v1.1)."""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from raman_bench.preprocessing.raman_preprocessing import bin_columns


def test_bin_columns_averages_neighbours():
    X = np.arange(12, dtype=float).reshape(2, 6)
    out = bin_columns(X, 3)
    assert out.shape == (2, 3)
    np.testing.assert_allclose(out, [[0.5, 2.5, 4.5], [6.5, 8.5, 10.5]])


def test_bin_columns_keeps_uneven_widths_and_leaves_narrow_input_alone():
    X = np.random.default_rng(0).normal(size=(3, 10))
    out = bin_columns(X, 4)
    assert out.shape == (3, 4)
    # Every input column lands in exactly one bin: the bin means weighted by bin size give the row sum.
    edges = np.linspace(0, 10, 5).round().astype(int)
    np.testing.assert_allclose((out * np.diff(edges)).sum(axis=1), X.sum(axis=1))
    assert bin_columns(X, 10) is X or np.array_equal(bin_columns(X, 10), X)
    assert bin_columns(X, 20).shape == X.shape


def test_bin_columns_is_stateless():
    """Train and test rows go through the same function of the column count only."""
    rng = np.random.default_rng(1)
    train, test = rng.normal(size=(5, 9)), rng.normal(size=(2, 9))
    both = bin_columns(np.vstack([train, test]), 4)
    np.testing.assert_allclose(bin_columns(train, 4), both[:5])
    np.testing.assert_allclose(bin_columns(test, 4), both[5:])


def _stub(cap):
    return SimpleNamespace(_max_input_features=cap, __class__=SimpleNamespace(__name__="Stub"))


def test_mixin_bins_wide_frames_only():
    from raman_bench.preprocessing.mixin import RamanPreprocessingMixin

    wide = pd.DataFrame(np.arange(20, dtype=float).reshape(2, 10), columns=[f"{i}.0" for i in range(10)])
    stub = _stub(4)
    out = RamanPreprocessingMixin._bin_to_feature_cap(stub, wide)
    assert out.shape == (2, 4)
    assert list(out.columns) == [f"feature_{i}" for i in range(4)]
    assert list(out.index) == list(wide.index)
    assert stub._prep_binned_width == 4

    narrow = wide.iloc[:, :4]
    assert RamanPreprocessingMixin._bin_to_feature_cap(_stub(4), narrow) is narrow
    assert RamanPreprocessingMixin._bin_to_feature_cap(_stub(None), wide) is wide
