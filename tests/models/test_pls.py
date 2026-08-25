"""Tests for PLSModel."""

import numpy as np
import pytest

pytest.importorskip("autogluon")

from raman_bench.models.custom.pls.model import PLSModel  # noqa: E402


def _clf(n=60, f=30, n_classes=3, seed=0):
    rng = np.random.RandomState(seed)
    return rng.randn(n, f).astype(np.float32), rng.choice(n_classes, size=n).astype(str)


def _bin(n=60, f=30, seed=0):
    rng = np.random.RandomState(seed)
    return rng.randn(n, f).astype(np.float32), rng.choice(["A", "B"], size=n)


def _reg(n=60, f=30, seed=0):
    rng = np.random.RandomState(seed)
    return rng.randn(n, f).astype(np.float32), rng.randn(n).astype(np.float32)


def test_predict_regression():
    X, y = _reg()
    preds = PLSModel(n_components=5).fit(X, y).predict(X)
    assert len(preds) == len(X)
    assert np.issubdtype(preds.dtype, np.floating)


def test_predict_multiclass():
    X, y = _clf()
    m = PLSModel(n_components=5).fit(X, y)
    preds = m.predict(X)
    assert len(preds) == len(X)
    assert set(preds).issubset(set(m.classes_))


def test_predict_binary():
    X, y = _bin()
    m = PLSModel(n_components=5).fit(X, y)
    preds = m.predict(X)
    assert set(preds).issubset({"A", "B"})


def test_predict_proba_binary():
    X, y = _bin()
    proba = PLSModel(n_components=5).fit(X, y).predict_proba(X)
    assert proba.ndim == 1
    assert np.all((proba >= 0) & (proba <= 1))


def test_predict_proba_multiclass():
    X, y = _clf(n_classes=3)
    proba = PLSModel(n_components=5).fit(X, y).predict_proba(X)
    assert proba.shape == (len(X), 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_n_components_clipped():
    X, y = _reg(n=10, f=5)
    m = PLSModel(n_components=100).fit(X, y)
    assert m.model_.n_components <= min(5, 9)


def test_accepts_dataframe():
    import pandas as pd

    X_np, y_np = _reg()
    X = pd.DataFrame(X_np)
    m = PLSModel(n_components=5).fit(X, y_np)
    preds = m.predict(X)
    assert len(preds) == len(X)


def test_invalid_problem_type_not_raised():
    """Float y → regression is inferred without error."""
    X, y = _reg()
    m = PLSModel().fit(X, y)
    assert m.problem_type_ == "regression"
