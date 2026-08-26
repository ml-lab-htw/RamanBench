"""Tests for HydraModel (issue #4 investigation: Hydra transform + closed-form
ridge as a candidate replacement for ROCKET/Arsenal)."""

import numpy as np
import pytest

pytest.importorskip("autogluon")

from raman_bench.models.custom.hydra.model import HydraModel  # noqa: E402


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
    preds = HydraModel(random_state=0).fit(X, y).predict(X)
    assert len(preds) == len(X)
    assert np.issubdtype(preds.dtype, np.floating)
    assert not np.isnan(preds).any()


def test_predict_multiclass():
    X, y = _clf()
    m = HydraModel(random_state=0).fit(X, y)
    preds = m.predict(X)
    assert len(preds) == len(X)
    assert set(preds).issubset(set(m.classes_))


def test_predict_binary():
    X, y = _bin()
    m = HydraModel(random_state=0).fit(X, y)
    preds = m.predict(X)
    assert set(preds).issubset({"A", "B"})


def test_predict_proba_binary():
    X, y = _bin()
    proba = HydraModel(random_state=0).fit(X, y).predict_proba(X)
    assert proba.ndim == 1
    assert np.all((proba >= 0) & (proba <= 1))


def test_predict_proba_multiclass():
    X, y = _clf(n_classes=3)
    proba = HydraModel(random_state=0).fit(X, y).predict_proba(X)
    assert proba.shape == (len(X), 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_accepts_dataframe():
    import pandas as pd

    X_np, y_np = _reg()
    X = pd.DataFrame(X_np)
    m = HydraModel(random_state=0).fit(X, y_np)
    preds = m.predict(X)
    assert len(preds) == len(X)


def test_infers_regression_from_float_dtype():
    X, y = _reg()
    m = HydraModel(random_state=0).fit(X, y)
    assert m.problem_type_ == "regression"


def test_n_less_than_p_and_n_greater_than_p_both_fit():
    """Exercise both closed-form ridge branches: default k/g gives thousands
    of Hydra features (n < p, exact-LOOCV branch); a tiny k/g collapses well
    below n (n >= p, held-out-split branch)."""
    X, y = _reg(n=60, f=30)

    m_np = HydraModel(k=8, g=64, random_state=0).fit(X, y)
    assert m_np.transform_.num_features > 60
    preds_np = m_np.predict(X)
    assert not np.isnan(preds_np).any()

    m_pn = HydraModel(k=2, g=4, random_state=0).fit(X, y)
    assert m_pn.transform_.num_features < 60
    preds_pn = m_pn.predict(X)
    assert not np.isnan(preds_pn).any()
