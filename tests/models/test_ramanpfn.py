"""Tests for RamanPFNModel."""

import numpy as np
import pytest

pytest.importorskip("autogluon")

from raman_bench.models.custom.ramanpfn.model import RamanPFNModel  # noqa: E402


def _clf(n=60, f=120, n_classes=3, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.rand(n, f).astype(np.float32)
    y = rng.choice(n_classes, size=n).astype(str)
    return X, y


def _bin(n=60, f=120, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.rand(n, f).astype(np.float32)
    y = rng.choice(["A", "B"], size=n)
    return X, y


def _reg(n=60, f=120, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.rand(n, f).astype(np.float32)
    y = rng.randn(n).astype(np.float32)
    return X, y


def _small_kwargs():
    # Keep GCU/LVSE views tiny for fast TabPFN forward passes in tests.
    return dict(rho=4, n_regions=4, k_per_region=2, n_estimators=2)


def test_predict_regression():
    X, y = _reg()
    m = RamanPFNModel(**_small_kwargs()).fit(X, y)
    preds = m.predict(X)
    assert len(preds) == len(X)
    assert np.issubdtype(preds.dtype, np.floating)
    assert not np.isnan(preds).any()


def test_predict_multiclass():
    X, y = _clf()
    m = RamanPFNModel(**_small_kwargs()).fit(X, y)
    preds = m.predict(X)
    assert len(preds) == len(X)
    assert set(preds).issubset(set(m.classes_))


def test_predict_binary():
    X, y = _bin()
    m = RamanPFNModel(**_small_kwargs()).fit(X, y)
    preds = m.predict(X)
    assert set(preds).issubset({"A", "B"})


def test_predict_proba_binary():
    X, y = _bin()
    m = RamanPFNModel(**_small_kwargs()).fit(X, y)
    proba = m.predict_proba(X)
    assert proba.ndim == 1
    assert np.all((proba >= 0) & (proba <= 1))


def test_predict_proba_multiclass():
    X, y = _clf(n_classes=3)
    m = RamanPFNModel(**_small_kwargs()).fit(X, y)
    proba = m.predict_proba(X)
    assert proba.shape == (len(X), 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_accepts_dataframe():
    import pandas as pd

    X_np, y_np = _reg()
    X = pd.DataFrame(X_np)
    m = RamanPFNModel(**_small_kwargs()).fit(X, y_np)
    preds = m.predict(X)
    assert len(preds) == len(X)


def test_lam_extremes_match_single_view():
    """lam=0 -> pure GCU-view prediction; lam=1 -> pure LVSE-view prediction."""
    X, y = _reg()
    kwargs = _small_kwargs()

    m0 = RamanPFNModel(lam=0.0, **kwargs).fit(X, y)
    preds0 = m0.predict(X)
    W_gcu, _ = m0._views(X)
    pred_gcu_only = m0._backbone_gcu.predict(W_gcu)
    np.testing.assert_allclose(preds0, pred_gcu_only, atol=1e-6)

    m1 = RamanPFNModel(lam=1.0, **kwargs).fit(X, y)
    preds1 = m1.predict(X)
    _, W_lvse = m1._views(X)
    pred_lvse_only = m1._backbone_lvse.predict(W_lvse)
    np.testing.assert_allclose(preds1, pred_lvse_only, atol=1e-6)
