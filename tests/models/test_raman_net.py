"""Tests for RamanNetModel."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from raman_bench.models.custom.ramannet.model import (  # noqa: E402
    RamanNetModel,
    _RamanNetNetwork,
)


def _clf(n=60, f=256, seed=42):
    rng = np.random.RandomState(seed)
    return rng.randn(n, f).astype(np.float32), rng.choice([0, 1, 2], size=n)


def _bin(n=60, f=256, seed=42):
    rng = np.random.RandomState(seed)
    return rng.randn(n, f).astype(np.float32), rng.choice([0, 1], size=n)


def _reg(n=60, f=256, seed=42):
    rng = np.random.RandomState(seed)
    return rng.randn(n, f).astype(np.float32), rng.randn(n).astype(np.float32)


def _model(**kwargs):
    defaults = dict(
        n_epochs=2,
        window_size=50,
        fc_dim=64,
        fc_dropout=0.5,
        patience=100,
        val_fraction=0.2,
        warmup_epochs=1,
    )
    return RamanNetModel(**{**defaults, **kwargs})


def test_network_forward_shapes():
    for n_out in (3, 1):
        net = _RamanNetNetwork(n_features=256, n_outputs=n_out, window_size=50, fc_dim=512)
        net.eval()
        assert net(torch.randn(4, 256)).shape == (4, n_out)


def test_n_windows_calculated_correctly():
    for n_features, window_size, expected in [
        (256, 50, 9),
        (100, 50, 3),
        (128, 32, 7),
    ]:
        net = _RamanNetNetwork(
            n_features=n_features, n_outputs=1, window_size=window_size, fc_dim=64
        )
        assert net.n_windows == expected


def test_predict_proba_multiclass():
    X, y = _clf()
    proba = _model().fit(X, y).predict_proba(X)
    assert proba.shape == (len(X), 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_predict_proba_binary():
    X, y = _bin()
    proba = _model().fit(X, y).predict_proba(X)
    assert proba.ndim == 1
    assert np.all((proba >= 0) & (proba <= 1))


def test_predict_regression():
    X, y = _reg()
    preds = _model().fit(X, y).predict(X)
    assert len(preds) == len(X)
    assert np.issubdtype(preds.dtype, np.floating)


def test_hyperparameters_applied():
    X, y = _clf()
    m = _model(window_size=30, fc_dim=128, fc_dropout=0.3).fit(X, y)
    net = m.model
    assert net.window_size == 30
    assert net.head[1].out_features == 128
    assert net.head[5].out_features == 64
