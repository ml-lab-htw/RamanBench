"""Tests for CoAtNetModel."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from raman_bench.models.custom.coatnet.model import (  # noqa: E402
    CoAtNetModel,
    _CoAtNetNetwork,
)


def _clf(n=60, f=50, seed=42):
    rng = np.random.RandomState(seed)
    return rng.randn(n, f).astype(np.float32), rng.choice([0, 1, 2], size=n)


def _bin(n=60, f=50, seed=42):
    rng = np.random.RandomState(seed)
    return rng.randn(n, f).astype(np.float32), rng.choice([0, 1], size=n)


def _reg(n=60, f=50, seed=42):
    rng = np.random.RandomState(seed)
    return rng.randn(n, f).astype(np.float32), rng.randn(n).astype(np.float32)


def _model(**kwargs):
    defaults = dict(
        n_epochs=2,
        n_blocks=2,
        base_channels=8,
        nhead=2,
        patience=100,
        val_fraction=0.2,
        warmup_epochs=1,
    )
    return CoAtNetModel(**{**defaults, **kwargs})


def test_network_forward_shapes():
    for n_out in (3, 1):
        net = _CoAtNetNetwork(n_outputs=n_out, n_blocks=2, base_channels=8, nhead=2)
        assert net(torch.randn(4, 100)).shape == (4, n_out)


def test_rezero_alphas_init_zero():
    net = _CoAtNetNetwork(n_outputs=2, n_blocks=3, base_channels=8, nhead=2)
    for block in net.blocks:
        assert torch.allclose(block.alpha, torch.zeros(1))


def test_predict_multiclass():
    X, y = _clf()
    m = _model().fit(X, y)
    preds = m.predict(X)
    assert len(preds) == len(X)
    assert set(preds).issubset(set(m.classes_))


def test_predict_proba_multiclass():
    X, y = _clf()
    proba = _model().fit(X, y).predict_proba(X)
    assert proba.shape == (len(X), 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_predict_binary():
    X, y = _bin()
    preds = _model().fit(X, y).predict(X)
    assert set(preds).issubset({0, 1})


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
    m = _model(n_blocks=3, base_channels=16, kernel_size=5, nhead=2, fc_dropout=0.4).fit(X, y)
    assert len(m.model.blocks) == 3
    assert m.model.stem[0].out_channels == 16
    assert m.model.blocks[0].conv[0].depthwise.kernel_size[0] == 5
    assert m.model.head[0].p == 0.4
