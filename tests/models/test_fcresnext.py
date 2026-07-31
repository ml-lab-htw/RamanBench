"""Tests for FCResNeXtModel."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from raman_bench.models.custom.fcresnext.model import (  # noqa: E402
    FCResNeXtModel,
    _FCResNeXtNetwork,
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
        hidden_dim=16,
        pool_size=5,
        patience=100,
        val_fraction=0.2,
        warmup_epochs=1,
    )
    return FCResNeXtModel(**{**defaults, **kwargs})


def test_network_forward_shapes():
    for n_out in (3, 1):
        net = _FCResNeXtNetwork(
            n_features=100, n_outputs=n_out, hidden_dim=32, n_blocks=2, pool_size=4
        )
        assert net(torch.randn(4, 100)).shape == (4, n_out)


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
    assert set(_model().fit(X, y).predict(X)).issubset({0, 1})


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
    m = _model(n_blocks=3, hidden_dim=32, cardinality=2, pool_size=10).fit(X, y)
    assert len(m.model.blocks) == 3
    assert m.model.input_proj[0].out_features == 32
    assert len(m.model.blocks[0].branches) == 2
