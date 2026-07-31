"""Tests for SANetModel."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from raman_bench.models.custom.sanet.model import (  # noqa: E402
    SANetModel,
    _SANetNetwork,
)


def _clf(n=60, f=64, seed=42):
    rng = np.random.RandomState(seed)
    return rng.randn(n, f).astype(np.float32), rng.choice([0, 1, 2], size=n)


def _bin(n=60, f=64, seed=42):
    rng = np.random.RandomState(seed)
    return rng.randn(n, f).astype(np.float32), rng.choice([0, 1], size=n)


def _reg(n=60, f=64, seed=42):
    rng = np.random.RandomState(seed)
    return rng.randn(n, f).astype(np.float32), rng.randn(n).astype(np.float32)


def _model(**kwargs):
    defaults = dict(
        n_epochs=2,
        num_blocks=3,
        initial_channels=4,
        num_branches=3,
        reduction=4,
        patience=100,
        val_fraction=0.2,
        warmup_epochs=1,
    )
    return SANetModel(**{**defaults, **kwargs})


def test_network_forward_shapes():
    for n_out in (3, 1):
        net = _SANetNetwork(
            n_outputs=n_out, num_blocks=3, initial_channels=4, num_branches=3, reduction=4
        )
        assert net(torch.randn(4, 64)).shape == (4, n_out)


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


def test_predict_regression():
    X, y = _reg()
    preds = _model().fit(X, y).predict(X)
    assert len(preds) == len(X)
    assert np.issubdtype(preds.dtype, np.floating)
