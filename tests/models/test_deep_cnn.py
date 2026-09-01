"""Unit tests for DeepCNNModel."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("autogluon")

from raman_bench.models.custom.deepcnn.model import (  # noqa: E402
    DeepCNNModel,
    _DeepCNNNetwork,
)


def _clf(n=60, f=128, n_classes=3, seed=0):
    rng = np.random.RandomState(seed)
    return rng.randn(n, f).astype(np.float32), rng.choice(n_classes, size=n)


def _bin(n=60, f=128, seed=0):
    rng = np.random.RandomState(seed)
    return rng.randn(n, f).astype(np.float32), rng.choice([0, 1], size=n)


def _reg(n=60, f=128, seed=0):
    rng = np.random.RandomState(seed)
    return rng.randn(n, f).astype(np.float32), rng.randn(n).astype(np.float32)


def _model(**kwargs):
    defaults = dict(
        n_epochs=2,
        initial_channels=8,
        dense_dim=32,
        batch_size=16,
        patience=100,
        val_fraction=0.2,
        warmup_epochs=0,
    )
    return DeepCNNModel(**{**defaults, **kwargs})


def test_network_output_shape():
    net = _DeepCNNNetwork(128, 5, initial_channels=8, dense_dim=32)
    out = torch.randn(8, 128)
    assert net(out).shape == (8, 5)


def test_predict_multiclass():
    X, y = _clf()
    m = _model().fit(X, y)
    preds = m.predict(X)
    assert len(preds) == len(X)
    assert set(preds).issubset(set(m.classes_))


def test_predict_proba_multiclass():
    X, y = _clf()
    m = _model().fit(X, y)
    proba = m.predict_proba(X)
    assert proba.shape == (len(X), m.n_classes_)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_predict_binary():
    X, y = _bin()
    m = _model().fit(X, y)
    assert set(m.predict(X)).issubset({0, 1})


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


def test_inferred_problem_type():
    X, y = _clf()
    assert _model().fit(X, y).problem_type_ == "multiclass"
    X, y = _reg()
    assert _model().fit(X, y).problem_type_ == "regression"


def test_early_stopping():
    X, y = _reg()
    assert _model(n_epochs=500, patience=1).fit(X, y).model is not None
