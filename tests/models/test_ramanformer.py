"""Tests for RamanFormerModel."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from raman_bench.models.custom.ramanformer.model import (  # noqa: E402
    RamanFormerModel,
    _RamanFormerNetwork,
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
        patch_size=64,
        d_model=32,
        nhead=4,
        dim_feedforward=64,
        n_layers=1,
        post_processing_dim=32,
        patience=100,
        val_fraction=0.2,
        warmup_epochs=1,
    )
    return RamanFormerModel(**{**defaults, **kwargs})


def test_network_forward_shapes():
    for n_out in (3, 1):
        net = _RamanFormerNetwork(
            256,
            n_out,
            patch_size=64,
            d_model=64,
            nhead=4,
            dim_feedforward=128,
            n_layers=1,
            post_processing_dim=64,
        )
        assert net(torch.randn(4, 256)).shape == (4, n_out)


def test_n_patches_calculated():
    for n_feat, ps, expected in [(128, 32, 4), (100, 32, 4), (256, 64, 4)]:
        net = _RamanFormerNetwork(
            n_feat, 1, patch_size=ps, d_model=32, nhead=4, dim_feedforward=64, n_layers=1
        )
        assert net.n_patches == expected


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
    m = _model(patch_size=32, d_model=64, nhead=4, n_layers=2, post_processing_dim=64).fit(X, y)
    assert m.model.patch_proj.in_features == 32
    assert len(m.model.transformer.layers) == 2
