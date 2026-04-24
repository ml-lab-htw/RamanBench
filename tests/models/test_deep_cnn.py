"""Unit tests for raman_bench.custom_models.deepcnn"""

import numpy as np
import pandas as pd
import pytest
import torch

pytest.importorskip("torch")

from raman_bench.models.custom.deepcnn import _DeepCNNNetwork, DeepCNNModel

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_classification_data(n_samples=50, n_features=128, n_classes=3, seed=0):
    rng = np.random.RandomState(seed)
    X = pd.DataFrame(rng.randn(n_samples, n_features).astype(np.float32))
    y = pd.Series(rng.choice(n_classes, size=n_samples))
    return X, y


def _make_regression_data(n_samples=50, n_features=128, seed=0):
    rng = np.random.RandomState(seed)
    X = pd.DataFrame(rng.randn(n_samples, n_features).astype(np.float32))
    y = pd.Series(rng.randn(n_samples).astype(np.float32))
    return X, y


def _make_fitted_model(problem_type="multiclass", n_features=128, n_epochs=2):
    model = DeepCNNModel(problem_type=problem_type)
    model.params = {
        "n_epochs": n_epochs,
        "lr": 1e-3,
        "batch_size": 16,
        "initial_channels": 16,
        "dropout": 0.0,
        "patience": 5,
        "val_fraction": 0.1,
        "dense_dim": 128,
        "warmup_epochs": 0,
        "weight_decay": 0.0,
        "aug_noise_sigma": 0.0,
        "aug_mixup_alpha": 0.0,
        "per_epoch_augmentation": True,
    }
    if problem_type in ("multiclass", "binary"):
        X, y = _make_classification_data(
            n_features=n_features,
            n_classes=3 if problem_type == "multiclass" else 2,
        )
    else:
        X, y = _make_regression_data(n_features=n_features)
    model._fit(X, y)
    return model, X


# ---------------------------------------------------------------------------
# 1. Network output shape (classification)
# ---------------------------------------------------------------------------


def test_network_output_shape_classification():
    """Forward pass should produce (batch, n_outputs) tensor."""
    n_features, n_outputs, batch = 128, 5, 8
    net = _DeepCNNNetwork(n_features, n_outputs)
    x = torch.randn(batch, n_features)
    out = net(x)
    assert out.shape == (batch, n_outputs)


# ---------------------------------------------------------------------------
# 2. _fit sets expected attributes
# ---------------------------------------------------------------------------


def test_fit_multiclass_sets_classes():
    """After fitting a multiclass model _classes and _n_classes are set."""
    model, _ = _make_fitted_model(problem_type="multiclass")
    assert hasattr(model, "_classes")
    assert model._n_classes == 3


# ---------------------------------------------------------------------------
# 3. _predict returns correct shape and valid classes
# ---------------------------------------------------------------------------


def test_predict_multiclass_returns_known_labels():
    """Every predicted label must be one of the training classes."""
    model, X = _make_fitted_model(problem_type="multiclass")
    preds = model._predict(X)
    assert len(preds) == len(X)
    assert set(preds).issubset(set(model._classes))


# ---------------------------------------------------------------------------
# 4. _predict_proba returns valid probability array
# ---------------------------------------------------------------------------


def test_proba_multiclass_shape_and_sums():
    """Multiclass probabilities: shape (n, k), rows sum to ~1."""
    model, X = _make_fitted_model(problem_type="multiclass")
    proba = model._predict_proba(X)
    assert proba.shape == (len(X), model._n_classes)
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(X)), atol=1e-5)
