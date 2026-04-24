"""Tests for raman_bench.custom_models.coatnet — CoAtNetModel."""

import numpy as np
import pandas as pd
import pytest
import pytest
import torch
pytest.importorskip("torch")

from raman_bench.models.custom.coatnet import CoAtNetModel, _CoAtNetNetwork


class TestCoAtNetNetwork:

    def test_forward_shape_multiclass(self):
        net = _CoAtNetNetwork(n_outputs=3, n_blocks=2, base_channels=8, nhead=2)
        out = net(torch.randn(4, 100))
        assert out.shape == (4, 3)

    def test_forward_shape_regression(self):
        net = _CoAtNetNetwork(n_outputs=1, n_blocks=2, base_channels=8, nhead=2)
        out = net(torch.randn(4, 100))
        assert out.shape == (4, 1)

    def test_rezero_alphas_init_zero(self):
        net = _CoAtNetNetwork(n_outputs=2, n_blocks=3, base_channels=8, nhead=2)
        for block in net.blocks:
            assert torch.allclose(block.alpha, torch.zeros(1))


class TestCoAtNetModel:

    @pytest.fixture()
    def classification_data(self):
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(60, 50).astype(np.float32))
        y = pd.Series(np.random.choice([0, 1, 2], size=60))
        return X, y

    @pytest.fixture()
    def binary_data(self):
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(60, 50).astype(np.float32))
        y = pd.Series(np.random.choice([0, 1], size=60))
        return X, y

    @pytest.fixture()
    def regression_data(self):
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(60, 50).astype(np.float32))
        y = pd.Series(np.random.randn(60).astype(np.float32))
        return X, y

    def _make_model(self, problem_type, **kwargs):
        hp = {
            "n_epochs": 2,
            "n_blocks": 2,
            "base_channels": 8,
            "nhead": 2,
            "patience": 100,
            "val_fraction": 0.2,
            "warmup_epochs": 1,
            **kwargs,
        }
        return CoAtNetModel(
            path="test_coatnet",
            name="CoAtNet",
            problem_type=problem_type,
            hyperparameters=hp,
        )

    def test_predict_multiclass(self, classification_data):
        X, y = classification_data
        model = self._make_model("multiclass")
        model.fit(X=X, y=y)
        preds = model.predict(X)
        assert len(preds) == len(X)
        assert set(preds).issubset(set(y))

    def test_predict_proba_multiclass(self, classification_data):
        X, y = classification_data
        model = self._make_model("multiclass")
        model.fit(X=X, y=y)
        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 3)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_fit_binary(self, binary_data):
        X, y = binary_data
        model = self._make_model("binary")
        model.fit(X=X, y=y)
        preds = model.predict(X)
        assert len(preds) == len(X)
        assert set(preds).issubset({0, 1})

    def test_predict_proba_binary(self, binary_data):
        X, y = binary_data
        model = self._make_model("binary")
        model.fit(X=X, y=y)
        proba = model.predict_proba(X)
        assert proba.shape == (len(X),)
        assert np.all((proba >= 0) & (proba <= 1))

    def test_predict_regression(self, regression_data):
        X, y = regression_data
        model = self._make_model("regression")
        model.fit(X=X, y=y)
        preds = model.predict(X)
        assert len(preds) == len(X)
        assert np.issubdtype(preds.dtype, np.floating)

    def test_early_stopping(self, regression_data):
        X, y = regression_data
        model = self._make_model("regression", n_epochs=100, patience=1)
        model.fit(X=X, y=y)
        assert model.model is not None

    def test_time_limit(self, regression_data):
        X, y = regression_data
        model = self._make_model("regression", n_epochs=10000)
        model.fit(X=X, y=y, time_limit=0.5)
        assert model.model is not None

    def test_hyperparameters_are_applied(self, classification_data):
        """Verify that custom HPs propagate to the network architecture,
        simulating what AutoGluon's HP tuning algorithm would suggest."""
        X, y = classification_data
        custom_hp = {
            "n_blocks": 3,
            "base_channels": 16,
            "kernel_size": 5,
            "nhead": 2,
            "attn_dropout": 0.3,
            "fc_dropout": 0.4,
        }
        model = self._make_model("multiclass", **custom_hp)
        model.fit(X=X, y=y)
        net = model.model

        # Number of ReZero encoder blocks
        assert len(net.blocks) == custom_hp["n_blocks"]

        # Stem output channels == base_channels
        assert net.stem[0].out_channels == custom_hp["base_channels"]

        # Kernel size in the first block's depthwise conv
        assert net.blocks[0].conv[0].depthwise.kernel_size[0] == custom_hp["kernel_size"]

        # Attention: number of heads
        assert net.attention.attn.num_heads == custom_hp["nhead"]

        # Head dropout probability
        assert net.head[0].p == custom_hp["fc_dropout"]
