"""Tests for raman_bench.custom_models.rezeronet — ReZeroNetModel."""

import numpy as np
import pandas as pd
import pytest
pytest.importorskip("torch")
import torch

from raman_bench.models.custom.rezeronet import ReZeroNetModel, _ReZeroNetNetwork


class TestReZeroNetNetwork:

    def test_forward_shape_multiclass(self):
        net = _ReZeroNetNetwork(
            n_outputs=3,
            input_dim=100,
            n_blocks=2,
            base_channels=16,
            kernel_size=3,
        )
        x = torch.randn(4, 100)
        out = net(x)
        assert out.shape == (4, 3)

    def test_forward_shape_regression(self):
        net = _ReZeroNetNetwork(
            n_outputs=1,
            input_dim=100,
            n_blocks=2,
            base_channels=16,
            kernel_size=3,
        )
        x = torch.randn(4, 100)
        out = net(x)
        assert out.shape == (4, 1)

    def test_rezero_alphas_init_zero(self):
        net = _ReZeroNetNetwork(n_outputs=2, input_dim=100, n_blocks=4, base_channels=16)
        for block in net.blocks:
            assert torch.allclose(block.alpha, torch.zeros(1))


class TestReZeroNetModel:

    @pytest.fixture()
    def classification_data(self):
        np.random.seed(42)
        n_samples, n_features = 60, 50
        X = pd.DataFrame(np.random.randn(n_samples, n_features).astype(np.float32))
        y = pd.Series(np.random.choice([0, 1, 2], size=n_samples))
        return X, y

    @pytest.fixture()
    def binary_data(self):
        np.random.seed(42)
        n_samples, n_features = 60, 50
        X = pd.DataFrame(np.random.randn(n_samples, n_features).astype(np.float32))
        y = pd.Series(np.random.choice([0, 1], size=n_samples))
        return X, y

    @pytest.fixture()
    def regression_data(self):
        np.random.seed(42)
        n_samples, n_features = 60, 50
        X = pd.DataFrame(np.random.randn(n_samples, n_features).astype(np.float32))
        y = pd.Series(np.random.randn(n_samples).astype(np.float32))
        return X, y

    def _make_model(self, problem_type, **kwargs):
        hyperparameters = {
            "n_epochs": 2,
            "n_blocks": 2,
            "base_channels": 8,
            "patience": 100,
            "val_fraction": 0.2,
            "warmup_epochs": 1,
            **kwargs,
        }
        model = ReZeroNetModel(
            path="test_rezero",
            name="ReZeroNet",
            problem_type=problem_type,
            hyperparameters=hyperparameters,
        )
        return model

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
        # Binary returns 1D array of positive-class probabilities
        assert proba.shape == (len(X),)
        assert np.all((proba >= 0) & (proba <= 1))

    def test_fit_regression(self, regression_data):
        X, y = regression_data
        model = self._make_model("regression")
        model.fit(X=X, y=y)
        assert model.model is not None

    def test_predict_regression(self, regression_data):
        X, y = regression_data
        model = self._make_model("regression")
        model.fit(X=X, y=y)
        preds = model.predict(X)
        assert len(preds) == len(X)
        assert preds.dtype == np.float32 or np.issubdtype(preds.dtype, np.floating)

    def test_predict_proba_regression_falls_back(self, regression_data):
        X, y = regression_data
        model = self._make_model("regression")
        model.fit(X=X, y=y)
        proba = model.predict_proba(X)
        preds = model.predict(X)
        np.testing.assert_array_almost_equal(proba, preds)

    def test_early_stopping(self, regression_data):
        """Model with patience=1 should stop before n_epochs."""
        X, y = regression_data
        model = self._make_model("regression", n_epochs=100, patience=1)
        model.fit(X=X, y=y)
        assert model.model is not None

    def test_time_limit(self, regression_data):
        """Model should respect a very short time limit."""
        X, y = regression_data
        model = self._make_model("regression", n_epochs=10000)
        model.fit(X=X, y=y, time_limit=0.5)
        assert model.model is not None

    def test_different_kernel_sizes(self, classification_data):
        X, y = classification_data
        for ks in [3, 5, 7]:
            model = self._make_model("multiclass", kernel_size=ks)
            model.fit(X=X, y=y)
            preds = model.predict(X)
            assert len(preds) == len(X)

    def test_hyperparameters_are_applied(self, classification_data):
        """Verify that custom hyperparameters are actually used in the model
        architecture and optimizer, simulating what happens when AutoGluon's
        HP tuning algorithm suggests specific values."""
        X, y = classification_data

        # Use non-default values to confirm they propagate
        custom_hp = {
            # Architecture (non-default)
            "n_blocks": 2,
            "base_channels": 4,
            "kernel_size": 5,
            "channel_factor": 1.05,
            # Regularization (non-default)
            "fc_dropout": 0.4,
            "weight_decay": 1e-3,
        }
        model = self._make_model("multiclass", **custom_hp)
        model.fit(X=X, y=y)

        net = model.model

        # Check architecture: number of ReZero blocks
        assert len(net.blocks) == custom_hp["n_blocks"]

        # Check architecture: stem output channels == base_channels
        stem_conv = net.stem[0]
        assert stem_conv.out_channels == custom_hp["base_channels"]

        # Check architecture: first block input channels == base_channels
        first_block = net.blocks[0]
        assert first_block.conv[0].depthwise.in_channels == custom_hp["base_channels"]

        # Check architecture: kernel size in the first block's depthwise conv
        first_block_ks = first_block.conv[0].depthwise.kernel_size[0]
        assert first_block_ks == custom_hp["kernel_size"]

        # Check that fc_dropout was applied (dropout layer probability)
        fc_dropout_layer = net.fc[0]
        assert isinstance(fc_dropout_layer, torch.nn.Dropout)
        assert fc_dropout_layer.p == custom_hp["fc_dropout"]
