import math

import numpy as np
import pandas as pd
import pytest
import pytest
import torch

pytest.importorskip("torch")

from raman_bench.models.custom.ramantransformer import (
    RamanTransformerModel,
    _RamanTransformerNetwork,
)


class TestRamanTransformerNetwork:

    def test_forward_shape_multiclass(self):
        net = _RamanTransformerNetwork(
            n_features=256,
            n_outputs=3,
            patch_size=64,
            d_model=64,
            nhead=4,
            dim_feedforward=128,
            n_layers=1,
            dropout=0.0,
        )
        x = torch.randn(4, 256)
        out = net(x)
        assert out.shape == (4, 3)

    def test_forward_shape_regression(self):
        net = _RamanTransformerNetwork(
            n_features=256,
            n_outputs=1,
            patch_size=64,
            d_model=64,
            nhead=4,
            dim_feedforward=128,
            n_layers=1,
            dropout=0.0,
        )
        x = torch.randn(4, 256)
        out = net(x)
        assert out.shape == (4, 1)

    def test_n_patches_calculated_correctly(self):
        for n_features, patch_size, expected in [(128, 32, 4), (100, 32, 4), (256, 64, 4)]:
            net = _RamanTransformerNetwork(
                n_features=n_features,
                n_outputs=1,
                patch_size=patch_size,
                d_model=32,
                nhead=4,
                dim_feedforward=64,
                n_layers=1,
            )
            assert net.n_patches == expected

    def test_padding_when_not_divisible(self):
        """Spectrum length not divisible by patch_size should be padded."""
        net = _RamanTransformerNetwork(
            n_features=100,
            n_outputs=2,
            patch_size=32,
            d_model=32,
            nhead=4,
            dim_feedforward=64,
            n_layers=1,
            dropout=0.0,
        )
        assert net.padded_len == 128  # ceil(100/32)*32
        x = torch.randn(2, 100)
        out = net(x)
        assert out.shape == (2, 2)


class TestRamanTransformerModel:

    @pytest.fixture()
    def classification_data(self):
        np.random.seed(42)
        n_samples, n_features = 60, 256
        X = pd.DataFrame(np.random.randn(n_samples, n_features).astype(np.float32))
        y = pd.Series(np.random.choice([0, 1, 2], size=n_samples))
        return X, y

    @pytest.fixture()
    def binary_data(self):
        np.random.seed(42)
        n_samples, n_features = 60, 256
        X = pd.DataFrame(np.random.randn(n_samples, n_features).astype(np.float32))
        y = pd.Series(np.random.choice([0, 1], size=n_samples))
        return X, y

    @pytest.fixture()
    def regression_data(self):
        np.random.seed(42)
        n_samples, n_features = 60, 256
        X = pd.DataFrame(np.random.randn(n_samples, n_features).astype(np.float32))
        y = pd.Series(np.random.randn(n_samples).astype(np.float32))
        return X, y

    def _make_model(self, problem_type, **kwargs):
        hyperparameters = {
            "n_epochs": 2,
            "patch_size": 64,
            "d_model": 32,
            "nhead": 4,
            "dim_feedforward": 64,
            "n_layers": 1,
            "patience": 100,
            "val_fraction": 0.2,
            "warmup_epochs": 1,
            **kwargs,
        }
        model = RamanTransformerModel(
            path="test_ramantransformer",
            name="RamanTransformer",
            problem_type=problem_type,
            hyperparameters=hyperparameters,
        )
        return model

    def test_predict_proba_multiclass(self, classification_data):
        X, y = classification_data
        model = self._make_model("multiclass")
        model.fit(X=X, y=y)
        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 3)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

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
        assert model.model is not None
        assert len(preds) == len(X)
        assert np.issubdtype(preds.dtype, np.floating)

    def test_time_limit(self, regression_data):
        """Model should respect a very short time limit."""
        X, y = regression_data
        model = self._make_model("regression", n_epochs=10000)
        model.fit(X=X, y=y, time_limit=0.5)
        assert model.model is not None

    def test_hyperparameters_are_applied(self, classification_data):
        """Verify that custom hyperparameters are actually used in the model
        architecture, simulating what happens when AutoGluon's HP tuning
        algorithm suggests specific values."""
        X, y = classification_data

        custom_hp = {
            "patch_size": 32,
            "d_model": 64,
            "nhead": 4,
            "dim_feedforward": 128,
            "n_layers": 2,
            "dropout": 0.2,
        }
        model = self._make_model("multiclass", **custom_hp)
        model.fit(X=X, y=y)

        net = model.model

        # Patch projection input size == patch_size
        assert net.patch_proj.in_features == custom_hp["patch_size"]

        # Patch projection output size == d_model
        assert net.patch_proj.out_features == custom_hp["d_model"]

        # Number of transformer encoder layers
        assert len(net.transformer.layers) == custom_hp["n_layers"]

        # Head input features == d_model
        assert net.head.in_features == custom_hp["d_model"]

        # Positional encoding shape: (1, n_patches+1, d_model)
        expected_n_patches = math.ceil(256 / custom_hp["patch_size"])
        assert net.pos_embed.shape == (1, expected_n_patches + 1, custom_hp["d_model"])
