"""Tests for raman_bench.custom_models.ramannet — RamanNetModel."""

import numpy as np
import pandas as pd
import pytest
pytest.importorskip("torch")
import torch

from raman_bench.models.custom.ramannet import RamanNetModel, _RamanNetNetwork


class TestRamanNetNetwork:

    def test_forward_shape_multiclass(self):
        net = _RamanNetNetwork(
            n_features=256,
            n_outputs=3,
            window_size=50,
            fc_dim=512,
            fc_dropout=0.5,
        )
        x = torch.randn(4, 256)
        net.eval()
        out = net(x)
        assert out.shape == (4, 3)

    def test_forward_shape_regression(self):
        net = _RamanNetNetwork(
            n_features=256,
            n_outputs=1,
            window_size=50,
            fc_dim=512,
            fc_dropout=0.5,
        )
        x = torch.randn(4, 256)
        net.eval()
        out = net(x)
        assert out.shape == (4, 1)

    def test_n_windows_calculated_correctly(self):
        # dw is always window_size // 2
        for n_features, window_size, expected in [
            (256, 50, 9),  # dw=25, (256-50)//25 + 1 = 9
            (100, 50, 3),  # dw=25, (100-50)//25 + 1 = 3
            (128, 32, 7),  # dw=16, (128-32)//16 + 1 = 7
        ]:
            net = _RamanNetNetwork(
                n_features=n_features,
                n_outputs=1,
                window_size=window_size,
                fc_dim=64,
                fc_dropout=0.5,
            )
            assert net.n_windows == expected

    def test_custom_fc_dim(self):
        net = _RamanNetNetwork(
            n_features=256,
            n_outputs=5,
            window_size=50,
            fc_dim=128,
            fc_dropout=0.3,
        )
        # head: Dropout, Linear(concat_dim, 128), BN, LeakyReLU,
        #        Dropout, Linear(128, 64), BN, LeakyReLU,
        #        Dropout, Linear(64, 5)
        assert net.head[1].out_features == 128
        assert net.head[5].out_features == 64  # 128 // 2
        assert net.head[9].out_features == 5


class TestRamanNetModel:

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
            "window_size": 50,
            "fc_dim": 64,
            "fc_dropout": 0.5,
            "patience": 100,
            "val_fraction": 0.2,
            "warmup_epochs": 1,
            **kwargs,
        }
        model = RamanNetModel(
            path="test_ramannet",
            name="RamanNet",
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
        # Binary returns 1D array of positive-class probabilities
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
            "window_size": 30,
            "fc_dim": 128,
            "fc_dropout": 0.3,
        }
        model = self._make_model("multiclass", **custom_hp)
        model.fit(X=X, y=y)

        net = model.model

        # Window size and derived stride
        assert net.window_size == custom_hp["window_size"]
        assert net.dw == custom_hp["window_size"] // 2

        # Number of windows = (256 - 30) // 15 + 1 = 16
        expected_dw = custom_hp["window_size"] // 2
        expected_n_windows = (256 - custom_hp["window_size"]) // expected_dw + 1
        assert net.n_windows == expected_n_windows

        # Each feature extractor's Linear input size == window_size
        for fe in net.feature_extractors:
            assert fe[0].in_features == custom_hp["window_size"]

        # Head fc_dim: first Linear maps to fc_dim, second to fc_dim // 2
        assert net.head[1].out_features == custom_hp["fc_dim"]
        assert net.head[5].out_features == custom_hp["fc_dim"] // 2

        # Final output Linear maps to n_classes
        assert net.head[9].out_features == 3
