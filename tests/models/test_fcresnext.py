"""Tests for raman_bench.custom_models.fcresnext — FCResNeXtModel."""

import numpy as np
import pandas as pd
import pytest
import torch

from raman_bench.models.custom.fcresnext import FCResNeXtModel, _FCResNeXtNetwork


class TestFCResNeXtNetwork:

    def test_forward_shape_multiclass(self):
        net = _FCResNeXtNetwork(n_features=100, n_outputs=3, hidden_dim=32, n_blocks=2, pool_size=4)
        out = net(torch.randn(4, 100))
        assert out.shape == (4, 3)

    def test_forward_shape_regression(self):
        net = _FCResNeXtNetwork(n_features=100, n_outputs=1, hidden_dim=32, n_blocks=2, pool_size=4)
        out = net(torch.randn(4, 100))
        assert out.shape == (4, 1)
        assert len(net.blocks) == 2


class TestFCResNeXtModel:

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
            "hidden_dim": 16,
            "patience": 100,
            "val_fraction": 0.2,
            "warmup_epochs": 1,
            "pool_size": 5,
            **kwargs,
        }
        return FCResNeXtModel(
            path="test_fcresnext",
            name="FCResNeXt",
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
        model = self._make_model("regression", n_epochs=1000)
        model.fit(X=X, y=y, time_limit=0.5)
        assert model.model is not None

    def test_hyperparameters_are_applied(self, classification_data):
        """Custom HP values propagate to the network architecture."""
        X, y = classification_data
        custom_hp = {
            "n_blocks": 3,
            "hidden_dim": 32,
            "cardinality": 2,
            "pool_size": 10,
            "fc_dropout": 0.4,
            "weight_decay": 1e-3,
        }
        model = self._make_model("multiclass", **custom_hp)
        model.fit(X=X, y=y)
        net = model.model

        # Architecture checks
        assert len(net.blocks) == custom_hp["n_blocks"]
        assert net.input_proj[0].out_features == custom_hp["hidden_dim"]
        assert len(net.blocks[0].branches) == custom_hp["cardinality"]
