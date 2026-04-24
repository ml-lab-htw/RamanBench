"""Tests for raman_bench.custom_models.sanet — SANetModel."""

import numpy as np
import pandas as pd
import pytest
import pytest
import torch

pytest.importorskip("torch")

from raman_bench.models.custom.sanet import SANetModel, _MultiScaleBlock, _SANetNetwork


class TestMultiScaleBlock:

    def test_forward_shape(self):
        block = _MultiScaleBlock(
            in_channels=1,
            out_channels=16,
            reduction=4,
            num_branches=4,
            stride=2,
        )
        x = torch.randn(4, 1, 128)
        out = block(x)
        assert out.shape == (4, 16, 64)  # stride=2 should halve the length

    def test_branch_count_matches_num_branches(self):
        for n in [3, 4, 6]:
            block = _MultiScaleBlock(
                in_channels=1,
                out_channels=8,
                reduction=4,
                num_branches=n,
                stride=1,
            )
            assert len(block.branches) == n


class TestSANetNetwork:

    def test_forward_shape_multiclass(self):
        net = _SANetNetwork(
            n_outputs=3,
            num_blocks=2,
            channel_factor=2.0,
            initial_channels=8,
            num_branches=3,
            reduction=4,
        )
        x = torch.randn(4, 100)
        out = net(x)
        assert out.shape == (4, 3)

    def test_forward_shape_regression(self):
        net = _SANetNetwork(
            n_outputs=1,
            num_blocks=2,
            channel_factor=2.0,
            initial_channels=8,
            num_branches=3,
            reduction=4,
        )
        x = torch.randn(4, 100)
        out = net(x)
        assert out.shape == (4, 1)

    def test_num_blocks_creates_correct_layers(self):
        for n_blocks in [2, 3, 5]:
            net = _SANetNetwork(
                n_outputs=2,
                num_blocks=n_blocks,
                channel_factor=2.0,
                initial_channels=8,
                num_branches=3,
                reduction=4,
            )
            assert len(net.blocks) == n_blocks

    def test_channel_sequence(self):
        net = _SANetNetwork(
            n_outputs=2,
            num_blocks=3,
            channel_factor=2.0,
            initial_channels=8,
            num_branches=3,
            reduction=4,
        )
        # Expected channel_seq: [1, 8, 16, 32]
        # First block: in=1, out=8
        first_branch_conv = net.blocks[0].branches[0][0]
        assert first_branch_conv.in_channels == 1
        assert first_branch_conv.out_channels == 8
        # Second block: in=8, out=16
        second_branch_conv = net.blocks[1].branches[0][0]
        assert second_branch_conv.in_channels == 8
        assert second_branch_conv.out_channels == 16
        # Second block: in=16, out=32
        third_branch_conv = net.blocks[2].branches[0][0]
        assert third_branch_conv.in_channels == 16
        assert third_branch_conv.out_channels == 32


class TestSANetModel:

    @pytest.fixture()
    def classification_data(self):
        np.random.seed(42)
        n_samples, n_features = 30, 50
        X = pd.DataFrame(np.random.randn(n_samples, n_features).astype(np.float32))
        y = pd.Series(np.random.choice([0, 1, 2], size=n_samples))
        return X, y

    @pytest.fixture()
    def binary_data(self):
        np.random.seed(42)
        n_samples, n_features = 30, 50
        X = pd.DataFrame(np.random.randn(n_samples, n_features).astype(np.float32))
        y = pd.Series(np.random.choice([0, 1], size=n_samples))
        return X, y

    @pytest.fixture()
    def regression_data(self):
        np.random.seed(42)
        n_samples, n_features = 30, 50
        X = pd.DataFrame(np.random.randn(n_samples, n_features).astype(np.float32))
        y = pd.Series(np.random.randn(n_samples).astype(np.float32))
        return X, y

    def _make_model(self, problem_type, **kwargs):
        hyperparameters = {
            "n_epochs": 2,
            "num_blocks": 2,
            "initial_channels": 8,
            "num_branches": 3,
            "reduction": 4,
            "patience": 100,
            "val_fraction": 0.2,
            "warmup_epochs": 1,
            **kwargs,
        }
        model = SANetModel(
            path="test_sanet",
            name="SANet",
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

    def test_predict_regression(self, regression_data):
        X, y = regression_data
        model = self._make_model("regression")
        model.fit(X=X, y=y)
        preds = model.predict(X)
        assert len(preds) == len(X)
        assert preds.dtype == np.float32 or np.issubdtype(preds.dtype, np.floating)

    def test_time_limit(self, regression_data):
        """Model should respect a very short time limit."""
        X, y = regression_data
        model = self._make_model("regression", n_epochs=10000)
        model.fit(X=X, y=y, time_limit=0.5)
        assert model.model is not None

    def test_different_num_branches(self, classification_data):
        X, y = classification_data
        for nb in [3, 6]:
            model = self._make_model("multiclass", num_branches=nb)
            model.fit(X=X, y=y)
            preds = model.predict(X)
            assert len(preds) == len(X)

    def test_hyperparameters_are_applied(self, classification_data):
        """Verify that custom hyperparameters are actually used in the model
        architecture, simulating what happens when AutoGluon's HP tuning
        algorithm suggests specific values."""
        X, y = classification_data

        # Use non-default values to confirm they propagate
        custom_hp = {
            # Architecture (non-default)
            "num_blocks": 3,
            "channel_factor": 2.0,
            "initial_channels": 8,
            "num_branches": 4,
            "reduction": 4,
        }
        model = self._make_model("multiclass", **custom_hp)
        model.fit(X=X, y=y)

        net = model.model

        # Check architecture: number of multi-scale blocks
        assert len(net.blocks) == custom_hp["num_blocks"]

        # Check architecture: first block has correct number of branches
        first_block = net.blocks[0]
        assert len(first_block.branches) == custom_hp["num_branches"]

        # Check architecture: first block input channels == 1 (from channel_seq[0])
        first_branch_conv = first_block.branches[0][0]
        assert first_branch_conv.in_channels == 1

        # Check architecture: first block output channels == initial_channels
        assert first_branch_conv.out_channels == custom_hp["initial_channels"]

        # Check architecture: second block output channels == initial_channels * channel_factor
        second_block = net.blocks[1]
        second_branch_conv = second_block.branches[0][0]
        expected_second_channels = int(custom_hp["initial_channels"] * custom_hp["channel_factor"])
        assert second_branch_conv.out_channels == expected_second_channels

        # Check architecture: all blocks have the requested number of branches
        for block in net.blocks:
            assert len(block.branches) == custom_hp["num_branches"]
