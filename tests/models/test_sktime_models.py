"""Tests for RocketModel and ArsenalModel."""

import numpy as np
import pytest

pytest.importorskip("sktime")
pytest.importorskip("autogluon")

from raman_bench.models.custom.arsenal.model import ArsenalModel  # noqa: E402
from raman_bench.models.custom.rocket.model import RocketModel  # noqa: E402


def _clf(n=60, f=50, seed=42):
    rng = np.random.RandomState(seed)
    return rng.randn(n, f).astype(np.float32), rng.choice(["A", "B", "C"], size=n)


def _bin(n=60, f=50, seed=42):
    rng = np.random.RandomState(seed)
    return rng.randn(n, f).astype(np.float32), rng.choice(["X", "Y"], size=n)


class TestRocketModel:

    def test_fit_predict_multiclass(self):
        X, y = _clf()
        m = RocketModel(num_kernels=200).fit(X, y)
        preds = m.predict(X)
        assert len(preds) == len(X)
        assert set(preds).issubset({"A", "B", "C"})

    def test_fit_predict_binary(self):
        X, y = _bin()
        m = RocketModel(num_kernels=200).fit(X, y)
        assert set(m.predict(X)).issubset({"X", "Y"})

    def test_predict_proba_multiclass(self):
        X, y = _clf()
        proba = RocketModel(num_kernels=200).fit(X, y).predict_proba(X)
        assert proba.shape == (len(X), 3)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_predict_proba_binary(self):
        X, y = _bin()
        proba = RocketModel(num_kernels=200).fit(X, y).predict_proba(X)
        assert proba.ndim == 1
        assert np.all((proba >= 0) & (proba <= 1))

    def test_accepts_numpy_array(self):
        X, y = _clf()
        m = RocketModel(num_kernels=200).fit(X, y)
        preds = m.predict(X)
        assert len(preds) == len(X)


class TestArsenalModel:

    def test_fit_predict_multiclass(self):
        X, y = _clf()
        m = ArsenalModel(num_kernels=200, n_estimators=5).fit(X, y)
        preds = m.predict(X)
        assert len(preds) == len(X)
        assert set(preds).issubset({"A", "B", "C"})

    def test_predict_proba_binary(self):
        X, y = _bin()
        proba = ArsenalModel(num_kernels=200, n_estimators=5).fit(X, y).predict_proba(X)
        assert proba.ndim == 1
        assert np.all((proba >= 0) & (proba <= 1))
