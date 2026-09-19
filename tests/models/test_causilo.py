"""Tests for CausiloModel (pretrained tabular foundation model, classification + regression)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("causilo")

from raman_bench.models.custom.causilo.model import CausiloModel  # noqa: E402


def _clf(n=30, f=20, seed=42):
    rng = np.random.RandomState(seed)
    return rng.randn(n, f).astype(np.float32), rng.choice(["A", "B", "C"], size=n)


def _bin(n=30, f=20, seed=42):
    rng = np.random.RandomState(seed)
    return rng.randn(n, f).astype(np.float32), rng.choice(["X", "Y"], size=n)


def _reg(n=30, f=20, seed=42):
    rng = np.random.RandomState(seed)
    return rng.randn(n, f).astype(np.float32), rng.randn(n).astype(np.float32)


class TestCausiloModel:
    def test_fit_predict_multiclass(self):
        X, y = _clf()
        m = CausiloModel(device="cpu", n_estimators=2).fit(X, y)
        preds = m.predict(X)
        assert len(preds) == len(X)
        assert set(preds).issubset({"A", "B", "C"})

    def test_fit_predict_binary(self):
        X, y = _bin()
        m = CausiloModel(device="cpu", n_estimators=2).fit(X, y)
        assert set(m.predict(X)).issubset({"X", "Y"})

    def test_predict_proba_binary(self):
        X, y = _bin()
        proba = CausiloModel(device="cpu", n_estimators=2).fit(X, y).predict_proba(X)
        assert proba.ndim == 1
        assert np.all((proba >= 0) & (proba <= 1))

    def test_predict_proba_multiclass(self):
        X, y = _clf()
        proba = CausiloModel(device="cpu", n_estimators=2).fit(X, y).predict_proba(X)
        assert proba.shape == (len(X), 3)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_fit_predict_regression(self):
        X, y = _reg()
        m = CausiloModel(device="cpu", n_estimators=2).fit(X, y)
        preds = m.predict(X)
        assert len(preds) == len(X)
        assert np.issubdtype(preds.dtype, np.floating)

    def test_predict_proba_raises_for_regression(self):
        X, y = _reg()
        m = CausiloModel(device="cpu", n_estimators=2).fit(X, y)
        with pytest.raises(AttributeError, match="not available for regression"):
            m.predict_proba(X)

    def test_many_class_uses_ecoc(self):
        # >many_class_threshold classes should route through
        # ManyClassClassifier (ECOC), not raise -- matches MitraModel/
        # TabPFNModel's own native pattern.
        pytest.importorskip("tabpfn_extensions")
        rng = np.random.RandomState(0)
        X = rng.randn(60, 20).astype(np.float32)
        y = rng.choice([str(i) for i in range(12)], size=60)
        m = CausiloModel(device="cpu", n_estimators=2, many_class_threshold=10).fit(X, y)
        preds = m.predict(X)
        assert len(preds) == len(X)
        assert set(preds).issubset(set(str(i) for i in range(12)))

    def test_accepts_dataframe(self):
        import pandas as pd

        X_np, y_np = _reg()
        X = pd.DataFrame(X_np)
        m = CausiloModel(device="cpu", n_estimators=2).fit(X, y_np)
        preds = m.predict(X)
        assert len(preds) == len(X)
