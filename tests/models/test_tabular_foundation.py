"""Tests for tabular foundation model wrappers (TabPFN, RealMLP, TabM, TabDPT)."""

import numpy as np
import pytest


def _clf(n=60, f=20, seed=42):
    rng = np.random.RandomState(seed)
    return rng.randn(n, f).astype(np.float32), rng.choice(["A", "B", "C"], size=n)


def _bin(n=60, f=20, seed=42):
    rng = np.random.RandomState(seed)
    return rng.randn(n, f).astype(np.float32), rng.choice(["X", "Y"], size=n)


def _reg(n=60, f=20, seed=42):
    rng = np.random.RandomState(seed)
    return rng.randn(n, f).astype(np.float32), rng.randn(n).astype(np.float32)


# ---------------------------------------------------------------------------
# TabPFNModel
# ---------------------------------------------------------------------------

class TestTabPFNModel:
    @pytest.fixture(autouse=True)
    def _skip(self):
        pytest.importorskip("tabpfn")

    def test_fit_predict_multiclass(self):
        from raman_bench.models.custom.tabular_foundation import TabPFNModel

        X, y = _clf()
        m = TabPFNModel().fit(X, y)
        preds = m.predict(X)
        assert len(preds) == len(X)
        assert set(preds).issubset({"A", "B", "C"})

    def test_fit_predict_binary(self):
        from raman_bench.models.custom.tabular_foundation import TabPFNModel

        X, y = _bin()
        m = TabPFNModel().fit(X, y)
        assert set(m.predict(X)).issubset({"X", "Y"})

    def test_fit_predict_regression(self):
        from raman_bench.models.custom.tabular_foundation import TabPFNModel

        X, y = _reg()
        preds = TabPFNModel().fit(X, y).predict(X)
        assert len(preds) == len(X)
        assert np.issubdtype(preds.dtype, np.floating)

    def test_predict_proba_multiclass(self):
        from raman_bench.models.custom.tabular_foundation import TabPFNModel

        X, y = _clf()
        proba = TabPFNModel().fit(X, y).predict_proba(X)
        assert proba.shape == (len(X), 3)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_predict_proba_binary(self):
        from raman_bench.models.custom.tabular_foundation import TabPFNModel

        X, y = _bin()
        proba = TabPFNModel().fit(X, y).predict_proba(X)
        assert proba.ndim == 1
        assert np.all((proba >= 0) & (proba <= 1))

    def test_problem_type_regression(self):
        from raman_bench.models.custom.tabular_foundation import TabPFNModel

        X, y = _reg()
        m = TabPFNModel().fit(X, y)
        assert m.problem_type_ == "regression"

    def test_predict_proba_raises_for_regression(self):
        from raman_bench.models.custom.tabular_foundation import TabPFNModel

        X, y = _reg()
        m = TabPFNModel().fit(X, y)
        with pytest.raises(ValueError, match="predict_proba"):
            m.predict_proba(X)


# ---------------------------------------------------------------------------
# RealMLPModel
# ---------------------------------------------------------------------------

class TestRealMLPModel:
    @pytest.fixture(autouse=True)
    def _skip(self):
        pytest.importorskip("pytabkit")

    def test_fit_predict_multiclass(self):
        from raman_bench.models.custom.tabular_foundation import RealMLPModel

        X, y = _clf()
        m = RealMLPModel().fit(X, y)
        preds = m.predict(X)
        assert len(preds) == len(X)
        assert set(preds).issubset({"A", "B", "C"})

    def test_fit_predict_regression(self):
        from raman_bench.models.custom.tabular_foundation import RealMLPModel

        X, y = _reg()
        preds = RealMLPModel().fit(X, y).predict(X)
        assert len(preds) == len(X)
        assert np.issubdtype(preds.dtype, np.floating)

    def test_predict_proba_binary(self):
        from raman_bench.models.custom.tabular_foundation import RealMLPModel

        X, y = _bin()
        proba = RealMLPModel().fit(X, y).predict_proba(X)
        assert proba.ndim == 1
        assert np.all((proba >= 0) & (proba <= 1))


# ---------------------------------------------------------------------------
# TabMModel
# ---------------------------------------------------------------------------

class TestTabMModel:
    @pytest.fixture(autouse=True)
    def _skip(self):
        pytest.importorskip("pytabkit")

    def test_fit_predict_multiclass(self):
        from raman_bench.models.custom.tabular_foundation import TabMModel

        X, y = _clf()
        m = TabMModel().fit(X, y)
        preds = m.predict(X)
        assert len(preds) == len(X)
        assert set(preds).issubset({"A", "B", "C"})

    def test_fit_predict_regression(self):
        from raman_bench.models.custom.tabular_foundation import TabMModel

        X, y = _reg()
        preds = TabMModel().fit(X, y).predict(X)
        assert len(preds) == len(X)
        assert np.issubdtype(preds.dtype, np.floating)

    def test_predict_proba_binary(self):
        from raman_bench.models.custom.tabular_foundation import TabMModel

        X, y = _bin()
        proba = TabMModel().fit(X, y).predict_proba(X)
        assert proba.ndim == 1
        assert np.all((proba >= 0) & (proba <= 1))


# ---------------------------------------------------------------------------
# TabDPTModel
# ---------------------------------------------------------------------------

class TestTabDPTModel:
    @pytest.fixture(autouse=True)
    def _skip(self):
        pytest.importorskip("tabdpt")

    def test_fit_predict_multiclass(self):
        from raman_bench.models.custom.tabular_foundation import TabDPTModel

        X, y = _clf()
        m = TabDPTModel().fit(X, y)
        preds = m.predict(X)
        assert len(preds) == len(X)
        assert set(preds).issubset({"A", "B", "C"})

    def test_fit_predict_regression(self):
        from raman_bench.models.custom.tabular_foundation import TabDPTModel

        X, y = _reg()
        preds = TabDPTModel().fit(X, y).predict(X)
        assert len(preds) == len(X)
        assert np.issubdtype(preds.dtype, np.floating)

    def test_predict_proba_binary(self):
        from raman_bench.models.custom.tabular_foundation import TabDPTModel

        X, y = _bin()
        proba = TabDPTModel().fit(X, y).predict_proba(X)
        assert proba.ndim == 1
        assert np.all((proba >= 0) & (proba <= 1))
