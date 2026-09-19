"""Tests for TabPFNWideModel (classification-only, wide-feature TabPFN variant)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("tabpfnwide")

from raman_bench.models.custom.tabpfn_wide.model import (  # noqa: E402
    TabPFNWideModel,
    _resolve_device,
)


def _clf(n=30, f=50, seed=42):
    rng = np.random.RandomState(seed)
    return rng.randn(n, f).astype(np.float32), rng.choice(["A", "B", "C"], size=n)


def _bin(n=30, f=50, seed=42):
    rng = np.random.RandomState(seed)
    return rng.randn(n, f).astype(np.float32), rng.choice(["X", "Y"], size=n)


def _reg(n=30, f=50, seed=42):
    rng = np.random.RandomState(seed)
    return rng.randn(n, f).astype(np.float32), rng.randn(n).astype(np.float32)


class TestResolveDevice:
    """_resolve_device is the fix for the CPU-hardcoded device bug (see model.py
    docstring): an explicit device always wins, ``None`` auto-detects CUDA.
    """

    def test_explicit_device_passes_through(self):
        assert _resolve_device("cpu") == "cpu"
        assert _resolve_device("cuda") == "cuda"

    def test_none_auto_detects(self):
        import torch

        expected = "cuda" if torch.cuda.is_available() else "cpu"
        assert _resolve_device(None) == expected


class TestTabPFNWideModel:
    def test_fit_predict_multiclass(self):
        X, y = _clf()
        m = TabPFNWideModel(device="cpu").fit(X, y)
        preds = m.predict(X)
        assert len(preds) == len(X)
        assert set(preds).issubset({"A", "B", "C"})

    def test_fit_predict_binary(self):
        X, y = _bin()
        m = TabPFNWideModel(device="cpu").fit(X, y)
        assert set(m.predict(X)).issubset({"X", "Y"})

    def test_predict_proba_binary(self):
        X, y = _bin()
        proba = TabPFNWideModel(device="cpu").fit(X, y).predict_proba(X)
        assert proba.ndim == 1
        assert np.all((proba >= 0) & (proba <= 1))

    def test_predict_proba_multiclass(self):
        X, y = _clf()
        proba = TabPFNWideModel(device="cpu").fit(X, y).predict_proba(X)
        assert proba.shape == (len(X), 3)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_default_device_is_none_until_fit(self):
        # Constructor no longer hardcodes "cpu" -- see _resolve_device's docstring.
        assert TabPFNWideModel().device is None

    def test_regression_raises(self):
        X, y = _reg()
        with pytest.raises(ValueError, match="does not support regression"):
            TabPFNWideModel(device="cpu").fit(X, y)

    def test_many_class_uses_ecoc_when_narrow(self):
        # >many_class_threshold classes, but well under _ECOC_MAX_FEATURES --
        # should route through ManyClassClassifier (ECOC) rather than raise.
        pytest.importorskip("tabpfn_extensions")
        rng = np.random.RandomState(0)
        X = rng.randn(60, 20).astype(np.float32)
        y = rng.choice([str(i) for i in range(12)], size=60)
        m = TabPFNWideModel(device="cpu", many_class_threshold=10).fit(X, y)
        preds = m.predict(X)
        assert len(preds) == len(X)

    def test_many_class_raises_when_wide(self):
        # >many_class_threshold classes AND over _ECOC_MAX_FEATURES -- ECOC's
        # per-sub-model cost on top of an already-wide fit is the OOM
        # combination this guard exists for (see _ECOC_MAX_FEATURES's
        # docstring), so this must still fail fast rather than attempt ECOC.
        rng = np.random.RandomState(0)
        n_features = TabPFNWideModel._ECOC_MAX_FEATURES + 1
        X = rng.randn(30, n_features).astype(np.float32)
        y = rng.choice([str(i) for i in range(12)], size=30)
        with pytest.raises(ValueError, match="ECOC-safe limit"):
            TabPFNWideModel(device="cpu", many_class_threshold=10).fit(X, y)
