"""Regression tests for source-defined MSC fitting regions."""

import numpy as np
import pytest

pytest.importorskip("autogluon")

from raman_bench.preprocessing import mixin  # noqa: E402


def test_msc_region_params_are_used_consistently_for_fit_and_transform(monkeypatch):
    calls = []

    def fake_fit(X):
        return X.mean(axis=0)

    def fake_transform(X, reference, start=0.0, end=1.0):
        calls.append((start, end, X.shape))
        return X.copy()

    monkeypatch.setattr(mixin, "multiplicative_scatter_correction_fit", fake_fit)
    monkeypatch.setattr(mixin, "multiplicative_scatter_correction_transform", fake_transform)

    class Dummy(mixin.RamanPreprocessingMixin):
        def _get_model_params(self):
            return {
                "prep_msc_enabled": True,
                "prep_msc_start_frac": 0.25,
                "prep_msc_end_frac": 0.5,
            }

    dummy = Dummy.__new__(Dummy)
    dummy._preprocess_fit(np.arange(24, dtype=float).reshape(4, 6))
    dummy._preprocess_transform(np.arange(12, dtype=float).reshape(2, 6))

    assert calls == [(0.25, 0.5, (4, 6)), (0.25, 0.5, (2, 6))]


def test_msc_region_defaults_preserve_full_spectrum_behavior():
    defaults = mixin._PREP_STEP_DEFINITIONS["msc"]["defaults"]
    assert defaults["prep_msc_start_frac"] == 0.0
    assert defaults["prep_msc_end_frac"] == 1.0
