"""Tests for GCU/LVSE wiring through RamanPreprocessingMixin._preprocess_fit
and _preprocess_transform (as opposed to the standalone gcu_fit/lvse_fit
functions, covered in test_preprocessing.py).

GCU and LVSE are representation-replacing, terminal steps: their output
width differs from n_features, and they always run last (after every
shape-preserving step), regardless of _PREP_STEP_DEFINITIONS dict order —
see the composition-order design note in mixin.py's _preprocess_fit.
"""

import numpy as np
import pytest

pytest.importorskip("autogluon")

from raman_bench.preprocessing.mixin import RamanPreprocessingMixin  # noqa: E402


class _FakeModel(RamanPreprocessingMixin):
    def __init__(self, params=None):
        self.params = dict(params or {})

    def _get_model_params(self):
        return self.params


def _spectra(n_samples=20, n_features=60, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_samples, n_features)).astype(np.float64) + 5.0


def test_gcu_fit_replaces_representation():
    X = _spectra(n_samples=15, n_features=40)
    model = _FakeModel({"prep_gcu_enabled": True, "prep_gcu_rho": 8})
    out = model._preprocess_fit(X)
    assert out.shape == (15, 8)
    assert hasattr(model, "_gcu_shift")
    assert hasattr(model, "_gcu_nmf")


def test_gcu_transform_reuses_fitted_state():
    X_train = _spectra(n_samples=15, n_features=40, seed=1)
    X_test = _spectra(n_samples=5, n_features=40, seed=2)
    model = _FakeModel({"prep_gcu_enabled": True, "prep_gcu_rho": 8})
    model._preprocess_fit(X_train)
    out = model._preprocess_transform(X_test)
    assert out.shape == (5, 8)


def test_lvse_fit_replaces_representation():
    X = _spectra(n_samples=15, n_features=64)
    model = _FakeModel(
        {"prep_lvse_enabled": True, "prep_lvse_n_regions": 8, "prep_lvse_k_per_region": 4}
    )
    out = model._preprocess_fit(X)
    assert out.shape == (15, 8 * 4)
    assert hasattr(model, "_lvse_components")


def test_lvse_transform_reuses_fitted_state():
    X_train = _spectra(n_samples=15, n_features=64, seed=1)
    X_test = _spectra(n_samples=5, n_features=64, seed=2)
    model = _FakeModel(
        {"prep_lvse_enabled": True, "prep_lvse_n_regions": 8, "prep_lvse_k_per_region": 4}
    )
    model._preprocess_fit(X_train)
    out = model._preprocess_transform(X_test)
    assert out.shape == (5, 8 * 4)


def test_gcu_and_lvse_together_are_concatenated_not_chained():
    """Both enabled runs each as an independent block on the same
    pre-GCU/LVSE input (paper's "separate forward pass" design) and
    concatenates — not LVSE-on-top-of-GCU's rho-wide output."""
    X = _spectra(n_samples=12, n_features=48)
    model = _FakeModel(
        {
            "prep_gcu_enabled": True,
            "prep_gcu_rho": 6,
            "prep_lvse_enabled": True,
            "prep_lvse_n_regions": 4,
            "prep_lvse_k_per_region": 3,
        }
    )
    out = model._preprocess_fit(X)
    assert out.shape == (12, 6 + 4 * 3)


def test_gcu_runs_after_shape_preserving_steps():
    """SNV enabled alongside GCU must run first (SNV output feeds GCU),
    not error, and not leave a stale n_features-wide representation."""
    X = _spectra(n_samples=10, n_features=30)
    model = _FakeModel({"prep_snv_enabled": True, "prep_gcu_enabled": True, "prep_gcu_rho": 5})
    out = model._preprocess_fit(X)
    assert out.shape == (10, 5)


def test_gcu_no_nan_end_to_end():
    X = _spectra(n_samples=10, n_features=30)
    model = _FakeModel({"prep_gcu_enabled": True, "prep_gcu_rho": 5})
    out = model._preprocess_fit(X)
    assert not np.isnan(out).any()


def test_lvse_no_nan_end_to_end():
    X = _spectra(n_samples=10, n_features=32)
    model = _FakeModel(
        {"prep_lvse_enabled": True, "prep_lvse_n_regions": 4, "prep_lvse_k_per_region": 4}
    )
    out = model._preprocess_fit(X)
    assert not np.isnan(out).any()


def test_gcu_disabled_by_default_no_shape_change():
    X = _spectra(n_samples=10, n_features=30)
    model = _FakeModel({})
    out = model._preprocess_fit(X)
    assert out.shape == X.shape
