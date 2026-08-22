"""Tests for the preprocessing-ensemble mechanism.

``_preprocess_fit_ensemble``/``_preprocess_transform_ensemble`` run several
independent preprocessing "blocks" (each a preprocessing_config-shaped dict)
through the existing ``_preprocess_fit``/``_preprocess_transform`` pipeline
and concatenate their outputs column-wise. This is orthogonal to
``_PREP_STEP_DEFINITIONS`` — it fans out into several copies of the ordinary
pipeline rather than being one more sequential step — so it is exercised
directly against ``RamanPreprocessingMixin``, independent of any concrete
AutoGluon model class.
"""

import numpy as np

from raman_bench.preprocessing.mixin import RamanPreprocessingMixin


class _FakeModel(RamanPreprocessingMixin):
    """Minimal stand-in exposing just what the mixin's preprocessing methods
    need: a ``params`` dict and ``_get_model_params()``. Bypasses AutoGluon's
    ``AbstractModel`` entirely — the ensemble methods only touch ``self.params``
    and the ``_STATEFUL_PREP_ATTRS`` instance attributes."""

    def __init__(self, params=None):
        self.params = dict(params or {})

    def _get_model_params(self):
        return self.params


def _spectra(n_samples=6, n_features=40, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_samples, n_features)).astype(np.float64) + 5.0


def test_two_blocks_concatenated_width():
    X = _spectra(n_samples=5, n_features=30)
    model = _FakeModel()
    blocks = [{}, {"snv": True}]  # raw + snv
    out = model._preprocess_fit_ensemble(X, blocks)
    assert out.shape == (5, 30 * 2)


def test_three_blocks_concatenated_width():
    X = _spectra(n_samples=4, n_features=20)
    model = _FakeModel()
    blocks = [{"snv": True}, {"denoising": True}, {"airpls": True, "snv": True}]
    out = model._preprocess_fit_ensemble(X, blocks)
    assert out.shape == (4, 20 * 3)


def test_raw_block_matches_identity():
    X = _spectra(n_samples=3, n_features=15)
    model = _FakeModel()
    out = model._preprocess_fit_ensemble(X, [{}])
    np.testing.assert_allclose(out, X)


def test_snv_block_matches_direct_snv_call():
    from raman_bench.preprocessing import snv

    X = _spectra(n_samples=3, n_features=15)
    model = _FakeModel()
    out = model._preprocess_fit_ensemble(X, [{"snv": True}])
    np.testing.assert_allclose(out, snv(X))


def test_transform_reuses_fitted_state_no_refit():
    """MSC's reference spectrum must come from the training call, not be
    recomputed from the (different) transform-time data."""
    X_train = _spectra(n_samples=6, n_features=25, seed=1)
    X_test = _spectra(n_samples=4, n_features=25, seed=2)

    model = _FakeModel()
    train_out = model._preprocess_fit_ensemble(X_train, [{"msc": True}])
    assert train_out.shape == (6, 25)

    # Fitted state was captured, not left on self after fit.
    assert not hasattr(model, "_msc_reference")

    test_out = model._preprocess_transform_ensemble(X_test)
    assert test_out.shape == (4, 25)

    from raman_bench.preprocessing import (
        multiplicative_scatter_correction_fit,
        multiplicative_scatter_correction_transform,
    )

    ref = multiplicative_scatter_correction_fit(X_train)
    expected = multiplicative_scatter_correction_transform(X_test, ref, start=0.0, end=1.0)
    np.testing.assert_allclose(test_out, expected)


def test_blocks_do_not_leak_state_into_each_other():
    """A block that doesn't enable MSC must not pick up another block's MSC
    reference/output."""
    X = _spectra(n_samples=5, n_features=20)
    model = _FakeModel()
    out = model._preprocess_fit_ensemble(X, [{"msc": True}, {}])
    raw_block = out[:, 20:]
    np.testing.assert_allclose(raw_block, X)


def test_ensemble_disabled_falls_back_to_single_recipe():
    """When prep_ensemble_enabled is False (or blocks are empty), normal
    single-recipe _preprocess_fit/_preprocess_transform behaviour is
    unchanged — the ensemble machinery is not invoked at all."""
    from raman_bench.preprocessing import snv

    X = _spectra(n_samples=5, n_features=20)
    model = _FakeModel(params={"prep_snv_enabled": True, "prep_ensemble_enabled": False})
    out = model._preprocess_fit(X)
    np.testing.assert_allclose(out, snv(X))
    assert out.shape == X.shape
