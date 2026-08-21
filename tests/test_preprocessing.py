"""Tests for Raman preprocessing functions."""

import numpy as np
import pytest

from raman_bench.preprocessing import (
    baseline_correction_airpls,
    baseline_correction_arpls,
    baseline_correction_asls,
    cosmic_ray_removal,
    crop_spectra,
    denoise_savgol,
    emsc_fit,
    emsc_transform,
    multiplicative_scatter_correction_fit,
    multiplicative_scatter_correction_transform,
    rubberband_correction,
    savgol_derivative,
    snv,
    vector_normalize,
    wavelet_denoise,
)


@pytest.fixture
def spectra():
    rng = np.random.default_rng(0)
    return rng.standard_normal((10, 100)).astype(np.float64) + 5.0


@pytest.fixture
def drifted_spectra():
    """Synthetic spectra with a peak on top of a smooth, strictly-positive
    convex baseline drift, plus a little noise."""
    rng = np.random.default_rng(0)
    n_samples, n_features = 10, 200
    x = np.linspace(0, 1, n_features)
    peak = 10.0 * np.exp(-((x - 0.5) ** 2) / (2 * 0.02**2))
    baseline = 5.0 + 8.0 * (x - 0.5) ** 2  # convex drift, always positive
    noise = rng.standard_normal((n_samples, n_features)) * 0.05
    return baseline[None, :] + peak[None, :] + noise


def test_snv_zero_mean(spectra):
    out = snv(spectra)
    means = out.mean(axis=1)
    np.testing.assert_allclose(means, 0.0, atol=1e-10)


def test_snv_unit_std(spectra):
    out = snv(spectra)
    stds = out.std(axis=1)
    np.testing.assert_allclose(stds, 1.0, atol=1e-10)


def test_savgol_shape(spectra):
    out = denoise_savgol(spectra)
    assert out.shape == spectra.shape


def test_baseline_shape(spectra):
    out = baseline_correction_asls(spectra)
    assert out.shape == spectra.shape


def test_msc_shape(spectra):
    ref = multiplicative_scatter_correction_fit(spectra)
    out = multiplicative_scatter_correction_transform(spectra, ref)
    assert out.shape == spectra.shape


def test_crr_shape(spectra):
    out = cosmic_ray_removal(spectra)
    assert out.shape == spectra.shape


def test_crr_no_change_clean(spectra):
    """Clean spectra (no spikes) should pass through unchanged."""
    # Use a very high threshold to ensure nothing gets flagged
    out = cosmic_ray_removal(spectra, threshold=1000)
    np.testing.assert_allclose(out, spectra)


# ---------------------------------------------------------------------------
# airPLS
# ---------------------------------------------------------------------------


def test_airpls_shape(spectra):
    out = baseline_correction_airpls(spectra)
    assert out.shape == spectra.shape


def test_airpls_no_nan(spectra):
    out = baseline_correction_airpls(spectra)
    assert not np.isnan(out).any()


def test_airpls_reduces_baseline_drift(drifted_spectra):
    out = baseline_correction_airpls(drifted_spectra, lam=1e5, max_iter=15)
    # The baseline "shoulders" (far from the peak, dominated by baseline
    # drift rather than signal) should be pulled toward 0.
    assert np.abs(out[:, :20]).mean() < np.abs(drifted_spectra[:, :20]).mean()


# ---------------------------------------------------------------------------
# arPLS
# ---------------------------------------------------------------------------


def test_arpls_shape(spectra):
    out = baseline_correction_arpls(spectra)
    assert out.shape == spectra.shape


def test_arpls_no_nan(spectra):
    out = baseline_correction_arpls(spectra)
    assert not np.isnan(out).any()


def test_arpls_reduces_baseline_drift(drifted_spectra):
    out = baseline_correction_arpls(drifted_spectra, lam=1e5, max_iter=15)
    assert np.abs(out[:, :20]).mean() < np.abs(drifted_spectra[:, :20]).mean()


# ---------------------------------------------------------------------------
# Rubberband (convex hull)
# ---------------------------------------------------------------------------


def test_rubberband_shape(spectra):
    out = rubberband_correction(spectra)
    assert out.shape == spectra.shape


def test_rubberband_no_nan(spectra):
    out = rubberband_correction(spectra)
    assert not np.isnan(out).any()


def test_rubberband_reduces_baseline_drift(drifted_spectra):
    out = rubberband_correction(drifted_spectra)
    assert np.abs(out[:, :20]).mean() < np.abs(drifted_spectra[:, :20]).mean()


def test_rubberband_nonnegative_residual():
    """The rubberband baseline never lies above the spectrum, so the
    corrected spectrum should be (numerically) non-negative everywhere."""
    x = np.linspace(0, 1, 50)
    y = 5.0 + 8.0 * (x - 0.5) ** 2
    out = rubberband_correction(y[None, :])
    assert (out >= -1e-8).all()


# ---------------------------------------------------------------------------
# EMSC (compare qualitatively against MSC)
# ---------------------------------------------------------------------------


def test_emsc_shape(spectra):
    ref = emsc_fit(spectra)
    out = emsc_transform(spectra, ref, poly_order=4)
    assert out.shape == spectra.shape


def test_emsc_no_nan(spectra):
    ref = emsc_fit(spectra)
    out = emsc_transform(spectra, ref, poly_order=4)
    assert not np.isnan(out).any()


def test_emsc_reference_shape(spectra):
    ref = emsc_fit(spectra)
    assert ref.shape == (spectra.shape[1],)


def test_emsc_reduces_scatter_variance_like_msc(spectra):
    """EMSC (poly_order=2, the simplest case) should reduce inter-spectrum
    variance at least as effectively as plain MSC, since MSC is the
    poly_order=0 special case of EMSC's additive term."""
    # Introduce multiplicative + additive scatter effects
    rng = np.random.default_rng(1)
    scale = rng.uniform(0.5, 1.5, size=(spectra.shape[0], 1))
    offset = rng.uniform(-1.0, 1.0, size=(spectra.shape[0], 1))
    scattered = spectra * scale + offset

    msc_ref = multiplicative_scatter_correction_fit(scattered)
    msc_out = multiplicative_scatter_correction_transform(scattered, msc_ref)

    emsc_ref = emsc_fit(scattered)
    emsc_out = emsc_transform(scattered, emsc_ref, poly_order=2)

    raw_var = scattered.var(axis=0).mean()
    msc_var = msc_out.var(axis=0).mean()
    emsc_var = emsc_out.var(axis=0).mean()

    assert msc_var < raw_var
    assert emsc_var < raw_var


# ---------------------------------------------------------------------------
# Savitzky-Golay derivative
# ---------------------------------------------------------------------------


def test_savgol_derivative_shape(spectra):
    out = savgol_derivative(spectra, deriv=1)
    assert out.shape == spectra.shape


def test_savgol_derivative_no_nan(spectra):
    out = savgol_derivative(spectra, deriv=1)
    assert not np.isnan(out).any()


def test_savgol_derivative_removes_constant_offset():
    """A first derivative of a spectrum with a constant additive offset
    should be identical (up to smoothing) to the derivative without it —
    i.e. the offset itself is suppressed."""
    x = np.linspace(0, 1, 200)
    y = np.sin(2 * np.pi * 3 * x)
    y_offset = y + 100.0

    d1 = savgol_derivative(y[None, :], window_length=11, polyorder=2, deriv=1)
    d2 = savgol_derivative(y_offset[None, :], window_length=11, polyorder=2, deriv=1)
    np.testing.assert_allclose(d1, d2, atol=1e-8)


def test_savgol_derivative_second_order(spectra):
    out = savgol_derivative(spectra, window_length=15, polyorder=3, deriv=2)
    assert out.shape == spectra.shape
    assert not np.isnan(out).any()


# ---------------------------------------------------------------------------
# Wavelet denoising
# ---------------------------------------------------------------------------


def test_wavelet_denoise_shape(spectra):
    pytest.importorskip("pywt")
    out = wavelet_denoise(spectra)
    assert out.shape == spectra.shape


def test_wavelet_denoise_no_nan(spectra):
    pytest.importorskip("pywt")
    out = wavelet_denoise(spectra)
    assert not np.isnan(out).any()


def test_wavelet_denoise_reduces_noise():
    pytest.importorskip("pywt")
    rng = np.random.default_rng(0)
    x = np.linspace(0, 1, 256)
    clean = np.sin(2 * np.pi * 3 * x)
    noisy = clean + rng.normal(0, 0.3, size=(5, 256))

    out = wavelet_denoise(noisy, wavelet="sym7", level=4)

    err_noisy = np.mean((noisy - clean[None, :]) ** 2)
    err_denoised = np.mean((out - clean[None, :]) ** 2)
    assert err_denoised < err_noisy


# ---------------------------------------------------------------------------
# Crop (fingerprint-region, fractional-index proxy)
# ---------------------------------------------------------------------------


def test_crop_fewer_features(spectra):
    out = crop_spectra(spectra, start_frac=0.15, end_frac=0.75)
    assert out.shape[1] < spectra.shape[1]
    assert out.shape[0] == spectra.shape[0]


def test_crop_indices_within_bounds():
    n_features = 100
    x = np.arange(n_features, dtype=np.float64)[None, :]
    out = crop_spectra(x, start_frac=0.15, end_frac=0.75)
    assert out.min() >= 0
    assert out.max() < n_features
    # Values are a contiguous sub-range matching the fractional bounds.
    expected_start = int(round(0.15 * n_features))
    expected_end = int(round(0.75 * n_features))
    np.testing.assert_array_equal(out[0], x[0, expected_start:expected_end])


def test_crop_full_range_default_no_op_bounds():
    out = crop_spectra(np.ones((3, 20)), start_frac=0.0, end_frac=1.0)
    assert out.shape == (3, 20)


def test_crop_clips_degenerate_fractions():
    """start_frac >= end_frac must still return at least one column, not crash."""
    out = crop_spectra(np.ones((2, 10)), start_frac=0.9, end_frac=0.1)
    assert out.shape[0] == 2
    assert out.shape[1] >= 1


# ---------------------------------------------------------------------------
# Vector normalization (L2)
# ---------------------------------------------------------------------------


def test_vecnorm_unit_norm(spectra):
    out = vector_normalize(spectra)
    norms = np.linalg.norm(out, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-10)


def test_vecnorm_shape(spectra):
    out = vector_normalize(spectra)
    assert out.shape == spectra.shape


def test_vecnorm_zero_row_no_nan():
    """An all-zero row must not produce NaN (0/0)."""
    X = np.zeros((2, 10))
    X[1] = 1.0
    out = vector_normalize(X)
    assert not np.isnan(out).any()
    np.testing.assert_allclose(out[0], 0.0)
