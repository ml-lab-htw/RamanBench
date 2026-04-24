"""Tests for Raman preprocessing functions."""

import numpy as np
import pytest

from raman_bench.preprocessing import (
    baseline_correction_asls,
    cosmic_ray_removal,
    denoise_savgol,
    multiplicative_scatter_correction_fit,
    multiplicative_scatter_correction_transform,
    snv,
)


@pytest.fixture
def spectra():
    rng = np.random.default_rng(0)
    return rng.standard_normal((10, 100)).astype(np.float64) + 5.0


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
