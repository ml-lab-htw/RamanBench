"""The spectrum cache must return exactly what the uncached per-spectrum steps return."""

import numpy as np
import pytest

from raman_bench.preprocessing import spectrum_cache as sc
from raman_bench.preprocessing.mixin import RamanPreprocessingMixin as M

WN = np.linspace(200.0, 2200.0, 600)

STEPS = {
    "crop_physical": {
        "prep_crop_physical_enabled": True,
        "prep_crop_physical_start_cm": 400.0,
        "prep_crop_physical_end_cm": 1800.0,
    },
    "crop": {"prep_crop_enabled": True, "prep_crop_start_frac": 0.15, "prep_crop_end_frac": 0.75},
    "crr": {"prep_crr_enabled": True, "prep_crr_threshold": 6, "prep_crr_kernel_size": 3},
    "sg": {"prep_denoise_enabled": True, "prep_denoise_wl": 9, "prep_denoise_po": 3},
    "wavelet": {"prep_wavelet_enabled": True, "prep_wavelet_name": "sym7", "prep_wavelet_level": 4},
    "asls": {"prep_bl_enabled": True, "prep_bl_lam": 1e5, "prep_bl_p": 0.01},
    "airpls": {
        "prep_airpls_enabled": True,
        "prep_airpls_lam": 1e5,
        "prep_airpls_max_iter": 15,
        "prep_airpls_tol": 1e-3,
    },
    "arpls": {"prep_arpls_enabled": True, "prep_arpls_lam": 1e5, "prep_arpls_max_iter": 15},
    "rubberband": {"prep_rubberband_enabled": True},
}
ALL = {k: v for step in STEPS.values() for k, v in step.items()}


def _spectra(n=40, seed=0):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, WN.size)
    peaks = sum(
        np.exp(-((x - c) ** 2) / 2e-4) * rng.uniform(0.5, 2, (n, 1)) for c in (0.3, 0.5, 0.7)
    )
    X = peaks + rng.uniform(0, 3, (n, 1)) * x**2 + rng.normal(0, 0.02, (n, WN.size))
    X[rng.integers(0, n, 5), rng.integers(0, WN.size, 5)] += 5.0  # cosmic-ray spikes
    return X


def _model():
    m = object.__new__(M)
    m._wavenumbers = WN
    return m


@pytest.fixture
def memory_cache(monkeypatch):
    monkeypatch.setenv(sc.ENV_MODE, "memory")
    monkeypatch.setattr(sc, "_CACHE", None)
    yield sc.get_cache()
    monkeypatch.setattr(sc, "_CACHE", None)


@pytest.mark.parametrize("name", [*STEPS, "all"])
def test_cached_equals_uncached_bitwise(name, memory_cache):
    params = ALL if name == "all" else STEPS[name]
    X = _spectra()
    m = _model()
    ref = m._apply_spectrumwise_steps(X, params, "Fit")
    rng = np.random.default_rng(1)
    subsets = [np.arange(25), rng.permutation(40), np.arange(10, 40), rng.permutation(40)[:7]]
    for idx in subsets:  # partial overlaps: some rows cached, some computed
        got = m._spectrumwise(X[idx], params, "Transform")
        assert got.dtype == ref.dtype and got.shape == ref[idx].shape
        if "prep_denoise_enabled" in params:
            # scipy's savgol_filter(axis=1) is not batch-invariant: the same spectrum smoothed
            # within a different set of rows differs by a few ulp (vectorised convolution), so
            # the uncached pipeline already varies at this level between folds and bag children.
            assert np.allclose(got, ref[idx], rtol=0, atol=1e-12 * np.abs(ref).max()), name
        else:
            assert np.array_equal(got, ref[idx]), name
    assert memory_cache.hits > 0 and memory_cache.misses == 40


def test_cache_off_by_default(monkeypatch):
    monkeypatch.delenv(sc.ENV_MODE, raising=False)
    monkeypatch.setattr(sc, "_CACHE", None)
    assert sc.get_cache() is None
    X = _spectra(5)
    assert np.array_equal(
        _model()._spectrumwise(X, STEPS["sg"], "Fit"),
        _model()._apply_spectrumwise_steps(X, STEPS["sg"], "Fit"),
    )


def test_step_key():
    assert sc.step_key({"prep_snv_enabled": True, "prep_msc_enabled": True}, WN) is None
    k = sc.step_key(STEPS["sg"], WN)
    assert k == sc.step_key(dict(STEPS["sg"]), WN[::-1])  # axis only matters for crop_physical
    assert k != sc.step_key({**STEPS["sg"], "prep_denoise_wl": 11}, WN)
    assert k != sc.step_key({**STEPS["sg"], **STEPS["asls"]}, WN)
    kp = sc.step_key(STEPS["crop_physical"], WN)
    assert kp != sc.step_key(STEPS["crop_physical"], WN + 1.0)


def test_precomputed_directory_is_read(tmp_path, monkeypatch):
    params = {**STEPS["sg"], **STEPS["asls"]}
    X = _spectra(12)
    m = _model()
    key = sc.step_key(params, WN)
    ref = m._apply_spectrumwise_steps(X, params, "Fit")
    sc.save_table(tmp_path, key, dict(zip(sc.row_digests(X), ref)), "ds")
    monkeypatch.setenv(sc.ENV_MODE, str(tmp_path))
    monkeypatch.setattr(sc, "_CACHE", None)

    def fail(Z):
        raise AssertionError("row should have come from the precomputed file")

    cache = sc.get_cache()
    got = cache.apply(X[::-1], key, fail)
    assert np.array_equal(got, ref[::-1]) and cache.misses == 0
    monkeypatch.setattr(sc, "_CACHE", None)
