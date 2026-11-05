"""raman_bench.preprocessing.parallel must give the serial results (2026-11-05)."""

import numpy as np
import pytest
from sklearn.preprocessing import PowerTransformer

from raman_bench.preprocessing import parallel
from raman_bench.preprocessing import spectrum_cache as sc
from test_spectrum_cache import ALL, STEPS, _model, _spectra


@pytest.fixture(scope="module")
def pool():
    parallel.configure(3)
    parallel.patch_power_transformer()
    yield
    parallel.configure(1)


@pytest.mark.parametrize("name", [*STEPS, "all"])
def test_rows_in_pool_equal_serial(name, pool):
    params = ALL if name == "all" else STEPS[name]
    X = _spectra(n=31)
    m = _model()
    parallel.configure(1)
    ref = m._spectrumwise(X, params, "Fit")
    parallel.configure(3)
    got = m._spectrumwise(X, params, "Fit")
    assert got.shape == ref.shape and got.dtype == ref.dtype
    if name in ("sg", "all"):
        # savgol_filter(axis=1) differs in the last bits with the rows that share a call
        np.testing.assert_allclose(got, ref, rtol=0, atol=1e-12 * np.abs(ref).max())
    else:
        assert np.array_equal(got, ref)


def _yj_data(dtype):
    rng = np.random.default_rng(0)
    X = np.exp(rng.normal(0, 1, (30, 1500))).astype(dtype) * 100
    X[:, 7] = 3.0  # constant column: lambda 1, skipped by the optimiser
    X[:, 900] = 0.0
    return X


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_yeo_johnson_pool_equals_serial(dtype, pool):
    X = _yj_data(dtype)
    parallel.configure(1)
    ref = PowerTransformer(method="yeo-johnson").fit(X)
    parallel.configure(3)
    got = PowerTransformer(method="yeo-johnson").fit(X)
    assert got.lambdas_.dtype == ref.lambdas_.dtype
    assert np.array_equal(got.lambdas_, ref.lambdas_)
    assert got.lambdas_[7] == 1.0 and got.lambdas_[900] == 1.0
    assert np.array_equal(got.transform(X), ref.transform(X))
    parallel.configure(1)
    Xt_ref = PowerTransformer(method="yeo-johnson").fit_transform(X)
    parallel.configure(3)
    assert np.array_equal(PowerTransformer(method="yeo-johnson").fit_transform(X), Xt_ref)
    assert "_yeo_johnson_optimize" not in vars(got)


def test_cache_store_false_serves_but_does_not_store(monkeypatch):
    monkeypatch.setenv(sc.ENV_MODE, "memory")
    monkeypatch.setattr(sc, "_CACHE", None)
    cache = sc.get_cache()
    X = _spectra(n=12)
    m = _model()
    params = STEPS["airpls"]
    ref = m._apply_spectrumwise_steps(X, params, "Fit")
    key = sc.step_key(params, m._wavenumbers)
    cache.apply(X[:4], key, lambda Z: m._apply_spectrumwise_steps(Z, params, "Fit"))
    got = cache.apply(X, key, lambda Z: m._apply_spectrumwise_steps(Z, params, "Fit"), store=False)
    assert np.array_equal(got, ref)
    assert len(cache.tables()[key]) == 4 and cache.hits == 4 and cache.misses == 12


def test_augmented_fit_rows_are_not_stored(monkeypatch):
    monkeypatch.setenv(sc.ENV_MODE, "memory")
    monkeypatch.setattr(sc, "_CACHE", None)
    m = _model()
    params = STEPS["airpls"]
    m._prep_fitting_augmented = True
    m._spectrumwise(_spectra(n=6), params, "Transform")
    assert len(sc.get_cache().tables()[sc.step_key(params, m._wavenumbers)]) == 0
    m._prep_fitting_augmented = False
    m._spectrumwise(_spectra(n=6), params, "Transform")
    assert len(sc.get_cache().tables()[sc.step_key(params, m._wavenumbers)]) == 6
    monkeypatch.setattr(sc, "_CACHE", None)


def test_tabicl_predictions_unchanged_by_parallel_yeo_johnson(pool):
    """End to end: TabICL (default norm_methods include 'power') predicts the same with the pool."""
    tabicl = pytest.importorskip("tabicl")
    import torch

    rng = np.random.default_rng(0)
    X = (rng.random((30, 800)) * 1000).astype(np.float32)
    y = rng.random(30)
    Xt = (rng.random((10, 800)) * 1000).astype(np.float32)

    def run():
        torch.manual_seed(0)
        m = tabicl.TabICLRegressor(device="cpu", n_estimators=2, random_state=0)
        return m.fit(X, y).predict(Xt)

    parallel.configure(1)
    ref = run()
    parallel.configure(3)
    assert np.array_equal(run(), ref)
