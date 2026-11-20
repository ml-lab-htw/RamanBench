"""Average pooling and decimation along the spectral axis (steps ``avgpool`` / ``decimate``).

Function contract (values, widths, truncation), the mixin wiring (fit and transform agree, the
two steps exclude each other, both run after SNV), the opt-in-only config behaviour (the
``preprocessing: true`` shorthand must not switch them on), and a real AutoGluon fit for a
non-HPO model, whose ``_fit`` gate clears every step the user did not set.
"""

import numpy as np
import pandas as pd
import pytest

from raman_bench.preprocessing.raman_preprocessing import avg_pool_spectra, decimate_spectra, snv


def test_avg_pool_means_each_window_and_drops_the_tail():
    X = np.arange(12, dtype=float).reshape(1, 12)  # 0..11, two full windows of 5, tail 10, 11
    out = avg_pool_spectra(X, size=5)
    np.testing.assert_allclose(out, [[2.0, 7.0]])


def test_decimate_keeps_every_kth_point_of_the_same_range():
    X = np.arange(12, dtype=float).reshape(1, 12)
    np.testing.assert_array_equal(decimate_spectra(X, step=5), [[0.0, 5.0]])
    assert decimate_spectra(X, 5).shape == avg_pool_spectra(X, 5).shape


@pytest.mark.parametrize("fn", [avg_pool_spectra, decimate_spectra])
def test_too_few_features_raise(fn):
    with pytest.raises(ValueError):
        fn(np.ones((2, 4)), 5)


pytest.importorskip("autogluon")

from raman_bench.config import _normalize_preprocessing_config  # noqa: E402
from raman_bench.preprocessing import mixin  # noqa: E402
from raman_bench.preprocessing.wrapped_models import PREPROCESSED_MODELS  # noqa: E402


class _Dummy(mixin.RamanPreprocessingMixin):
    def __init__(self, params):
        self._params = params

    def _get_model_params(self):
        return self._params


def test_steps_registered_as_transform_steps():
    assert mixin.STEP_ENABLED_PARAMS["avgpool"] == "prep_avgpool_enabled"
    assert mixin.STEP_ENABLED_PARAMS["decimate"] == "prep_decimate_enabled"
    assert {"prep_avgpool_enabled", "prep_decimate_enabled"} <= set(mixin._TRANSFORM_ENABLED_PARAMS)


@pytest.mark.parametrize("key, fn", [("avgpool", avg_pool_spectra), ("decimate", decimate_spectra)])
def test_fit_and_transform_agree_and_run_after_snv(key, fn):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(6, 103)) + np.linspace(0, 5, 103)
    dummy = _Dummy({"prep_snv_enabled": True, f"prep_{key}_enabled": True})
    fitted = dummy._preprocess_fit(X.copy())
    transformed = dummy._preprocess_transform(X.copy())
    np.testing.assert_allclose(fitted, transformed)
    np.testing.assert_allclose(fitted, fn(snv(X), 5))
    assert fitted.shape == (6, 20)


def test_both_steps_together_raise():
    dummy = _Dummy({"prep_avgpool_enabled": True, "prep_decimate_enabled": True})
    with pytest.raises(ValueError, match="mutually exclusive"):
        dummy._preprocess_fit(np.ones((2, 20)))


def test_enable_everything_shorthand_leaves_pooling_off():
    on = _normalize_preprocessing_config({"preprocessing": True})["preprocessing_config"]
    assert not on.get("avgpool") and not on.get("decimate")
    explicit = _normalize_preprocessing_config({"preprocessing": {"avgpool": True}})["preprocessing_config"]
    assert explicit["avgpool"] is True


@pytest.mark.parametrize("model_key", ["RF", "PLS"])
def test_real_fit_pools_the_input(model_key):
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.rand(40, 101), columns=np.linspace(200.0, 2200.0, 101))
    y = pd.Series(rng.rand(40)) if model_key == "PLS" else pd.Series(rng.randint(0, 3, 40))
    model = PREPROCESSED_MODELS[model_key](
        hyperparameters={"prep_snv_enabled": True, "prep_avgpool_enabled": True},
        problem_type="regression" if model_key == "PLS" else "multiclass",
    )
    model.fit(X=X, y=y)
    assert model._prep_n_features_in == 101
    assert model._prep_n_features_out == 20
    preds = model.predict(X.iloc[:10])
    assert len(preds) == 10 and not pd.isna(preds).any()
