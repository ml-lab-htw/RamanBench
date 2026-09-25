"""Tests for physical-axis (true cm^-1) cropping wired through the mixin.

Complements the direct-function tests for ``crop_spectra_physical`` in
``test_preprocessing.py`` (full coverage / partial overlap / no overlap /
exact boundary — the pure-function contract) with:

1. Unit-level tests of ``_preprocess_fit``/``_preprocess_transform``'s
   ``prep_crop_physical_enabled`` wiring, using the same lightweight
   ``Dummy(mixin.RamanPreprocessingMixin)`` pattern as
   ``test_preprocessing_msc_region.py``.
2. An integration-level test confirming the true wavenumber axis actually
   survives from a DataFrame with cm^-1 column headers, through the real
   ``AbstractModel.fit()`` -> ``RamanPreprocessingMixin._fit()`` ->
   ``self._wavenumbers`` capture -> ``_preprocess`` path, and crops
   correctly — matching the real AutoGluon-fit pattern used in
   ``test_preprocessing_mixin_real_fit.py``.
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("autogluon")

from raman_bench.preprocessing import mixin  # noqa: E402
from raman_bench.preprocessing.wrapped_models import PREPROCESSED_MODELS  # noqa: E402


# ---------------------------------------------------------------------------
# Step definition sanity
# ---------------------------------------------------------------------------


def test_crop_physical_step_definition_defaults():
    defaults = mixin._PREP_STEP_DEFINITIONS["crop_physical"]["defaults"]
    assert defaults["prep_crop_physical_enabled"] is False
    assert defaults["prep_crop_physical_start_cm"] < defaults["prep_crop_physical_end_cm"]


def test_crop_physical_enabled_param_registered():
    assert mixin.STEP_ENABLED_PARAMS["crop_physical"] == "prep_crop_physical_enabled"
    assert "prep_crop_physical_enabled" in mixin._TRANSFORM_ENABLED_PARAMS


def test_crop_physical_does_not_touch_fractional_crop_defaults():
    """Regression guard: adding crop_physical must not alter "crop"'s own defaults."""
    crop_defaults = mixin._PREP_STEP_DEFINITIONS["crop"]["defaults"]
    assert crop_defaults == {
        "prep_crop_enabled": False,
        "prep_crop_start_frac": 0.15,
        "prep_crop_end_frac": 0.75,
    }


# ---------------------------------------------------------------------------
# _preprocess_fit / _preprocess_transform wiring (Dummy pattern, mirrors
# test_preprocessing_msc_region.py)
# ---------------------------------------------------------------------------


class _Dummy(mixin.RamanPreprocessingMixin):
    def __init__(self, params):
        self._params = params

    def _get_model_params(self):
        return self._params


def test_crop_physical_missing_wavenumbers_raises_in_fit():
    dummy = _Dummy(
        {
            "prep_crop_physical_enabled": True,
            "prep_crop_physical_start_cm": 400.0,
            "prep_crop_physical_end_cm": 1800.0,
        }
    )
    # self._wavenumbers deliberately never set.
    with pytest.raises(ValueError, match="self._wavenumbers is unset"):
        dummy._preprocess_fit(np.ones((3, 10)))


def test_crop_physical_missing_wavenumbers_raises_in_transform():
    dummy = _Dummy(
        {
            "prep_crop_physical_enabled": True,
            "prep_crop_physical_start_cm": 400.0,
            "prep_crop_physical_end_cm": 1800.0,
        }
    )
    with pytest.raises(ValueError, match="self._wavenumbers is unset"):
        dummy._preprocess_transform(np.ones((3, 10)))


def test_crop_physical_fit_then_transform_crop_consistently():
    wavenumbers = np.linspace(200.0, 2200.0, 41)  # 50 cm^-1 spacing
    dummy = _Dummy(
        {
            "prep_crop_physical_enabled": True,
            "prep_crop_physical_start_cm": 400.0,
            "prep_crop_physical_end_cm": 1800.0,
        }
    )
    dummy._wavenumbers = wavenumbers

    X_train = np.tile(wavenumbers, (5, 1))
    X_test = np.tile(wavenumbers, (2, 1))

    out_train = dummy._preprocess_fit(X_train)
    out_test = dummy._preprocess_transform(X_test)

    expected_kept = wavenumbers[(wavenumbers >= 400.0) & (wavenumbers <= 1800.0)]
    assert out_train.shape == (5, len(expected_kept))
    assert out_test.shape == (2, len(expected_kept))
    np.testing.assert_array_equal(out_train[0], expected_kept)
    np.testing.assert_array_equal(out_test[0], expected_kept)


def test_crop_physical_runs_before_fractional_crop_when_both_enabled(caplog):
    """crop_physical must see the *original* full-width axis, so it has to run
    first; the fractional crop then narrows whatever crop_physical produced.
    """
    wavenumbers = np.linspace(0.0, 2000.0, 21)  # 100 cm^-1 spacing, 21 columns
    dummy = _Dummy(
        {
            "prep_crop_physical_enabled": True,
            "prep_crop_physical_start_cm": 500.0,
            "prep_crop_physical_end_cm": 1500.0,
            "prep_crop_enabled": True,
            "prep_crop_start_frac": 0.0,
            "prep_crop_end_frac": 0.5,
        }
    )
    dummy._wavenumbers = wavenumbers
    X = np.tile(wavenumbers, (3, 1))

    with caplog.at_level("WARNING"):
        out = dummy._preprocess_fit(X)

    # crop_physical keeps wavenumbers in [500, 1500] -> 11 columns (500..1500
    # step 100), then fractional crop takes the first half of *that* (11
    # columns -> round(0*11)=0 to round(0.5*11)=6 -> 6 columns).
    physical_kept = wavenumbers[(wavenumbers >= 500.0) & (wavenumbers <= 1500.0)]
    assert len(physical_kept) == 11
    assert out.shape == (3, 6)
    assert any("unusual" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# Integration: true wavenumber axis survives DataFrame -> _fit -> _preprocess
# ---------------------------------------------------------------------------


def _make_wavenumber_data(n_samples=40, n_classes=3, seed=0):
    rng = np.random.RandomState(seed)
    wavenumbers = np.linspace(200.0, 2200.0, 101)  # 20 cm^-1 spacing
    X = pd.DataFrame(rng.rand(n_samples, len(wavenumbers)), columns=wavenumbers)
    y = pd.Series(rng.randint(0, n_classes, n_samples))
    return X, y, wavenumbers


@pytest.mark.parametrize("model_key", ["KNN", "PLS"])
def test_crop_physical_survives_real_autogluon_fit_and_crops_correctly(model_key):
    cls = PREPROCESSED_MODELS[model_key]
    X, y, wavenumbers = _make_wavenumber_data()

    hyperparameters = {
        "prep_crop_physical_enabled": True,
        "prep_crop_physical_start_cm": 500.0,
        "prep_crop_physical_end_cm": 1500.0,
    }
    model = cls(hyperparameters=hyperparameters)
    model.fit(X=X, y=y)

    expected_kept = int(((wavenumbers >= 500.0) & (wavenumbers <= 1500.0)).sum())
    assert getattr(model, "_wavenumbers", None) is not None
    np.testing.assert_array_equal(model._wavenumbers, wavenumbers)
    assert model._prep_n_features_out == expected_kept

    preds = model.predict(X.iloc[:10])
    assert len(preds) == 10
    assert not pd.isna(preds).any()
