"""Exercise ``Prep_RAMANPFN`` through the real AutoGluon ``AbstractModel``
``fit()``/``predict()``/``predict_proba()`` path (same style as
``tests/test_preprocessing_mixin_real_fit.py``), for both a classification
and a regression target.

Note: the regression case depends on TabPFN's *regressor* backbone weights,
which in this sandbox are blocked by a one-time interactive license
acceptance (``tabpfn.errors.TabPFNLicenseError`` -- see the pre-existing,
unrelated ``TestTabPFNModel`` failures in
``tests/models/test_tabular_foundation.py``). That failure is expected here
too and is not specific to this model's implementation; it reflects the same
sandbox limitation.
"""

import numpy as np
import pandas as pd
import pytest

from raman_bench.preprocessing.wrapped_models import PREPROCESSED_MODELS

N_SAMPLES = 40
N_FEATURES = 120

# GCU/LVSE views this small so TabPFN forward passes stay fast in CI.
_SMALL_PREP = {
    "hyperparameters": {
        "rho": 4,
        "n_regions": 4,
        "k_per_region": 2,
        "n_estimators": 2,
    }
}


def _make_data(n_samples=N_SAMPLES, n_features=N_FEATURES, n_classes=3, seed=0):
    rng = np.random.RandomState(seed)
    X = pd.DataFrame(rng.rand(n_samples, n_features))
    y_class = pd.Series(rng.randint(0, n_classes, n_samples))
    y_reg = pd.Series(rng.randn(n_samples))
    return X, y_class, y_reg


def test_ramanpfn_real_autogluon_fit_classification():
    cls = PREPROCESSED_MODELS["RAMANPFN"]
    X, y, _ = _make_data()

    model = cls(hyperparameters=dict(_SMALL_PREP["hyperparameters"]))
    model.fit(X=X, y=y)

    preds = model.predict(X.iloc[:10])
    assert len(preds) == 10
    assert not pd.isna(preds).any()

    proba = model.predict_proba(X.iloc[:10])
    proba = np.asarray(proba)
    assert not np.isnan(proba).any()
    assert len(set(np.asarray(preds))) > 1 or proba.shape[-1] == 1


def test_ramanpfn_real_autogluon_fit_regression():
    cls = PREPROCESSED_MODELS["RAMANPFN"]
    X, _, y = _make_data()

    model = cls(problem_type="regression", hyperparameters=dict(_SMALL_PREP["hyperparameters"]))
    try:
        model.fit(X=X, y=y)
    except Exception as exc:  # noqa: BLE001 - see module docstring
        if "TabPFNLicenseError" in type(exc).__name__ or "license" in str(exc).lower():
            pytest.skip(f"TabPFN regressor weights blocked by license acceptance: {exc}")
        raise

    preds = model.predict(X.iloc[:10])
    preds = np.asarray(preds)
    assert len(preds) == 10
    assert not np.isnan(preds).any()
