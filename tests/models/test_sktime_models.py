"""Unit tests for raman_bench.custom_models.sktime_models (RocketModel, ArsenalModel)."""

import numpy as np
import pandas as pd
import pytest

from raman_bench.models.custom.sktime_models import ArsenalModel, RocketModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_clf_data(n_samples=60, n_features=100, n_classes=3, seed=0):
    rng = np.random.RandomState(seed)
    X = pd.DataFrame(rng.randn(n_samples, n_features).astype(np.float32))
    y = pd.Series(rng.choice([f"class{i}" for i in range(n_classes)], size=n_samples))
    return X, y


def _make_binary_data(n_samples=40, n_features=100, seed=1):
    rng = np.random.RandomState(seed)
    X = pd.DataFrame(rng.randn(n_samples, n_features).astype(np.float32))
    y = pd.Series(rng.choice(["healthy", "stroke"], size=n_samples))
    return X, y


def _fit_rocket(problem_type="multiclass"):
    model = RocketModel(problem_type=problem_type)
    model.params = {"rocket_transform": "minirocket", "num_kernels": 100}
    X, y = _make_clf_data() if problem_type == "multiclass" else _make_binary_data()
    model._fit(X, y)
    return model, X, y


def _fit_arsenal(problem_type="multiclass"):
    model = ArsenalModel(problem_type=problem_type)
    model.params = {"rocket_transform": "rocket", "num_kernels": 100, "n_estimators": 3}
    X, y = _make_clf_data() if problem_type == "multiclass" else _make_binary_data()
    model._fit(X, y)
    return model, X, y


# ---------------------------------------------------------------------------
# supported_problem_types
# ---------------------------------------------------------------------------

def test_rocket_supported_problem_types():
    assert "multiclass" in RocketModel.supported_problem_types()
    assert "binary" in RocketModel.supported_problem_types()
    assert "regression" not in RocketModel.supported_problem_types()


def test_arsenal_supported_problem_types():
    assert "multiclass" in ArsenalModel.supported_problem_types()
    assert "binary" in ArsenalModel.supported_problem_types()
    assert "regression" not in ArsenalModel.supported_problem_types()


# ---------------------------------------------------------------------------
# RocketModel: fit + predict (multiclass)
# ---------------------------------------------------------------------------

def test_rocket_fit_sets_feature_names():
    model, X, _ = _fit_rocket("multiclass")
    assert hasattr(model, "_feature_names")
    assert len(model._feature_names) == X.shape[1]


def test_rocket_predict_multiclass_returns_known_labels():
    model, X, y = _fit_rocket("multiclass")
    preds = model._predict(X)
    assert len(preds) == len(X)
    assert set(preds).issubset(set(y))


def test_rocket_predict_proba_multiclass_shape_and_sums():
    model, X, _ = _fit_rocket("multiclass")
    proba = model._predict_proba(X)
    assert proba.shape == (len(X), 3)
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(X)), atol=1e-5)


# ---------------------------------------------------------------------------
# RocketModel: binary
# ---------------------------------------------------------------------------

def test_rocket_predict_binary_returns_known_labels():
    model, X, y = _fit_rocket("binary")
    preds = model._predict(X)
    assert set(preds).issubset(set(y))


def test_rocket_predict_proba_binary_is_1d():
    model, X, _ = _fit_rocket("binary")
    proba = model._predict_proba(X)
    assert proba.ndim == 1
    assert len(proba) == len(X)
    assert ((proba >= 0) & (proba <= 1)).all()


# ---------------------------------------------------------------------------
# RocketModel: regression raises
# ---------------------------------------------------------------------------

def test_rocket_fit_raises_on_regression():
    model = RocketModel(problem_type="regression")
    model.params = {"rocket_transform": "minirocket", "num_kernels": 100}
    X, y = _make_clf_data()
    with pytest.raises(ValueError, match="classification"):
        model._fit(X, y)


# ---------------------------------------------------------------------------
# RocketModel: _select_features handles extra columns
# ---------------------------------------------------------------------------

def test_rocket_select_features_drops_extra_column():
    model, X, _ = _fit_rocket("multiclass")
    X_extra = X.copy()
    X_extra["extra_col"] = np.nan
    preds = model._predict(X_extra)
    assert len(preds) == len(X)


# ---------------------------------------------------------------------------
# ArsenalModel: fit + predict (multiclass)
# ---------------------------------------------------------------------------

def test_arsenal_fit_sets_feature_names():
    model, X, _ = _fit_arsenal("multiclass")
    assert hasattr(model, "_feature_names")
    assert len(model._feature_names) == X.shape[1]


def test_arsenal_predict_multiclass_returns_known_labels():
    model, X, y = _fit_arsenal("multiclass")
    preds = model._predict(X)
    assert len(preds) == len(X)
    assert set(preds).issubset(set(y))


def test_arsenal_predict_proba_multiclass_shape_and_sums():
    model, X, _ = _fit_arsenal("multiclass")
    proba = model._predict_proba(X)
    assert proba.shape == (len(X), 3)
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(X)), atol=1e-5)


# ---------------------------------------------------------------------------
# ArsenalModel: binary
# ---------------------------------------------------------------------------

def test_arsenal_predict_proba_binary_is_1d():
    model, X, _ = _fit_arsenal("binary")
    proba = model._predict_proba(X)
    assert proba.ndim == 1
    assert len(proba) == len(X)
    assert ((proba >= 0) & (proba <= 1)).all()


# ---------------------------------------------------------------------------
# ArsenalModel: regression raises
# ---------------------------------------------------------------------------

def test_arsenal_fit_raises_on_regression():
    model = ArsenalModel(problem_type="regression")
    model.params = {"rocket_transform": "rocket", "num_kernels": 100, "n_estimators": 3}
    X, y = _make_clf_data()
    with pytest.raises(ValueError, match="classification"):
        model._fit(X, y)
