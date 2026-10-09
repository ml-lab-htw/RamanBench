"""raman_bench.fold_metrics and metric selection in raman_bench.compare."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn import metrics as skm

from raman_bench.compare import select_metric
from raman_bench.fold_metrics import METRICS, metric_names, metrics_from_result, to_error


def _result(problem_type, y, proba):
    return {
        "simulation_artifacts": {
            "problem_type": problem_type,
            "pred_proba_dict_test": {"M_c1_BAG_L1": np.asarray(proba)},
            "y_test": np.asarray(y),
        }
    }


def test_binary_metrics_from_one_column_probabilities():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    p1 = np.clip(0.3 * y + 0.35 + rng.normal(0, 0.2, 200), 0.01, 0.99)
    m = metrics_from_result(_result("binary", y, p1))
    pred = (p1 > 0.5).astype(int)
    assert m["accuracy"] == pytest.approx(skm.accuracy_score(y, pred))
    assert m["balanced_accuracy"] == pytest.approx(skm.balanced_accuracy_score(y, pred))
    assert m["f1_macro"] == pytest.approx(skm.f1_score(y, pred, average="macro"))
    assert m["mcc"] == pytest.approx(skm.matthews_corrcoef(y, pred))
    assert m["roc_auc"] == pytest.approx(skm.roc_auc_score(y, p1))
    assert m["log_loss"] == pytest.approx(skm.log_loss(y, np.column_stack([1 - p1, p1])))
    assert np.isnan(m["rmse"]) and np.isnan(m["r2"])


def test_multiclass_metrics_and_missing_class():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 3, 150)
    proba = rng.dirichlet(np.ones(3), 150)
    proba[np.arange(150), y] += 1.0
    proba /= proba.sum(axis=1, keepdims=True)
    m = metrics_from_result(_result("multiclass", y, proba))
    assert m["accuracy"] == pytest.approx(skm.accuracy_score(y, proba.argmax(1)))
    assert m["roc_auc"] == pytest.approx(skm.roc_auc_score(y, proba, multi_class="ovr"))
    assert m["brier"] == pytest.approx(np.mean(np.sum((proba - np.eye(3)[y]) ** 2, axis=1)))

    # A class absent from the test fold: one-vs-rest AUC is undefined, the rest still work.
    m = metrics_from_result(_result("multiclass", np.where(y == 2, 0, y), proba))
    assert np.isnan(m["roc_auc"]) and not np.isnan(m["log_loss"]) and not np.isnan(m["accuracy"])


def test_regression_metrics():
    rng = np.random.default_rng(2)
    y = rng.normal(5, 2, 100)
    pred = y + rng.normal(0, 0.5, 100)
    m = metrics_from_result(_result("regression", y, pred))
    rmse = np.sqrt(np.mean((y - pred) ** 2))
    assert m["rmse"] == pytest.approx(rmse)
    assert m["mae"] == pytest.approx(skm.mean_absolute_error(y, pred))
    assert m["r2"] == pytest.approx(skm.r2_score(y, pred))
    assert m["rpd"] == pytest.approx(np.std(y, ddof=1) / rmse)
    assert np.isnan(m["accuracy"])


def test_errors_are_lower_is_better_and_non_negative():
    assert set(metric_names("classification")) | set(metric_names("regression")) == set(METRICS)
    assert to_error([0.9], "accuracy")[0] == pytest.approx(0.1)
    assert to_error([0.3], "log_loss")[0] == pytest.approx(0.3)
    assert to_error([4.0], "rpd")[0] == pytest.approx(0.25)
    assert to_error([-0.5], "r2")[0] == pytest.approx(1.5)


def _tidy():
    return pd.DataFrame({
        "dataset": ["a__0", "a__0", "b__0", "b__0"],
        "fold": [0, 0, 0, 0],
        "model": ["X", "Y", "X", "Y"],
        "task": ["classification", "classification", "regression", "regression"],
        "metric": ["roc_auc", "roc_auc", "rmse", "rmse"],
        "metric_error": [0.1, 0.2, 1.0, 2.0],
        "accuracy": [0.6, 0.9, np.nan, np.nan],
        "mae": [np.nan, np.nan, 0.5, np.nan],
    })


def test_select_metric_replaces_the_error_per_task_type():
    out = select_metric(_tidy(), classification_metric="accuracy")
    clf = out[out["task"] == "classification"].set_index("model")
    assert clf["metric_error"].to_dict() == pytest.approx({"X": 0.4, "Y": 0.1})  # Y is now better
    assert set(clf["metric"]) == {"accuracy"}
    assert out[out["task"] == "regression"]["metric_error"].tolist() == [1.0, 2.0]  # untouched

    out = select_metric(_tidy(), regression_metric="mae")
    assert out[out["task"] == "regression"]["model"].tolist() == ["X"]  # Y's NaN fold dropped


def test_select_metric_rejects_unknown_or_unavailable_metrics():
    with pytest.raises(ValueError, match="classification_metric must be one of"):
        select_metric(_tidy(), classification_metric="rmse")
    with pytest.raises(ValueError, match="No 'r2' column"):
        select_metric(_tidy(), regression_metric="r2")
