"""Tests for classification and regression metrics."""

import numpy as np
import pytest

from raman_bench.metrics import ClassificationMetrics, RegressionMetrics, compute_metrics
from raman_data import TASK_TYPE


class TestClassificationMetrics:
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0])

    def test_accuracy(self):
        m = ClassificationMetrics()
        assert 0.0 <= m.accuracy(self.y_true, self.y_pred) <= 1.0

    def test_f1(self):
        m = ClassificationMetrics()
        assert 0.0 <= m.f1_score(self.y_true, self.y_pred) <= 1.0

    def test_balanced_accuracy(self):
        m = ClassificationMetrics()
        assert 0.0 <= m.balanced_accuracy(self.y_true, self.y_pred) <= 1.0

    def test_compute_all_keys(self):
        m = ClassificationMetrics()
        result = m.compute_all(self.y_true, self.y_pred)
        for key in ("accuracy", "f1_score", "balanced_accuracy", "cohen_kappa"):
            assert key in result

    def test_perfect_score(self):
        m = ClassificationMetrics()
        result = m.compute_all(self.y_true, self.y_true)
        assert result["accuracy"] == pytest.approx(1.0)


class TestRegressionMetrics:
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])

    def test_rmse_nonneg(self):
        m = RegressionMetrics()
        assert m.rmse(self.y_true, self.y_pred) >= 0.0

    def test_r2_perfect(self):
        m = RegressionMetrics()
        assert m.r2(self.y_true, self.y_true) == pytest.approx(1.0)

    def test_compute_all_keys(self):
        m = RegressionMetrics()
        result = m.compute_all(self.y_true, self.y_pred)
        for key in ("rmse", "mae", "r2", "mse"):
            assert key in result

    def test_mape_zero_targets(self):
        m = RegressionMetrics()
        y = np.array([0.0, 0.0])
        assert m.mape(y, y) == 0.0


def test_compute_metrics_classification():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    result = compute_metrics(y_true, y_pred, task_type=TASK_TYPE.Classification)
    assert "accuracy" in result


def test_compute_metrics_regression():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 2.1, 2.9])
    result = compute_metrics(y_true, y_pred, task_type=TASK_TYPE.Regression)
    assert "rmse" in result
