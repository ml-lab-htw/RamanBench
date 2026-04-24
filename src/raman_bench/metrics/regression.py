"""Regression metrics for RamanBench."""

from typing import Dict

import numpy as np
from sklearn.metrics import (
    explained_variance_score,
    max_error,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)


class RegressionMetrics:
    """Compute regression evaluation metrics."""

    def compute_all(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Return a dict of all standard regression metrics.

        Parameters
        ----------
        y_true : np.ndarray
            Ground-truth values.
        y_pred : np.ndarray
            Predicted values.
        """
        return {
            "mse": self.mse(y_true, y_pred),
            "rmse": self.rmse(y_true, y_pred),
            "mae": self.mae(y_true, y_pred),
            "r2": self.r2(y_true, y_pred),
            "mape": self.mape(y_true, y_pred),
            "explained_variance": self.explained_variance(y_true, y_pred),
            "max_error": self.max_error(y_true, y_pred),
            "median_ae": self.median_ae(y_true, y_pred),
        }

    def mse(self, y_true, y_pred) -> float:
        return float(mean_squared_error(y_true, y_pred))

    def rmse(self, y_true, y_pred) -> float:
        return float(np.sqrt(self.mse(y_true, y_pred)))

    def mae(self, y_true, y_pred) -> float:
        return float(mean_absolute_error(y_true, y_pred))

    def r2(self, y_true, y_pred) -> float:
        return float(r2_score(y_true, y_pred))

    def mape(self, y_true, y_pred) -> float:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        mask = y_true != 0
        if not mask.any():
            return 0.0
        return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

    def explained_variance(self, y_true, y_pred) -> float:
        return float(explained_variance_score(y_true, y_pred))

    def max_error(self, y_true, y_pred) -> float:
        return float(max_error(y_true, y_pred))

    def median_ae(self, y_true, y_pred) -> float:
        return float(median_absolute_error(y_true, y_pred))
