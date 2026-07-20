"""Regression metrics for RamanBench."""

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

    def compute_all(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
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
        r"""Coefficient of determination, guarded against near-constant test folds.

        R^2 = 1 - SS_res / SS_tot, and SS_tot = n * Var(y_true). When a test fold's
        targets are (near-)constant, SS_tot ~ 0, so an ordinary residual sends R^2 to
        huge negatives (observed: -650 on a fold with target std 0.04, RMSE ~1.0). The
        value is not a bad model, it is R^2 being undefined for a degenerate fold — but
        those outliers wreck any mean over folds. Report NaN there so downstream
        aggregation (mean, per-dataset normalisation) simply skips them; RMSE stays the
        honest signal, matching how TabArena ranks regression on error, not R^2.

        The gate is scale-free: coefficient of variation below 0.1% means the targets
        vary by under a thousandth of their magnitude, i.e. effectively one value.
        """
        y_true = np.asarray(y_true, dtype=float)
        ref = np.abs(y_true).mean()
        if np.std(y_true) < 1e-3 * ref or np.ptp(y_true) == 0.0:
            return float("nan")
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
