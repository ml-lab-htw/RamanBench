"""Utility functions for metrics computation."""

import numpy as np
from raman_data import TASK_TYPE

from raman_bench.metrics.classification import ClassificationMetrics
from raman_bench.metrics.regression import RegressionMetrics


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_type: TASK_TYPE = TASK_TYPE.Classification,
    y_proba: np.ndarray | None = None,
    **kwargs,
) -> dict[str, float]:
    """Compute all metrics for the given task type.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth labels or values.
    y_pred : np.ndarray
        Predicted labels or values.
    task_type : TASK_TYPE
        ``TASK_TYPE.Classification`` or ``TASK_TYPE.Regression``.
    y_proba : np.ndarray, optional
        Class probabilities (classification only, used for ROC-AUC).
    **kwargs
        Forwarded to :class:`ClassificationMetrics` (e.g. ``average``).

    Returns
    -------
    dict[str, float]
        Metric names mapped to values.
    """
    if task_type == TASK_TYPE.Classification:
        return ClassificationMetrics(**kwargs).compute_all(y_true, y_pred, y_proba)
    if task_type == TASK_TYPE.Regression:
        return RegressionMetrics().compute_all(y_true, y_pred)
    raise ValueError(f"Unknown task type: {task_type}")
