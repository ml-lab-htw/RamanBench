"""Utility functions for metrics computation."""

from typing import Any

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


def compare_metrics(
    results: dict[str, dict[str, float]],
    metric: str,
    higher_is_better: bool = True,
) -> dict[str, Any]:
    """Compare a single metric across multiple models.

    Parameters
    ----------
    results : dict[str, dict[str, float]]
        Mapping of model name → metrics dict.
    metric : str
        Metric to compare.
    higher_is_better : bool
        Sort direction (default ``True``).

    Returns
    -------
    dict
        Keys: ``best_model``, ``worst_model``, ``ranking``, ``values``,
        ``mean``, ``std``.
    """
    values = {name: m[metric] for name, m in results.items() if metric in m}
    if not values:
        return {"error": f"Metric '{metric}' not found in any results"}

    ranking = sorted(values, key=values.__getitem__, reverse=higher_is_better)
    return {
        "best_model": ranking[0],
        "best_value": values[ranking[0]],
        "worst_model": ranking[-1],
        "worst_value": values[ranking[-1]],
        "ranking": ranking,
        "values": values,
        "mean": float(np.mean(list(values.values()))),
        "std": float(np.std(list(values.values()))),
    }
