"""More metrics per fold, computed from the test predictions every v1 result stores.

The leaderboard ranks classification by ROC AUC (binary) or log loss (multiclass) and
regression by RMSE, TabArena's conventions. Every cached ``results.pkl`` also keeps the
test-set predictions (``simulation_artifacts``: ``pred_proba_dict_test``, ``y_test``),
so any other metric can be computed afterwards without rerunning a model.
:func:`metrics_from_result` does that for one result, :func:`fold_metrics` for a results
directory; :func:`raman_bench.compare.compare` and :func:`raman_bench.compare.leaderboard`
rank by any of them (``classification_metric=``, ``regression_metric=``).

Predicted class: the most probable class (probability above 0.5 for binary). ``f1_macro``,
``precision_macro`` and ``recall_macro`` average over classes unweighted, ``f1_weighted``
by class frequency. ``roc_auc`` is one-vs-rest and macro-averaged for multiclass; it is
NaN on a fold where a class is missing from the test set. ``brier`` is the multiclass
Brier score (squared error summed over classes). ``rpd`` (ratio of performance to
deviation) is the test targets' standard deviation over the RMSE, common in chemometrics.

For ranking, each metric is turned into an error (lower is better, never negative):
the value itself where lower is better, ``1 - value`` for the bounded higher-is-better
metrics, ``1 / value`` for ``rpd``.
"""

from __future__ import annotations

import logging
import os
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from glob import glob

import numpy as np
import pandas as pd
from scipy import stats
from sklearn import metrics as skm

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Metric:
    task: str  # "classification" or "regression"
    higher_is_better: bool
    fn: Callable
    description: str


def _labels(y, proba):
    """Predicted classes from stored probabilities (1-D: P(class 1) for binary)."""
    return (proba > 0.5).astype(int) if proba.ndim == 1 else proba.argmax(axis=1)


def _proba_2d(proba):
    proba = np.asarray(proba, dtype=np.float64)
    proba = np.column_stack([1.0 - proba, proba]) if proba.ndim == 1 else proba
    return proba / proba.sum(axis=1, keepdims=True)


def _roc_auc(y, proba):
    if len(np.unique(y)) < 2:
        return np.nan
    if proba.ndim == 1:
        return skm.roc_auc_score(y, proba)
    if len(np.unique(y)) < proba.shape[1]:
        return np.nan  # a class missing from this test fold: one-vs-rest AUC undefined
    return skm.roc_auc_score(y, _proba_2d(proba), multi_class="ovr", average="macro")


def _log_loss(y, proba):
    # As the stored metric_error: probabilities clipped at float32 epsilon, not renormalised.
    # sklearn's log_loss clips at float64 epsilon instead, which differs by up to ~1e-3 when
    # a model puts ~1e-7 on the true class.
    proba = np.asarray(proba, dtype=np.float64)
    p = np.column_stack([1.0 - proba, proba]) if proba.ndim == 1 else proba
    eps = np.finfo(np.float32).eps
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], eps, 1 - eps))))


def _brier(y, proba):
    p = _proba_2d(proba)
    onehot = np.eye(p.shape[1])[y]
    return float(np.mean(np.sum((p - onehot) ** 2, axis=1)))


def _clf(fn):
    return lambda y, proba: fn(y, _labels(y, proba))


def _rmse(y, pred):
    return float(np.sqrt(np.mean((y - pred) ** 2)))


def _corr(fn):
    def corr(y, pred):
        if np.std(y) == 0 or np.std(pred) == 0:
            return np.nan
        return float(fn(y, pred)[0])

    return corr


METRICS: dict[str, Metric] = {
    "accuracy": Metric("classification", True, _clf(skm.accuracy_score), "Accuracy"),
    "balanced_accuracy": Metric(
        "classification",
        True,
        _clf(skm.balanced_accuracy_score),
        "Balanced accuracy (mean recall per class)",
    ),
    "f1_macro": Metric(
        "classification",
        True,
        _clf(lambda y, p: skm.f1_score(y, p, average="macro", zero_division=0)),
        "F1, unweighted mean over classes",
    ),
    "f1_weighted": Metric(
        "classification",
        True,
        _clf(lambda y, p: skm.f1_score(y, p, average="weighted", zero_division=0)),
        "F1, mean over classes weighted by frequency",
    ),
    "precision_macro": Metric(
        "classification",
        True,
        _clf(lambda y, p: skm.precision_score(y, p, average="macro", zero_division=0)),
        "Precision, unweighted mean over classes",
    ),
    "recall_macro": Metric(
        "classification",
        True,
        _clf(lambda y, p: skm.recall_score(y, p, average="macro", zero_division=0)),
        "Recall, unweighted mean over classes",
    ),
    "mcc": Metric(
        "classification", True, _clf(skm.matthews_corrcoef), "Matthews correlation coefficient"
    ),
    "cohen_kappa": Metric("classification", True, _clf(skm.cohen_kappa_score), "Cohen's kappa"),
    "roc_auc": Metric("classification", True, _roc_auc, "ROC AUC (multiclass: one-vs-rest, macro)"),
    "log_loss": Metric("classification", False, _log_loss, "Log loss (cross-entropy)"),
    "brier": Metric("classification", False, _brier, "Brier score (multiclass)"),
    "rmse": Metric("regression", False, _rmse, "Root mean squared error"),
    "mae": Metric(
        "regression", False, lambda y, p: float(np.mean(np.abs(y - p))), "Mean absolute error"
    ),
    "median_ae": Metric("regression", False, skm.median_absolute_error, "Median absolute error"),
    "max_error": Metric("regression", False, skm.max_error, "Largest absolute error"),
    "r2": Metric("regression", True, skm.r2_score, "Coefficient of determination R²"),
    "explained_variance": Metric(
        "regression", True, skm.explained_variance_score, "Explained variance"
    ),
    "pearson_r": Metric(
        "regression", True, _corr(stats.pearsonr), "Pearson correlation of prediction and target"
    ),
    "spearman_r": Metric(
        "regression",
        True,
        _corr(stats.spearmanr),
        "Spearman rank correlation of prediction and target",
    ),
    "rpd": Metric(
        "regression",
        True,
        lambda y, p: float(np.std(y, ddof=1) / _rmse(y, p)),
        "Ratio of performance to deviation: SD(y) / RMSE",
    ),
}
METRIC_COLUMNS = list(METRICS)

# The protocol's own metric for each problem type, as stored in a result's ``metric``.
PROTOCOL_METRIC = {"binary": "roc_auc", "multiclass": "log_loss", "regression": "rmse"}


def metric_names(task: str | None = None) -> list[str]:
    """The available metrics, optionally only those for ``"classification"``/``"regression"``."""
    return [name for name, m in METRICS.items() if task is None or m.task == task]


def to_error(values, metric: str):
    """*values* of *metric* as an error for ranking: lower is better, never negative."""
    m = METRICS[metric]
    values = np.asarray(values, dtype=float)
    if not m.higher_is_better:
        return values
    if metric == "rpd":
        with np.errstate(divide="ignore"):
            return 1.0 / values
    return 1.0 - values


def metrics_from_result(result: dict) -> dict[str, float]:
    """Every metric of :data:`METRICS` for one cached result (NaN where not applicable)."""
    sa = result["simulation_artifacts"]
    proba = np.asarray(next(iter(sa["pred_proba_dict_test"].values())))
    y = np.asarray(sa["y_test"])
    task = "regression" if sa["problem_type"] == "regression" else "classification"
    out = dict.fromkeys(METRIC_COLUMNS, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, m in METRICS.items():
            if m.task != task:
                continue
            try:
                out[name] = float(
                    m.fn(y.astype(int) if task == "classification" else y.astype(float), proba)
                )
            except Exception as e:  # one bad fold must not lose the rest
                logger.debug("%s failed: %s", name, e)
    return out


def fold_metrics(results_dir: str | os.PathLike) -> pd.DataFrame:
    """One row per cached result under *results_dir*: ``ta_name``, ``dataset``, ``fold`` and every metric.

    ``ta_name`` is the result directory's model name without the ``_c1_BAG_L1`` suffix
    (the ``ta_name`` of ``hpo_results.csv``); ``fold`` is the split index, as there.
    Results without stored test predictions (e.g. the AutoGluon reference runs) are skipped.
    """
    from tabarena.utils.pickle_utils import load_pickle

    rows = []
    for path in sorted(glob(os.path.join(results_dir, "*", "*", "*", "results.pkl"))):
        try:
            result = load_pickle(path)
            values = metrics_from_result(result)
        except Exception as e:
            logger.warning("No metrics for %s: %s", path, e)
            continue
        tm = result["task_metadata"]
        framework = result.get("framework") or path.split(os.sep)[-4]
        rows.append(
            {
                "ta_name": framework.removesuffix("_c1_BAG_L1"),
                "dataset": tm["name"],
                "fold": int(tm.get("split_idx", tm["fold"])),
                **values,
            }
        )
    return pd.DataFrame(rows, columns=["ta_name", "dataset", "fold", *METRIC_COLUMNS])
