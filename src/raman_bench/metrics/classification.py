"""Classification metrics for RamanBench."""

import warnings

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import LabelBinarizer


class ClassificationMetrics:
    """Compute classification evaluation metrics.

    Parameters
    ----------
    average : str
        Averaging strategy for multi-class metrics (default ``"weighted"``).
    """

    def __init__(self, average: str = "weighted"):
        self.average = average

    def compute_all(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Return a dict of all standard classification metrics.

        Parameters
        ----------
        y_true : np.ndarray
            Ground-truth labels.
        y_pred : np.ndarray
            Predicted labels.
        y_proba : np.ndarray, optional
            Class probabilities.  Required for ROC-AUC.
        """
        metrics = {
            "accuracy": self.accuracy(y_true, y_pred),
            "precision": self.precision(y_true, y_pred),
            "recall": self.recall(y_true, y_pred),
            "f1_score": self.f1_score(y_true, y_pred),
            "balanced_accuracy": self.balanced_accuracy(y_true, y_pred),
            "cohen_kappa": self.cohen_kappa(y_true, y_pred),
            "matthews_corrcoef": self.matthews_corrcoef(y_true, y_pred),
        }
        if y_proba is not None:
            try:
                metrics["roc_auc"] = self.roc_auc(y_true, y_proba)
            except Exception:
                metrics["roc_auc"] = np.nan
            try:
                metrics["log_loss"] = self.log_loss(y_true, y_proba)
            except Exception:
                metrics["log_loss"] = np.nan
        return metrics

    def accuracy(self, y_true, y_pred) -> float:
        return float(accuracy_score(y_true, y_pred))

    def precision(self, y_true, y_pred) -> float:
        return float(precision_score(y_true, y_pred, average=self.average, zero_division=0))

    def recall(self, y_true, y_pred) -> float:
        return float(recall_score(y_true, y_pred, average=self.average, zero_division=0))

    def f1_score(self, y_true, y_pred) -> float:
        return float(f1_score(y_true, y_pred, average=self.average, zero_division=0))

    def balanced_accuracy(self, y_true, y_pred) -> float:
        return float(balanced_accuracy_score(y_true, y_pred))

    def cohen_kappa(self, y_true, y_pred) -> float:
        return float(cohen_kappa_score(y_true, y_pred))

    def matthews_corrcoef(self, y_true, y_pred) -> float:
        return float(matthews_corrcoef(y_true, y_pred))

    def roc_auc(self, y_true, y_proba, multi_class: str = "ovr") -> float:
        n_classes = len(np.unique(y_true))
        if n_classes == 2:
            if y_proba.ndim == 2:
                y_proba = y_proba[:, 1]
            return float(roc_auc_score(y_true, y_proba))
        lb = LabelBinarizer()
        y_true_bin = lb.fit_transform(y_true)
        return float(
            roc_auc_score(y_true_bin, y_proba, multi_class=multi_class, average=self.average)
        )

    def log_loss(self, y_true, y_proba) -> float:
        """Cross-entropy. Accepts 1-D proba for binary or 2-D (n_samples, n_classes)."""
        if y_proba.ndim == 1:
            # Binary: sklearn log_loss wants 2-D for >=2 classes; build it.
            y_proba = np.column_stack([1.0 - y_proba, y_proba])
        y_proba = np.asarray(y_proba, dtype=np.float64)
        row_sums = y_proba.sum(axis=1, keepdims=True)

        # sklearn's log_loss warns whenever row sums stray from 1 by more than
        # sqrt(float64 eps) ~= 1.5e-8. Probabilities are stored from float32 model
        # outputs, so their sums drift ~1e-7 — genuinely normalized, but past that
        # threshold, so the warning fires on nearly every multiclass dataset and drowns
        # out anything real. Replace it with our own check at a tolerance that only
        # flags a genuinely un-normalized proba (a real model bug), then renormalize so
        # the loss is computed on exact probabilities regardless.
        max_dev = float(np.abs(row_sums - 1.0).max()) if row_sums.size else 0.0
        if max_dev > 1e-3:
            warnings.warn(
                f"Probability rows deviate from 1.0 by up to {max_dev:.2e}; "
                "check the model's predict_proba output.",
                UserWarning,
                stacklevel=2,
            )
        y_proba = np.divide(
            y_proba, row_sums, out=np.zeros_like(y_proba), where=row_sums != 0
        )
        with warnings.catch_warnings():
            # Belt-and-suspenders: rows now sum to 1 to machine precision, but suppress
            # sklearn's sum-to-one warning so a future dtype change can't reintroduce it.
            warnings.filterwarnings(
                "ignore", message="The y_pred values do not sum to one", category=UserWarning
            )
            return float(log_loss(y_true, y_proba, labels=np.unique(y_true)))

    def confusion_matrix(self, y_true, y_pred, normalize=None) -> np.ndarray:
        return confusion_matrix(y_true, y_pred, normalize=normalize)

    def classification_report(self, y_true, y_pred, output_dict=True) -> str | dict:
        return classification_report(y_true, y_pred, output_dict=output_dict, zero_division=0)
