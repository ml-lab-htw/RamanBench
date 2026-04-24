"""Classification metrics for RamanBench."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
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

    def confusion_matrix(self, y_true, y_pred, normalize=None) -> np.ndarray:
        return confusion_matrix(y_true, y_pred, normalize=normalize)

    def classification_report(self, y_true, y_pred, output_dict=True) -> str | dict:
        return classification_report(y_true, y_pred, output_dict=output_dict, zero_division=0)
