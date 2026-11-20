from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator

from raman_bench.preprocessing.bridge_bases import SklearnAutoGluonBridge, _NoAugBase


def _to_numpy(X) -> np.ndarray:
    return X.values if hasattr(X, "values") else np.asarray(X)


def _resolve_device(device: str | None) -> str:
    if device is not None:
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


class _CloneSafeTabOmics(BaseEstimator):
    """The published TabOmics classifier behind a constructor that ``sklearn.base.clone`` accepts.

    ``ManyClassClassifier`` clones its estimator for every codebook row, so the wrapped class has to
    keep its constructor arguments as plain attributes.
    """

    def __init__(self, model: str = "v2.5", device: str | None = None):
        self.model = model
        self.device = device

    def fit(self, X, y):
        from tabomics import Classifier

        self._estimator = Classifier(model=self.model, device=_resolve_device(self.device))
        self._estimator.fit(X, y)
        self.classes_ = self._estimator.classes_
        return self

    def predict(self, X):
        return self._estimator.predict(X)

    def predict_proba(self, X):
        return self._estimator.predict_proba(X)


class TabOmicsModel(BaseEstimator):
    """TabOmics (classification only): a tabular foundation model for wide omics tables.

    Pretrained on synthetic tables, it predicts the test rows in one forward pass over the training rows,
    like TabPFN. It is built for thousands of features and few samples, the regime of Raman spectra. The
    published model takes up to 10 classes; above ``many_class_threshold`` classes it is wrapped in
    tabpfn-extensions' output-coding ``ManyClassClassifier``, as TabPFN-Wide is.

    Requires the ``tabomics`` package (``requirements-tabomics-git.txt``); the weights come from Hugging
    Face on first use.

    Reference:
        https://github.com/lbotecur/tabomics (GPL-3.0-or-later; model cards on huggingface.co/lbotecur)
    """

    def __init__(self, model: str = "v2.5", device: str | None = None, many_class_threshold: int = 10):
        self.model = model
        self.device = device
        self.many_class_threshold = many_class_threshold

    def fit(self, X, y):
        try:
            import tabomics  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "TabOmicsModel requires tabomics. Install with: pip install -r requirements-tabomics-git.txt"
            ) from e

        y_arr = np.asarray(y)
        if np.issubdtype(y_arr.dtype, np.floating):
            raise ValueError("TabOmics does not support regression.")
        self.classes_ = np.unique(y_arr)
        self.model_ = _CloneSafeTabOmics(model=self.model, device=self.device)
        if len(self.classes_) > self.many_class_threshold:
            from tabpfn_extensions.many_class import ManyClassClassifier

            self.model_ = ManyClassClassifier(estimator=self.model_, alphabet_size=self.many_class_threshold)
        self.model_.fit(_to_numpy(X), y_arr)
        return self

    def predict(self, X):
        return self.model_.predict(_to_numpy(X))

    def predict_proba(self, X):
        proba = self.model_.predict_proba(_to_numpy(X))
        return proba[:, 1] if len(self.classes_) == 2 else proba


class _TabOmicsBridge(SklearnAutoGluonBridge):
    _sklearn_cls = TabOmicsModel
    ag_key = "TABOMICS"
    ag_name = "TabOmics"


class Prep_TABOMICS(_NoAugBase, _TabOmicsBridge):  # noqa: N801
    """TabOmics, classification only; built for wide tables with few samples."""

    pass
