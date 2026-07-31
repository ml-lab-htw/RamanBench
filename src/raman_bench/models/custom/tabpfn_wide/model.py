from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator

from raman_bench.preprocessing.bridge_bases import SklearnAutoGluonBridge, _NoAugBase


def _to_numpy(X) -> np.ndarray:
    return X.values if hasattr(X, "values") else np.asarray(X)


def _infer_problem_type(y) -> str:
    arr = np.asarray(y)
    if np.issubdtype(arr.dtype, np.floating):
        return "regression"
    return "binary" if len(np.unique(arr)) == 2 else "multiclass"


class _CloneSafeTabPFNWide(BaseEstimator):
    """sklearn-clone-safe wrapper around TabPFNWideClassifier.

    TabPFNWideClassifier mutates its own params during __init__ (resolves
    model_name → model_path), so get_params() returns both, and re-init
    via sklearn.clone() trips its mutually-exclusive XOR check.
    This wrapper exposes only the user-supplied constructor args, so
    cloning round-trips cleanly. Required by ManyClassClassifier, which
    clones the base estimator per ECOC sub-task.
    """

    def __init__(self, model_name: str = "wide-v2-5k", device: str = "cpu"):
        self.model_name = model_name
        self.device = device

    def fit(self, X, y):
        from tabpfnwide.classifier import TabPFNWideClassifier

        self._estimator = TabPFNWideClassifier(model_name=self.model_name, device=self.device)
        self._estimator.fit(X, y)
        self.classes_ = self._estimator.classes_
        return self

    def predict(self, X):
        return self._estimator.predict(X)

    def predict_proba(self, X):
        return self._estimator.predict_proba(X)


class TabPFNWideModel(BaseEstimator):
    """TabPFN-Wide for Raman spectra — sklearn-compatible (classification only).

    Built with PriorLabs-TabPFN. TabPFN-Wide targets datasets with many
    features and few samples — a regime Raman spectroscopy frequently
    occupies (2000+ wavenumber columns, often <1000 spectra).

    Requires the ``tabpfnwide`` package.

    Reference:
        TabPFN-Wide: https://github.com/not-a-feature/TabPFN-Wide
        DOI: 10.48550/arXiv.2510.06162
    """

    def __init__(
        self,
        model_name: str = "wide-v2-5k",
        device: str = "cpu",
        many_class_threshold: int = 10,
    ):
        self.model_name = model_name
        self.device = device
        self.many_class_threshold = many_class_threshold

    def fit(self, X, y):
        try:
            import tabpfnwide.classifier  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "TabPFNWideModel requires tabpfnwide. Install with: pip install tabpfnwide"
            ) from e

        X_arr, y_arr = _to_numpy(X), np.asarray(y)
        self.problem_type_ = _infer_problem_type(y)
        if self.problem_type_ == "regression":
            raise ValueError("TabPFN-Wide does not support regression.")

        self.classes_ = np.unique(y_arr)

        # TabPFN natively supports up to `many_class_threshold` classes. The
        # ECOC ManyClassClassifier workaround OOMs on wide Raman spectra, so for
        # now we skip datasets above the limit by failing fast (AutoGluon then
        # records no prediction for this dataset/model).
        if len(self.classes_) > self.many_class_threshold:
            raise ValueError(
                f"TabPFN-Wide: {len(self.classes_)} classes exceeds the native limit "
                f"({self.many_class_threshold}); skipping this dataset."
            )

        self.model_ = _CloneSafeTabPFNWide(model_name=self.model_name, device=self.device)

        # Many-class support via ECOC ManyClassClassifier — disabled for now
        # because it OOMs on wide Raman spectra even at 256G. Pending advice
        # from the TabPFN-Wide authors on memory-efficient many-class usage.
        # if len(self.classes_) > self.many_class_threshold:
        #     try:
        #         from tabpfn_extensions.many_class import ManyClassClassifier
        #     except ImportError as e:
        #         raise ImportError(
        #             f"TabPFN-Wide: {len(self.classes_)} classes exceeds native limit "
        #             f"({self.many_class_threshold}). Install tabpfn-extensions: "
        #             "pip install tabpfn-extensions"
        #         ) from e
        #     self.model_ = ManyClassClassifier(
        #         estimator=self.model_, alphabet_size=self.many_class_threshold
        #     )

        self.model_.fit(X_arr, y_arr)
        return self

    def predict(self, X):
        return self.model_.predict(_to_numpy(X))

    def predict_proba(self, X):
        proba = self.model_.predict_proba(_to_numpy(X))
        return proba[:, 1] if self.problem_type_ == "binary" else proba


class _TabPFNWideBridge(SklearnAutoGluonBridge):
    _sklearn_cls = TabPFNWideModel
    ag_key = "TABPFN-WIDE"
    ag_name = "TabPFN-Wide"


class Prep_TABPFN_WIDE(_NoAugBase, _TabPFNWideBridge):  # noqa: N801
    """TabPFN-Wide — classification-only, targets wide datasets (many features, few samples).

    Built with PriorLabs-TabPFN.
    """

    pass
