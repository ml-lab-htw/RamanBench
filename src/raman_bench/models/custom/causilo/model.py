"""Causilo -- pretrained tabular foundation model (Nums AI, github.com/nums-ai/causilo).

RamanBench-only integration (not added to TabArena upstream -- as of tabarena
0.1.0 there is no ``tabarena.models.causilo``; confirmed by enumerating every
``tabarena.models.<key>`` directory in the installed wheel before writing this
file). Causilo is a general-purpose, in-context tabular foundation model in
the TabPFN family: a fixed pretrained transformer conditions on the training
rows at inference time (no gradient training), supporting both classification
(official checkpoint: up to 10 classes) and regression, with a
scikit-learn-compatible ``CausiloClassifier``/``CausiloRegressor`` pair
(``pip install causilo``; first fit downloads and caches the checkpoint from
Hugging Face). Unlike ``TabPFNWideModel`` (this repo's ``tabpfn_wide/model.py``),
Causilo's own ``device="auto"`` already resolves CUDA-vs-CPU internally, so no
RamanBench-side device-resolution wrapper is needed here.

Model weights ship under a separate non-commercial "Causilo License v1.0"
(code itself is Apache-2.0) -- fine for RamanBench's own research use; see the
note next to the ``causilo`` pin in ``pyproject.toml``.
"""

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


class CausiloModel(BaseEstimator):
    """Causilo for tabular data -- sklearn-compatible, classification + regression.

    Dispatches to ``causilo.CausiloClassifier`` or ``causilo.CausiloRegressor``
    based on the target dtype, mirroring how ``RidgeModel`` and other dual-task
    wrappers in this codebase pick their sklearn delegate at ``fit`` time.

    Requires the ``causilo`` package.

    Reference:
        https://github.com/nums-ai/causilo
    """

    def __init__(
        self,
        n_estimators: int = 8,
        random_state: int = 42,
        device: str = "auto",
        use_kv_cache: bool = False,
        retain_preprocessing: bool = True,
        many_class_threshold: int = 10,
    ):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.device = device
        self.use_kv_cache = use_kv_cache
        self.retain_preprocessing = retain_preprocessing
        self.many_class_threshold = many_class_threshold

    def fit(self, X, y, **kwargs):
        try:
            import causilo  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "CausiloModel requires causilo. Install with: pip install causilo"
            ) from e

        X_arr, y_arr = _to_numpy(X), np.asarray(y)
        self.problem_type_ = _infer_problem_type(y)

        estimator_kwargs = dict(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            device=self.device,
            use_kv_cache=self.use_kv_cache,
            retain_preprocessing=self.retain_preprocessing,
        )

        if self.problem_type_ == "regression":
            from causilo import CausiloRegressor

            self.model_ = CausiloRegressor(**estimator_kwargs)
        else:
            from causilo import CausiloClassifier

            self.classes_ = np.unique(y_arr)
            # Official checkpoint natively supports up to `many_class_threshold`
            # classes (matches TabPFNWideModel's own fail-fast convention for
            # foundation models with a hard class-count ceiling: AutoGluon then
            # records no prediction for this dataset/model rather than crashing
            # the whole run).
            if len(self.classes_) > self.many_class_threshold:
                raise ValueError(
                    f"Causilo: {len(self.classes_)} classes exceeds the native limit "
                    f"({self.many_class_threshold}); skipping this dataset."
                )
            self.model_ = CausiloClassifier(**estimator_kwargs)

        self.model_.fit(X_arr, y_arr)
        return self

    def predict(self, X):
        return self.model_.predict(_to_numpy(X))

    def predict_proba(self, X):
        if self.problem_type_ == "regression":
            raise AttributeError("predict_proba is not available for regression.")
        proba = self.model_.predict_proba(_to_numpy(X))
        return proba[:, 1] if self.problem_type_ == "binary" else proba


class _CausiloBridge(SklearnAutoGluonBridge):
    _sklearn_cls = CausiloModel
    ag_key = "CAUSILO"
    ag_name = "Causilo"

    def _get_default_searchspace(self):
        from autogluon.common import space

        return {
            "n_estimators": space.Int(lower=1, upper=32),
        }


class Prep_CAUSILO(_NoAugBase, _CausiloBridge):  # noqa: N801
    """Causilo -- pretrained tabular foundation model, classification + regression."""

    pass
