"""Causilo 1.5 -- pretrained tabular foundation model (Nums AI, github.com/nums-ai/causilo, PyPI ``causilo>=1.5.0``).

Separate from ``models/custom/causilo`` (Causilo 1.0, whose results stay as they are): Causilo 1.5 is the package's
default network from 1.5.0 on, with the full training context (``mode=None``; ``mode="light"`` is the light variant)
and the library's own default ensemble size (``n_estimators=None``). Fits above the checkpoint's class limit use the
library's own error-correcting output codes, so no ``ManyClassClassifier`` wrapper is needed here, unlike in the 1.0
wrapper. The weights have a separate non-commercial licence (see the ``causilo`` pin in ``pyproject.toml``).
"""

from __future__ import annotations

import inspect

import numpy as np
from sklearn.base import BaseEstimator

from raman_bench.preprocessing.bridge_bases import SklearnAutoGluonBridge, _GPURequiredBridge, _NoAugBase


def _to_numpy(X) -> np.ndarray:
    return X.values if hasattr(X, "values") else np.asarray(X)


def _infer_problem_type(y) -> str:
    arr = np.asarray(y)
    if np.issubdtype(arr.dtype, np.floating):
        return "regression"
    return "binary" if len(np.unique(arr)) == 2 else "multiclass"


class Causilo15Model(BaseEstimator):
    """Causilo 1.5 for tabular data -- sklearn-compatible, classification + regression.

    Dispatches to ``causilo.CausiloClassifier`` or ``causilo.CausiloRegressor`` by the target dtype, like the other
    dual-task wrappers in this codebase.

    Reference:
        https://github.com/nums-ai/causilo
    """

    def __init__(
        self,
        n_estimators: int | None = None,
        mode: str | None = None,
        random_state: int = 42,
        device: str = "auto",
        use_kv_cache: bool = False,
        retain_preprocessing: bool = True,
        max_context: int | None = None,
    ):
        self.n_estimators = n_estimators
        self.mode = mode
        self.random_state = random_state
        self.device = device
        self.use_kv_cache = use_kv_cache
        self.retain_preprocessing = retain_preprocessing
        self.max_context = max_context

    def fit(self, X, y, **kwargs):
        try:
            import causilo
        except ImportError as e:
            raise ImportError("Causilo15Model requires causilo>=1.5.0. Install with: pip install 'causilo>=1.5.0'") from e
        if "model_version" not in inspect.signature(causilo.CausiloClassifier).parameters:
            raise ImportError("Causilo15Model needs causilo>=1.5.0 (the installed causilo has no model_version)")

        X_arr, y_arr = _to_numpy(X), np.asarray(y)
        self.problem_type_ = _infer_problem_type(y)
        kwargs = dict(
            n_estimators=self.n_estimators,
            mode=self.mode,
            random_state=self.random_state,
            device=self.device,
            use_kv_cache=self.use_kv_cache,
            retain_preprocessing=self.retain_preprocessing,
            max_context=self.max_context,
            model_version="1.5",
        )
        if self.problem_type_ == "regression":
            self.model_ = causilo.CausiloRegressor(**kwargs)
        else:
            self.classes_ = np.unique(y_arr)
            self.model_ = causilo.CausiloClassifier(**kwargs)
        self.model_.fit(X_arr, y_arr)
        return self

    def predict(self, X):
        return self.model_.predict(_to_numpy(X))

    def predict_proba(self, X):
        if self.problem_type_ == "regression":
            raise AttributeError("predict_proba is not available for regression.")
        proba = self.model_.predict_proba(_to_numpy(X))
        return proba[:, 1] if self.problem_type_ == "binary" else proba


class Causilo15LightModel(Causilo15Model):
    """Causilo 1.5 (Light): the library's smaller, faster ``mode="light"`` variant of the same 1.5 checkpoint family."""

    def __init__(
        self,
        n_estimators: int | None = None,
        mode: str | None = "light",
        random_state: int = 42,
        device: str = "auto",
        use_kv_cache: bool = False,
        retain_preprocessing: bool = True,
        max_context: int | None = None,
    ):
        super().__init__(
            n_estimators=n_estimators, mode=mode, random_state=random_state, device=device,
            use_kv_cache=use_kv_cache, retain_preprocessing=retain_preprocessing, max_context=max_context,
        )


class _Causilo15Bridge(_GPURequiredBridge, SklearnAutoGluonBridge):
    _sklearn_cls = Causilo15Model
    ag_key = "CAUSILO-1.5"
    ag_name = "Causilo-1.5"

    def _get_default_searchspace(self):
        from autogluon.common import space

        return {"n_estimators": space.Int(lower=1, upper=32)}


class Prep_CAUSILO15(_NoAugBase, _Causilo15Bridge):  # noqa: N801
    """Causilo 1.5 -- pretrained tabular foundation model, classification + regression."""

    pass


class _Causilo15LightBridge(_Causilo15Bridge):
    _sklearn_cls = Causilo15LightModel
    ag_key = "CAUSILO-1.5-LIGHT"
    ag_name = "Causilo-1.5-Light"


class Prep_CAUSILO15_LIGHT(_NoAugBase, _Causilo15LightBridge):  # noqa: N801
    """Causilo 1.5 (Light) -- pretrained tabular foundation model, classification + regression."""

    pass
