"""AutoGluon wrapper for any scikit-learn estimator, used by :func:`raman_bench.evaluate.evaluate_estimator`.

Unlike the registered models (``models/custom/<key>/``), the estimator is not fixed by the
class: it comes in as the ``classifier``/``regressor`` hyperparameter, and each fit uses a
fresh ``sklearn.base.clone`` of the one matching the task's problem type. RamanBench
preprocessing is off by default, as for every model; ``prep_*`` hyperparameters switch
steps on.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd

from raman_bench.preprocessing.bridge_bases import SklearnAutoGluonBridge, _NoAugBase
from raman_bench.splitting import GROUP_COL


def row_groups(X) -> np.ndarray | None:
    """The group id of every row of *X*, or ``None`` when the task has no group column.

    The column is in X when it reaches the model (``RamanBenchTaskWrapper(keep_group_col=True)``); the
    preprocessing drops it, so it has to be read first.
    """
    if isinstance(X, pd.DataFrame) and GROUP_COL in X.columns:
        return X[GROUP_COL].to_numpy()
    return None


def accepts_groups(method) -> bool:
    """Whether the estimator method has a ``groups`` parameter, so it can use the replicate structure."""
    return "groups" in inspect.signature(method).parameters


class _EstimatorBridge(SklearnAutoGluonBridge):
    def _fit(self, X, y, time_limit=None, **kwargs):
        from sklearn.base import clone

        groups = row_groups(X)
        X = self.preprocess(X, y=y)
        X_np = X.values.astype(np.float32) if hasattr(X, "values") else np.asarray(X, dtype=np.float32)
        y_arr = y.values if hasattr(y, "values") else np.asarray(y)
        key = "regressor" if self.problem_type == "regression" else "classifier"
        estimator = self._get_model_params().get(key)
        if estimator is None:
            raise ValueError(f"No {key} given for a {self.problem_type} task")
        self._estimator = clone(estimator)
        # An estimator with a ``groups`` argument in fit/predict (the hosted TabPFN) gets the replicate
        # structure on the grouped tasks; every other estimator is called as before.
        if groups is not None and accepts_groups(self._estimator.fit):
            self._estimator.fit(X_np, y_arr, groups=groups)
        else:
            self._estimator.fit(X_np, y_arr)

    def _predict_proba(self, X, **kwargs):
        groups = row_groups(X)
        method = self._estimator.predict if self.problem_type == "regression" else self._estimator.predict_proba
        if groups is None or not accepts_groups(method):
            pred = super()._predict_proba(X, **kwargs)
        else:
            Xp = self.preprocess(X, **kwargs)
            X_np = Xp.values.astype(np.float32) if hasattr(Xp, "values") else np.asarray(Xp, dtype=np.float32)
            pred = method(X_np, groups=groups)
            if self.problem_type == "binary":
                pred = pred[:, 1] if pred.ndim == 2 else pred
        # Some regressors (e.g. PLSRegression) predict a column vector.
        return np.asarray(pred).ravel() if self.problem_type == "regression" else pred


class Prep_SKLEARN(_NoAugBase, _EstimatorBridge):  # noqa: N801
    """A scikit-learn ``classifier``/``regressor`` passed in as hyperparameters."""

    ag_key = "SKLEARN"
    ag_name = "SKLEARN"


def estimator_model_cls(name: str) -> type[Prep_SKLEARN]:
    """``Prep_SKLEARN`` under the model key *name*.

    TabArena names result directories after the class's ``ag_name``
    (``<name>_c1_BAG_L1``) and ranks results by its ``ag_key``, so each evaluated model
    gets its own subclass. It is registered in this module so AutoGluon can pickle it.
    """
    attr = "Prep_SKLEARN__" + "".join(c if c.isalnum() else "_" for c in name)
    cls = globals().get(attr)
    if cls is None:
        cls = type(attr, (Prep_SKLEARN,), {"__module__": __name__, "ag_key": name, "ag_name": name})
        globals()[attr] = cls
    return cls
