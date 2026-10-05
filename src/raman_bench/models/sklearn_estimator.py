"""AutoGluon wrapper for any scikit-learn estimator, used by :func:`raman_bench.evaluate.evaluate_estimator`.

Unlike the registered models (``models/custom/<key>/``), the estimator is not fixed by the
class: it comes in as the ``classifier``/``regressor`` hyperparameter, and each fit uses a
fresh ``sklearn.base.clone`` of the one matching the task's problem type. RamanBench
preprocessing is off by default, as for every model; ``prep_*`` hyperparameters switch
steps on.
"""

from __future__ import annotations

import numpy as np

from raman_bench.preprocessing.bridge_bases import SklearnAutoGluonBridge, _NoAugBase


class _EstimatorBridge(SklearnAutoGluonBridge):
    def _fit(self, X, y, time_limit=None, **kwargs):
        from sklearn.base import clone

        X = self.preprocess(X, y=y)
        X_np = X.values.astype(np.float32) if hasattr(X, "values") else np.asarray(X, dtype=np.float32)
        y_arr = y.values if hasattr(y, "values") else np.asarray(y)
        key = "regressor" if self.problem_type == "regression" else "classifier"
        estimator = self._get_model_params().get(key)
        if estimator is None:
            raise ValueError(f"No {key} given for a {self.problem_type} task")
        self._estimator = clone(estimator)
        self._estimator.fit(X_np, y_arr)

    def _predict_proba(self, X, **kwargs):
        pred = super()._predict_proba(X, **kwargs)
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
