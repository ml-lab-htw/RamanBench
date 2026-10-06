"""aeon's HIVE-COTE 2 time-series classifier.

HIVE-COTE 2 (Middlehurst et al., 2021) is a weighted ensemble of four classifiers,
each built on a different representation of the series:

- STC (Shapelet Transform Classifier): discriminative subsequences,
- DrCIF (Diverse Representation Canonical Interval Forest): interval features,
- Arsenal: an ensemble of ROCKET classifiers,
- TDE (Temporal Dictionary Ensemble): bag-of-words patterns.

HC2 weights each component's probabilities by its train-set accuracy estimate to the
power 4 (CAWPE). ``HIVECOTEV2`` runs aeon's ``HIVECOTEV2`` as one model, under a time
contract derived from the fit's time limit (the scope file gives it a longer budget
than the protocol).

**Classification only**: the key is in ``CLASSIFICATION_ONLY_MODELS``
(``preprocessing/wrapped_models.py``); ``fit`` also raises on a continuous target.

aeon is imported inside ``fit`` so the registry imports without it.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator

from raman_bench.preprocessing.bridge_bases import SklearnAutoGluonBridge, _NoAugBase

# Share of the fit's time limit given to aeon's own time contract. The rest covers
# AutoGluon's out-of-fold prediction of the bag child, which the contract does not
# include: AutoGluon aborts the whole bag if its first child takes more than a third
# of the bag's budget.
_CONTRACT_FRACTION = 0.5


def _make_estimator(estimator: str, time_limit_in_minutes: float, n_jobs: int, random_state: int):
    if estimator != "HIVECOTEV2":
        raise ValueError(f"Unknown aeon estimator {estimator!r}")
    from aeon.classification.hybrid import HIVECOTEV2

    return HIVECOTEV2(time_limit_in_minutes=time_limit_in_minutes, n_jobs=n_jobs, random_state=random_state)


def _to_3d(X) -> np.ndarray:
    # float64: aeon's catch22 features (used by DrCIF) fail numba typing on float32.
    arr = np.asarray(X.values if hasattr(X, "values") else X, dtype=np.float64)
    return arr.reshape(arr.shape[0], 1, arr.shape[1])


class AeonClassifierModel(BaseEstimator):
    """An aeon classifier (HIVE-COTE 2), sklearn-compatible.

    ``estimator`` is ``"HIVECOTEV2"``. Without a ``time_limit`` the classifier runs
    uncontracted (aeon's ``time_limit_in_minutes=0``).
    """

    def __init__(self, estimator="HIVECOTEV2", n_jobs=1, random_state=0):
        self.estimator = estimator
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, X, y, time_limit=None):
        y_arr = np.asarray(y)
        if np.issubdtype(y_arr.dtype, np.floating):
            raise ValueError("AeonClassifierModel is classification-only; got a continuous target.")
        minutes = 0 if time_limit is None else max(time_limit * _CONTRACT_FRACTION / 60, 0.1)
        self.model_ = _make_estimator(self.estimator, minutes, self.n_jobs, self.random_state)
        self.model_.fit(_to_3d(X), y_arr)
        self.classes_ = self.model_.classes_
        return self

    def predict(self, X):
        return self.model_.predict(_to_3d(X))

    def predict_proba(self, X):
        return self.model_.predict_proba(_to_3d(X))


class _AeonBridge(SklearnAutoGluonBridge):
    """Fits :class:`AeonClassifierModel` with the fold's CPUs and time limit."""

    _sklearn_cls = AeonClassifierModel
    _estimator_key: str = None  # set per subclass

    def _fit(self, X, y, time_limit=None, num_cpus=1, **kwargs):
        X = self.preprocess(X, y=y)
        params = {
            k: v for k, v in self._get_model_params().items() if not k.startswith(("ag.", "_"))
        }
        params.setdefault("random_state", 0)
        self._estimator = AeonClassifierModel(
            estimator=self._estimator_key, n_jobs=max(int(num_cpus), 1), **params
        )
        self._estimator.fit(X, np.asarray(y), time_limit=time_limit)

    def _predict_proba(self, X, **kwargs):
        X = self.preprocess(X, **kwargs)
        raw = self._estimator.predict_proba(X)
        # Columns follow the classes this bag child saw; place them in AutoGluon's
        # 0..num_classes-1 label order (a child can miss a rare class).
        proba = np.zeros((raw.shape[0], self.num_classes))
        proba[:, np.asarray(self._estimator.classes_, dtype=int)] = raw
        if self.problem_type == "binary":
            return proba[:, 1]
        return proba


class _HIVECOTEV2Bridge(_AeonBridge):
    ag_key = "HIVECOTEV2"
    ag_name = "HIVECOTEV2"
    _estimator_key = "HIVECOTEV2"


class Prep_HIVECOTEV2(_NoAugBase, _HIVECOTEV2Bridge):  # noqa: N801
    pass
