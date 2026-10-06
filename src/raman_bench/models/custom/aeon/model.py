"""aeon time-series classifiers: HIVE-COTE 2 and its four components.

HIVE-COTE 2 (Middlehurst et al., 2021) is a weighted ensemble of four classifiers,
each built on a different representation of the series:

- STC (Shapelet Transform Classifier): discriminative subsequences,
- DrCIF (Diverse Representation Canonical Interval Forest): interval features,
- Arsenal: an ensemble of ROCKET classifiers,
- TDE (Temporal Dictionary Ensemble): bag-of-words patterns.

HC2 weights each component's probabilities by its train-set accuracy estimate to the
power 4 (CAWPE). Fitting it in one go is slow, so it is run two ways here:

- ``HIVECOTEV2``: aeon's ``HIVECOTEV2`` as one model, under a time contract derived
  from the fit's time limit (the scope file gives it a longer budget than the protocol).
- ``STC``/``DRCIF``/``ARSENAL``/``TDE``: each component as its own model on the
  normal protocol, with HC2's default component settings. Their stored out-of-fold
  and test predictions are combined afterwards into HC2 with
  ``scripts/assemble_hivecote.py``, using the out-of-fold accuracy as the train
  estimate.

**Classification only**: these keys are in ``CLASSIFICATION_ONLY_MODELS``
(``preprocessing/wrapped_models.py``); ``fit`` also raises on a continuous target.

aeon is imported inside ``fit`` so the registry imports without it.
"""

from __future__ import annotations

import importlib

import numpy as np
from sklearn.base import BaseEstimator

from raman_bench.preprocessing.bridge_bases import SklearnAutoGluonBridge, _NoAugBase

# HIVECOTEV2's own component defaults (aeon 1.6, HIVECOTEV2._DEFAULT_*), so a component
# run standalone is the same classifier HC2 builds internally. Under a time contract
# aeon keeps adding ensemble members until the time is up (Arsenal up to 100, STC and
# TDE without limit), which makes prediction slow; the contract_max_* caps hold each
# component at its default size, so the contract can only make it smaller.
HC2_COMPONENT_PARAMS = {
    "STC": {"n_shapelet_samples": 10000, "contract_max_n_shapelet_samples": 10000},
    "DRCIF": {"n_estimators": 500, "contract_max_n_estimators": 500},
    "ARSENAL": {"n_kernels": 2000, "n_estimators": 25, "contract_max_n_estimators": 25},
    "TDE": {
        "n_parameter_samples": 250,
        "max_ensemble_size": 50,
        "randomly_selected_params": 50,
        "contract_max_n_parameter_samples": 250,
    },
}

_COMPONENT_CLASSES = {
    "STC": ("aeon.classification.shapelet_based", "ShapeletTransformClassifier"),
    "DRCIF": ("aeon.classification.interval_based", "DrCIFClassifier"),
    "ARSENAL": ("aeon.classification.convolution_based", "Arsenal"),
    "TDE": ("aeon.classification.dictionary_based", "TemporalDictionaryEnsemble"),
}

# Share of the fit's time limit given to aeon's own time contract. The rest covers
# AutoGluon's out-of-fold prediction of the bag child, which the contract does not
# include: AutoGluon aborts the whole bag if its first child takes more than a third
# of the bag's budget.
_CONTRACT_FRACTION = 0.5


def _make_estimator(estimator: str, time_limit_in_minutes: float, n_jobs: int, random_state: int):
    if estimator == "HIVECOTEV2":
        from aeon.classification.hybrid import HIVECOTEV2

        return HIVECOTEV2(
            time_limit_in_minutes=time_limit_in_minutes, n_jobs=n_jobs, random_state=random_state
        )
    if estimator not in _COMPONENT_CLASSES:
        raise ValueError(f"Unknown aeon estimator {estimator!r}")
    module, name = _COMPONENT_CLASSES[estimator]
    cls = getattr(importlib.import_module(module), name)
    return cls(
        **HC2_COMPONENT_PARAMS[estimator],
        time_limit_in_minutes=time_limit_in_minutes,
        n_jobs=n_jobs,
        random_state=random_state,
    )


def _to_3d(X) -> np.ndarray:
    # float64: aeon's catch22 features (used by DrCIF) fail numba typing on float32.
    arr = np.asarray(X.values if hasattr(X, "values") else X, dtype=np.float64)
    return arr.reshape(arr.shape[0], 1, arr.shape[1])


class AeonClassifierModel(BaseEstimator):
    """An aeon classifier (HIVE-COTE 2 or one of its components), sklearn-compatible.

    ``estimator`` is ``"HIVECOTEV2"``, ``"STC"``, ``"DRCIF"``, ``"ARSENAL"`` or
    ``"TDE"``. Without a ``time_limit`` the classifier runs uncontracted (aeon's
    ``time_limit_in_minutes=0``).
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


class _STCBridge(_AeonBridge):
    ag_key = "STC"
    ag_name = "STC"
    _estimator_key = "STC"


class Prep_STC(_NoAugBase, _STCBridge):  # noqa: N801
    pass


class _DRCIFBridge(_AeonBridge):
    ag_key = "DRCIF"
    ag_name = "DRCIF"
    _estimator_key = "DRCIF"


class Prep_DRCIF(_NoAugBase, _DRCIFBridge):  # noqa: N801
    pass


class _ARSENALBridge(_AeonBridge):
    ag_key = "ARSENAL"
    ag_name = "ARSENAL"
    _estimator_key = "ARSENAL"


class Prep_ARSENAL(_NoAugBase, _ARSENALBridge):  # noqa: N801
    pass


class _TDEBridge(_AeonBridge):
    ag_key = "TDE"
    ag_name = "TDE"
    _estimator_key = "TDE"


class Prep_TDE(_NoAugBase, _TDEBridge):  # noqa: N801
    pass
