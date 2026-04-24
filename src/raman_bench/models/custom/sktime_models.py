import logging

import numpy as np
from autogluon.common import space

from raman_bench.models.custom.base import BaseCustomModel

logger = logging.getLogger(__name__)

_CLF_ONLY_MSG = "{cls} only supports classification tasks (binary/multiclass)."


def _to_3d(X):
    """Reshape tabular X (n_samples, n_features) to sktime numpy3D (n_samples, 1, n_features)."""
    return X.values.reshape(X.shape[0], 1, X.shape[1])


class _SktimeMixin:
    """Shared helpers for sktime-based AutoGluon wrappers."""

    def _select_features(self, X):
        if hasattr(self, "_feature_names") and hasattr(X, "columns"):
            missing = [c for c in self._feature_names if c not in X.columns]
            if missing:
                logger.warning(
                    "%s predict: %d training column(s) missing — filling with 0: %s",
                    self.__class__.__name__,
                    len(missing),
                    missing[:5],
                )
            X = X.reindex(columns=self._feature_names, fill_value=0.0)
        return X

    def _predict(self, X, **kwargs):
        X = self._select_features(X)
        return self.model.predict(_to_3d(X))

    def _predict_proba(self, X, **kwargs):
        X = self._select_features(X)
        proba = self.model.predict_proba(_to_3d(X))
        if self.problem_type == "binary":
            return proba[:, 1]
        return proba


class RocketModel(_SktimeMixin, BaseCustomModel):
    """AutoGluon-compatible ROCKET classifier for Raman spectra.

    Applies random convolutional kernels (ROCKET / MiniRocket / MultiRocket)
    to extract features, then trains a ridge classifier.  Extremely fast and
    competitive with the SOTA on most UCR benchmarks.

    Classification-only (binary and multiclass).

    Reference:
        Dempster, A., Petitjean, F., & Webb, G. I. (2020). ROCKET:
        exceptionally fast and accurate time series classification using
        random convolutional kernels. Data Mining and Knowledge Discovery,
        34(5), 1454–1495. https://doi.org/10.1007/s10618-020-00701-z
    """

    @classmethod
    def supported_problem_types(cls):
        return ["binary", "multiclass"]

    def _set_default_params(self):
        default_params = {
            "rocket_transform": "minirocket",
            "num_kernels": 10_000,
        } | getattr(self, "_fixed_params", {})
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    def _get_default_searchspace(self):
        return {
            "rocket_transform": space.Categorical("rocket", "minirocket", "multirocket"),
            "num_kernels": space.Categorical(2_000, 5_000, 10_000),
        }

    def _fit(self, X, y, **kwargs):
        if self.problem_type not in ("multiclass", "binary"):
            raise ValueError(_CLF_ONLY_MSG.format(cls="RocketModel"))

        try:
            from sktime.classification.kernel_based import RocketClassifier
        except ImportError:
            raise ImportError("RocketModel requires sktime. Install with: pip install sktime")

        params = self._get_model_params()
        self._log_params(params)
        self._feature_names = list(X.columns)

        self.model = RocketClassifier(
            rocket_transform=params["rocket_transform"],
            num_kernels=params["num_kernels"],
            n_jobs=1,
        )
        self.model.fit(_to_3d(X), np.array(y))


class ArsenalModel(_SktimeMixin, BaseCustomModel):
    """AutoGluon-compatible Arsenal classifier for Raman spectra.

    An ensemble of multiple ROCKET transforms trained with RidgeClassifierCV
    and combined via cross-validation-weighted voting.  Unlike the plain
    RocketClassifier, Arsenal produces well-calibrated probability estimates
    via ``predict_proba``.  It is one of the four components inside
    HIVE-COTE 2.0, the current SOTA on the UCR archive.

    Classification-only (binary and multiclass).

    Reference:
        Middlehurst, M., Large, J., Flynn, M., Lines, J., Bagnall, A. (2021).
        HIVE-COTE 2.0: a new meta ensemble for time series classification.
        Machine Learning, 110(11), 3211–3243.
        https://doi.org/10.1007/s10994-021-06055-9
    """

    @classmethod
    def supported_problem_types(cls):
        return ["binary", "multiclass"]

    def _set_default_params(self):
        default_params = {
            "num_kernels": 2_000,
            "n_estimators": 25,
            "rocket_transform": "rocket",
        } | getattr(self, "_fixed_params", {})
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    def _get_default_searchspace(self):
        return {
            "num_kernels": space.Categorical(2_000, 5_000, 10_000),
            "n_estimators": space.Categorical(10, 25, 50),
            "rocket_transform": space.Categorical("rocket", "minirocket", "multirocket"),
        }

    def _fit(self, X, y, time_limit=None, **kwargs):
        if self.problem_type not in ("multiclass", "binary"):
            raise ValueError(_CLF_ONLY_MSG.format(cls="ArsenalModel"))

        try:
            from sktime.classification.kernel_based import Arsenal
        except ImportError:
            raise ImportError("ArsenalModel requires sktime. Install with: pip install sktime")

        params = self._get_model_params()
        self._log_params(params)
        self._feature_names = list(X.columns)

        time_limit_minutes = (time_limit / 60.0 * 0.9) if time_limit is not None else 0.0

        self.model = Arsenal(
            num_kernels=params["num_kernels"],
            n_estimators=params["n_estimators"],
            rocket_transform=params["rocket_transform"],
            time_limit_in_minutes=time_limit_minutes,
            n_jobs=1,
        )
        self.model.fit(_to_3d(X), np.array(y))
