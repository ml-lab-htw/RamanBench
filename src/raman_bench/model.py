"""AutoGluon-based model wrapper for the benchmark pipeline.

The :class:`AutoGluonModel` class wraps AutoGluon's ``TabularPredictor`` with
a simplified fit/predict API and adds:

- Automatic Raman preprocessing (via :class:`~raman_bench.preprocessing.mixin.RamanPreprocessingMixin`)
- Target rescaling for regression (zero-mean / unit-variance)
- GPU/CPU resource detection (SLURM-aware)
- Fit statistics extraction

See Also
--------
- raman-bench: https://github.com/ml-lab-htw/RamanBench
- AutoGluon: https://auto.gluon.ai
"""

import logging
import os
import shutil
import uuid
from typing import Any

import torch
from autogluon.common import TabularDataset
from autogluon.common.space import Space
from autogluon.tabular import TabularPredictor
from pandas import DataFrame
from raman_data import TASK_TYPE

from raman_bench.models.custom.base import BaseCustomModel
from raman_bench.preprocessing.mixin import RamanPreprocessingMixin, build_restricted_searchspace
from raman_bench.preprocessing.wrapped_models import create_preprocessed_hyperparameters

logger = logging.getLogger(__name__)


def _build_foundation_hyperparameters(num_gpus: int) -> dict:
    """Build hyperparameters for the AUTOGLUON meta-model using the zeroshot portfolio.

    Uses AutoGluon's ``zeroshot_2025_12_18_gpu`` portfolio (TabPFN v2, TABDPT,
    TABM, TABICL, MITRA, GBM, CatBoost) and adds REALTABPFN-V2.5.
    """
    from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config
    from autogluon.tabular.registry import ag_model_registry

    hp: dict = get_hyperparameter_config("zeroshot_2025_12_18_gpu").copy()
    try:
        cls_v25 = ag_model_registry.key_to_cls("REALTABPFN-V2.5")
        gpu_arg = {"ag.num_gpus": 1} if num_gpus > 0 else {}
        hp[cls_v25] = [{**gpu_arg}]
    except Exception:
        pass
    return hp


class AutoGluonModel:
    """Wrapper around AutoGluon's ``TabularPredictor`` with Raman preprocessing.

    Exposes a ``fit / predict`` API compatible with the benchmark pipeline.

    Parameters
    ----------
    models : list[str]
        Model names (e.g. ``["GBM", "PLS", "RAMANNET"]``).  Pass
        ``["AUTOGLUON"]`` to run AutoGluon natively with its full preset.
    ensemble : bool
        Enable AutoGluon bagging/stacking (default ``True``).  Automatically
        disabled for tiny or singleton-class datasets.
    optimize : bool
        Enable HPO via Bayesian search (default ``True``).  Mutually exclusive
        with bagging.
    num_hpo_trials : int
        Maximum HPO trials when *optimize* is ``True`` (default 0 = unlimited).
    task_type : TASK_TYPE
        Task type for metric selection and problem type mapping.
    autogluon_time_limit : int
        Total training time budget in seconds (default 60).
    autogluon_presets : str
        AutoGluon preset string (default ``"best_quality"``).
    autogluon_path : str | None
        Base directory for AutoGluon model artefacts.
    preprocessing_config : dict | None
        Preprocessing restriction dict from the benchmark config.
    model_extra_params : dict | None
        Extra parameters forwarded to custom neural-network models
        (e.g. ``{"per_epoch_augmentation": True}``).

    Examples
    --------
    ::

        model = AutoGluonModel(
            models=["GBM"],
            task_type=TASK_TYPE.Regression,
            autogluon_time_limit=300,
        )
        model.fit(train_df)
        predictions = model.predict(test_df)
    """

    def __init__(
        self,
        models: list[str],
        ensemble: bool = True,
        optimize: bool = True,
        num_hpo_trials: int = 0,
        task_type: TASK_TYPE = TASK_TYPE.Regression,
        autogluon_time_limit: int = 60,
        autogluon_presets: str = "best_quality",
        autogluon_path: str | None = None,
        preprocessing_config: dict | None = None,
        model_extra_params: dict | None = None,
    ) -> None:
        self.task_type = task_type
        self.preprocessing_config = preprocessing_config

        self._autogluon_native = "AUTOGLUON" in [m.upper() for m in models]

        if task_type == TASK_TYPE.Regression:
            self.metric: str = "rmse"
            self.problem_type: str = "regression"
        elif task_type == TASK_TYPE.Classification:
            self.metric = "f1_macro"
            self.problem_type = "multiclass"
        else:
            raise ValueError(f"Unsupported task type: {task_type}")

        self.custom_models = (
            {}
            if self._autogluon_native
            else create_preprocessed_hyperparameters(models, prep_restriction=preprocessing_config)
        )
        self.builtin_models: list = []

        self.label = "target"
        self.autogluon_path = os.path.join(
            autogluon_path or os.path.join(".cache", "autogluon"),
            uuid.uuid4().hex,
        )

        self.time_limit = autogluon_time_limit
        self.presets: Any = autogluon_presets
        self.ensemble = ensemble
        self.optimize = optimize
        self.num_hpo_trials = num_hpo_trials
        self.model_extra_params = model_extra_params or {}

        self._target_mean: float | None = None
        self._target_std: float | None = None

        if os.path.exists(self.autogluon_path):
            shutil.rmtree(self.autogluon_path)

        self.predictor = TabularPredictor(
            label=self.label,
            eval_metric=self.metric,
            problem_type=self.problem_type,
            path=self.autogluon_path,
        )

    def fit(
        self, data_train: DataFrame, raise_on_no_models_fitted: bool = True
    ) -> TabularPredictor:
        """Fit the predictor on *data_train*.

        Parameters
        ----------
        data_train : DataFrame
            Training data.  The last column must be ``"target"``.
        raise_on_no_models_fitted : bool
            Passed directly to AutoGluon (default ``True``).

        Returns
        -------
        autogluon.tabular.TabularPredictor
        """
        hyperparameters = {}
        for cls, cfg in self.custom_models.items():
            extra = self.model_extra_params if issubclass(cls, BaseCustomModel) else {}
            merged = {**cfg, **extra}
            restriction = merged.pop("_prep_restriction", None)

            if restriction and restriction.get("standard_scaling"):
                merged["prep_scaling_enabled"] = True

            if self.optimize:
                search_space = build_restricted_searchspace(restriction)
                for base_cls in cls.__mro__[1:]:
                    if issubclass(base_cls, RamanPreprocessingMixin):
                        continue
                    if hasattr(base_cls, "_get_default_searchspace"):
                        try:
                            stub = object.__new__(base_cls)
                            stub.params = {}
                            model_own_ss = base_cls._get_default_searchspace(stub)
                            search_space = {**model_own_ss, **search_space}
                        except Exception:
                            pass
                        break
                merged = {**merged, **search_space}
                hyperparameters[cls] = merged if merged else "default"
            else:
                hyperparameters[cls] = {k: v for k, v in merged.items() if not isinstance(v, Space)}

        num_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))
        num_gpus = min(1, torch.cuda.device_count())

        fit_args = {
            "time_limit": self.time_limit,
            "presets": self.presets,
            "raise_on_no_models_fitted": raise_on_no_models_fitted,
            "verbosity": 1,
            "ag_args_fit": {"ag.max_memory_usage_ratio": 100.0},
            "ag_args_ensemble": {"fold_fitting_strategy": "sequential_local"},
            "num_gpus": num_gpus,
            "num_cpus": num_cpus,
        }

        if self._autogluon_native:
            fit_args["hyperparameters"] = _build_foundation_hyperparameters(num_gpus)
        else:
            fit_args["hyperparameters"] = hyperparameters

        if not self._autogluon_native:
            if not self.ensemble:
                fit_args["num_bag_folds"] = 0
                fit_args["num_stack_levels"] = 0
            elif self.optimize:
                fit_args["num_bag_folds"] = 0
                fit_args["num_stack_levels"] = 0

        if self.optimize:
            fit_args["hyperparameter_tune_kwargs"] = {
                "searcher": "bayes",
                "scheduler": "local",
                "resources": {
                    "num_cpus": max(1, num_cpus // 4),
                    "num_gpus": 1 if num_gpus > 0 else 0,
                },
            }
            if self.num_hpo_trials > 0:
                fit_args["hyperparameter_tune_kwargs"]["num_trials"] = self.num_hpo_trials
        else:
            fit_args["hyperparameter_tune_kwargs"] = None

        tabular_data = TabularDataset(data_train)
        n_samples = len(tabular_data)

        # Disable bagging for tiny datasets or singleton-class datasets to avoid
        # empty OOF folds.  best_quality uses 8-fold stratified bagging by default.
        n_bag_folds = 8
        if "num_bag_folds" not in fit_args:
            disable_reason = None
            if n_samples < 20:
                disable_reason = f"only {n_samples} training samples (threshold 20)"
            elif self.problem_type in ("binary", "multiclass"):
                min_class = int(tabular_data[self.label].value_counts().min())
                if min_class < n_bag_folds:
                    disable_reason = f"min class count {min_class} < {n_bag_folds} folds"
            if disable_reason:
                fit_args["num_bag_folds"] = 0
                fit_args["num_stack_levels"] = 0
                logger.warning("Disabled bagging: %s.", disable_reason)

        # Rescale regression target to zero-mean / unit-variance for stability
        if self.problem_type == "regression":
            self._target_mean = float(tabular_data[self.label].mean())
            self._target_std = float(tabular_data[self.label].std()) or 1.0
            tabular_data = tabular_data.copy()
            tabular_data[self.label] = (
                tabular_data[self.label] - self._target_mean
            ) / self._target_std

        self.predictor.fit(tabular_data, **fit_args)

        try:
            self._leaderboard = self.predictor.leaderboard(silent=True)
        except Exception:
            self._leaderboard = None

        return self.predictor

    def predict(self, data_test: DataFrame):
        """Predict labels/values for *data_test*.

        For regression tasks, predictions are inverse-transformed back to the
        original target scale.
        """
        test_input = (
            data_test if isinstance(data_test, TabularDataset) else TabularDataset(data_test)
        )
        predictions = self.predictor.predict(test_input)

        if (
            self.problem_type == "regression"
            and self._target_mean is not None
            and self._target_std is not None
        ):
            predictions = predictions * self._target_std + self._target_mean

        return predictions

    def get_fit_stats(self) -> dict:
        """Return training statistics from AutoGluon's leaderboard.

        Returns
        -------
        dict
            Keys: ``n_models_trained``, ``n_base_models``,
            ``ag_total_fit_time_s``, ``ag_time_per_model_s``.
            Empty dict when no leaderboard is available.
        """
        lb = getattr(self, "_leaderboard", None)
        if lb is None or lb.empty:
            return {}

        n_total = len(lb)
        is_ensemble = lb["model"].str.contains("Ensemble|Stack", case=False, na=False)
        n_base = int((~is_ensemble).sum())
        ag_total = float(lb["fit_time"].sum()) if "fit_time" in lb.columns else None
        ag_per = (ag_total / n_base) if (ag_total is not None and n_base > 0) else None

        return {
            "n_models_trained": n_total,
            "n_base_models": n_base,
            "ag_total_fit_time_s": round(ag_total, 3) if ag_total is not None else None,
            "ag_time_per_model_s": round(ag_per, 3) if ag_per is not None else None,
        }
