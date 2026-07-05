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
from pandas import DataFrame
from raman_data import TASK_TYPE

try:
    from autogluon.common import TabularDataset
    from autogluon.common.space import Space
    from autogluon.tabular import TabularPredictor
except ImportError as _ag_err:
    raise ImportError(
        "raman_bench.model requires autogluon. "
        "Install with: pip install -r requirements-autogluon-fork.txt && pip install 'raman-bench[autogluon]'"
    ) from _ag_err
from raman_bench.models.custom.base import BaseRamanEstimator as BaseCustomModel
from raman_bench.preprocessing.mixin import (
    STEP_ENABLED_PARAMS,
    RamanPreprocessingMixin,
    build_restricted_searchspace,
)
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

    # Pin CatBoost to CPU. Its GPU implementation competes with the foundation
    # models (TabPFN/TabICL/TabDPT) for VRAM and triggered CUDA out-of-memory
    # crashes that aborted the whole run (e.g. during refit_full). CatBoost is
    # fast on CPU, so the only cost is a small bit of CPU time. The portfolio is
    # keyed by short model strings ("CAT"), so we patch that key directly.
    try:
        cat_cfgs = hp.get("CAT")
        if cat_cfgs is not None:
            if isinstance(cat_cfgs, dict):
                cat_cfgs = [cat_cfgs]
            patched = []
            for cfg in cat_cfgs:
                ag_args_fit = {**cfg.get("ag_args_fit", {}), "ag.num_gpus": 0}
                patched.append({**cfg, "ag_args_fit": ag_args_fit})
            hp["CAT"] = patched
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
            # HPO trial scoring in autogluon_fork passes the (N, K) proba
            # matrix into sklearn f1_score without argmax-converting it to
            # labels, which raises a length-mismatch error on multi-class
            # datasets. log_loss is proba-native and bypasses that codepath.
            # Headline F1 is recomputed downstream from predictions, so this
            # only affects model selection during HPO.
            self.metric = "log_loss" if optimize else "f1_macro"
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
            sklearn_cls = getattr(cls, "_sklearn_cls", None)
            is_custom = issubclass(cls, BaseCustomModel) or (
                sklearn_cls is not None and issubclass(sklearn_cls, BaseCustomModel)
            )
            extra = self.model_extra_params if is_custom else {}
            merged = {**cfg, **extra}
            restriction = merged.pop("_prep_restriction", None)

            if restriction and restriction.get("standard_scaling"):
                merged["prep_scaling_enabled"] = True

            # Enforce restriction for each step:
            #  - Disabled (False) → force the enabled flag off.  This overrides
            #    model-class defaults from _set_default_params (e.g. Prep_PLS
            #    defaults prep_bl_enabled=True) so HPO cannot silently activate
            #    a step the user turned off.
            #  - Augmentation enabled (True) + model supports it → force on.
            #    _NoAugBase defaults prep_aug_enabled=False; without this a
            #    capable model (NN_TORCH, FASTAI, REALMLP) would never see
            #    augmented data even when the config enables it.
            if restriction is not None:
                for step_key, enabled_param in STEP_ENABLED_PARAMS.items():
                    if not restriction.get(step_key, True):
                        merged[enabled_param] = False
                    elif (
                        step_key == "augmentation"
                        and restriction.get("augmentation", False)
                        and getattr(cls, "_supports_augmentation", False)
                    ):
                        merged[enabled_param] = True

            if self.optimize:
                optimize_prep = getattr(cls, "_optimize_preprocessing", False)
                search_space = build_restricted_searchspace(restriction) if optimize_prep else {}
                # Strip augmentation params from the HPO search space for models
                # that don't support augmentation.  build_restricted_searchspace
                # adds prep_aug_* whenever augmentation=True in the config; without
                # this filter, HPO would sample aug params for PLS/KNN/LR even
                # though _supports_augmentation=False.
                if not getattr(cls, "_supports_augmentation", False):
                    search_space = {
                        k: v for k, v in search_space.items() if not k.startswith("prep_aug_")
                    }
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
                hyperparameters[cls] = merged
            else:
                hyperparameters[cls] = {k: v for k, v in merged.items() if not isinstance(v, Space)}

        num_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))
        num_gpus = min(1, torch.cuda.device_count())

        fit_args = {
            "time_limit": self.time_limit,
            "presets": self.presets,
            "raise_on_no_models_fitted": raise_on_no_models_fitted,
            "verbosity": 1,
            "ag_args_fit": {"ag.max_memory_usage_ratio": 100.0, "time_limit": self.time_limit},
            "ag_args_ensemble": {"fold_fitting_strategy": "sequential_local"},
            "num_gpus": num_gpus,
            "num_cpus": num_cpus,
            # Refit the best HPO/ensemble configuration on full train+val data
            # and use the refit model for downstream predictions.
            "refit_full": True,
            "set_best_to_refit_full": True,
            # Disable post-fit probability calibration. The `extreme_quality`
            # preset enables it, but its calibrate_model() step calls
            # get_model_best(can_infer=True) and raises "Trainer has no fit
            # models that can infer" whenever no model carries a validation
            # score — which happens on large classification datasets where HPO
            # trials exhaust the time budget (slow custom models) or the single
            # trained model's val score is not registered. Calibration is not
            # needed for the benchmark metrics, so turn it off for robustness.
            "calibrate": False,
        }

        if self._autogluon_native:
            fit_args["hyperparameters"] = _build_foundation_hyperparameters(num_gpus)
        else:
            fit_args["hyperparameters"] = hyperparameters

        if not self._autogluon_native:
            if not self.ensemble:
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

        # Augment small datasets so bagging has enough samples per class/total.
        # Only runs when ensemble=True (bagging intended); skipped otherwise because
        # fit_args already has num_bag_folds=0.
        n_bag_folds = 8
        if "num_bag_folds" not in fit_args:
            data_train = self._augment_for_bagging(
                data_train, self.label, self.problem_type, n_bag_folds
            )

        tabular_data = TabularDataset(data_train)
        n_samples = len(tabular_data)

        # Final safety check: if augmentation still wasn't enough, disable bagging.
        if "num_bag_folds" not in fit_args:
            disable_reason = None
            if n_samples < 20:
                disable_reason = f"only {n_samples} training samples even after augmentation"
            elif self.problem_type in ("binary", "multiclass"):
                min_class = int(tabular_data[self.label].value_counts().min())
                if min_class < n_bag_folds:
                    disable_reason = (
                        f"min class count {min_class} < {n_bag_folds} folds even after augmentation"
                    )
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

        if self._leaderboard is not None and not self._leaderboard.empty:
            lb_str = self._leaderboard[["model", "score_val", "fit_time"]].to_string(index=False)
            logger.info("AutoGluon leaderboard:\n%s", lb_str)

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

    def predict_proba(self, data_test: DataFrame):
        """Return class probabilities for *data_test* (classification only).

        Returns a DataFrame with one column per class and the same index as
        the test set. Raises ValueError for regression tasks.
        """
        if self.problem_type == "regression":
            raise ValueError("predict_proba is not available for regression tasks.")
        test_input = (
            data_test if isinstance(data_test, TabularDataset) else TabularDataset(data_test)
        )
        return self.predictor.predict_proba(test_input)

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

    @staticmethod
    def _augment_for_bagging(
        df: DataFrame,
        label: str,
        problem_type: str,
        n_bag_folds: int = 8,
    ) -> DataFrame:
        """Augment training data so every class has >= n_bag_folds samples.

        Uses the existing Raman augmentation pipeline (Gaussian noise + optional
        shift/mixup) to generate extra samples only for under-populated classes
        (classification) or tiny datasets (regression).  The noise level is kept
        very small (0.5 % of global std) so augmented samples are nearly identical
        to their originals — the goal is to prevent empty bagging folds, not to
        artificially enrich the distribution.
        """
        import numpy as np
        import pandas as pd

        from raman_bench.preprocessing.raman_preprocessing import augment_spectra

        feature_cols = [c for c in df.columns if c != label]
        X = df[feature_cols].to_numpy(dtype=float)
        y = df[label].to_numpy()
        extra_frames = []

        if problem_type in ("binary", "multiclass"):
            classes, counts = np.unique(y, return_counts=True)
            if counts.min() >= n_bag_folds:
                return df
            for cls, n_cls in zip(classes, counts):
                if n_cls >= n_bag_folds:
                    continue
                n_needed = n_bag_folds - int(n_cls)
                X_cls = X[y == cls]
                y_cls = y[y == cls]
                n_aug = int(np.ceil(n_needed / n_cls))
                X_aug, y_aug = augment_spectra(
                    X_cls, y_cls, noise_sigma=0.005, n_augments=n_aug, label_type="classification"
                )
                # augment_spectra prepends originals; slice out exactly what we need
                frame = pd.DataFrame(X_aug[n_cls : n_cls + n_needed], columns=feature_cols)
                frame[label] = y_aug[n_cls : n_cls + n_needed]
                extra_frames.append(frame)
        else:
            min_needed = max(20, 2 * n_bag_folds)
            n_samples = len(df)
            if n_samples >= min_needed:
                return df
            n_needed = min_needed - n_samples
            n_aug = int(np.ceil(n_needed / n_samples))
            X_aug, y_aug = augment_spectra(
                X, y, noise_sigma=0.005, n_augments=n_aug, label_type="regression"
            )
            frame = pd.DataFrame(X_aug[n_samples : n_samples + n_needed], columns=feature_cols)
            frame[label] = y_aug[n_samples : n_samples + n_needed]
            extra_frames.append(frame)

        if not extra_frames:
            return df

        result = pd.concat([df, *extra_frames], ignore_index=True)
        logger.info(
            "Augmented training set for bagging: %d → %d rows (need %d/class).",
            len(df),
            len(result),
            n_bag_folds,
        )
        return result
