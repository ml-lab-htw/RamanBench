"""Preprocessing-restriction / HPO hyperparameter builder for ``Prep_*`` model classes.

Used by ``scripts/run_experiment.py`` (the TabArena-based benchmark runner) to
build the exact ``prep_*_enabled``/``prep_*`` hyperparameters for a single
model class before handing it to AutoGluon.

See Also
--------
- raman-bench: https://github.com/ml-lab-htw/RamanBench
- AutoGluon: https://auto.gluon.ai
"""

import logging

try:
    from autogluon.common.space import Space
except ImportError as _ag_err:
    raise ImportError(
        "raman_bench.model requires autogluon. Install with: pip install 'raman-bench[autogluon]'"
    ) from _ag_err
from raman_bench.models.custom.base import BaseRamanEstimator as BaseCustomModel
from raman_bench.preprocessing.mixin import (
    STEP_ENABLED_PARAMS,
    RamanPreprocessingMixin,
    build_restricted_searchspace,
)

logger = logging.getLogger(__name__)


def build_prep_model_hyperparameters(
    cls,
    cfg: dict,
    *,
    preprocessing_params: dict | None = None,
    optimize: bool = False,
    model_extra_params: dict | None = None,
) -> dict:
    """Apply the preprocessing-restriction / override / (optional) HPO logic for one
    ``Prep_*`` model class, returning the hyperparameters dict AutoGluon should fit it with.

    Used by ``scripts/run_experiment.py`` (driven by the k-fold orchestrator)
    to build the ``prep_*_enabled``/``prep_*`` hyperparameters for a single
    model class before handing it to AutoGluon. Precedence rules:
    restriction enable/disable > preprocessing_params overrides > class defaults;
    optimize=True additionally folds in the model's own + preprocessing's HPO
    search space.

    Parameters
    ----------
    cls : type
        A ``Prep_*`` model class (subclass of ``RamanPreprocessingMixin``).
    cfg : dict
        That class's base hyperparameters dict, as produced by
        :func:`~raman_bench.preprocessing.wrapped_models.create_preprocessed_hyperparameters`
        -- may carry a ``"_prep_restriction"`` key (the step-level enable/disable dict,
        i.e. the config's ``preprocessing_config``); this key is consumed (popped) here,
        not passed through to AutoGluon.
    preprocessing_params : dict | None
        Flat ``param_name -> value`` override dict (the config's ``preprocessing_params``).
    optimize : bool
        Whether to fold in the model's own + preprocessing's HPO search space (``Space``
        objects) instead of stripping them for a fixed-hyperparameter run.
    model_extra_params : dict | None
        Extra parameters forwarded to custom neural-network models only.
    """
    sklearn_cls = getattr(cls, "_sklearn_cls", None)
    is_custom = issubclass(cls, BaseCustomModel) or (
        sklearn_cls is not None and issubclass(sklearn_cls, BaseCustomModel)
    )
    extra = (model_extra_params or {}) if is_custom else {}
    merged = {**cfg, **extra}
    restriction = merged.pop("_prep_restriction", None)

    if restriction and restriction.get("standard_scaling"):
        merged["prep_scaling_enabled"] = True

    if restriction is not None:
        for step_key, enabled_param in STEP_ENABLED_PARAMS.items():
            want = restriction.get(step_key)
            if want is None:
                continue
            if not want:
                merged[enabled_param] = False
            elif step_key != "augmentation" or getattr(cls, "_supports_augmentation", False):
                merged[enabled_param] = True

    restriction_owned_enabled_params = set()
    if restriction is not None:
        for step_key, enabled_param in STEP_ENABLED_PARAMS.items():
            if restriction.get(step_key) is not None:
                restriction_owned_enabled_params.add(enabled_param)

    if preprocessing_params:
        for param_name, value in preprocessing_params.items():
            if param_name in restriction_owned_enabled_params:
                logger.warning(
                    "preprocessing_params override for %r ignored: "
                    "preprocessing_config already decides this step's "
                    "enabled state.",
                    param_name,
                )
                continue
            merged[param_name] = value

    if optimize:
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
        # Re-pin preprocessing_params overrides so they win over the
        # HPO search space too (a Space object for the same param
        # would otherwise silently clobber the fixed override just
        # applied above, since search_space is merged in last).
        if preprocessing_params:
            for param_name, value in preprocessing_params.items():
                if param_name in restriction_owned_enabled_params:
                    continue
                merged[param_name] = value
        return merged
    else:
        return {k: v for k, v in merged.items() if not isinstance(v, Space)}
