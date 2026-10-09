"""Self-contained systems benchmarked on the v1 protocol (``scripts/run_system.py``).

A system does its own validation, ensembling and model selection, so it is not wrapped in
AutoGluon's bagging like the leaderboard models: TabArena's ``ExternalSystemExperiment``
fits it once per outer fold on the whole training split and scores its test predictions,
on the same folds and time budget as every other v1 result. This is the flavour TabArena
uses for hosted APIs.

:data:`SYSTEMS` maps a model key to its system class, the configuration it runs with and
the results directory its results land in (``results/v1/data/<result_dir>/``).
"""

from __future__ import annotations

from dataclasses import dataclass, field


def with_test_artifacts(system_cls: type, key: str, name: str, hyperparameters: dict | None = None) -> type:
    """*system_cls*, made to store its test predictions like every other v1 result.

    TabArena builds a result's ``simulation_artifacts`` (test predictions and the test rows'
    ``y_test_idx``) only for models with out-of-fold predictions, which a system doesn't have,
    so a system's result would keep just its score. The fold check
    (:func:`raman_bench.folds.check_result_folds`) and the per-fold metrics
    (:mod:`raman_bench.fold_metrics`) need those artifacts, so this subclass reports an empty
    out-of-fold set: the artifacts then hold the test predictions, and no validation ones
    (:func:`raman_bench.aggregation.aggregate_results` records no validation error for them).
    A result with predictions is read as a model config, so its ``method_metadata`` also gets
    the ``model_type`` (the model *key*, which the leaderboard uses), ``name_prefix`` (*name*,
    the results directory) and ``model_hyperparameters`` a model records, next to what the
    system reports.
    """
    import numpy as np
    import pandas as pd

    class WithTestArtifacts(system_cls):
        can_get_oof = True

        def get_oof(self) -> dict:
            # The artifact TabArena's AGWrapper gives for a fit without validation rows
            # (_simulation_artifact_without_validation); the runner adds the test predictions.
            cleaner = self.label_cleaner
            num_classes = getattr(cleaner, "num_classes", None)
            if self.problem_type == "multiclass" and num_classes and num_classes > 2:
                pred_val = pd.DataFrame(np.empty((0, num_classes), dtype=np.float32), columns=list(range(num_classes)))
            else:
                pred_val = pd.Series(np.empty(0, dtype=np.float32), dtype=np.float32)
            return {
                "pred_proba_dict_val": pred_val,
                "y_val": pd.Series(np.empty(0, dtype=np.float64 if self.problem_type == "regression" else np.int64)),
                "problem_type": self.problem_type,
                "problem_type_transform": getattr(cleaner, "problem_type_transform", self.problem_type),
                "ordered_class_labels": getattr(cleaner, "ordered_class_labels", None),
                "ordered_class_labels_transformed": getattr(cleaner, "ordered_class_labels_transformed", None),
                "num_classes": num_classes,
            }

        def get_metadata(self) -> dict:
            return {
                "model_type": key,
                "name_prefix": name,
                "model_hyperparameters": dict(hyperparameters or {}),
                **super().get_metadata(),
            }

    WithTestArtifacts.__name__ = WithTestArtifacts.__qualname__ = system_cls.__name__
    WithTestArtifacts.__module__ = __name__
    return WithTestArtifacts


def _chakra_tab_cls():
    try:  # TabArena's own copy, once the pin includes autogluon/tabarena#654
        from tabarena.systems.chakra_tab.system import ChakraTabSystemModel
    except ImportError:
        from raman_bench.systems._chakra_tab_vendored import ChakraTabSystemModel
    return ChakraTabSystemModel


def _tabfm_plus_cls():
    from tabarena.systems.tabfm_plus.system import TabFMPlusSystemModel

    return TabFMPlusSystemModel


@dataclass(frozen=True)
class System:
    key: str
    import_cls: object  # () -> the ExternalSystemModel subclass, imported on use
    result_dir: str
    hyperparameters: dict = field(default_factory=dict)
    env: tuple[str, ...] = ()  # environment variables the system needs
    notes: str = ""
    num_gpus: int = 0  # GPUs the system fits with (scripts/run_system.py --num-gpus overrides)
    # Row sample per dataset, as a model's ``model_max_train_samples_overrides`` (the smaller
    # of this and the protocol's own sample wins; the fold check accepts it as match_capped).
    max_train_samples: int | None = None

    def load_cls(self) -> type:
        """The system class, storing its test predictions like the v1 models (:func:`with_test_artifacts`)."""
        return with_test_artifacts(self.import_cls(), self.key, self.result_dir, self.hyperparameters)


SYSTEMS: dict[str, System] = {
    # YHat Labs' hosted API (closed source, paid; https://yhatlabs.com). Every fold is one
    # fit_predict request with the training table and the test features; the API key comes
    # from CHAKRA_TAB_KEY. "medium": 3-fold bagging of several pretrained tabular foundation
    # models behind the API; "full": 8-fold bagging.
    "CHAKRA-TAB": System(
        "CHAKRA-TAB",
        _chakra_tab_cls,
        "Chakra-Tab-medium",
        {"preset": "medium"},
        env=("CHAKRA_TAB_KEY",),
        notes="hosted API; the data of every fold is sent to api.yhatlabs.com",
    ),
    "CHAKRA-TAB-FULL": System(
        "CHAKRA-TAB-FULL",
        _chakra_tab_cls,
        "Chakra-Tab-full",
        {"preset": "full"},
        env=("CHAKRA_TAB_KEY",),
        notes="hosted API; the data of every fold is sent to api.yhatlabs.com",
    ),
    # TabFM run through its heavier ``ensemble`` interface (TabArena's own TabFM+ system,
    # tabarena.systems.tabfm_plus; Google, non-commercial licence). Shares TABFM's
    # checkpoint and package. TabArena ran it with a 4 h limit per task; here it gets the
    # v1 budget like every other model.
    "TABFM-PLUS": System(
        "TABFM-PLUS",
        _tabfm_plus_cls,
        "TabFM-Plus",
        {"interface": "ensemble"},
        num_gpus=1,
        # TABFM's own cap (it ran out of memory above it); TabFM+ runs the same network.
        max_train_samples=5000,
    ),
}


def get_system(key: str) -> System:
    try:
        return SYSTEMS[key.upper()]
    except KeyError:
        raise KeyError(f"Unknown system {key!r}; one of {sorted(SYSTEMS)}") from None
