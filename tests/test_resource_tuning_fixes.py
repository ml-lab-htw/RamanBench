"""Regression tests for the 3 resource-tuning fixes found during batch-3
verification of the 14-model TabArena onboarding effort:

1. TabSTAR OOMs on RamanBench's widest spectra (memory scales with feature
   count, not row count -- confirmed real: OOM-killed on an 11,084-feature
   dataset even on a ~36GB machine). Fixed with a ``max_features`` cap
   (``wrapped_models._TABSTAR_MAX_FEATURES``) plus a job-level clean skip
   (``wrapped_models.MAX_FEATURES_MODELS``, consumed by
   ``scripts/run_experiment.py::run_one()``) -- AutoGluon's own
   ``ConstraintViolationError`` skip alone isn't enough here since RamanBench's
   cluster jobs always fit exactly one model, and
   ``raise_on_no_models_fitted=True`` turns "no models fit" into a job-crashing
   ``RuntimeError`` with nothing else in the hyperparameters dict to fall back
   on.
2. EBM's default ``interactions="3x"`` triggers an ``interpret``-internal FAST
   interaction-ranking pre-scan that is NOT bounded by ``time_limit`` and
   scales combinatorially with feature count -- confirmed real (millions of
   per-pair log lines, still running after 9+ minutes on an 11,084-feature
   dataset). Fixed by disabling interactions (``interactions=0`` -- confirmed
   the only value that actually skips the ranking loop, not just a small
   positive count) for wide feature counts.
3. ``cluster/opportunistic_scheduler.py``'s ``time_limit_overrides`` mechanism
   was dataset-keyed only, unable to express "this model needs more time on
   this dataset, but other models sharing it are fine" -- extended with a
   parallel ``model_time_limit_overrides`` (model -> dataset -> seconds) layer.

See ``wrapped_models.py``'s own comment blocks (``_TABSTAR_MAX_FEATURES``,
``Prep_EBM._fit``, ``MAX_FEATURES_MODELS``) and
``cluster/opportunistic_scheduler.py::effective_time_limit`` for the full
evidence/reasoning behind each fix -- this file only locks the resulting
behavior in.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_tabstar_max_features_cap():
    """Prep_TABSTAR must reject (via AutoGluon's own ag.max_features constraint,
    which produces a clean ConstraintViolationError skip, not a crash) any
    dataset wider than the confirmed-safe cap."""
    from raman_bench.preprocessing.wrapped_models import (
        _TABSTAR_MAX_FEATURES,
        Prep_TABSTAR,
    )

    if Prep_TABSTAR is None:
        pytest.skip("TabSTARModel unavailable in this tabarena build")

    assert _TABSTAR_MAX_FEATURES > 0
    model = Prep_TABSTAR(problem_type="regression")
    aux = model._get_default_auxiliary_params()
    assert aux["max_features"] == _TABSTAR_MAX_FEATURES


def test_max_features_models_registry():
    """MAX_FEATURES_MODELS is the job-level mirror of the AutoGluon-level cap --
    keeps scripts/run_experiment.py's clean skip in sync with the real cap
    value rather than a second hardcoded number."""
    from raman_bench.preprocessing.wrapped_models import (
        _TABSTAR_MAX_FEATURES,
        MAX_FEATURES_MODELS,
    )

    assert MAX_FEATURES_MODELS["TABSTAR"] == _TABSTAR_MAX_FEATURES


def test_ebm_interactions_disabled_for_wide_features():
    """Prep_EBM._fit must force interactions=0 above the wide-feature
    threshold, and leave interpret's own default alone below it.

    Stubs out the real (expensive) EBMModel._fit so this stays a fast, real
    local check of the parameter-setting logic, not a full model fit."""
    import pandas as pd

    from raman_bench.preprocessing.wrapped_models import (
        _EBM_WIDE_FEATURE_THRESHOLD,
        Prep_EBM,
    )

    def _make_model(n_features: int) -> tuple[Prep_EBM, pd.DataFrame, pd.Series]:
        model = Prep_EBM(problem_type="regression")
        model.params = {}
        X = pd.DataFrame(
            {f"f{i}": [0.0, 1.0] for i in range(n_features)}
        )
        y = pd.Series([0.0, 1.0])
        return model, X, y

    with patch(
        "raman_bench.preprocessing.wrapped_models.EBMModel._fit", return_value=None
    ):
        wide_model, wide_x, wide_y = _make_model(_EBM_WIDE_FEATURE_THRESHOLD + 1)
        wide_model._fit(wide_x, wide_y)
        assert wide_model.params["interactions"] == 0

        narrow_model, narrow_x, narrow_y = _make_model(_EBM_WIDE_FEATURE_THRESHOLD - 1)
        narrow_model._fit(narrow_x, narrow_y)
        assert "interactions" not in narrow_model.params


def test_ebm_interactions_not_reset_if_user_disabled_already():
    """A user-supplied interactions=0 (or a fixed int already at 0) is left as
    a no-op re-set, not accidentally left at some other value."""
    import pandas as pd

    from raman_bench.preprocessing.wrapped_models import (
        _EBM_WIDE_FEATURE_THRESHOLD,
        Prep_EBM,
    )

    model = Prep_EBM(problem_type="regression")
    model.params = {"interactions": 0}
    n_features = _EBM_WIDE_FEATURE_THRESHOLD + 1
    X = pd.DataFrame({f"f{i}": [0.0, 1.0] for i in range(n_features)})
    y = pd.Series([0.0, 1.0])

    with patch(
        "raman_bench.preprocessing.wrapped_models.EBMModel._fit", return_value=None
    ):
        model._fit(X, y)
    assert model.params["interactions"] == 0


def _load_opportunistic_scheduler():
    spec = importlib.util.spec_from_file_location(
        "_opportunistic_scheduler_under_test",
        REPO_ROOT / "cluster" / "opportunistic_scheduler.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def opportunistic_scheduler():
    return _load_opportunistic_scheduler()


def test_effective_time_limit_dataset_override_unaffected(opportunistic_scheduler):
    """Pre-existing dataset-keyed override (e.g. mlrod) still applies regardless
    of which model is running -- the new model-keyed layer is additive."""
    scope = {"time_limit": 3600, "time_limit_overrides": {"mlrod": 10800}}
    chunk = [("mlrod", 0, 0, 0, 0, 10)]
    assert opportunistic_scheduler.effective_time_limit(scope, "PLS", chunk) == 10800
    assert opportunistic_scheduler.effective_time_limit(scope, "EBM", chunk) == 10800


def test_effective_time_limit_model_scoped_override(opportunistic_scheduler):
    """A model_time_limit_overrides entry only applies to its own model, not to
    other models sharing the same dataset in a different chunk."""
    scope = {
        "time_limit": 3600,
        "model_time_limit_overrides": {
            "EBM": {"microgel_synthesis": 10800},
        },
    }
    chunk = [("microgel_synthesis", 0, 0, 0, 0, 10)]
    assert opportunistic_scheduler.effective_time_limit(scope, "EBM", chunk) == 10800
    # PLS shares the dataset but isn't in EBM's override sub-dict -- unaffected.
    assert opportunistic_scheduler.effective_time_limit(scope, "PLS", chunk) == 3600


def test_effective_time_limit_combines_both_override_sources(opportunistic_scheduler):
    """Dataset-keyed and model-keyed overrides compose (max of applicable
    values from both), never silently shadow each other."""
    scope = {
        "time_limit": 3600,
        "time_limit_overrides": {"mlrod": 10800},
        "model_time_limit_overrides": {"EBM": {"mlrod": 7200}},
    }
    chunk = [("mlrod", 0, 0, 0, 0, 10)]
    # The dataset-level override (10800) is larger than EBM's own (7200) here.
    assert opportunistic_scheduler.effective_time_limit(scope, "EBM", chunk) == 10800


def test_effective_time_limit_no_overrides(opportunistic_scheduler):
    scope = {"time_limit": 3600}
    chunk = [("wheat_lines", 0, 0, 0, 0, 10)]
    assert opportunistic_scheduler.effective_time_limit(scope, "PLS", chunk) == 3600


def _load_run_experiment():
    spec = importlib.util.spec_from_file_location(
        "_run_experiment_under_test_resource_fixes", REPO_ROOT / "scripts" / "run_experiment.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def run_experiment():
    return _load_run_experiment()


class _FakeDataset:
    """Minimal stand-in for raman_data.RamanDataset -- only the surface
    run_one() touches before its own clean-skip checks would fire."""

    def __init__(self, task_type, df):
        self.task_type = task_type
        self._df = df

    def to_dataframe(self, target_idx):
        return self._df.copy()


def _make_df(n_features: int, n_rows: int = 4, label_col: str = "target"):
    import numpy as np
    import pandas as pd

    data = {f"f{i}": np.random.default_rng(0).random(n_rows) for i in range(n_features)}
    data[label_col] = np.random.default_rng(1).integers(0, 2, n_rows)
    return pd.DataFrame(data)


def test_run_one_skips_classification_only_model_on_regression_dataset(run_experiment):
    """Regression test for a real crash: ORIONMSP (classification-only) against
    a regression dataset previously reached AutoGluon's own
    `raise_on_no_models_fitted` RuntimeError instead of a clean skip."""
    from raman_data import TASK_TYPE

    df = _make_df(n_features=10)
    fake = _FakeDataset(TASK_TYPE.Regression, df)

    with patch(
        "raman_bench.benchmark.RamanBenchmark._load_raman_dataset", return_value=fake
    ):
        out = run_experiment.run_one(
            dataset_name="fake_dataset",
            target_idx=0,
            model_key="ORIONMSP",
            repeat=0,
            fold=0,
            config_index=0,
        )
    assert out is None


def test_run_one_skips_regression_only_model_on_classification_dataset(run_experiment):
    from raman_data import TASK_TYPE

    df = _make_df(n_features=10)
    fake = _FakeDataset(TASK_TYPE.Classification, df)

    with patch(
        "raman_bench.benchmark.RamanBenchmark._load_raman_dataset", return_value=fake
    ):
        out = run_experiment.run_one(
            dataset_name="fake_dataset",
            target_idx=0,
            model_key="NORI",
            repeat=0,
            fold=0,
            config_index=0,
        )
    assert out is None


def test_run_one_skips_tabstar_above_max_features(run_experiment):
    from raman_data import TASK_TYPE

    from raman_bench.preprocessing.wrapped_models import MAX_FEATURES_MODELS

    if "TABSTAR" not in MAX_FEATURES_MODELS:
        pytest.skip("TABSTAR unavailable in this tabarena build")

    n_features = MAX_FEATURES_MODELS["TABSTAR"] + 1
    df = _make_df(n_features=n_features, n_rows=4)
    fake = _FakeDataset(TASK_TYPE.Regression, df)

    with patch(
        "raman_bench.benchmark.RamanBenchmark._load_raman_dataset", return_value=fake
    ):
        out = run_experiment.run_one(
            dataset_name="fake_wide_dataset",
            target_idx=0,
            model_key="TABSTAR",
            repeat=0,
            fold=0,
            config_index=0,
        )
    assert out is None
