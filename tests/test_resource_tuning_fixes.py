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
4. ORIONMSP OOMs on GPU VRAM (not host RAM) above a real, joint feature-count
   AND row-count threshold -- confirmed real on the cluster: "Tried to
   allocate 94.47 GiB" on a 79.25 GiB GPU for ``pharmaceutical_ingredients``
   (3,276 features, 2,340 train rows), traced to
   ``tabtune.models.orionmsp_v15.model.interaction.RowInteraction``'s
   per-row feature attention, whose score-buffer allocation is
   ``n_rows * nhead * L**2 * 2 bytes`` (``L = 6 + ceil(n_features/2)``) --
   confirmed by reproducing that formula almost exactly (94.24 GiB predicted,
   0.24% off the real failure). Unlike TabSTAR, a single ``max_features``
   number can't express this (row count is independently load-bearing --
   e.g. ``mlrod``, 1,836 features but 130,061 rows, predicts a *worse* OOM
   than the dataset that triggered this investigation). Fixed with a joint
   ``(n_features, n_rows) -> bool`` predicate
   (``wrapped_models.VRAM_CAPPED_MODELS``), the row-and-feature-aware
   counterpart to ``MAX_FEATURES_MODELS``, consumed the same way by
   ``scripts/run_experiment.py::run_one()``.

See ``wrapped_models.py``'s own comment blocks (``_TABSTAR_MAX_FEATURES``,
``Prep_EBM._fit``, ``MAX_FEATURES_MODELS``, ``VRAM_CAPPED_MODELS``) and
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


def test_orionmsp_formula_matches_real_production_failure():
    """The predicted-VRAM formula must reproduce the real cluster OOM almost
    exactly, not just be in the right order of magnitude: pharmaceutical_
    ingredients (3,276 features, 2,340 train rows) attempted to allocate
    94.47 GiB on a 79.25 GiB GPU."""
    from raman_bench.preprocessing.wrapped_models import _orionmsp_predicted_attn_gib

    predicted = _orionmsp_predicted_attn_gib(n_features=3276, n_rows=2340)
    assert predicted == pytest.approx(94.47, rel=0.01)


def test_orionmsp_vram_cap_known_real_datapoints():
    """Real, confirmed data points from the v1 target distribution -- not
    synthetic ones -- must land on the expected side of the cap:

    - pharmaceutical_ingredients (3,276 features, 2,340 train rows): the
      real production OOM. Must be flagged.
    - diabetes_skin_ear_lobe (2,803 processed features, ~13 train rows):
      confirmed working on GPU. Must NOT be flagged, despite having a
      similar feature count to pharmaceutical_ingredients -- proof the cap
      is row-aware, not just a max_features check in disguise.
    - mlrod (1,836 features, ~86,707 train rows) and wheat_lines (1,748
      features, ~35,423 train rows): FEWER features than pharmaceutical_
      ingredients but far more rows -- both must be flagged, which a
      features-only cap could never do (both are narrower than the known
      failure point).
    - sugar_mixtures_high_snr vs. sugar_mixtures_low_snr: identical 2,000
      features, but low_snr has ~5,227 train rows vs. high_snr's ~1,307 --
      must land on opposite sides of the cap, proving row count is
      independently load-bearing and not just a proxy for feature count.
    """
    from raman_bench.preprocessing.wrapped_models import VRAM_CAPPED_MODELS

    exceeds = VRAM_CAPPED_MODELS["ORIONMSP"]

    assert exceeds(3276, 2340) is True  # pharmaceutical_ingredients: real OOM
    assert exceeds(2803, 13) is False  # diabetes_skin_ear_lobe: confirmed working
    assert exceeds(1836, 86707) is True  # mlrod: fewer features, far more rows
    assert exceeds(1748, 35423) is True  # wheat_lines: same story
    assert exceeds(2000, 5227) is True  # sugar_mixtures_low_snr
    assert exceeds(2000, 1307) is False  # sugar_mixtures_high_snr: same features, fewer rows


def test_run_one_skips_orionmsp_above_vram_budget(run_experiment):
    """Regression test for the real cluster OOM: a dataset whose predicted
    OrionMSP attention buffer exceeds the VRAM budget must clean-skip
    (return None, no crash) before any model/CUDA code ever runs."""
    from raman_data import TASK_TYPE

    # n_features=5000, n_rows=700 -> n_rows_estimate = round(700 * 2/3) = 467,
    # comfortably above the ~428-row threshold at which 5,000 features alone
    # crosses the 40 GiB budget (see VRAM_CAPPED_MODELS's docstring).
    df = _make_df(n_features=5000, n_rows=700)
    fake = _FakeDataset(TASK_TYPE.Classification, df)

    with patch(
        "raman_bench.benchmark.RamanBenchmark._load_raman_dataset", return_value=fake
    ):
        out = run_experiment.run_one(
            dataset_name="fake_wide_and_tall_dataset",
            target_idx=0,
            model_key="ORIONMSP",
            repeat=0,
            fold=0,
            config_index=0,
        )
    assert out is None


def test_run_one_does_not_skip_orionmsp_below_vram_budget(run_experiment):
    """A dataset well under the VRAM budget (e.g. diabetes_skin_ear_lobe's
    real shape: ~2,800 features, tiny row count) must NOT be caught by the
    new clean-skip -- it should proceed exactly as before this fix. Patches
    OrionMSPModel._fit itself (not just the dataset load) so this stays a
    fast, real check of the skip *predicate*, not a full model fit."""
    from raman_data import TASK_TYPE

    from raman_bench.preprocessing.wrapped_models import Prep_ORIONMSP

    if Prep_ORIONMSP is None:
        pytest.skip("OrionMSPModel unavailable in this tabarena build")

    df = _make_df(n_features=2803, n_rows=20, label_col="target")
    # ORIONMSP is classification-only -- make the label look categorical.
    df["target"] = [0, 1] * 10
    fake = _FakeDataset(TASK_TYPE.Classification, df)

    reached_fit = {"called": False}

    def _fake_fit(self, X, y, **kwargs):
        reached_fit["called"] = True
        raise RuntimeError("stop before any real (CUDA/CPU) model work")

    with (
        patch(
            "raman_bench.benchmark.RamanBenchmark._load_raman_dataset", return_value=fake
        ),
        patch(
            "raman_bench.preprocessing.wrapped_models.OrionMSPModel._fit", _fake_fit
        ),
        pytest.raises(RuntimeError, match="stop before any real"),
    ):
        run_experiment.run_one(
            dataset_name="fake_narrow_dataset",
            target_idx=0,
            model_key="ORIONMSP",
            repeat=0,
            fold=0,
            config_index=0,
            # ORIONMSP has an empty AutoGluon search space (gen_orionmsp:
            # search_space={}, manual_configs=[{}]) -- requesting more than 1
            # random config from an empty space raises AutoGluon's own
            # ExhaustedSearchSpaceError before ever reaching _fit, unrelated
            # to this test. Matches the real single-config smoke-test
            # convention (see model-agent's own reference command).
            num_random_configs=1,
        )
    assert reached_fit["called"], "run_one() should have reached the real _fit call, not skipped"


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
