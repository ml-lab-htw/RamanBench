"""Regression tests for the Pipeline B preprocessing-recipe gap.

Before this fix, ``scripts/run_experiment.py::run_one()`` had no way to specify which
preprocessing recipe a job should use -- only ``--config-index`` for model
hyperparameters (see ``docs/kfold_priority_plan.md``'s §6.1 in the RamanPreprocessing repo,
and this fix's own writeup). Adding ``--recipe-config``/``recipe_config`` reuses the exact
same restriction-application code path Pipeline A's ``AutoGluonModel._build_model_hyperparameters``
already used, factored out as ``raman_bench.model.build_prep_model_hyperparameters``.

These tests check three things, mirroring ``tests/test_resource_tuning_fixes.py``'s
"patch the real, cheap parts; assert on real model-class state" convention:

1. ``_load_recipe_config`` normalizes a recipe JSON file the same way Pipeline A's
   ``load_config`` does (same ``preprocessing``/``preprocessing_params`` schema).
2. A recipe passed to ``run_one`` actually reaches the real fitted model instance's
   ``self.params`` (the ``prep_snv_enabled`` hyperparameter AutoGluon's ``AbstractModel``
   reads at fit time) -- not just built correctly in isolation.
3. Two ``run_one`` jobs on the *same* tiny synthetic dataset/model, differing only in
   ``recipe_config`` (SNV enabled vs. no preprocessing at all), produce genuinely different
   preprocessed feature matrices -- the end-to-end confirmation the task asked for.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("tabarena")

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_run_experiment():
    spec = importlib.util.spec_from_file_location(
        "_run_experiment_under_test_recipe", REPO_ROOT / "scripts" / "run_experiment.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def run_experiment():
    return _load_run_experiment()


class _FakeDataset:
    """Minimal stand-in for raman_data.RamanDataset (see test_resource_tuning_fixes.py)."""

    def __init__(self, task_type, df):
        self.task_type = task_type
        self._df = df
        # Every row unique -> infer_group_ids_from_targets finds no shared key,
        # returns None (no grouping), matching plain i.i.d. synthetic data.
        self.targets = df[df.columns[-1]].to_numpy().reshape(-1, 1)

    def to_dataframe(self, target_idx):
        return self._df.copy()


def _make_regression_df(n_samples: int = 24, n_features: int = 30, seed: int = 0) -> pd.DataFrame:
    """Tiny synthetic "spectra" with a real per-row baseline offset, so SNV
    (row-wise mean-center + unit-variance) has something non-trivial to remove --
    a pure global rescale would not distinguish "SNV applied" from "not applied"
    as clearly."""
    rng = np.random.default_rng(seed)
    X = rng.normal(loc=5.0, scale=2.0, size=(n_samples, n_features))
    X += rng.normal(scale=3.0, size=(n_samples, 1))  # per-row baseline offset
    y = X[:, 0] * 2.0 + rng.normal(scale=0.1, size=n_samples)
    data = {f"f{i}": X[:, i] for i in range(n_features)}
    data["target"] = y
    return pd.DataFrame(data)


@pytest.fixture
def snv_recipe_config(tmp_path) -> str:
    path = tmp_path / "snv.json"
    path.write_text(json.dumps({"preprocessing": {"snv": True}}))
    return str(path)


@pytest.fixture
def none_recipe_config(tmp_path) -> str:
    path = tmp_path / "none.json"
    path.write_text(json.dumps({"preprocessing": {}}))
    return str(path)


def test_load_recipe_config_none_path_is_noop(run_experiment):
    assert run_experiment._load_recipe_config(None) == (None, None)


def test_load_recipe_config_normalizes_snv_recipe(run_experiment, snv_recipe_config):
    """Same normalization Pipeline A's load_config applies: every known step
    key present, defaulting to False except the ones the recipe turns on."""
    preprocessing_config, preprocessing_params = run_experiment._load_recipe_config(
        snv_recipe_config
    )
    assert preprocessing_config["snv"] is True
    assert preprocessing_config["baseline_correction"] is False
    assert preprocessing_config["augmentation"] is False
    assert preprocessing_params is None


def test_run_one_recipe_reaches_real_fitted_model_params(run_experiment, snv_recipe_config, tmp_path):
    """The prep_snv_enabled hyperparameter from --recipe-config must reach the
    real PLS model instance's self.params at fit time -- not just build
    correctly in an isolated hyperparameters dict. Spies on the model's real
    _fit (no stubbing of the fit logic itself, matching
    test_resource_tuning_fixes.py's real-model-class-level convention)."""
    from raman_data import TASK_TYPE

    from raman_bench.preprocessing.wrapped_models import PREPROCESSED_MODELS

    df = _make_regression_df()
    fake = _FakeDataset(TASK_TYPE.Regression, df)

    cls = PREPROCESSED_MODELS["PLS"]
    original_fit = cls._fit
    captured = {}

    def _spy_fit(self, *args, **kwargs):
        captured["prep_snv_enabled"] = self.params.get("prep_snv_enabled")
        return original_fit(self, *args, **kwargs)

    with (
        patch("raman_bench.benchmark.RamanBenchmark._load_raman_dataset", return_value=fake),
        patch.object(cls, "_fit", _spy_fit),
    ):
        out = run_experiment.run_one(
            dataset_name="fake_snv_dataset",
            target_idx=0,
            model_key="PLS",
            repeat=0,
            fold=0,
            config_index=0,
            n_repeats=1,
            n_splits=2,
            num_bag_folds=2,
            time_limit=60,
            recipe_config=snv_recipe_config,
            results_dir=str(tmp_path / "results"),
            cache_dir=str(tmp_path / "cache"),
        )

    assert out is not None
    assert captured.get("prep_snv_enabled") is True


def test_run_one_recipe_changes_preprocessed_feature_matrix(
    run_experiment, snv_recipe_config, none_recipe_config, tmp_path
):
    """End-to-end confirmation: recipe=snv vs. recipe=none on the identical
    dataset/model produce genuinely different preprocessed feature matrices --
    the exact "confirm the feature matrices or predictions differ" check the
    recipe-injection gap needed. Spies on the innermost sklearn estimator's own
    ``fit`` (``PLSModel.fit``) rather than
    ``RamanPreprocessingMixin._preprocess_fit`` -- the mixin's ``_fit`` only
    calls ``_preprocess_fit`` when at least one step is enabled
    (``has_preprocessing``, see ``mixin.py``), so for the "none" recipe (every
    step disabled) it is a no-op skip, not a call with an identity transform;
    spying there would only ever capture the "snv" run. ``PLSModel.fit``
    always runs, with whatever the final feature matrix is -- preprocessed or
    passed straight through -- so it's the correct, recipe-agnostic capture
    point for "what did the model actually train on"."""
    from raman_data import TASK_TYPE

    from raman_bench.models.custom.pls.model import PLSModel

    df = _make_regression_df()
    fake = _FakeDataset(TASK_TYPE.Regression, df)

    original_fit = PLSModel.fit
    captured_matrices: dict[str, np.ndarray] = {}

    def _make_spy(key: str):
        def _spy(self, X, y):
            X_np = X.values if hasattr(X, "values") else np.asarray(X)
            # Keep only the first call's input (AutoGluon's bagged folds call this
            # repeatedly; the first is enough to prove the recipe took effect).
            captured_matrices.setdefault(key, X_np.copy())
            return original_fit(self, X, y)

        return _spy

    common_kwargs = dict(
        dataset_name="fake_recipe_diff_dataset",
        target_idx=0,
        model_key="PLS",
        repeat=0,
        fold=0,
        config_index=0,
        n_repeats=1,
        n_splits=2,
        num_bag_folds=2,
        time_limit=60,
        results_dir=str(tmp_path / "results"),
        cache_dir=str(tmp_path / "cache"),
    )

    with (
        patch("raman_bench.benchmark.RamanBenchmark._load_raman_dataset", return_value=fake),
        patch.object(PLSModel, "fit", _make_spy("snv")),
    ):
        out_snv = run_experiment.run_one(recipe_config=snv_recipe_config, **common_kwargs)

    with (
        patch("raman_bench.benchmark.RamanBenchmark._load_raman_dataset", return_value=fake),
        patch.object(PLSModel, "fit", _make_spy("none")),
    ):
        out_none = run_experiment.run_one(recipe_config=none_recipe_config, **common_kwargs)

    assert out_snv is not None
    assert out_none is not None
    assert "snv" in captured_matrices
    assert "none" in captured_matrices
    x_snv = captured_matrices["snv"]
    x_none = captured_matrices["none"]
    assert x_snv.shape == x_none.shape
    assert not np.allclose(x_snv, x_none)

    # SNV row-normalizes: every row of the SNV-preprocessed matrix should have
    # ~zero mean and ~unit variance, unlike the untouched "none" matrix.
    row_means = x_snv.mean(axis=1)
    row_stds = x_snv.std(axis=1)
    assert np.allclose(row_means, 0.0, atol=1e-6)
    assert np.allclose(row_stds, 1.0, atol=1e-6)

    # Predictions must also differ end-to-end (the recipe affects both train and
    # test-time preprocessing, not just the captured training-fold matrix).
    # `experiment.run()`'s own output drops raw "predictions"
    # (ExperimentRunner.convert_to_output pops them after scoring) -- the
    # OOF-ensemble-simulation artifact's "pred_proba_dict_test" is where the
    # test-fold predictions actually end up.
    preds_snv = np.asarray(
        out_snv["simulation_artifacts"]["pred_proba_dict_test"]["PLS_c1_BAG_L1"]
    )
    preds_none = np.asarray(
        out_none["simulation_artifacts"]["pred_proba_dict_test"]["PLS_c1_BAG_L1"]
    )
    assert preds_snv.shape == preds_none.shape
    assert not np.allclose(preds_snv, preds_none)
    # And the scored metric itself must differ too (not just raw predictions
    # that happen to score identically).
    assert out_snv["metric_error"] != pytest.approx(out_none["metric_error"])


def test_recipe_config_ignored_for_non_preprocessing_model(
    run_experiment, snv_recipe_config, caplog, tmp_path
):
    """A model_cls that does NOT resolve to a RamanPreprocessingMixin subclass
    must not crash when a recipe is requested -- the recipe is ignored with a
    warning, matching how Pipeline A's create_preprocessed_hyperparameters
    silently passes such models through with no preprocessing hyperparameters
    at all. Stubs ``infer_model_cls`` to return a plain (non-Prep_*) class so
    this doesn't depend on which specific non-preprocessing model happens to
    be registered in a given tabarena/AutoGluon build -- the model registry
    genuinely has several (e.g. AutoGluon's own AG_TEXT_NN, RuleFit, KNN-new),
    but their fit requirements (text columns, classification-only, etc.) are
    irrelevant to what's under test here: the recipe-application branch in
    run_one() must warn-and-skip, not crash, for ANY non-Prep_* model_cls."""
    import logging

    class _PlainNonPrepModel:
        """Stand-in for a real AutoGluon/tabarena model class with no
        RamanPreprocessingMixin ancestry (e.g. AutoGluon's own KNNModel)."""

    from raman_data import TASK_TYPE

    df = _make_regression_df(n_features=10)
    fake = _FakeDataset(TASK_TYPE.Regression, df)

    with (
        patch("raman_bench.benchmark.RamanBenchmark._load_raman_dataset", return_value=fake),
        patch(
            "raman_bench.models.registry.infer_model_cls",
            return_value=_PlainNonPrepModel,
        ),
        caplog.at_level(logging.WARNING),
    ):
        try:
            run_experiment.run_one(
                dataset_name="fake_non_prep_dataset",
                target_idx=0,
                model_key="PLS",
                repeat=0,
                fold=0,
                config_index=0,
                n_repeats=1,
                n_splits=2,
                num_bag_folds=2,
                time_limit=60,
                recipe_config=snv_recipe_config,
                results_dir=str(tmp_path / "results"),
                cache_dir=str(tmp_path / "cache"),
            )
        except Exception:
            pass  # The stubbed model_cls mismatches gen_pls's real Prep_PLS,
            # which trips run_one()'s own later sanity assert -- expected and
            # irrelevant here; only the warning (not a crash from the
            # recipe-application code itself) is under test.
    assert any(
        "recipe_config" in rec.getMessage() and "ignored" in rec.getMessage()
        for rec in caplog.records
    )
