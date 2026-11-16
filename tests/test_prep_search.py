"""Tests for the opt-in preprocessing search (``--prep-search-config``, raman_bench.prep_search).

Checks, following ``test_run_experiment_recipe.py``'s "real model classes, patched data" style:

1. config loading/validation and the pin set;
2. the wrapped generator keeps experiment names, is deterministic across calls (config ``rK``
   must mean the same thing in every job of a sweep), keeps the upstream model hyperparameters
   unchanged and only adds prep keys, and pins every non-searched step off -- including in
   ``c1`` (no class-default baseline correction leaking into PLS/LR);
3. end to end through ``run_one``: a sampled tree config (RF, which does NOT set
   ``_optimize_preprocessing``) with cosmic-ray removal enabled really runs cosmic-ray removal
   at fit time -- the silent-degrade failure mode where a gate drops the sampled flags -- and
   lands in its own ``__prepsearch_`` cache directory.
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

from raman_bench.prep_search import (  # noqa: E402
    load_prep_search_config,
    pinned_off_params,
    prep_search_slug,
    with_prep_search,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CRR_SG_SNV = REPO_ROOT / "configs" / "prep_search" / "crr_sg_snv.json"
SEARCHED_FLAGS = {"prep_crr_enabled", "prep_denoise_enabled", "prep_snv_enabled"}


def _load_run_experiment():
    spec = importlib.util.spec_from_file_location(
        "_run_experiment_under_test_prep_search", REPO_ROOT / "scripts" / "run_experiment.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def run_experiment():
    return _load_run_experiment()


# --- 1. config loading ------------------------------------------------------------------------


def test_load_shipped_config():
    assert load_prep_search_config(CRR_SG_SNV) == {
        "cosmic_ray_removal": True,
        "denoising": True,
        "snv": True,
    }


def test_load_bare_mapping_and_false_steps(tmp_path):
    path = tmp_path / "p.json"
    path.write_text(json.dumps({"snv": True, "denoising": False}))
    assert load_prep_search_config(path) == {"snv": True}


@pytest.mark.parametrize("content", [{"preprocessing": {"no_such_step": True}}, {"snv": False}, []])
def test_load_rejects_bad_configs(tmp_path, content):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(content))
    with pytest.raises(ValueError):
        load_prep_search_config(path)


def test_slug_depends_on_content(tmp_path):
    a, b = tmp_path / "a" / "x.json", tmp_path / "b" / "x.json"
    a.parent.mkdir(), b.parent.mkdir()
    a.write_text(json.dumps({"snv": True}))
    b.write_text(json.dumps({"snv": True, "denoising": True}))
    assert prep_search_slug(a) != prep_search_slug(b)
    assert prep_search_slug(a).startswith("x-")


def test_pins_cover_every_other_step_but_not_augmentation():
    pins = pinned_off_params(load_prep_search_config(CRR_SG_SNV))
    assert not SEARCHED_FLAGS & set(pins)
    assert "prep_aug_enabled" not in pins
    assert pins["prep_bl_enabled"] is False
    assert set(pins.values()) == {False}


# --- 2. wrapped generators --------------------------------------------------------------------


def _strip_prep(config: dict) -> dict:
    return {k: v for k, v in config.items() if not k.startswith("prep_")}


@pytest.mark.parametrize("model_key", ["PLS", "LR", "RF", "XGB", "CAT"])
def test_wrapped_generator_is_deterministic_and_additive(run_experiment, model_key):
    steps = load_prep_search_config(CRR_SG_SNV)
    upstream = run_experiment._import_generator(model_key)
    n = 6

    first = with_prep_search(upstream, steps).generate_all_configs_lst(num_random_configs=n)
    again = with_prep_search(
        run_experiment._import_generator(model_key), steps
    ).generate_all_configs_lst(num_random_configs=n)
    plain = upstream.generate_all_configs_lst(num_random_configs=n)

    # same names, same order, same configs in every job
    assert [c["ag_args"]["name_suffix"] for c in first] == [
        c["ag_args"]["name_suffix"] for c in plain
    ]
    assert first == again
    # model hyperparameters untouched -- only prep_* keys added
    assert [_strip_prep(c) for c in first] == [_strip_prep(c) for c in plain]

    pins = pinned_off_params(steps)
    c1, randoms = first[0], first[1:]
    assert c1["ag_args"]["name_suffix"] == "_c1"
    assert all(c1[k] is False for k in pins)
    assert not SEARCHED_FLAGS & set(c1)  # c1 keeps the model's own default for searched steps
    for cfg in randoms:
        assert all(cfg[k] is False for k in pins)
        assert SEARCHED_FLAGS <= set(cfg)
    # the sampler actually varies the recipe across configs
    assert len({tuple(sorted((k, c[k]) for k in SEARCHED_FLAGS)) for c in randoms}) > 1


# --- 3. end to end through run_one ------------------------------------------------------------


class _FakeDataset:
    """Minimal stand-in for raman_data.RamanDataset (see test_run_experiment_recipe.py)."""

    def __init__(self, task_type, df):
        self.task_type = task_type
        self._df = df
        self.targets = df[df.columns[-1]].to_numpy().reshape(-1, 1)

    def to_dataframe(self, target_idx):
        return self._df.copy()


def _make_regression_df(n_samples: int = 30, n_features: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = rng.normal(loc=5.0, scale=2.0, size=(n_samples, n_features))
    X += rng.normal(scale=3.0, size=(n_samples, 1))
    y = X[:, 0] * 2.0 + rng.normal(scale=0.1, size=n_samples)
    data = {f"f{i}": X[:, i] for i in range(n_features)}
    data["target"] = y
    return pd.DataFrame(data)


def test_run_one_tree_config_runs_sampled_cosmic_ray_removal(run_experiment, tmp_path):
    from raman_data import TASK_TYPE

    import raman_bench.preprocessing.mixin as mixin_mod
    from raman_bench.preprocessing.wrapped_models import PREPROCESSED_MODELS

    num_random_configs = 10
    steps = load_prep_search_config(CRR_SG_SNV)
    configs = with_prep_search(
        run_experiment._import_generator("RF"), steps
    ).generate_all_configs_lst(num_random_configs=num_random_configs)
    index = next(i for i, c in enumerate(configs) if c.get("prep_crr_enabled") is True)

    cls = PREPROCESSED_MODELS["RF"]
    assert not getattr(cls, "_optimize_preprocessing", False)  # the gated case
    original_fit = cls._fit
    original_crr = mixin_mod.cosmic_ray_removal
    captured = {"crr_calls": 0}

    def _spy_fit(self, *args, **kwargs):
        out = original_fit(self, *args, **kwargs)
        captured["prep_crr_enabled"] = self.params.get("prep_crr_enabled")
        captured["prep_bl_enabled"] = self.params.get("prep_bl_enabled")
        return out

    def _spy_crr(X, *args, **kwargs):
        captured["crr_calls"] += 1
        return original_crr(X, *args, **kwargs)

    fake = _FakeDataset(TASK_TYPE.Regression, _make_regression_df())
    results_dir = tmp_path / "results"
    with (
        patch("raman_bench.benchmark.RamanBenchmark._load_raman_dataset", return_value=fake),
        patch.object(cls, "_fit", _spy_fit),
        patch.object(mixin_mod, "cosmic_ray_removal", _spy_crr),
        patch.dict("os.environ", {"RAMAN_BENCH_SPECTRUM_CACHE": "off"}),
    ):
        out = run_experiment.run_one(
            dataset_name="fake_prep_search_dataset",
            target_idx=0,
            model_key="RF",
            repeat=0,
            fold=0,
            config_index=index,
            n_repeats=1,
            n_splits=2,
            num_random_configs=num_random_configs,
            num_bag_folds=2,
            time_limit=60,
            prep_search_config=str(CRR_SG_SNV),
            results_dir=str(results_dir),
            cache_dir=str(tmp_path / "cache"),
        )

    assert out is not None
    assert captured.get("prep_crr_enabled") is True, captured
    assert captured.get("prep_bl_enabled") is False, captured
    assert captured["crr_calls"] > 0, "sampled cosmic-ray removal never ran"
    assert any("__prepsearch_crr_sg_snv-" in p.name for p in results_dir.rglob("*") if p.is_dir())


def test_run_one_rejects_recipe_plus_prep_search(run_experiment, tmp_path):
    from raman_data import TASK_TYPE

    recipe = tmp_path / "snv.json"
    recipe.write_text(json.dumps({"preprocessing": {"snv": True}}))
    fake = _FakeDataset(TASK_TYPE.Regression, _make_regression_df())
    with (
        patch("raman_bench.benchmark.RamanBenchmark._load_raman_dataset", return_value=fake),
        pytest.raises(ValueError, match="mutually exclusive"),
    ):
        run_experiment.run_one(
            dataset_name="fake_prep_search_dataset",
            target_idx=0,
            model_key="PLS",
            repeat=0,
            fold=0,
            config_index=0,
            n_repeats=1,
            n_splits=2,
            num_bag_folds=2,
            time_limit=60,
            recipe_config=str(recipe),
            prep_search_config=str(CRR_SG_SNV),
            results_dir=str(tmp_path / "results"),
            cache_dir=str(tmp_path / "cache"),
        )
