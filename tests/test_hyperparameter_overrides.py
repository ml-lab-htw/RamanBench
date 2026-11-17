"""Tests for fixed hyperparameter overrides (``--hyperparameter-overrides``,
raman_bench.hyperparameter_overrides), in ``test_run_experiment_recipe.py``'s "real model
classes, patched data" style.

The motivating case: the Raman DL models' search spaces sample ``aug_noise_sigma`` /
``aug_mixup_alpha``, which only act when ``per_epoch_augmentation`` is on (it is off by default,
so they were inert). The shipped ``per_epoch_augmentation.json`` switches it on and the fixed
pre-fit noise copies off; the end-to-end test checks both really happen at fit time, with the
sampled values, and that the run lands in its own ``__hp_`` cache directory.
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

from raman_bench.hyperparameter_overrides import (  # noqa: E402
    hyperparameter_overrides_slug,
    load_hyperparameter_overrides,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PER_EPOCH_AUG = REPO_ROOT / "configs" / "hyperparameter_overrides" / "per_epoch_augmentation.json"


def _load_run_experiment():
    spec = importlib.util.spec_from_file_location(
        "_run_experiment_under_test_hp_overrides", REPO_ROOT / "scripts" / "run_experiment.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def run_experiment():
    return _load_run_experiment()


def test_load_shipped_config_drops_comment():
    assert load_hyperparameter_overrides(PER_EPOCH_AUG) == {
        "per_epoch_augmentation": True,
        "prep_aug_enabled": False,
    }


@pytest.mark.parametrize("content", [[], {}, {"_comment": "only"}, {"ag_args": {"name_suffix": "_x"}}])
def test_load_rejects_bad_files(tmp_path, content):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(content))
    with pytest.raises(ValueError):
        load_hyperparameter_overrides(path)


def test_slug_depends_on_values(tmp_path):
    a, b = tmp_path / "a" / "x.json", tmp_path / "b" / "x.json"
    a.parent.mkdir(), b.parent.mkdir()
    a.write_text(json.dumps({"per_epoch_augmentation": True}))
    b.write_text(json.dumps({"per_epoch_augmentation": False}))
    assert hyperparameter_overrides_slug(a) != hyperparameter_overrides_slug(b)
    assert hyperparameter_overrides_slug(a).startswith("x-")


class _FakeDataset:
    """Minimal stand-in for raman_data.RamanDataset (see test_run_experiment_recipe.py)."""

    def __init__(self, task_type, df):
        self.task_type = task_type
        self._df = df
        self.targets = df[df.columns[-1]].to_numpy().reshape(-1, 1)

    def to_dataframe(self, target_idx):
        return self._df.copy()


def _make_regression_df(n_samples: int = 36, n_features: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = rng.normal(loc=5.0, scale=2.0, size=(n_samples, n_features))
    y = X[:, 0] * 2.0 + rng.normal(scale=0.1, size=n_samples)
    data = {f"f{i}": X[:, i] for i in range(n_features)}
    data["target"] = y
    return pd.DataFrame(data)


@pytest.mark.parametrize("config_index", [0, 1])
def test_run_one_per_epoch_augmentation_uses_sampled_values(run_experiment, tmp_path, config_index):
    from raman_data import TASK_TYPE

    import raman_bench.preprocessing.mixin as mixin_mod
    import raman_bench.preprocessing.raman_preprocessing as rp

    num_random_configs = 2
    configs = run_experiment._import_generator("FCRESNEXT").generate_all_configs_lst(
        num_random_configs=num_random_configs
    )
    cfg = configs[config_index]

    calls = {"per_epoch": [], "fixed": 0}
    original_torch_aug = rp.augment_spectra_torch
    original_fixed_aug = mixin_mod.augment_spectra

    def _spy_torch_aug(X, y, **kwargs):
        calls["per_epoch"].append(kwargs)
        return original_torch_aug(X, y, **kwargs)

    def _spy_fixed_aug(*args, **kwargs):
        calls["fixed"] += 1
        return original_fixed_aug(*args, **kwargs)

    fake = _FakeDataset(TASK_TYPE.Regression, _make_regression_df())
    results_dir = tmp_path / "results"
    with (
        patch("raman_bench.benchmark.RamanBenchmark._load_raman_dataset", return_value=fake),
        patch.object(rp, "augment_spectra_torch", _spy_torch_aug),
        patch.object(mixin_mod, "augment_spectra", _spy_fixed_aug),
    ):
        out = run_experiment.run_one(
            dataset_name="fake_hp_overrides_dataset",
            target_idx=0,
            model_key="FCRESNEXT",
            repeat=0,
            fold=0,
            config_index=config_index,
            n_repeats=1,
            n_splits=2,
            num_random_configs=num_random_configs,
            num_bag_folds=2,
            time_limit=60,
            hyperparameter_overrides=str(PER_EPOCH_AUG),
            results_dir=str(results_dir),
            cache_dir=str(tmp_path / "cache"),
        )

    assert out is not None
    assert calls["fixed"] == 0, "fixed pre-fit noise copies still ran"
    assert calls["per_epoch"], "per-epoch augmentation never ran"
    first = calls["per_epoch"][0]
    if config_index > 0:  # a sampled config: its own values reach the augmentation
        assert first["noise_sigma"] == pytest.approx(cfg["aug_noise_sigma"])
        assert first["mixup_alpha"] == pytest.approx(cfg["aug_mixup_alpha"])
    assert first["label_type"] == "regression"
    stored = out["method_metadata"]["model_hyperparameters"]
    assert stored["per_epoch_augmentation"] is True and stored["prep_aug_enabled"] is False
    assert any("__hp_per_epoch_augmentation-" in p.name for p in results_dir.rglob("*") if p.is_dir())
