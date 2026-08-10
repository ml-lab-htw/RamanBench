"""Regression test for onboarding batch 2 of TabArena's newest models.

EBM, PERPETUAL_BOOSTER, XRFM, and CHIMERABOOST are tree/boosting/kernel-family
models (not tabular foundation models -- unlike
``test_generate_tabarena_foundation_models.py``'s batch 1, none of these four are
pretrained/zero-shot), so they get their own test module rather than living in
that one. EBM is a graduated AutoGluon-core class
(``autogluon.tabular.models.EBMModel``); PERPETUAL_BOOSTER, XRFM, and
CHIMERABOOST ship only from TabArena's own package (``tabarena.models.<key>.model``),
same situation as TABFM/TABPFN-V3/TABSWIFT in batch 1. Every one of these keys
must resolve a ``ConfigGenerator``/``CustomAGConfigGenerator`` whose ``model_cls``
matches the registry-resolved class, and generate at least the manual/default
config without error.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# PERPETUAL_BOOSTER's upstream search space is a single categorical knob
# (`budget`, 5 discrete values -- see generate/perpetual_booster.py's docstring):
# `LocalRandomSearcher._get_num_configs()` reports exactly 5 total configs, so
# requesting more than 5 random configs raises ExhaustedSearchSpaceError
# (confirmed empirically against the installed tabarena build). EBM, XRFM, and
# CHIMERABOOST all have real continuous search spaces (confirmed to support at
# least 200 unique random configs without exhaustion), so they use the same
# num_random_configs=2 every other non-empty-search-space model's test uses.
_MAX_RANDOM_CONFIGS = {"PERPETUAL_BOOSTER": 0}

# EBM, XRFM, CHIMERABOOST already existed under a short ag_key in TabArena's own
# (un-preprocessed baseline) registry ("EBM", "XRFM", "CHIMERA") -- only
# PERPETUAL_BOOSTER's and CHIMERABOOST's ag_key needed overriding to a
# spelled-out, readability-motivated form ("PB" -> "PERPETUAL_BOOSTER", "CHIMERA"
# -> "CHIMERABOOST"); see wrapped_models.py's batch-2 comment block.
TABARENA_BATCH2_KEYS = [
    "EBM",
    "PERPETUAL_BOOSTER",
    "XRFM",
    "CHIMERABOOST",
]

# Only XRFM is GPU-tier (auto-detects a GPU at fit time, falls back to CPU when
# none is available -- same shape as ModernNCA/RealMLP/NN_TORCH/FASTAI). EBM,
# PERPETUAL_BOOSTER, and CHIMERABOOST are CPU-only.
TABARENA_BATCH2_GPU_KEYS = ["XRFM"]


def _load_run_experiment():
    """Import ``scripts/run_experiment.py`` by path (it's a script, not a package)."""
    spec = importlib.util.spec_from_file_location(
        "_run_experiment_under_test", REPO_ROOT / "scripts" / "run_experiment.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def run_experiment():
    return _load_run_experiment()


def test_all_models_json_keys_covered():
    """Guards against ``configs/models/all.json`` silently dropping one of these keys."""
    all_json = json.loads((REPO_ROOT / "configs" / "models" / "all.json").read_text())
    missing = set(TABARENA_BATCH2_KEYS) - set(all_json)
    assert not missing, f"configs/models/all.json no longer lists: {missing}"


def test_traditional_ml_json_keys_covered():
    """Tree/boosting/kernel-family models belong in traditional_ml.json, not
    tabular_foundation.json (none of the four are pretrained/zero-shot)."""
    traditional_ml_json = json.loads(
        (REPO_ROOT / "configs" / "models" / "traditional_ml.json").read_text()
    )
    missing = set(TABARENA_BATCH2_KEYS) - set(traditional_ml_json)
    assert not missing, f"configs/models/traditional_ml.json no longer lists: {missing}"


def test_gpu_models_json_keys_covered():
    """Only XRFM is GPU-tier -- guards against ``cluster/gpu_models.json`` drift."""
    gpu_models = json.loads((REPO_ROOT / "cluster" / "gpu_models.json").read_text())
    missing = set(TABARENA_BATCH2_GPU_KEYS) - set(gpu_models)
    assert not missing, f"cluster/gpu_models.json no longer lists: {missing}"
    extra = {k for k in TABARENA_BATCH2_KEYS if k not in TABARENA_BATCH2_GPU_KEYS} & set(gpu_models)
    assert not extra, f"cluster/gpu_models.json unexpectedly lists CPU-only model(s): {extra}"


@pytest.mark.parametrize("key", TABARENA_BATCH2_KEYS)
def test_generator_resolves_and_matches_registry(key, run_experiment):
    from raman_bench.models.registry import infer_model_cls

    try:
        model_cls = infer_model_cls(key)
    except AssertionError:
        pytest.skip(f"{key} unavailable in this tabarena/AutoGluon build")

    gen = run_experiment._import_generator(key)
    assert gen.model_cls is model_cls, (
        f"{key}: generator's model_cls ({gen.model_cls}) doesn't match the "
        f"registry-resolved class ({model_cls})"
    )

    num_random_configs = _MAX_RANDOM_CONFIGS.get(key, 2)
    experiments = gen.generate_all_bag_experiments(
        num_random_configs=num_random_configs,
        time_limit=60,
        num_bag_folds=2,
        fold_fitting_strategy="sequential_local",
        add_seed="fold-config-wise",
    )
    assert len(experiments) >= 1
    assert experiments[0].method_kwargs["model_cls"] is model_cls


def test_perpetual_booster_search_space_exhausts_past_five():
    """Documents the real ceiling generate/perpetual_booster.py's docstring warns
    about, so a future HPO sweep for this model doesn't have to rediscover it."""
    from autogluon.core.searcher.exceptions import ExhaustedSearchSpaceError

    from raman_bench.models.generate.perpetual_booster import gen_perpetual_booster

    assert len(gen_perpetual_booster.get_searcher_configs(5)) == 5
    with pytest.raises(ExhaustedSearchSpaceError):
        gen_perpetual_booster.get_searcher_configs(6)


def test_ebm_memory_estimate_ignores_prep_params():
    """Regression test: found via a real local smoke test (``run_experiment.py
    --model EBM``), not anticipated up front. ``EBMModel._estimate_memory_usage``
    (unlike CAT/XGB/RF/XT's purely arithmetic estimators) instantiates the real
    ``interpret`` estimator from ``self._get_model_params()`` to call its own
    ``.estimate_mem()`` -- AutoGluon's own ``_validate_fit_memory_usage`` calls
    this *before* ``RamanPreprocessingMixin._fit()`` gets a chance to strip
    ``prep_*`` keys, so without ``Prep_EBM``'s ``_estimate_memory_usage``
    override this raised ``TypeError: ExplainableBoostingClassifier.__init__()
    got an unexpected keyword argument 'prep_aug_enabled'`` on every real bagged
    fit. See wrapped_models.py's ``Prep_EBM._estimate_memory_usage`` docstring.
    """
    import numpy as np
    import pandas as pd

    from raman_bench.preprocessing.wrapped_models import Prep_EBM

    model = Prep_EBM(
        problem_type="binary",
        hyperparameters={"prep_aug_enabled": True, "prep_snv_enabled": True, "prep_bl_lam": 1e5},
    )
    X = pd.DataFrame(np.random.RandomState(0).randn(10, 2803))
    model._features_internal = list(X.columns)
    estimate = model._estimate_memory_usage(X, y=None)
    assert isinstance(estimate, int)
    assert estimate > 0
