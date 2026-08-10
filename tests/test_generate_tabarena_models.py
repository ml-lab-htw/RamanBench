"""Regression test for the TabArena-native model -> ConfigGenerator wiring gap.

Before this fix, ``scripts/run_experiment.py::_import_generator`` had no
``raman_bench.models.generate.<key>`` (nor ``custom/<key>/hpo.py``) module for any
of the "built-in" AutoGluon models in ``configs/models/all.json`` (RF, CAT, XGB,
KNN, LR, NN_TORCH, FASTAI, DUMMY, REALMLP, MITRA, TABM, TABDPT, TABICL,
REALTABPFN-V2, REALTABPFN-V2.5, XT) -- every one of them could be resolved to a
``Prep_*`` model class via ``raman_bench.models.registry.infer_model_cls``, but
had no way to actually generate a hyperparameter config to run, so none of them
could go through the real v1 cluster pipeline (see
``raman_bench/models/generate/_tabarena_adapter.py`` for the fix). This checks
every one of them now resolves a ``ConfigGenerator``/``CustomAGConfigGenerator``
whose ``model_cls`` matches the registry's resolved class, and that at least the
manual/default config can be generated without error.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Keys with an empty upstream search space (see generate/dummy.py,
# generate/mitra.py, generate/realtabpfn_v2.py) -- requesting more than the
# manual/default config from an empty space raises ExhaustedSearchSpaceError.
_NO_RANDOM_HPO = {"DUMMY", "MITRA", "REALTABPFN-V2"}

# The 16 "built-in" AutoGluon models this fix wires up (the 15 originally
# reported + XT, caught by the exact same gap -- see generate/xt.py's
# docstring). AUTOGLUON (the whole-predictor meta-preset) is deliberately
# excluded: it's not a per-model ConfigGenerator at all (see
# tabarena.models.automl.generate_autogluon_experiments), a different, separate
# gap not addressed here.
TABARENA_NATIVE_KEYS = [
    "RF",
    "CAT",
    "XGB",
    "XT",
    "KNN",
    "LR",
    "NN_TORCH",
    "FASTAI",
    "DUMMY",
    "REALMLP",
    "MITRA",
    "TABM",
    "TABDPT",
    "TABICL",
    "REALTABPFN-V2",
    "REALTABPFN-V2.5",
]


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
    missing = set(TABARENA_NATIVE_KEYS) - set(all_json)
    assert not missing, f"configs/models/all.json no longer lists: {missing}"


@pytest.mark.parametrize("key", TABARENA_NATIVE_KEYS)
def test_generator_resolves_and_matches_registry(key, run_experiment):
    from raman_bench.models.registry import infer_model_cls

    try:
        model_cls = infer_model_cls(key)
    except AssertionError:
        pytest.skip(f"{key} unavailable in this AutoGluon build")

    gen = run_experiment._import_generator(key)
    assert gen.model_cls is model_cls, (
        f"{key}: generator's model_cls ({gen.model_cls}) doesn't match the "
        f"registry-resolved class ({model_cls})"
    )

    num_random_configs = 0 if key in _NO_RANDOM_HPO else 2
    experiments = gen.generate_all_bag_experiments(
        num_random_configs=num_random_configs,
        time_limit=60,
        num_bag_folds=2,
        fold_fitting_strategy="sequential_local",
        add_seed="fold-config-wise",
    )
    assert len(experiments) >= 1
    assert experiments[0].method_kwargs["model_cls"] is model_cls
