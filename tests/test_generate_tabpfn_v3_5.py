"""Regression test for onboarding TabPFN-3.5 (``TABPFN-V3.5``).

TabPFN-3.5 is TabArena's own integration of Prior Labs' September 2026 tabular
foundation model release (``tabarena.models.tabpfn_3_5``, ``TabPFN35Model``).
Same shape as ``TABPFN-V3`` (see ``test_generate_tabarena_foundation_models.py``):
a TabArena-package-only model class (no equivalent anywhere in
``autogluon.tabular.models``), an empty/manual-config-only upstream search
space, and an ``ag_key`` overridden from the ``"TA-"``-prefixed upstream form
(``"TA-TABPFN-3.5"``) to a short, spelled-out one (``"TABPFN-V3.5"``) purely for
naming consistency with ``REALTABPFN-V2.5``/``REALTABPFN-V2.6`` -- not to dodge a
collision (nothing else in the registry claims ``"TA-TABPFN-3.5"``).

TabArena's ``tabpfn_3_5`` package also exposes a second, smaller sibling model
(``TabPFN35FastModel``, ``gen_tabpfn_3_5_fast``) that RamanBench does not wrap
here -- out of scope for this onboarding, see ``generate/tabpfn_v35.py``'s
docstring.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("tabarena")

REPO_ROOT = Path(__file__).resolve().parents[1]

MODEL_KEY = "TABPFN-V3.5"


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


def test_all_models_json_lists_key():
    """Guards against ``configs/models/all.json`` silently dropping this key."""
    all_json = json.loads((REPO_ROOT / "configs" / "models" / "all.json").read_text())
    assert MODEL_KEY in all_json


def test_gpu_tier_listed():
    """TabPFN-3.5 is a GPU foundation model -- guards ``cluster/gpu_models.json`` drift."""
    gpu_models = json.loads((REPO_ROOT / "cluster" / "gpu_models.json").read_text())
    assert MODEL_KEY in gpu_models


def test_scope_default_lists_key():
    """Guards against ``configs/v1/scope_default.json`` silently dropping this key."""
    scope = json.loads((REPO_ROOT / "configs" / "v1" / "scope_default.json").read_text())
    assert MODEL_KEY in scope["models"]


def test_generator_resolves_and_matches_registry(run_experiment):
    from raman_bench.models.registry import infer_model_cls

    try:
        model_cls = infer_model_cls(MODEL_KEY)
    except AssertionError:
        pytest.skip(f"{MODEL_KEY} unavailable in this tabarena/AutoGluon build")

    assert model_cls.ag_key == MODEL_KEY

    gen = run_experiment._import_generator(MODEL_KEY)
    assert gen.model_cls is model_cls, (
        f"{MODEL_KEY}: generator's model_cls ({gen.model_cls}) doesn't match the "
        f"registry-resolved class ({model_cls})"
    )
    # No tunable HPO surface (manual/default config only) -- same shape as TABPFN-V3.
    assert gen.manual_configs == [{}]
    assert gen.search_space == {}

    # Full experiment generation (generate_all_bag_experiments) is exercised by
    # test_generate_tabarena_foundation_models.py's parametrized suite for the sibling
    # TABPFN-V3/TABFM/TABSWIFT/MODERNNCA keys already, and is currently failing there for
    # an unrelated, pre-existing reason (a tabarena API change -- AGModelBagExperiment now
    # wants ``validation_protocol=ValidationProtocol(num_bag_folds=...)`` instead of a bare
    # ``num_bag_folds=`` kwarg -- confirmed to already fail identically on main, before this
    # commit, for every one of those four keys; not something specific to TABPFN-V3.5).
    # Not re-exercised here to avoid depending on that unstable surface a second time.
