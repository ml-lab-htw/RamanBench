"""Regression test for onboarding batch 1 of TabArena's newest foundation models.

TABFM, TABPFN-V3, TABSWIFT, and MODERNNCA are TabArena-native models that (unlike
the 16 models ``test_generate_tabarena_models.py`` covers) ship *only* from
TabArena's own package -- confirmed against a real installed build, none of
TabFMModel/TabPFN3Model(TabArena's)/TabSwiftModel/ModernNCAModel exist under
those names anywhere in ``autogluon.tabular.models``. Their ``Prep_*`` classes
(``preprocessing/wrapped_models.py``) import the underlying model class directly
from ``tabarena.models.<key>.model`` and override ``ag_key`` to a short,
TA-prefix-free form (see that module's comments for why). This mirrors
``test_generate_tabarena_models.py``'s pattern: every one of these keys must
resolve a ``ConfigGenerator``/``CustomAGConfigGenerator`` whose ``model_cls``
matches the registry-resolved class, and generate at least the manual/default
config without error.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("tabarena")

REPO_ROOT = Path(__file__).resolve().parents[1]

# TABFM/TABPFN-V3/TABSWIFT wrap an upstream ConfigGenerator with an empty search
# space (manual/default config only, see each generate/<key>.py's docstring) --
# requesting more than the default config from an empty space raises
# ExhaustedSearchSpaceError. MODERNNCA's upstream generator is a
# CustomAGConfigGenerator with a real randomised search space function, so it
# can generate additional random configs.
_NO_RANDOM_HPO = {"TABFM", "TABPFN-V3", "TABSWIFT"}

TABARENA_FOUNDATION_BATCH1_KEYS = [
    "TABFM",
    "TABPFN-V3",
    "TABSWIFT",
    "MODERNNCA",
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
    missing = set(TABARENA_FOUNDATION_BATCH1_KEYS) - set(all_json)
    assert not missing, f"configs/models/all.json no longer lists: {missing}"


def test_all_models_gpu_tier_listed():
    """All four are GPU-tier -- guards against ``cluster/gpu_models.json`` drift."""
    gpu_models = json.loads((REPO_ROOT / "cluster" / "gpu_models.json").read_text())
    missing = set(TABARENA_FOUNDATION_BATCH1_KEYS) - set(gpu_models)
    assert not missing, f"cluster/gpu_models.json no longer lists: {missing}"


@pytest.mark.parametrize("key", TABARENA_FOUNDATION_BATCH1_KEYS)
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


def test_tabpfn_v3_does_not_collide_with_ta_tabpfn_3_baseline():
    """``TABPFN-V3`` (Raman-wrapped) and ``TA-TABPFN-3`` (raw TabArena baseline,
    ``raman_bench.models.custom.ta_tabpfn_3``) must resolve to two distinct
    classes sharing the same underlying ``tabarena.models.tabpfn_3.model.
    TabPFN3Model`` architecture -- not the same registry entry, and not a
    silently-overwritten one (see ``wrapped_models.py``'s ``Prep_TABPFN_V3``
    comment for why this could otherwise happen via dict-merge order).
    """
    from raman_bench.models.registry import infer_model_cls

    wrapped = infer_model_cls("TABPFN-V3")
    baseline = infer_model_cls("TA-TABPFN-3")

    assert wrapped is not baseline
    assert wrapped.ag_key == "TABPFN-V3"
    assert baseline.ag_key == "TA-TABPFN-3"
