"""Regression test for onboarding batch 3 of TabArena's newest models -- the last
of the 14-model TabArena-native onboarding effort (batch 1:
``test_generate_tabarena_foundation_models.py``, batch 2:
``test_generate_tabarena_tree_models.py``).

NORI, SAP_RPT_OSS, ORIONMSP, ILTM, LIMIX, and TABSTAR ship only from TabArena's
own package (``tabarena.models.<key>.model``), same "not (yet) graduated into
AutoGluon core" situation as every batch-1/2 model. All six are tabular
*foundation* models (pretrained in-context-learning or LoRA-fine-tuned), so this
lives alongside batch 1's file rather than batch 2's tree/boosting one. Every one
of these keys must resolve a ``ConfigGenerator``/``CustomAGConfigGenerator``
whose ``model_cls`` matches the registry-resolved class, and generate at least
the manual/default config without error.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# NORI, SAP_RPT_OSS, ORIONMSP, and LIMIX all wrap an upstream ConfigGenerator with
# an empty search space (manual/default config only -- see each generate/<key>.py's
# docstring); requesting more than the default config from an empty space raises
# ExhaustedSearchSpaceError. ILTM's upstream generator is a CustomAGConfigGenerator
# with a real (24-dimension, mostly continuous) ConfigSpace search space, and
# TABSTAR's is a plain ConfigGenerator with a real (6-knob, all-categorical)
# autogluon.common.space search space -- both can generate additional random
# configs.
_NO_RANDOM_HPO = {"NORI", "SAP_RPT_OSS", "ORIONMSP", "LIMIX"}

TABARENA_BATCH3_KEYS = [
    "NORI",
    "SAP_RPT_OSS",
    "ORIONMSP",
    "ILTM",
    "LIMIX",
    "TABSTAR",
]

# All six are pretrained/fine-tuned tabular foundation models -- unlike batch 2's
# tree/boosting family, they belong in tabular_foundation.json, not
# traditional_ml.json.
TABARENA_BATCH3_FOUNDATION_KEYS = list(TABARENA_BATCH3_KEYS)

# All six are GPU-tier in cluster/gpu_models.json. Five (NORI, SAP_RPT_OSS,
# ORIONMSP, ILTM, TABSTAR) auto-detect a GPU at fit time and fall back to CPU
# when none is available (same shape as ModernNCA/RealMLP/XRFM) -- confirmed
# via each model's own ``_fit``: every one only raises if more GPUs are
# *requested* than are actually available, never merely because none are
# present. LIMIX is the exception: confirmed via a real local run that its
# bundled default (retrieval-mode) config is rejected outright on CPU by
# ``LimiXPredictor.__init__`` regardless of requested GPU count -- see
# generate/limix.py's docstring. It's still GPU-tier here (more strictly so
# than the other five, since it has no CPU fallback at all for its default
# config).
TABARENA_BATCH3_GPU_KEYS = list(TABARENA_BATCH3_KEYS)


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
    missing = set(TABARENA_BATCH3_KEYS) - set(all_json)
    assert not missing, f"configs/models/all.json no longer lists: {missing}"


def test_tabular_foundation_json_keys_covered():
    """All six are pretrained/fine-tuned foundation models -- belong in
    tabular_foundation.json, not traditional_ml.json."""
    foundation_json = json.loads(
        (REPO_ROOT / "configs" / "models" / "tabular_foundation.json").read_text()
    )
    missing = set(TABARENA_BATCH3_FOUNDATION_KEYS) - set(foundation_json)
    assert not missing, f"configs/models/tabular_foundation.json no longer lists: {missing}"


def test_gpu_models_json_keys_covered():
    """All six are GPU-tier -- guards against ``cluster/gpu_models.json`` drift."""
    gpu_models = json.loads((REPO_ROOT / "cluster" / "gpu_models.json").read_text())
    missing = set(TABARENA_BATCH3_GPU_KEYS) - set(gpu_models)
    assert not missing, f"cluster/gpu_models.json no longer lists: {missing}"


def test_orionmsp_is_classification_only():
    """OrionMSPModel.supported_problem_types() == ["binary", "multiclass"] --
    ORIONMSP must be listed so predictions.py's old v0 path skips it (rather than
    crashing) on regression datasets."""
    from raman_bench.preprocessing.wrapped_models import CLASSIFICATION_ONLY_MODELS

    assert "ORIONMSP" in CLASSIFICATION_ONLY_MODELS


def test_nori_is_regression_only():
    """NoriModel.supported_problem_types() == ["regression"] -- NORI must be
    listed so predictions.py's old v0 path skips it (rather than crashing) on
    classification datasets. This is the first model needing
    REGRESSION_ONLY_MODELS (the mirror of CLASSIFICATION_ONLY_MODELS)."""
    from raman_bench.preprocessing.wrapped_models import REGRESSION_ONLY_MODELS

    assert "NORI" in REGRESSION_ONLY_MODELS


@pytest.mark.parametrize("key", TABARENA_BATCH3_KEYS)
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


def test_limix_max_classes_cap_lifted():
    """Regression test: found via a real local check, not anticipated up front.

    ``LimiXModel._get_default_auxiliary_params`` (unlike Mitra/TabDPT/TabICL/
    RealTabPFN's classes) fully overrides the method and unconditionally
    ``dict.update()``s ``max_classes=10`` back in after calling ``super()`` --
    that clobbers a declarative ``_default_auxiliary_params_extra`` class
    attribute no matter where it sits in the MRO (confirmed: passing
    ``_default_auxiliary_params_extra=_NO_FOUNDATION_MODEL_FEATURE_CAP`` to
    ``_make_optional_prep_class`` had no effect). ``Prep_LIMIX`` is instead a
    hand-written class (not built via ``_make_optional_prep_class``) that
    overrides ``_get_default_auxiliary_params`` itself and re-clobbers the cap
    back to ``None`` afterwards -- see wrapped_models.py's batch-3 comment
    block. Real RamanBench classification datasets exceed the original cap of
    10 (``bacteria_identification``: 30 classes, ``rruff_mineral_raw``: 79).
    """
    from raman_bench.preprocessing.wrapped_models import Prep_LIMIX

    model = Prep_LIMIX(problem_type="multiclass")
    aux = model._get_default_auxiliary_params()
    assert aux["max_classes"] is None
    assert aux["max_rows"] is None
    assert aux["max_features"] is None


def test_limix_picklable():
    """Regression guard for the ``__module__`` pickling trap
    ``_make_optional_prep_class``'s docstring describes -- ``Prep_LIMIX`` is a
    hand-written ``class`` statement (not built via ``type()``), so it should
    never have been at risk, but this locks that in given the unusual
    conditional-class-definition shape it needed (see wrapped_models.py)."""
    import pickle

    from raman_bench.preprocessing.wrapped_models import Prep_LIMIX

    assert Prep_LIMIX.__module__ == "raman_bench.preprocessing.wrapped_models"
    assert pickle.loads(pickle.dumps(Prep_LIMIX)) is Prep_LIMIX
