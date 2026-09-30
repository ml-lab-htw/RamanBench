"""Regression test for LIMIX2 (``Prep_LIMIX2``, wrapping
``tabarena.models.limix_2.model.LimiX2Model``), onboarded after ``LIMIX`` (v1)
-- same "TabArena-package-only" staging situation, but a genuinely separate
model/checkpoint (Stable AI's second-generation tabular foundation model, not
a version bump of the first). See ``raman_bench/models/generate/limix2.py``
and ``wrapped_models.py``'s ``Prep_LIMIX2`` comment block for the full
rationale, and ``Dockerfile.limix2``/``requirements-limix2-git.txt`` for why
this model needs a dedicated container (Python >=3.12, torch>=2.9.1 -- both
incompatible with the main image).

Unlike ``LIMIX`` (v1), ``LimiX2Model`` caps ``max_classes=10`` via the normal
DECLARATIVE ``_default_auxiliary_params_extra`` class attribute (confirmed by
reading ``tabarena/models/limix_2/model.py`` directly), so ``Prep_LIMIX2`` is
built the ordinary way via ``_make_optional_prep_class`` -- no hand-written
class/method override needed, unlike ``Prep_LIMIX``.
"""

from __future__ import annotations

import importlib.util
import json
import pickle
from pathlib import Path

import pytest

pytest.importorskip("tabarena")

REPO_ROOT = Path(__file__).resolve().parents[2]


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


def _prep_limix2():
    from raman_bench.preprocessing.wrapped_models import Prep_LIMIX2

    if Prep_LIMIX2 is None:
        pytest.skip("LimiX2Model unavailable in this tabarena build")
    return Prep_LIMIX2


def test_all_json_keys_covered():
    all_json = json.loads((REPO_ROOT / "configs" / "models" / "all.json").read_text())
    assert "LIMIX2" in all_json

    foundation_json = json.loads(
        (REPO_ROOT / "configs" / "models" / "tabular_foundation.json").read_text()
    )
    assert "LIMIX2" in foundation_json

    gpu_models = json.loads((REPO_ROOT / "cluster" / "gpu_models.json").read_text())
    assert "LIMIX2" in gpu_models


def test_generator_resolves_and_matches_registry(run_experiment):
    from raman_bench.models.registry import infer_model_cls

    Prep_LIMIX2 = _prep_limix2()

    model_cls = infer_model_cls("LIMIX2")
    assert model_cls is Prep_LIMIX2

    gen = run_experiment._import_generator("LIMIX2")
    assert gen.model_cls is model_cls

    # Empty upstream search space (can_hpo=False, manual_configs=[{}]) -- same
    # shape as LIMIX (v1), NORI, SAP_RPT_OSS, ORIONMSP.
    experiments = gen.generate_all_bag_experiments(
        num_random_configs=0,
        time_limit=60,
        num_bag_folds=2,
        fold_fitting_strategy="sequential_local",
        add_seed="fold-config-wise",
    )
    assert len(experiments) >= 1
    assert experiments[0].method_kwargs["model_cls"] is model_cls


def test_max_classes_cap_lifted():
    """LimiX2Model declares ``_default_auxiliary_params_extra = {"max_classes": 10}``
    -- confirmed against real RamanBench classification datasets that exceed
    it (``bacteria_identification``: 30 classes, ``rruff_mineral_raw``: 79).
    Unlike ``Prep_LIMIX``, this is the DECLARATIVE class-attribute shape, so
    ``_make_optional_prep_class``'s ``_default_auxiliary_params_extra`` kwarg
    lifts it normally (same mechanism as Mitra/TabDPT/TabICL/RealTabPFN),
    without needing a hand-written method override.
    """
    Prep_LIMIX2 = _prep_limix2()

    model = Prep_LIMIX2(problem_type="multiclass")
    aux = model._get_default_auxiliary_params()
    assert aux["max_classes"] is None
    assert aux["max_rows"] is None
    assert aux["max_features"] is None


def test_picklable():
    """Guards against the ``__module__`` pickling trap
    ``_make_optional_prep_class``'s docstring describes (AutoGluon's
    ``TabularPredictor.save()`` pickles the model class at the end of every
    real fit)."""
    Prep_LIMIX2 = _prep_limix2()

    assert Prep_LIMIX2.__module__ == "raman_bench.preprocessing.wrapped_models"
    assert pickle.loads(pickle.dumps(Prep_LIMIX2)) is Prep_LIMIX2
