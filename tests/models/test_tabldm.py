"""Tests for Prep_TABLDM.

TabLDM is a GPU-only tabular foundation model (dual-stream column embedder + MoE
backbone) whose checkpoints are downloaded from Hugging Face (``occams/Xiaomi-TabLDM``)
on first fit, so (matching the existing GPU-foundation-model test posture in this repo --
e.g. ``test_ta_mitra_v2.py``, no ``test_gbm.py``/``test_ta_tabpfn_3.py`` fit test) this
only checks the wrapper's structure and registration, not an actual fit/predict cycle.
End-to-end verification is via ``scripts/run_experiment.py`` on a GPU machine.

Unlike ``TA-MITRA-V2`` (a migrated per-model-directory custom model, discovered via
``raman_bench.models.discover.discover_custom_models``), ``TABLDM`` follows the older
flat-file convention: ``Prep_TABLDM`` lives in ``preprocessing/wrapped_models.py`` and
its search-space generator in ``models/generate/tabldm.py`` -- see that module's own
docstring, and ``raman_bench.models.registry`` (which merges both conventions into one
``raman_bench_model_registry``).
"""

from __future__ import annotations

import pytest

pytest.importorskip("autogluon")
pytest.importorskip("tabarena")

from raman_bench.preprocessing.mixin import RamanPreprocessingMixin  # noqa: E402
from raman_bench.preprocessing.wrapped_models import (  # noqa: E402
    PREPROCESSED_MODELS,
    Prep_TABLDM,
    TabLDMModel,
)


def test_available_in_this_tabarena_build():
    # Prep_TABLDM is None (not raised) if this tabarena build is missing TabLDMModel --
    # see wrapped_models.py's _OPTIONAL_TABARENA_MODEL_IMPORTS handling.
    assert Prep_TABLDM is not None
    assert TabLDMModel is not None


def test_ag_key_and_name_set():
    # ConfigGenerator asserts both are non-None -- see raman_bench.models._model_info.ModelInfo.
    assert Prep_TABLDM.ag_key == "TABLDM"
    assert Prep_TABLDM.ag_name == "TA-Xiaomi-TabLDM"


def test_reuses_tabarena_model_directly():
    assert issubclass(Prep_TABLDM, TabLDMModel)


def test_combines_raman_preprocessing_mixin():
    assert issubclass(Prep_TABLDM, RamanPreprocessingMixin)


def test_supports_classification_and_regression():
    assert set(Prep_TABLDM.supported_problem_types()) >= {
        "binary",
        "multiclass",
        "regression",
    }


def test_gpu_only():
    assert Prep_TABLDM.minimum_num_gpus == 1
    assert Prep_TABLDM.default_num_gpus == 1


def test_registered_in_preprocessed_models():
    assert PREPROCESSED_MODELS["TABLDM"] is Prep_TABLDM


def test_resolves_via_registry():
    from raman_bench.models.registry import raman_bench_model_registry

    assert raman_bench_model_registry.key_to_cls(key="TABLDM") is Prep_TABLDM


def test_generate_module_search_space():
    from raman_bench.models.generate.tabldm import gen_tabldm

    assert gen_tabldm.model_cls is Prep_TABLDM
