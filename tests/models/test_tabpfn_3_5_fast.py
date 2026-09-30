"""Tests for Prep_TABPFN_V3_5_FAST.

TabPFN-3.5-Fast is the smaller and faster sibling of TabPFN-3.5, released alongside
it from the same Hugging Face repo (Prior Labs reports up to 6x faster inference; the
model is marked alpha upstream). Same limits, license and estimator surface as
TabPFN-3.5 -- it subclasses ``TabPFN35Model`` directly -- only the checkpoint differs,
so (matching the existing GPU-foundation-model test posture in this repo -- e.g.
``test_tabldm.py``, ``test_kumo_tabular.py``, no real fit test) this only checks the
wrapper's structure and registration, not an actual fit/predict cycle. End-to-end
verification is via ``scripts/run_experiment.py`` on a GPU machine.

Follows the older flat-file convention (like TABPFN-V3.5, not the migrated per-model-
directory convention): ``Prep_TABPFN_V3_5_FAST`` lives in
``preprocessing/wrapped_models.py`` and its search-space generator in
``models/generate/tabpfn_v35_fast.py``.
"""

from __future__ import annotations

import pytest

pytest.importorskip("autogluon")
pytest.importorskip("tabarena")

from raman_bench.preprocessing.mixin import RamanPreprocessingMixin  # noqa: E402
from raman_bench.preprocessing.wrapped_models import (  # noqa: E402
    PREPROCESSED_MODELS,
    Prep_TABPFN_V3_5_FAST,
    TabPFN35FastModel,
    TabPFN35Model,
)


def test_available_in_this_tabarena_build():
    # Prep_TABPFN_V3_5_FAST is None (not raised) if this tabarena build is missing
    # TabPFN35FastModel -- see wrapped_models.py's _OPTIONAL_TABARENA_MODEL_IMPORTS
    # handling.
    assert Prep_TABPFN_V3_5_FAST is not None
    assert TabPFN35FastModel is not None


def test_ag_key_and_name_set():
    # ConfigGenerator asserts both are non-None -- see raman_bench.models._model_info.ModelInfo.
    assert Prep_TABPFN_V3_5_FAST.ag_key == "TABPFN-V3.5-FAST"
    assert Prep_TABPFN_V3_5_FAST.ag_name == "TA-TabPFN-3.5-Fast"


def test_reuses_tabarena_model_directly():
    assert issubclass(Prep_TABPFN_V3_5_FAST, TabPFN35FastModel)


def test_is_a_tabpfn_35_subclass():
    # TabPFN35FastModel subclasses TabPFN35Model directly (same file) -- same limits,
    # license and estimator surface, only the checkpoint differs.
    assert issubclass(TabPFN35FastModel, TabPFN35Model)


def test_combines_raman_preprocessing_mixin():
    assert issubclass(Prep_TABPFN_V3_5_FAST, RamanPreprocessingMixin)


def test_supports_classification_and_regression():
    assert set(Prep_TABPFN_V3_5_FAST.supported_problem_types()) >= {
        "binary",
        "multiclass",
        "regression",
    }


def test_gpu_only():
    assert Prep_TABPFN_V3_5_FAST.minimum_num_gpus == 1
    assert Prep_TABPFN_V3_5_FAST.default_num_gpus == 1


def test_registered_in_preprocessed_models():
    assert PREPROCESSED_MODELS["TABPFN-V3.5-FAST"] is Prep_TABPFN_V3_5_FAST


def test_resolves_via_registry():
    from raman_bench.models.registry import raman_bench_model_registry

    assert raman_bench_model_registry.key_to_cls(key="TABPFN-V3.5-FAST") is Prep_TABPFN_V3_5_FAST


def test_generate_module_search_space():
    from raman_bench.models.generate.tabpfn_v35_fast import gen_tabpfn_v35_fast

    assert gen_tabpfn_v35_fast.model_cls is Prep_TABPFN_V3_5_FAST
