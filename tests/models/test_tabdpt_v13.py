"""Tests for Prep_TABDPT_V13.

TabDPT v1.3 (``tabarena.models.tabdpt.model.TabDPTv13Model``) is a GPU-only tabular
foundation model whose checkpoint is downloaded from Hugging Face (``Layer6/TabDPT``,
``tabdpt1_3.safetensors``) on first fit, so (matching the existing GPU-foundation-model
test posture in this repo -- e.g. ``test_tabldm.py``, ``test_kumo_tabular.py``, no real
fit test) this only checks the wrapper's structure and registration, not an actual
fit/predict cycle. End-to-end verification is via ``scripts/run_experiment.py`` on a
GPU machine.

IMPORTANT: ``TabDPTv13Model`` is a DIFFERENT class from the plain ``TabDPTModel``
RamanBench's existing ``TABDPT`` key wraps -- that one is
``autogluon.tabular.models.TabDPTModel``, an already-graduated-into-AutoGluon-core
class that floats with whatever ``tabdpt`` package version is installed, NOT
tabarena's own (checkpoint-version-pinned, ``tabdpt<1.2``, ``superseded=True``)
``tabarena.models.tabdpt.model.TabDPTModel``. ``TabDPTv13Model`` instead subclasses
``TabDPTTurboModel`` (v1.2), itself a sibling of tabarena's own v1.1 ``TabDPTModel``
under a shared ``TabDPTModelBase``. See ``wrapped_models.py``'s
``_OPTIONAL_TABARENA_MODEL_IMPORTS`` entry for ``TabDPTv13Model`` for the full
checkpoint-compatibility investigation performed before this onboarding (confirmed:
the cluster's installed ``tabdpt`` is already 1.3.1, matching what
``TabDPTv13Model`` needs, and the existing ``TABDPT`` entry already empirically works
fine against that same installed version since it isn't hardcoded to an older
checkpoint format -- no coexistence conflict).

Follows the older flat-file convention (like TABDPT, not the migrated per-model-
directory convention): ``Prep_TABDPT_V13`` lives in
``preprocessing/wrapped_models.py`` and its search-space generator in
``models/generate/tabdpt_v13.py``.
"""

from __future__ import annotations

import pytest

pytest.importorskip("autogluon")
pytest.importorskip("tabarena")

from raman_bench.preprocessing.mixin import RamanPreprocessingMixin  # noqa: E402
from raman_bench.preprocessing.wrapped_models import (  # noqa: E402
    PREPROCESSED_MODELS,
    Prep_TABDPT,
    Prep_TABDPT_V13,
    TabDPTModel,
    TabDPTv13Model,
)


def test_available_in_this_tabarena_build():
    # Prep_TABDPT_V13 is None (not raised) if this tabarena build is missing
    # TabDPTv13Model -- see wrapped_models.py's _OPTIONAL_TABARENA_MODEL_IMPORTS
    # handling.
    assert Prep_TABDPT_V13 is not None
    assert TabDPTv13Model is not None


def test_ag_key_and_name_set():
    # ConfigGenerator asserts both are non-None -- see raman_bench.models._model_info.ModelInfo.
    assert Prep_TABDPT_V13.ag_key == "TABDPT-V1.3"
    assert Prep_TABDPT_V13.ag_name == "TA-TabDPT-1.3"


def test_reuses_tabarena_model_directly():
    assert issubclass(Prep_TABDPT_V13, TabDPTv13Model)


def test_distinct_from_existing_tabdpt_wrapper():
    # TabDPTv13Model (tabarena's own package) is NOT the same class as TabDPTModel
    # (AutoGluon-core, already graduated) -- the existing TABDPT key wraps the latter.
    # See this module's own docstring for the full investigation.
    assert Prep_TABDPT_V13 is not Prep_TABDPT
    assert not issubclass(TabDPTv13Model, TabDPTModel)
    assert Prep_TABDPT_V13.ag_key != Prep_TABDPT.ag_key


def test_combines_raman_preprocessing_mixin():
    assert issubclass(Prep_TABDPT_V13, RamanPreprocessingMixin)


def test_supports_classification_and_regression():
    assert set(Prep_TABDPT_V13.supported_problem_types()) >= {
        "binary",
        "multiclass",
        "regression",
    }


def test_gpu_only():
    assert Prep_TABDPT_V13.minimum_num_gpus == 0.5
    assert Prep_TABDPT_V13.default_num_gpus == 1


def test_registered_in_preprocessed_models():
    assert PREPROCESSED_MODELS["TABDPT-V1.3"] is Prep_TABDPT_V13


def test_resolves_via_registry():
    from raman_bench.models.registry import raman_bench_model_registry

    assert raman_bench_model_registry.key_to_cls(key="TABDPT-V1.3") is Prep_TABDPT_V13


def test_generate_module_search_space():
    from raman_bench.models.generate.tabdpt_v13 import gen_tabdpt_v13

    assert gen_tabdpt_v13.model_cls is Prep_TABDPT_V13
