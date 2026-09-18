"""Tests for Prep_MITRA_V2.

Mitra-v2 is a GPU-only fine-tuned-per-bag-child foundation model whose checkpoints are
downloaded from Hugging Face on first fit, so (matching the existing GPU-foundation-model
test posture in this repo -- e.g. no ``test_gbm.py``/``test_ta_tabpfn_3.py``/
``test_ta_exaone_tabular.py`` fit test) this only checks the wrapper's structure and
registration, not an actual fit/predict cycle. End-to-end verification is via
``scripts/run_experiment.py`` on a GPU machine.
"""

from __future__ import annotations

import pytest

pytest.importorskip("autogluon")
pytest.importorskip("tabarena")

from raman_bench.models.custom.ta_mitra_v2.model import (  # noqa: E402
    MitraV2Model,
    Prep_MITRA_V2,
)
from raman_bench.models.discover import discover_custom_models  # noqa: E402
from raman_bench.preprocessing.mixin import RamanPreprocessingMixin  # noqa: E402


def test_ag_key_and_name_set():
    # ConfigGenerator asserts both are non-None -- see raman_bench.models._model_info.ModelInfo.
    assert Prep_MITRA_V2.ag_key == "TA-MITRA-V2"
    assert Prep_MITRA_V2.ag_name == "TA-Mitra-v2"


def test_reuses_tabarena_model_directly():
    assert issubclass(Prep_MITRA_V2, MitraV2Model)


def test_combines_raman_preprocessing_mixin():
    assert issubclass(Prep_MITRA_V2, RamanPreprocessingMixin)


def test_supports_classification_and_regression():
    assert set(Prep_MITRA_V2.supported_problem_types()) >= {
        "binary",
        "multiclass",
        "regression",
    }


def test_gpu_only():
    assert Prep_MITRA_V2.minimum_num_gpus == 1


def test_discovered_by_registry():
    registry = discover_custom_models()
    assert "TA-MITRA-V2" in registry
    info = registry["TA-MITRA-V2"]
    assert info.model_cls is Prep_MITRA_V2
    assert info.compute == "gpu"
    assert info.display_name == "Mitra-v2"
