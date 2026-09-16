"""Tests for Prep_EXAONE_TABULAR.

EXAONE-Tabular is a GPU-only in-context-learning foundation model whose weights are
downloaded from Hugging Face on first fit, so (matching the existing GPU-foundation-model
test posture in this repo -- e.g. no ``test_gbm.py``/``test_ta_tabpfn_3.py`` fit test) this
only checks the wrapper's structure and registration, not an actual fit/predict cycle.
End-to-end verification is via ``scripts/run_experiment.py`` (Pipeline B) on a GPU machine.
"""

from __future__ import annotations

import pytest

pytest.importorskip("autogluon")
pytest.importorskip("tabarena")

from raman_bench.models.custom.ta_exaone_tabular.model import (  # noqa: E402
    EXAONETabularModel,
    Prep_EXAONE_TABULAR,
)
from raman_bench.models.discover import discover_custom_models  # noqa: E402
from raman_bench.preprocessing.mixin import RamanPreprocessingMixin  # noqa: E402


def test_ag_key_and_name_set():
    # ConfigGenerator asserts both are non-None -- see raman_bench.models._model_info.ModelInfo.
    assert Prep_EXAONE_TABULAR.ag_key == "TA-EXAONE-TABULAR"
    assert Prep_EXAONE_TABULAR.ag_name == "TA-EXAONE-Tabular"


def test_reuses_tabarena_model_directly():
    assert issubclass(Prep_EXAONE_TABULAR, EXAONETabularModel)


def test_combines_raman_preprocessing_mixin():
    assert issubclass(Prep_EXAONE_TABULAR, RamanPreprocessingMixin)


def test_supports_classification_and_regression():
    assert set(Prep_EXAONE_TABULAR._supported_problem_types) >= {
        "binary",
        "multiclass",
        "regression",
    }


def test_gpu_only():
    assert Prep_EXAONE_TABULAR.minimum_num_gpus == 1


def test_discovered_by_registry():
    registry = discover_custom_models()
    assert "TA-EXAONE-TABULAR" in registry
    info = registry["TA-EXAONE-TABULAR"]
    assert info.model_cls is Prep_EXAONE_TABULAR
    assert info.compute == "gpu"
    assert info.display_name == "EXAONE-Tabular"
