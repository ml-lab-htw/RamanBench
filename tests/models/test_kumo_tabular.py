"""Tests for Prep_KUMO_TABULAR / Prep_KUMO_TABULAR_MEDIUM / Prep_KUMO_TABULAR_SMALL.

Kumo Tabular (NVIDIA, https://huggingface.co/blog/nvidia/kumo-tabular) is a GPU-only
tabular foundation model whose checkpoints are downloaded from Hugging Face
(``nvidia/Kumo-Tabular``) on first fit, so (matching the existing GPU-foundation-model
test posture in this repo -- e.g. ``test_tabldm.py``, ``test_ta_mitra_v2.py``, no real
fit test) this only checks the wrapper's structure and registration, not an actual
fit/predict cycle. End-to-end verification is via ``scripts/run_experiment.py`` on a
GPU machine.

PROVISIONAL onboarding: sourced from an unmerged upstream PR (autogluon/tabarena#625,
branch ``kumo-tabular``) rather than a merged-main release -- see
``requirements-tabarena-git.txt``'s pin comment.

Follows the older flat-file convention (like TABLDM, not the migrated per-model-
directory convention): the three ``Prep_*`` classes live in
``preprocessing/wrapped_models.py`` and their search-space generators in
``models/generate/kumo_tabular{,_medium,_small}.py``.
"""

from __future__ import annotations

import pytest

pytest.importorskip("autogluon")
pytest.importorskip("tabarena")

from raman_bench.preprocessing.mixin import RamanPreprocessingMixin  # noqa: E402
from raman_bench.preprocessing.wrapped_models import (  # noqa: E402
    PREPROCESSED_MODELS,
    KumoTabularMediumModel,
    KumoTabularModel,
    KumoTabularSmallModel,
    Prep_KUMO_TABULAR,
    Prep_KUMO_TABULAR_MEDIUM,
    Prep_KUMO_TABULAR_SMALL,
)

_VARIANTS = [
    ("KUMO-TABULAR", Prep_KUMO_TABULAR, KumoTabularModel, "TA-Kumo-Tabular"),
    ("KUMO-TABULAR-MEDIUM", Prep_KUMO_TABULAR_MEDIUM, KumoTabularMediumModel, "TA-Kumo-Tabular-Medium"),
    ("KUMO-TABULAR-SMALL", Prep_KUMO_TABULAR_SMALL, KumoTabularSmallModel, "TA-Kumo-Tabular-Small"),
]


@pytest.mark.parametrize("key,prep_cls,base_cls,ag_name", _VARIANTS)
def test_available_in_this_tabarena_build(key, prep_cls, base_cls, ag_name):
    # Prep_KUMO_TABULAR* is None (not raised) if this tabarena build is missing the
    # underlying class -- see wrapped_models.py's _OPTIONAL_TABARENA_MODEL_IMPORTS
    # handling.
    assert prep_cls is not None
    assert base_cls is not None


@pytest.mark.parametrize("key,prep_cls,base_cls,ag_name", _VARIANTS)
def test_ag_key_and_name_set(key, prep_cls, base_cls, ag_name):
    # ConfigGenerator asserts both are non-None -- see raman_bench.models._model_info.ModelInfo.
    assert prep_cls.ag_key == key
    assert prep_cls.ag_name == ag_name


@pytest.mark.parametrize("key,prep_cls,base_cls,ag_name", _VARIANTS)
def test_reuses_tabarena_model_directly(key, prep_cls, base_cls, ag_name):
    assert issubclass(prep_cls, base_cls)


@pytest.mark.parametrize("key,prep_cls,base_cls,ag_name", _VARIANTS)
def test_combines_raman_preprocessing_mixin(key, prep_cls, base_cls, ag_name):
    assert issubclass(prep_cls, RamanPreprocessingMixin)


@pytest.mark.parametrize("key,prep_cls,base_cls,ag_name", _VARIANTS)
def test_supports_classification_and_regression(key, prep_cls, base_cls, ag_name):
    assert set(prep_cls.supported_problem_types()) >= {
        "binary",
        "multiclass",
        "regression",
    }


@pytest.mark.parametrize("key,prep_cls,base_cls,ag_name", _VARIANTS)
def test_gpu_only(key, prep_cls, base_cls, ag_name):
    assert prep_cls.minimum_num_gpus == 1
    assert prep_cls.default_num_gpus == 1


@pytest.mark.parametrize("key,prep_cls,base_cls,ag_name", _VARIANTS)
def test_registered_in_preprocessed_models(key, prep_cls, base_cls, ag_name):
    assert PREPROCESSED_MODELS[key] is prep_cls


@pytest.mark.parametrize("key,prep_cls,base_cls,ag_name", _VARIANTS)
def test_resolves_via_registry(key, prep_cls, base_cls, ag_name):
    from raman_bench.models.registry import raman_bench_model_registry

    assert raman_bench_model_registry.key_to_cls(key=key) is prep_cls


def test_generate_module_search_space():
    from raman_bench.models.generate.kumo_tabular import gen_kumo_tabular
    from raman_bench.models.generate.kumo_tabular_medium import gen_kumo_tabular_medium
    from raman_bench.models.generate.kumo_tabular_small import gen_kumo_tabular_small

    assert gen_kumo_tabular.model_cls is Prep_KUMO_TABULAR
    assert gen_kumo_tabular_medium.model_cls is Prep_KUMO_TABULAR_MEDIUM
    assert gen_kumo_tabular_small.model_cls is Prep_KUMO_TABULAR_SMALL
