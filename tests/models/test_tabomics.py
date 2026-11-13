"""Tests for TabOmicsModel (classification-only tabular foundation model for wide tables).

The weights come from Hugging Face on first use, so the real-model tests are skipped when
``tabomics`` is not installed; the registration tests need only the repo.
"""

from __future__ import annotations

import numpy as np
import pytest

from raman_bench.models.discover import discover_custom_models
from raman_bench.preprocessing.wrapped_models import CLASSIFICATION_ONLY_MODELS


def _clf(n=40, f=60, k=3, seed=42):
    rng = np.random.RandomState(seed)
    y = np.tile(np.arange(k), n // k + 1)[:n]
    X = rng.randn(n, f).astype(np.float32)
    X[:, 0] += 2.0 * y  # one informative column
    return X, y.astype(str)


def test_tabomics_is_registered_as_a_classification_only_gpu_model():
    info = discover_custom_models()["TABOMICS"]
    assert info.display_name == "TabOmics" and info.compute == "gpu"
    assert "TABOMICS" in CLASSIFICATION_ONLY_MODELS
    import json
    from pathlib import Path

    gpu = json.loads((Path(__file__).parents[2] / "cluster" / "gpu_models.json").read_text())
    assert "TABOMICS" in (gpu if isinstance(gpu, list) else gpu.get("models", gpu))


def test_regression_target_is_rejected():
    pytest.importorskip("tabomics")
    from raman_bench.models.custom.tabomics.model import TabOmicsModel

    X, _ = _clf()
    with pytest.raises(ValueError, match="regression"):
        TabOmicsModel(device="cpu").fit(X, np.random.RandomState(0).randn(len(X)))


class TestTabOmicsModel:
    @pytest.fixture(autouse=True)
    def _need_package(self):
        pytest.importorskip("tabomics")

    def test_fit_predict_multiclass(self):
        from raman_bench.models.custom.tabomics.model import TabOmicsModel

        X, y = _clf()
        m = TabOmicsModel(device="cpu").fit(X, y)
        assert set(m.predict(X)).issubset(set(y))
        proba = m.predict_proba(X)
        assert proba.shape == (len(X), 3) and np.allclose(proba.sum(1), 1, atol=1e-4)

    def test_binary_proba_is_the_positive_class_column(self):
        from raman_bench.models.custom.tabomics.model import TabOmicsModel

        X, y = _clf(k=2)
        proba = TabOmicsModel(device="cpu").fit(X, y).predict_proba(X)
        assert proba.shape == (len(X),)

    def test_more_classes_than_the_native_limit_use_output_coding(self):
        pytest.importorskip("tabpfn_extensions")
        from raman_bench.models.custom.tabomics.model import TabOmicsModel

        X, y = _clf(n=60, k=5)
        m = TabOmicsModel(device="cpu", many_class_threshold=3).fit(X, y)
        assert m.predict_proba(X).shape == (len(X), 5)
