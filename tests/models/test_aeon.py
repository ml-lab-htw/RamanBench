"""Tests for the aeon model (HIVE-COTE 2)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("autogluon")
pytest.importorskip("tabarena")


def test_registered_and_classification_only():
    from raman_bench.models.discover import discover_custom_models
    from raman_bench.preprocessing.wrapped_models import (
        CLASSIFICATION_ONLY_MODELS,
        PREPROCESSED_MODELS,
    )

    registry = discover_custom_models()
    assert "HIVECOTEV2" in registry
    assert "HIVECOTEV2" in PREPROCESSED_MODELS
    assert "HIVECOTEV2" in CLASSIFICATION_ONLY_MODELS
    assert registry["HIVECOTEV2"].model_cls.ag_key == "HIVECOTEV2"


def test_hivecotev2_fit_predict_proba():
    pytest.importorskip("aeon")
    from raman_bench.models.custom.aeon.model import AeonClassifierModel

    rng = np.random.default_rng(0)
    y = np.repeat([0, 1, 2], 10)
    X = rng.normal(size=(30, 60)).astype(np.float32)  # float32 like the bridge's input
    X[np.arange(30), 10 + 15 * y] += 5
    model = AeonClassifierModel("HIVECOTEV2", n_jobs=1).fit(X, y, time_limit=12)
    proba = model.predict_proba(X)
    assert proba.shape == (30, 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1)
    assert list(model.classes_) == [0, 1, 2]


def test_continuous_target_raises():
    from raman_bench.models.custom.aeon.model import AeonClassifierModel

    with pytest.raises(ValueError, match="classification-only"):
        AeonClassifierModel().fit(np.zeros((4, 10)), np.array([0.1, 0.2, 0.3, 0.4]))
