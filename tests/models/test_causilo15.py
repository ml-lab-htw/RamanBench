"""Tests for Causilo15Model (Causilo 1.5, needs causilo>=1.5.0) and for the 1.0 wrapper staying on Causilo 1.0."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

causilo = pytest.importorskip("causilo")
if "model_version" not in inspect.signature(causilo.CausiloClassifier).parameters:
    pytest.skip("needs causilo>=1.5.0", allow_module_level=True)

from raman_bench.models.custom.causilo15.model import Causilo15LightModel, Causilo15Model  # noqa: E402


def _data(n=40, f=12, classes=None, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, f).astype(np.float32)
    y = rng.choice(classes, size=n) if classes else rng.randn(n).astype(np.float32)
    return X, y


def test_registered_under_its_own_key():
    from raman_bench.models.discover import discover_custom_models

    registry = discover_custom_models()
    assert {"CAUSILO", "CAUSILO-1.5", "CAUSILO-1.5-LIGHT"} <= set(registry)
    assert registry["CAUSILO-1.5"].model_cls.ag_name == "Causilo-1.5"
    assert registry["CAUSILO-1.5-LIGHT"].model_cls.ag_name == "Causilo-1.5-Light"


@pytest.mark.parametrize("classes", [["A", "B"], ["A", "B", "C"]])
def test_classification(classes):
    X, y = _data(classes=classes)
    m = Causilo15Model(device="cpu", n_estimators=2).fit(X, y)
    assert set(m.predict(X)).issubset(set(classes))
    proba = m.predict_proba(X)
    if len(classes) == 2:
        assert proba.ndim == 1 and np.all((proba >= 0) & (proba <= 1))
    else:
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_many_classes_use_the_librarys_ecoc():
    classes = [f"c{i}" for i in range(14)]  # above the checkpoint's native class limit
    X, y = _data(n=70, classes=classes)
    m = Causilo15Model(device="cpu", n_estimators=2).fit(X, y)
    assert m.predict_proba(X).shape == (len(X), len(np.unique(y)))


def test_regression():
    X, y = _data()
    preds = Causilo15Model(device="cpu", n_estimators=2).fit(X, y).predict(X)
    assert preds.shape == (len(X),) and np.issubdtype(preds.dtype, np.floating)


def test_models_use_their_own_network_version():
    """CAUSILO-1.5 asks the library for 1.5, CAUSILO for 1.0, whatever the installed package defaults to."""
    from raman_bench.models.custom.causilo.model import CausiloModel

    X, y = _data(classes=["A", "B"])
    new = Causilo15Model(device="cpu", n_estimators=2).fit(X, y)
    old = CausiloModel(device="cpu", n_estimators=2).fit(X, y)
    assert new.model_.model_version == "1.5"
    assert old.model_._estimator.model_version == "1.0"  # the 1.0 wrapper nests the library estimator


def test_light_variant_asks_for_light_mode():
    X, y = _data(classes=["A", "B"])
    assert Causilo15Model().get_params()["mode"] is None
    light = Causilo15LightModel(device="cpu").fit(X, y)  # mode and n_estimators are exclusive in causilo
    assert light.get_params()["mode"] == "light" and light.model_.mode == "light"
    assert light.predict(X).shape == (len(X),)
