"""Tests for the aeon models (HIVE-COTE 2 and its components) and the HC2 assembly."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("autogluon")
pytest.importorskip("tabarena")

from raman_bench.models.custom.aeon.assemble import (  # noqa: E402
    ALPHA,
    ASSEMBLED_KEY,
    COMPONENTS,
    assemble,
    assemble_result,
)

KEYS = {"HIVECOTEV2", "STC", "DRCIF", "ARSENAL", "TDE"}


def test_registered_and_classification_only():
    from raman_bench.models.discover import discover_custom_models
    from raman_bench.preprocessing.wrapped_models import (
        CLASSIFICATION_ONLY_MODELS,
        PREPROCESSED_MODELS,
    )

    registry = discover_custom_models()
    assert KEYS <= set(registry)
    assert KEYS <= set(PREPROCESSED_MODELS)
    assert KEYS <= CLASSIFICATION_ONLY_MODELS
    assert {registry[k].model_cls.ag_key for k in KEYS} == KEYS


def test_arsenal_fit_predict_proba():
    pytest.importorskip("aeon")
    from raman_bench.models.custom.aeon.model import AeonClassifierModel

    rng = np.random.default_rng(0)
    y = np.repeat([0, 1, 2], 10)
    X = rng.normal(size=(30, 60)).astype(np.float32)  # float32 like the bridge's input
    X[np.arange(30), 10 + 15 * y] += 5
    model = AeonClassifierModel("ARSENAL", n_jobs=1).fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (30, 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1)
    assert list(model.classes_) == [0, 1, 2]


def test_hivecotev2_fit_predict_proba():
    pytest.importorskip("aeon")
    from raman_bench.models.custom.aeon.model import AeonClassifierModel

    rng = np.random.default_rng(0)
    y = np.repeat([0, 1, 2], 10)
    X = rng.normal(size=(30, 60)).astype(np.float32)
    X[np.arange(30), 10 + 15 * y] += 5
    model = AeonClassifierModel("HIVECOTEV2", n_jobs=1).fit(X, y, time_limit=12)
    assert model.predict_proba(X).shape == (30, 3)


def test_continuous_target_raises():
    from raman_bench.models.custom.aeon.model import AeonClassifierModel

    with pytest.raises(ValueError, match="classification-only"):
        AeonClassifierModel("TDE").fit(np.zeros((4, 10)), np.array([0.1, 0.2, 0.3, 0.4]))


def _result(
    name, pred_val, pred_test, y_val, y_test, *, problem_type="multiclass", metric="log_loss"
):
    fw = f"{name}_c1_BAG_L1"
    return {
        "framework": fw,
        "metric": metric,
        "problem_type": problem_type,
        "metric_error": 0.0,
        "metric_error_val": 0.0,
        "time_train_s": 10.0,
        "time_infer_s": 1.0,
        "task_metadata": {"name": "t__0", "fold": 0, "repeat": 0},
        "method_metadata": {"model_type": name, "name_prefix": name, "model_cls": f"Prep_{name}"},
        "simulation_artifacts": {
            "y_val": y_val,
            "y_test": y_test,
            "y_val_idx": np.arange(len(y_val)),
            "y_test_idx": np.arange(len(y_test)),
            "pred_proba_dict_val": {fw: pred_val},
            "pred_proba_dict_test": {fw: pred_test},
            "bag_info": {"pred_proba_test_per_child": [pred_test, pred_test]},
        },
    }


def test_assemble_weights_components_by_oof_accuracy_to_the_fourth():
    y_val, y_test = np.array([0, 1, 1, 0]), np.array([0, 1])
    right = np.array([[0.9, 0.1], [0.2, 0.8], [0.3, 0.7], [0.6, 0.4]])  # 4/4 correct
    half = np.array([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.4, 0.6]])  # 2/4 correct
    tests = {
        c: np.array([[0.7, 0.3], [0.4, 0.6]]) if c != "TDE" else np.array([[0.1, 0.9], [0.9, 0.1]])
        for c in COMPONENTS
    }
    results = {
        c: _result(c, half if c == "TDE" else right, tests[c], y_val, y_test) for c in COMPONENTS
    }

    out = assemble_result(results)
    w = np.array([1.0, 1.0, 1.0, 0.5**ALPHA])
    expected = sum(wi * tests[c] for wi, c in zip(w, COMPONENTS)) / w.sum()
    fw = f"{ASSEMBLED_KEY}_c1_BAG_L1"
    np.testing.assert_allclose(out["simulation_artifacts"]["pred_proba_dict_test"][fw], expected)
    np.testing.assert_allclose(
        out["simulation_artifacts"]["bag_info"]["pred_proba_test_per_child"][0], expected
    )
    assert out["framework"] == fw and out["method_metadata"]["model_type"] == ASSEMBLED_KEY
    assert out["time_train_s"] == 40.0
    assert out["metric_error"] == pytest.approx(-np.mean(np.log(expected[[0, 1], [0, 1]])))
    assert out["method_metadata"]["assembled_from"]["TDE"] == pytest.approx(0.5**ALPHA)


def test_assemble_binary_keeps_positive_class_column():
    y_val, y_test = np.array([0, 1, 1, 0]), np.array([0, 1, 1])
    results = {
        c: _result(c, np.array([0.2, 0.7, 0.8, 0.3]), np.array([0.1, 0.9, 0.6]), y_val, y_test,
                   problem_type="binary", metric="roc_auc")
        for c in COMPONENTS
    }  # fmt: skip
    out = assemble_result(results)
    pred = out["simulation_artifacts"]["pred_proba_dict_test"][f"{ASSEMBLED_KEY}_c1_BAG_L1"]
    np.testing.assert_allclose(pred, [0.1, 0.9, 0.6])
    assert out["metric_error"] == pytest.approx(0.0)  # 1 - AUC of a perfect ranking


def test_assemble_writes_only_complete_splits(tmp_path):
    from tabarena.utils.pickle_utils import dumps_pickle, load_pickle

    y_val, y_test = np.array([0, 1, 1, 0]), np.array([0, 1])
    pv, pt = (
        np.array([[0.9, 0.1], [0.2, 0.8], [0.3, 0.7], [0.6, 0.4]]),
        np.array([[0.7, 0.3], [0.4, 0.6]]),
    )
    for split, comps in {"0_0": COMPONENTS, "0_1": COMPONENTS[:3]}.items():
        for c in comps:
            d = tmp_path / f"{c}_c1_BAG_L1" / "t__0" / split
            d.mkdir(parents=True)
            (d / "results.pkl").write_bytes(
                dumps_pickle(_result(c, pv, pt, y_val, y_test), compress=True)
            )
    written = assemble(str(tmp_path))
    assert [p.split("/")[-2] for p in written] == ["0_0"]
    assert load_pickle(written[0])["framework"] == f"{ASSEMBLED_KEY}_c1_BAG_L1"
    assert assemble(str(tmp_path)) == []  # already there
