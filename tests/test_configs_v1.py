"""Benchmark v1's config files agree with each other: scopes, target lists, exclusions and dataset lists."""

import json
from pathlib import Path

import pytest

V1 = Path(__file__).resolve().parents[1] / "configs" / "v1"


def _load(*parts):
    return json.loads(V1.joinpath(*parts).read_text())


@pytest.fixture(scope="module")
def targets():
    return _load("target_list.json")


@pytest.fixture(scope="module")
def datasets():
    classification, regression = _load("datasets", "classification_all.json"), _load("datasets", "regression_all.json")
    assert len(classification) == len(set(classification)) and len(regression) == len(set(regression))
    assert not set(classification) & set(regression), "a dataset is in both lists"
    return set(classification) | set(regression)


def _key(t):
    return f"{t['dataset']}_{t['target_idx']}"


def test_targets_are_unique_and_every_dataset_is_listed(targets, datasets):
    keys = [_key(t) for t in targets]
    assert len(keys) == len(set(keys))
    assert {t["dataset"] for t in targets} <= datasets


def test_exclusions_and_target_list_agree(targets):
    exclusions = _load("exclusions.json")
    by_key = {_key(t): t for t in targets}
    listed = {}
    for reason in ("trivial", "not_learnable"):
        for key in exclusions[reason]:
            if key.startswith("_"):
                continue
            assert key in by_key, f"{reason} key {key} is not in the target list"
            assert by_key[key]["excluded"] and by_key[key]["exclusion_reason"] == reason, key
            listed[key] = reason
    for key, t in by_key.items():
        if t["excluded"] and t["exclusion_reason"] in ("trivial", "not_learnable"):
            assert key in listed, f"{key} is excluded as {t['exclusion_reason']} but missing from exclusions.json"
        if not t["excluded"]:
            assert t["exclusion_reason"] is None, key


def test_core_target_list_is_the_core_part_of_the_target_list(targets):
    core = _load("core_target_list.json")
    expected = {_key(t) for t in targets if t["is_core"] and not t["excluded"]}
    assert {_key(t) for t in core} == expected
    assert all(t["is_core"] and not t["excluded"] for t in core)


@pytest.mark.parametrize("scope_file, targets_file", [("scope_default.json", "target_list.json"),
                                                      ("scope_core.json", "core_target_list.json")])
def test_scope_points_at_its_target_list_and_names_known_datasets_and_models(scope_file, targets_file, datasets):
    scope = _load(scope_file)
    assert scope["targets_file"].endswith(targets_file) and (V1 / targets_file).is_file()
    models = scope["models"]
    assert len(models) == len(set(models)), "a model is listed twice"
    named_datasets = set(scope["large_datasets"]) | set(scope["max_train_samples_overrides"]) | set(scope["time_limit_overrides"])
    assert named_datasets <= datasets, named_datasets - datasets
    if scope_file == "scope_default.json":  # scope_core.json keeps a HIVECOTEV2 time override for a model it does not run
        for field in ("model_time_limit_overrides", "model_max_train_samples_overrides"):
            assert set(scope[field]) <= set(models), (field, set(scope[field]) - set(models))


def test_core_scope_runs_only_models_of_the_full_scope():
    assert set(_load("scope_core.json")["models"]) <= set(_load("scope_default.json")["models"])
