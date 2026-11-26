"""Benchmark v2's config files agree with each other, and with v1 where v2 reuses it."""

import json
from pathlib import Path

import pytest

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _load(*parts):
    return json.loads(CONFIGS.joinpath(*parts).read_text())


def _key(t):
    return f"{t['dataset']}_{t['target_idx']}"


@pytest.fixture(scope="module")
def targets():
    return _load("v2", "target_list.json")


def test_targets_are_unique(targets):
    keys = [_key(t) for t in targets]
    assert len(keys) == len(set(keys))


def test_the_v1_targets_are_those_of_v1_unchanged(targets):
    v1 = {_key(t): t for t in _load("v1", "target_list.json")}
    kept = {_key(t): t for t in targets if t["since"] == "v1"}
    assert kept.keys() == v1.keys()
    for key, t in kept.items():
        assert {f: t[f] for f in v1[key]} == v1[key], key


def test_exclusions_and_target_list_agree(targets):
    exclusions = {r: ks for r, ks in _load("v2", "exclusions.json").items() if not r.startswith("_")}
    by_key = {_key(t): t for t in targets}
    seen = set()
    for reason, keys in exclusions.items():
        for key in keys:
            assert key not in seen, f"{key} is listed under two reasons"
            seen.add(key)
            assert key in by_key, f"{reason} key {key} is not in the target list"
            t = by_key[key]
            assert t["since"] == "v2" and t["excluded"] and t["exclusion_reason"] == reason, key
    for key, t in by_key.items():
        if t["since"] == "v2" and t["excluded"]:
            assert key in seen, f"{key} is excluded in the target list but missing from exclusions.json"
        if not t["excluded"]:
            assert t["exclusion_reason"] is None, key


def test_dataset_lists_are_v1_plus_the_new_datasets(targets):
    v1 = set(_load("v1", "datasets", "classification_all.json")) | set(_load("v1", "datasets", "regression_all.json"))
    all_, new = _load("v2", "datasets", "all.json"), _load("v2", "datasets", "new.json")
    assert len(all_) == len(set(all_)) and len(new) == len(set(new))
    assert set(all_) == v1 | set(new)
    assert set(new) == {t["dataset"] for t in targets if t["since"] == "v2"}
    assert {t["dataset"] for t in targets} == set(all_)


def test_scope_points_at_its_target_list_and_names_known_datasets_and_models():
    scope = _load("v2", "scope_default.json")
    assert scope["targets_file"].endswith("target_list.json") and (CONFIGS / "v2" / "target_list.json").is_file()
    models = scope["models"]
    assert len(models) == len(set(models)), "a model is listed twice"
    all_ = set(_load("v2", "datasets", "all.json"))
    named = set(scope.get("large_datasets", [])) | set(scope.get("time_limit_overrides", {}))
    assert named <= all_, named - all_
    for field in ("model_time_limit_overrides", "model_max_train_samples_overrides"):
        assert set(scope.get(field, {})) <= set(models), (field, set(scope.get(field, {})) - set(models))
