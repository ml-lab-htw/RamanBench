"""Benchmark v2's config files agree with each other, with v1 where v2 reuses it, and with the fit settings the runners apply."""

import json
import sys
from pathlib import Path

import pytest

from raman_bench.compare import load_protocol

REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "configs"


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


def test_scope_and_protocol_agree_on_the_fit_settings():
    scope, protocol = _load("v2", "scope_default.json"), _load("v2", "protocol.json")
    for key in ("n_splits", "num_bag_folds", "time_limit", "max_run_time", "num_random_configs", "max_train_samples"):
        assert scope[key] == protocol[key], key
    assert protocol["version"] == "v2"


def test_every_model_gets_the_same_cap_and_time_limit():
    for name in ("scope_default.json", "protocol.json"):
        config = _load("v2", name)
        for key in (
            "time_limit_overrides", "model_time_limit_overrides", "model_max_train_samples_overrides",
            "max_train_samples_overrides",
        ):
            assert not config.get(key), f"{name}: {key} must stay empty in v2"


def test_the_protocol_lists_exactly_the_tasks_v2_runs():
    included = [f"{t['dataset']}__{t['target_idx']}" for t in _load("v2", "target_list.json") if not t.get("excluded")]
    tasks = _load("v2", "protocol.json")["tasks"]
    assert [t["task"] for t in tasks] == included
    assert len({t["task"] for t in tasks}) == len(tasks)


def test_v2_keeps_every_v1_task_with_the_same_metric_and_folds():
    v1 = {t["task"]: t for t in load_protocol()["tasks"]}
    v2 = {t["task"]: t for t in _load("v2", "protocol.json")["tasks"]}
    assert set(v1) <= set(v2)
    for key, task in v1.items():
        for field in ("dataset", "target_idx", "problem_type", "metric", "n_folds", "n_repeats", "num_instances"):
            assert v2[key][field] == task[field], (key, field)


def test_one_row_cap_for_every_dataset():
    from raman_bench.compare import row_cap

    protocol = _load("v2", "protocol.json")
    assert protocol["max_train_samples"] == 30000
    assert {row_cap(protocol, t["dataset"]) for t in protocol["tasks"]} == {30000}


def test_max_run_time_caps_the_child_budget_and_raises_the_bag_limit():
    from raman_bench.experiment_utils import BAG_CHILD_RATIO, bag_experiment_kwargs

    plain = bag_experiment_kwargs(num_random_configs=0, time_limit=600, num_bag_folds=3)
    assert plain["time_limit"] == 600 and "extra_model_hyperparameters" not in plain
    split = bag_experiment_kwargs(num_random_configs=0, time_limit=600, num_bag_folds=3, max_run_time=10800)
    assert split["time_limit"] == 10800
    assert split["extra_model_hyperparameters"] == {"ag.max_time_limit": BAG_CHILD_RATIO * 600}


def test_autogluon_baseline_gets_the_row_cap():
    """The AutoGluon baselines ran uncapped until the entrypoint passed --max-train-samples (48 stale folds)."""
    sys.path.insert(0, str(REPO / "cluster"))
    import submit_v2_sweep

    manifest = {"spec": {"template": {"spec": {"containers": [{"env": []}]}}}}
    submit_v2_sweep.set_container(manifest, ["cmd"], 3, False, max_train_samples=30000)
    env = {e["name"]: e["value"] for e in manifest["spec"]["template"]["spec"]["containers"][0]["env"]}
    assert env["MAX_TRAIN_SAMPLES"] == "30000"
    assert '--max-train-samples "${MAX_TRAIN_SAMPLES}"' in (REPO / "cluster/k8s_entrypoint_autogluon.sh").read_text()
