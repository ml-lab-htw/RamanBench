"""Benchmark v2 has one protocol file, and the scope and the target list agree with it."""

import json
import sys
from pathlib import Path

from raman_bench.compare import load_protocol

REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "configs" / "v2"


def _load(name):
    return json.loads((CONFIGS / name).read_text())


def test_scope_and_protocol_agree_on_the_fit_settings():
    scope, protocol = _load("scope_default.json"), _load("protocol.json")
    for key in ("n_splits", "num_bag_folds", "time_limit", "max_run_time", "num_random_configs", "max_train_samples"):
        assert scope[key] == protocol[key], key
    assert protocol["version"] == "v2"


def test_every_model_gets_the_same_cap_and_time_limit():
    for name in ("scope_default.json", "protocol.json"):
        config = _load(name)
        for key in (
            "time_limit_overrides", "model_time_limit_overrides", "model_max_train_samples_overrides",
            "max_train_samples_overrides",
        ):
            assert not config.get(key), f"{name}: {key} must stay empty in v2"


def test_the_protocol_lists_exactly_the_tasks_v2_runs():
    included = [f"{t['dataset']}__{t['target_idx']}" for t in _load("target_list.json") if not t.get("excluded")]
    tasks = _load("protocol.json")["tasks"]
    assert [t["task"] for t in tasks] == included
    assert len({t["task"] for t in tasks}) == len(tasks)


def test_v2_keeps_every_v1_task_with_the_same_metric_and_folds():
    v1 = {t["task"]: t for t in load_protocol()["tasks"]}
    v2 = {t["task"]: t for t in _load("protocol.json")["tasks"]}
    assert set(v1) <= set(v2)
    for key, task in v1.items():
        for field in ("dataset", "target_idx", "problem_type", "metric", "n_folds", "n_repeats", "num_instances"):
            assert v2[key][field] == task[field], (key, field)


def test_one_row_cap_for_every_dataset():
    from raman_bench.compare import row_cap

    protocol = _load("protocol.json")
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
