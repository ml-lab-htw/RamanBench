"""The v2 scope and protocol files agree: the same row caps, the same time limit, no per-model exceptions."""

import json
from pathlib import Path

CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "v2"


def _load(name):
    return json.loads((CONFIGS / name).read_text())


def test_scope_and_protocol_agree_on_the_caps_and_the_time_limit():
    scope, protocol = _load("scope_default.json"), _load("protocol_v2_only.json")
    assert scope["max_train_samples_overrides"] == protocol["max_train_samples_overrides"]
    assert scope["time_limit"] == protocol["time_limit"]
    assert scope["n_splits"] == protocol["n_splits"] and scope["num_bag_folds"] == protocol["num_bag_folds"]


def test_every_model_gets_the_same_cap_and_time_limit():
    for name in ("scope_default.json", "protocol_v2_only.json"):
        config = _load(name)
        for key in ("time_limit_overrides", "model_time_limit_overrides", "model_max_train_samples_overrides"):
            assert not config.get(key), f"{name}: {key} must stay empty in v2"
