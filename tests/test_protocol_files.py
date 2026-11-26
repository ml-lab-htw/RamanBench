"""The v1 scope (what the sweeps run) and the bundled protocol (what compare() ranks against) agree."""

import json
from pathlib import Path

from raman_bench.compare import load_protocol

SCOPE = json.loads((Path(__file__).resolve().parents[1] / "configs" / "v1" / "scope_default.json").read_text())


def test_scope_and_protocol_agree_on_the_fit_settings():
    protocol = load_protocol()
    for key in ("n_splits", "num_bag_folds", "time_limit", "time_limit_overrides", "max_train_samples_overrides"):
        assert protocol[key] == SCOPE[key], key


def test_every_model_cap_in_the_protocol_is_the_scope_cap():
    # The protocol is rebuilt at a release, so the scope may hold newer models; it must not disagree on the ones it lists.
    scope_caps = SCOPE["model_max_train_samples_overrides"]
    for model, cap in load_protocol()["model_max_train_samples_overrides"].items():
        assert scope_caps[model] == cap, model
