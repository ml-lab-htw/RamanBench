"""Tests for config loading and normalisation."""

import json

import pytest

from raman_bench.config import _normalize_preprocessing_config, load_config


def test_normalize_preprocessing_false():
    config = {"preprocessing": False}
    out = _normalize_preprocessing_config(config)
    assert out["preprocessing_config"] is None


def test_normalize_preprocessing_true():
    config = {"preprocessing": True}
    out = _normalize_preprocessing_config(config)
    assert isinstance(out["preprocessing_config"], dict)
    assert out["preprocessing_config"]["baseline_correction"] is True


def test_normalize_preprocessing_dict():
    config = {"preprocessing": {"baseline_correction": True, "snv": False}}
    out = _normalize_preprocessing_config(config)
    assert out["preprocessing_config"]["baseline_correction"] is True
    assert out["preprocessing_config"]["snv"] is False
    assert out["preprocessing_config"]["msc"] is False  # default


def test_normalize_preprocessing_true_excludes_crop_physical():
    """crop_physical is opt-in only -- never auto-enabled by the "enable
    everything" bool shorthand (see the NOTE above _ALL_PREPROCESSING_STEPS
    in config.py: it requires a deliberately-chosen [start_cm, end_cm]).
    """
    config = {"preprocessing": True}
    out = _normalize_preprocessing_config(config)
    assert "crop_physical" not in out["preprocessing_config"]


def test_normalize_preprocessing_dict_can_explicitly_enable_crop_physical():
    """An explicit preprocessing_config dict IS able to opt into crop_physical
    (regression guard: the dict-branch comprehension must recognize
    "crop_physical" as a valid key, not silently drop it because it isn't
    part of _ALL_PREPROCESSING_STEPS).
    """
    config = {"preprocessing": {"crop_physical": True, "snv": True}}
    out = _normalize_preprocessing_config(config)
    assert out["preprocessing_config"]["crop_physical"] is True
    assert out["preprocessing_config"]["snv"] is True
    assert out["preprocessing_config"]["crop"] is False  # default, not requested


def test_load_config_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(str(tmp_path / "nonexistent.json"))


def test_load_config_minimal(tmp_path):
    cfg = {
        "datasets_regression": [],
        "datasets_classification": [],
        "test_size": 0.2,
        "n_repetitions": 1,
        "preprocessing": False,
        "models": ["RF"],
        "autogluon_time_limit": 60,
        "autogluon_presets": "medium_quality",
        "optimize": False,
        "ensemble": False,
        "output_dir": "results/test",
    }
    p = tmp_path / "config.json"
    p.write_text(json.dumps(cfg))
    loaded = load_config(str(p))
    assert loaded["preprocessing_config"] is None
    assert loaded["models"] == ["RF"]
    assert loaded["log_level"] == "INFO"
