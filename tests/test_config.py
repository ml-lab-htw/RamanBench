"""Tests for config loading and normalisation."""
import json
import os

import pytest

from raman_bench.config import load_config, _normalize_preprocessing_config


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
