"""Shared test fixtures for RamanBench."""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def small_clf_df():
    """Tiny classification DataFrame with 30 samples, 20 features, 3 classes."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((30, 20)).astype(np.float32)
    y = np.array(["A"] * 10 + ["B"] * 10 + ["C"] * 10)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(20)])
    df["target"] = y
    return df


@pytest.fixture
def small_reg_df():
    """Tiny regression DataFrame with 40 samples and 20 features."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((40, 20)).astype(np.float32)
    y = rng.standard_normal(40).astype(np.float32)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(20)])
    df["target"] = y
    return df


@pytest.fixture
def debug_config(tmp_path):
    """Minimal config dict for fast unit tests (no file I/O)."""
    return {
        "dataset_names_classification": ["wheat_lines"],
        "dataset_names_regression": [],
        "test_size": 0.2,
        "random_state": 42,
        "cache_dir": str(tmp_path / "cache"),
        "output_dir": str(tmp_path / "results"),
        "models": ["RF"],
        "autogluon_time_limit": 30,
        "autogluon_presets": "medium_quality",
        "optimize": False,
        "ensemble": False,
        "min_samples_per_class": 3,
        "group_regression_splits": True,
        "n_repetitions": 1,
        "preprocessing_config": None,
    }
