"""Tests for the Leaderboard class."""

import pandas as pd
import pytest

from raman_bench.leaderboard import Leaderboard


def _make_reg_metrics():
    return pd.DataFrame(
        {
            "seed": [0, 0, 0, 1, 1, 1],
            "key": ["ds_0", "ds_0", "ds_0", "ds_0", "ds_0", "ds_0"],
            "model": ["RF", "GBM", "PLS", "RF", "GBM", "PLS"],
            "rmse": [0.10, 0.08, 0.12, 0.11, 0.09, 0.13],
            "r2": [0.90, 0.92, 0.88, 0.89, 0.91, 0.87],
        }
    )


def _make_leaderboard():
    return Leaderboard(reg_metrics=_make_reg_metrics(), clf_metrics=pd.DataFrame())


def test_rank_returns_sorted():
    lb = _make_leaderboard()
    ranked = lb.rank()
    scores = ranked["Score"].tolist()
    assert scores == sorted(scores, reverse=True)


def test_rank_has_rank_column():
    lb = _make_leaderboard()
    ranked = lb.rank()
    assert "Rank" in ranked.columns
    assert ranked["Rank"].tolist() == [1, 2, 3]


def test_add_results():
    lb = _make_leaderboard()
    metrics_df = pd.DataFrame(
        {
            "seed": [0, 1],
            "key": ["ds_0", "ds_0"],
            "rmse": [0.05, 0.06],
            "r2": [0.95, 0.94],
        }
    )
    lb.add_results("MyModel", metrics_df)
    ranked = lb.rank()
    assert "MyModel" in ranked["Model"].tolist()


def test_summary_string():
    lb = _make_leaderboard()
    summary = lb.summary()
    assert "GBM" in summary
    assert "RF" in summary


def test_select_unknown_task():
    lb = _make_leaderboard()
    with pytest.raises(ValueError):
        lb.rank(task="unknown")


def test_from_precomputed():
    """Smoke test — checks that bundled CSV files are loadable."""
    try:
        lb = Leaderboard.from_precomputed()
        assert len(lb.rank()) > 0
    except FileNotFoundError:
        pytest.skip("Precomputed data not bundled in this environment")
