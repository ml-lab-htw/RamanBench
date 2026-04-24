"""Tests for the Leaderboard class."""

import pandas as pd
import pytest

from raman_bench.leaderboard import Leaderboard


def _make_leaderboard():
    overall = pd.DataFrame(
        {
            "Model": ["RF", "GBM", "PLS"],
            "Score": [0.5, 0.6, 0.4],
            "Elo": [1000, 1050, 950],
        }
    )
    clf = overall.copy()
    reg = overall.copy()
    return Leaderboard(overall, clf, reg)


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
            "rmse": [0.1, 0.12],
            "r2": [0.9, 0.88],
        }
    )
    lb.add_results("MyModel", metrics_df, task="overall")
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
