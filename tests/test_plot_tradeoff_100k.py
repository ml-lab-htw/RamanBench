"""scripts/plot_tradeoff_100k.py draws both figures from a leaderboard CSV and rejects unknown models."""
import runpy
import sys
from pathlib import Path

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "plot_tradeoff_100k.py"


def _leaderboard(path: Path) -> Path:
    rows = {
        "XGB": ("XGBoost", "Gradient Boosting", 1100.0, 20.0, 2.0),
        "CAT": ("CatBoost", "Gradient Boosting", 1200.0, 40.0, 4.0),
        "TABM": ("TabM", "Deep Learning", 1000.0, 10.0, 1.0),
    }
    lb = pd.DataFrame(
        {
            "display_name": [r[0] for r in rows.values()],
            "category": [r[1] for r in rows.values()],
            "elo": [r[2] for r in rows.values()],
            "median_time_total_per_1k_s": [r[3] for r in rows.values()],
            "median_infer_per_1k_s": [r[4] for r in rows.values()],
            "is_reference": False,
            "evaluated_on": "all",
            "contamination": "",
        },
        index=pd.Index(list(rows), name="model"),
    )
    lb.to_csv(path)
    return path


def _run(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["plot_tradeoff_100k.py", *argv])
    runpy.run_path(str(SCRIPT), run_name="__main__")


def test_writes_both_figures(tmp_path, monkeypatch):
    csv = _leaderboard(tmp_path / "lb.csv")
    _run(monkeypatch, "--leaderboard", str(csv), "--highlight", "XGB", "--output-dir", str(tmp_path / "out"))
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == [
        "elo_vs_time_100k_infer.png", "elo_vs_time_100k_train_infer.png",
    ]


def test_unknown_highlight_model_fails(tmp_path, monkeypatch):
    csv = _leaderboard(tmp_path / "lb.csv")
    with pytest.raises(AssertionError, match="NOPE"):
        _run(monkeypatch, "--leaderboard", str(csv), "--highlight", "NOPE", "--output-dir", str(tmp_path / "out"))
