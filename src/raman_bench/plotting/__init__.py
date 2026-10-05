"""Leaderboard figures for RamanBench v1 results: static (PNG/PDF) and interactive (HTML).

Scores come from TabArena's own evaluator (``bencheval``); see
:mod:`raman_bench.plotting.results`. Entry point: :func:`generate_all`, or
``scripts/plot_results.py`` on the command line.

Needs the ``plots`` extra: ``pip install "raman-bench[plots]"``.
"""

from raman_bench.plotting.pipeline import generate_all, generate_from_results
from raman_bench.plotting.results import load_results, score_all, score_group, select_focus

__all__ = ["generate_all", "generate_from_results", "load_results", "score_all", "score_group", "select_focus"]
