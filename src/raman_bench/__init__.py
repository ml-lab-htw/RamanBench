"""
RamanBench: A large-scale benchmark for machine learning on Raman spectroscopy data.

74 datasets · 163 prediction targets · 28+ baseline models

Ecosystem
---------
- **raman-data** (datasets):
  PyPI: ``pip install raman-data``
  Source: https://github.com/ml-lab-htw/raman_data

- **raman-bench** (this package):
  PyPI: ``pip install raman-bench``
  Source: https://github.com/ml-lab-htw/RamanBench

- **Live Leaderboard**:
  https://huggingface.co/spaces/HTW-KI-Werkstatt/RamanBench

- **Paper** (under review):
  https://arxiv.org/abs/2605.02003

Quick Start
-----------
::

    from raman_bench import RamanBenchmark, Leaderboard

    # Load precomputed v0.1 results
    lb = Leaderboard.from_precomputed()
    print(lb.rank())

    # Evaluate a new model against all 74 datasets
    results = lb.evaluate_and_add("MyModel", my_sklearn_model)
    lb.rank()
"""

import importlib
from pathlib import Path

from dotenv import load_dotenv

# Load credentials (TABPFN_TOKEN, HUGGING_FACE_HUB_TOKEN, etc.) from a .env file.
# Two locations, in priority order: this repo's own root (works for editable
# installs regardless of the caller's working directory), then whatever .env
# python-dotenv finds searching upward from the current working directory (e.g.
# a caller project's own .env). Neither overrides a variable already set in the
# real environment (load_dotenv's default override=False).
load_dotenv(Path(__file__).resolve().parents[2] / ".env")
load_dotenv()

__version__ = "0.1.0"
__author__ = "Mario Koddenbrock (HTW Berlin), Christoph Lange (TU Berlin)"

_public_map = {
    # Core benchmark
    "RamanBenchmark": ("raman_bench.benchmark", "RamanBenchmark"),
    "configure_benchmark": ("raman_bench.benchmark", "configure_benchmark"),
    # Models
    "AutoGluonModel": ("raman_bench.model", "AutoGluonModel"),
    # Leaderboard — rank new models against precomputed baselines
    "Leaderboard": ("raman_bench.leaderboard", "Leaderboard"),
    # Metrics
    "ClassificationMetrics": ("raman_bench.metrics", "ClassificationMetrics"),
    "RegressionMetrics": ("raman_bench.metrics", "RegressionMetrics"),
    "compute_metrics": ("raman_bench.metrics", "compute_metrics"),
    # Pipeline steps
    "compute_predictions": ("raman_bench.predictions", "compute_predictions"),
    "compute_metrics_from_predictions": (
        "raman_bench.evaluation",
        "compute_metrics_from_predictions",
    ),
    # Config
    "load_config": ("raman_bench.config", "load_config"),
}

__all__: list[str] = sorted(list(_public_map.keys()))


def __getattr__(name: str):
    if name in _public_map:
        module_name, attr = _public_map[name]
        module = importlib.import_module(module_name)
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + list(_public_map.keys()))
