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

__version__ = "1.5.1"
__author__ = "Mario Koddenbrock (HTW Berlin), Christoph Lange (TU Berlin)"

_public_map = {
    # Core benchmark
    "RamanBenchmark": ("raman_bench.benchmark", "RamanBenchmark"),
    "configure_benchmark": ("raman_bench.benchmark", "configure_benchmark"),
    # Leaderboard — rank new models against precomputed baselines
    "Leaderboard": ("raman_bench.leaderboard", "Leaderboard"),
    "compute_elo": ("raman_bench.leaderboard", "compute_elo"),
    # Metrics
    "ClassificationMetrics": ("raman_bench.metrics", "ClassificationMetrics"),
    "RegressionMetrics": ("raman_bench.metrics", "RegressionMetrics"),
    "compute_metrics": ("raman_bench.metrics", "compute_metrics"),
    "compute_metrics_from_predictions": ("raman_bench.evaluation", "compute_metrics_from_predictions"),
    # Seeds
    "get_seeds": ("raman_bench.seeds", "get_seeds"),
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
