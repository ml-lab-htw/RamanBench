"""LGBM-PREP-FTE25 (raman_bench.systems.lgbm_prep_fte) on a toy task, with a two-config portfolio.

The real portfolio has 26 bagged configs; the test keeps the LightGBM default and the first TabPrep config so it
runs in seconds. It checks that the vendored system runs through run_cached, stores test predictions like every
other v1 result, and that every bag gets the protocol's budget.
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("tabarena")
pytest.importorskip("lightgbm")

from raman_bench.experiment_utils import build_task, run_cached  # noqa: E402
from raman_bench.systems import get_system  # noqa: E402
from raman_bench.systems.lgbm_prep_fte import system as system_module  # noqa: E402


def test_portfolios_are_complete():
    for arena, n in {"tabarena": 201, "beyondarena": 26}.items():
        configs = system_module.load_configs(arena)
        assert len(configs) == n and configs[0]["model"] == "lightgbm_default"


def _task(problem_type, n=90, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n, 12)), columns=[f"{400 + i}" for i in range(12)])
    df["target"] = np.tile(["a", "b", "c"], n // 3) if problem_type == "classification" else df["400"] * 2 + rng.normal(size=n)
    return build_task(
        dataset_name="toy", target_idx=0, df=df, raw_targets=df[["target"]].to_numpy(),
        problem_type=problem_type, n_repeats=1, n_splits=3,
    )


@pytest.mark.parametrize("problem_type", ["classification", "regression"])
def test_toy_fold_runs_with_two_bags(monkeypatch, tmp_path, problem_type):
    from tabarena.benchmark.experiment.experiment_constructor import ExternalSystemExperiment

    configs = system_module.load_configs("beyondarena")
    monkeypatch.setattr(system_module, "load_configs", lambda arena: [configs[0], next(c for c in configs if c["model"] != "lightgbm_default")])
    system = get_system("lgbm-prep-fte25")
    task_name, task = _task(problem_type)
    experiment = ExternalSystemExperiment(
        name=system.result_dir,
        system_cls=system.load_cls(),
        system_hyperparameters=dict(system.hyperparameters),
        method_kwargs={"fit_kwargs": {"time_limit": 120, "num_cpus": 1}},
        experiment_kwargs={"require_warmup": False},
    )
    out = run_cached(experiment, task_name=task_name, task_wrapper=task, repeat=0, fold=1, results_dir=str(tmp_path))

    assert out["framework"] == "LightGBM-Prep-FTE25"
    assert out["method_metadata"]["bag_time_limit"] == 120  # each bag gets the whole budget
    assert len(out["method_metadata"]["weights"]) == 2
    assert len(out["simulation_artifacts"]["y_test_idx"]) > 0
