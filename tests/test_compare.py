"""Tests for the bundled v1 reference results and ``raman_bench.compare``.

No network and no model fits: the reference results ship with the package, and the
"own" results here are copies of reference rows under a new model key.
"""

from __future__ import annotations

import pandas as pd
import pytest

from raman_bench.compare import (
    TIDY_COLUMNS,
    coverage,
    load_own_results,
    load_protocol,
    load_reference,
    protocol_tasks,
)

pytest.importorskip("pyarrow")


def test_protocol_matches_the_published_sweep():
    p = load_protocol()
    assert (p["n_splits"], p["num_bag_folds"], p["time_limit"], p["num_random_configs"]) == (3, 3, 600, 0)
    assert p["time_limit_overrides"]["mlrod"] == 10800
    assert p["max_train_samples_overrides"]["wheat_lines"] == 10000
    assert len(p["tasks"]) == 135
    assert sum(t["problem_type"] == "classification" for t in p["tasks"]) == 21
    # Every leaderboard model has its preprocessing recorded.
    assert set(p["preprocessing"]) == set(p["models"])
    assert p["preprocessing"]["PLS"] == "baseline_correction, denoising, snv"
    assert p["preprocessing"]["RF"] == "none"


def test_reference_covers_every_task_and_fold():
    ref = load_reference(exclude_models=())
    assert set(TIDY_COLUMNS) <= set(ref.columns)
    p = load_protocol()
    assert set(ref["dataset"]) == {t["task"] for t in p["tasks"]}
    assert set(ref["model"]) == set(p["models"])
    folds = ref.groupby("dataset")["fold"].nunique()
    assert (folds == 3).all()
    assert (ref["metric_error"] >= 0).all()
    assert "DUMMY" not in set(load_reference()["model"])


def test_protocol_tasks_filters():
    reg = protocol_tasks("regression")
    assert len(reg) == 114 and all(t["problem_type"] == "regression" for t in reg)
    mlrod = protocol_tasks(tasks=["mlrod"])
    assert [t["task"] for t in mlrod] == ["mlrod__0"]
    with pytest.raises(ValueError, match="Not in the protocol"):
        protocol_tasks(tasks=["no_such_dataset"])
    with pytest.raises(ValueError, match="task_type"):
        protocol_tasks("clustering")


def _copy_of(model: str, name: str) -> pd.DataFrame:
    ref = load_reference()
    return ref[ref["model"] == model].assign(model=name).drop(columns=["num_instances"])


def test_load_own_results_drops_tasks_and_folds_outside_the_protocol():
    own = _copy_of("RF", "MINE")
    extra = own.head(2).assign(dataset="not_a_task__0")
    extra_fold = own.head(1).assign(fold=7)
    out = load_own_results(pd.concat([own, extra, extra_fold]))
    assert len(out) == len(own)
    assert out["num_instances"].notna().all()


def test_coverage_counts_tasks_per_type():
    own = _copy_of("RF", "MINE")
    own = own[own["dataset"].isin(own["dataset"].unique()[:10])]
    cov = coverage(load_own_results(own)).set_index("task")
    assert cov["tasks_run"].sum() == 10
    assert cov.loc["regression", "tasks_total"] == 114


@pytest.fixture(scope="module")
def scored_copy():
    pytest.importorskip("bencheval")
    from raman_bench.compare import compare

    return compare(_copy_of("RF", "MINE"), bootstrap_rounds=10)


def test_compare_ranks_a_copy_of_rf_like_rf(scored_copy):
    lb = scored_copy["all"].leaderboard
    assert lb.loc["MINE", "elo"] == pytest.approx(lb.loc["RF", "elo"])
    assert lb.loc["MINE", "imputed_pct"] == 0
    assert lb.loc["PLS", "preprocessing"] == "baseline_correction, denoising, snv"
    assert lb.loc["MINE", "preprocessing"] == "unknown"  # tidy input without the column


def test_compare_reproduces_the_published_ranking(scored_copy):
    lb = scored_copy["all"].leaderboard.drop(index="MINE")
    top = lb.sort_values("elo", ascending=False).index[:3].tolist()
    assert top[0] == "KUMO-TABULAR"
    assert len(lb) == len(load_protocol()["models"]) - 1  # DUMMY is left out


def test_compare_on_own_tasks_only():
    pytest.importorskip("bencheval")
    from raman_bench.compare import compare

    own = _copy_of("PLS", "MINE")
    keep = own["dataset"].unique()[:8]
    scores = compare(own[own["dataset"].isin(keep)], tasks="own", bootstrap_rounds=5)
    assert scores["all"].n_tasks == len(keep)


def test_compare_refuses_a_leaderboard_key_unless_replacing():
    pytest.importorskip("bencheval")
    from raman_bench.compare import compare

    own = _copy_of("RF", "PLS")
    with pytest.raises(ValueError, match="already on the leaderboard"):
        compare(own, bootstrap_rounds=5)
    lb = compare(own, replace_reference=True, bootstrap_rounds=5)["all"].leaderboard
    assert lb.loc["PLS", "elo"] == pytest.approx(lb.loc["RF", "elo"])
