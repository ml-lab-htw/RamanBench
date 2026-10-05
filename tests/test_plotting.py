"""Tests for raman_bench.plotting on a small synthetic hpo_results table."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("bencheval.evaluator")

from raman_bench.plotting import models as model_info  # noqa: E402
from raman_bench.plotting import results as res  # noqa: E402

MODELS = {
    # key: (skill, runs regression?)
    "RF": (0.0, True),
    "TABPFN-V3": (0.6, True),
    "RAMANPFN": (0.5, True),
    "LR": (0.2, True),
    "ROCKET": (0.3, False),  # classification only
    "DUMMY": (-0.5, True),
}


def _hpo_results(tmp_path, n_clf=4, n_reg=5, n_folds=3, missing=()):
    rng = np.random.default_rng(0)
    rows = []
    tasks = [(f"clf{i}__0", "binary") for i in range(n_clf)] + [(f"reg{i}__0", "regression") for i in range(n_reg)]
    for dataset, ptype in tasks:
        for fold in range(n_folds + 3):  # 3 extra folds an older sweep left behind
            for model, (skill, does_reg) in MODELS.items():
                if ptype == "regression" and not does_reg:
                    continue
                if (model, dataset) in missing:
                    continue
                rows.append(
                    {
                        "dataset": dataset,
                        "fold": fold,
                        "method": f"{model} (default)",
                        "metric_error": max(0.0, 0.5 - 0.3 * skill + rng.normal(0, 0.02)),
                        "time_train_s": 10.0 * (1 + skill),
                        "time_infer_s": 0.5,
                        "metric": "rmse" if ptype == "regression" else "roc_auc",
                        "problem_type": ptype,
                        "method_subtype": "default",
                        "config_type": model,
                    }
                )
    path = tmp_path / "hpo_results.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    targets = [
        {"dataset": d.split("__")[0], "target_idx": 0, "num_instances": 300, "n_repeats": 1,
         "excluded": d == "reg4__0"}
        for d, _ in tasks
    ]
    tl = tmp_path / "target_list.json"
    tl.write_text(json.dumps(targets))
    scope = tmp_path / "scope.json"
    scope.write_text(json.dumps({"models": list(MODELS), "n_splits": n_folds}))
    return path, scope, tl


def test_load_results_applies_scope_targets_and_folds(tmp_path):
    path, scope, tl = _hpo_results(tmp_path)
    df = res.load_results(path, scope=scope, target_list=tl)
    assert "reg4__0" not in set(df["dataset"])  # excluded target dropped
    assert df["fold"].max() == 2  # only n_repeats x n_splits folds
    assert set(df["task"]) == {"classification", "regression"}
    assert (df["metric_error"] >= 0).all()


def test_scores_rank_skill_and_skip_task_restricted_models(tmp_path):
    path, scope, tl = _hpo_results(tmp_path)
    df = res.load_results(path, scope=scope, target_list=tl, exclude_models=())
    scores = res.score_all(df, bootstrap_rounds=10)
    assert set(scores) == {"all", "classification", "regression"}

    reg = scores["regression"].leaderboard
    assert "ROCKET" not in reg.index  # never ran regression: left out, not imputed
    clf = scores["classification"].leaderboard
    assert "ROCKET" in clf.index

    # In the all-tasks ranking ROCKET is ranked on its classification tasks only, marked as such,
    # never imputed on regression; its Elo comes from comparisons on the tasks it ran.
    lb_all = scores["all"].leaderboard
    assert lb_all.loc["ROCKET", "evaluated_on"] == "classification"
    assert lb_all.loc["ROCKET", "display_name"].endswith("(clf. only)")
    assert lb_all.loc["ROCKET", "n_tasks"] == clf.loc["ROCKET", "n_tasks"]
    assert lb_all.loc["ROCKET", "winrate"] == pytest.approx(clf.loc["ROCKET", "winrate"])
    assert (lb_all.drop(index="ROCKET")["evaluated_on"] == "all").all()
    assert lb_all.loc["RF", "elo"] == pytest.approx(1000, abs=1)
    assert "ROCKET" in scores["all"].winrate_matrix.index
    all_results = scores["all"].results
    assert set(all_results.loc[all_results["model"] == "ROCKET", "dataset"]) <= set(
        scores["classification"].results["dataset"]
    )
    assert reg.index[0] == "TABPFN-V3"
    assert reg.index[-1] == "DUMMY"
    assert reg.loc["RF", "elo"] == pytest.approx(1000, abs=1)  # calibration anchor
    assert (reg["imputed_pct"] == 0).all()
    wr = scores["regression"].winrate_matrix
    assert list(wr.index) == list(reg.index)
    assert wr.loc["TABPFN-V3", "DUMMY"] == pytest.approx(1.0)


def test_dummy_excluded_and_autogluon_scored_as_reference(tmp_path):
    path, scope, tl = _hpo_results(tmp_path)
    df = pd.read_csv(path)
    ag = df[df["config_type"] == "TABPFN-V3"].assign(
        method="AutoGluon_extreme_1h", config_type=np.nan, method_subtype=np.nan, method_type="baseline"
    )
    pd.concat([df, ag]).to_csv(path, index=False)
    scope.write_text(json.dumps({"models": [*MODELS, "AUTOGLUON-EXTREME-1H"], "n_splits": 3}))

    loaded = res.load_results(path, scope=scope, target_list=tl)
    assert "DUMMY" not in set(loaded["model"])
    lb = res.score_all(loaded, bootstrap_rounds=10)["regression"].leaderboard
    assert bool(lb.loc["AUTOGLUON-EXTREME-1H", "is_reference"])
    models, refs = res.split_references(lb)
    assert list(refs.index) == ["AUTOGLUON-EXTREME-1H"]
    assert "AUTOGLUON-EXTREME-1H" not in res.select_focus(lb, 1)


def test_missing_runs_are_imputed_with_reference_model(tmp_path):
    path, scope, tl = _hpo_results(tmp_path, missing={("RAMANPFN", "reg0__0")})
    scores = res.score_all(res.load_results(path, scope=scope, target_list=tl), bootstrap_rounds=10)
    reg = scores["regression"].leaderboard
    assert reg.loc["RAMANPFN", "imputed_pct"] == pytest.approx(25.0)  # 1 of 4 kept regression tasks
    filled = scores["regression"].results
    imputed = filled[(filled["model"] == "RAMANPFN") & filled["imputed"]]
    assert imputed["time_train_s"].isna().all()  # times never come from the reference model


def test_mostly_imputed_models_are_dropped(tmp_path):
    missing = {("LR", f"reg{i}__0") for i in range(4)}
    path, scope, tl = _hpo_results(tmp_path, missing=missing)
    scores = res.score_all(res.load_results(path, scope=scope, target_list=tl), bootstrap_rounds=10)
    assert "LR" not in scores["regression"].leaderboard.index
    assert "LR" in scores["classification"].leaderboard.index


def test_select_focus_takes_top_k_per_category():
    lb = pd.DataFrame(
        {"elo": [1500, 1400, 1300, 1000, 900], "category": ["A", "A", "A", "B", "B"]},
        index=["a1", "a2", "a3", "b1", "b2"],
    )
    assert res.select_focus(lb, 1) == {"a1", "b1"}
    assert res.select_focus(lb, 2) == {"a1", "a2", "b1", "b2"}
    assert res.select_focus(lb, None) == set(lb.index)
    assert res.select_focus(lb, 1, always=("a3", "zzz")) == {"a1", "b1", "a3"}


def test_normalized_score_best_is_one_median_is_zero():
    per_task = pd.DataFrame(
        {"dataset": ["d"] * 3, "model": ["good", "mid", "bad"], "metric_error": [0.1, 0.2, 0.9]}
    )
    score = res.normalized_score(per_task)
    assert score["good"] == pytest.approx(1.0)
    assert score["mid"] == pytest.approx(0.0)
    assert score["bad"] == pytest.approx(0.0)  # clipped


def test_every_scope_model_has_metadata():
    with open(res.DEFAULT_SCOPE) as f:
        scope_models = json.load(f)["models"]
    unknown = [m for m in scope_models if model_info.category(m) == "Other"]
    # AutoGluon presets are a reference line, not a ranked model.
    assert [m for m in unknown if not m.startswith("AUTOGLUON")] == []


def test_generate_all_writes_static_interactive_and_index(tmp_path):
    pytest.importorskip("plotly")
    from raman_bench.plotting import generate_all

    path, scope, tl = _hpo_results(tmp_path)
    out = tmp_path / "figures"
    written = generate_all(
        path, out, scope=scope, target_list=tl, bootstrap_rounds=10, focus_top_k=1, dataset_figures=False
    )
    for stem in ("elo_ranking_combined", "metrics_vs_time", "improvability_vs_time",
                 "elo_vs_release_date", "pairwise_win_rates", "efficiency_overview"):
        assert (out / "static" / f"{stem}.png").exists()
        assert (out / "static" / f"{stem}.pdf").exists()
        assert (out / "interactive" / f"{stem}.html").exists()
    assert (out / "leaderboards" / "leaderboard_regression.csv").exists()
    assert written["index"][0].read_text().count("<section>") >= 6


def test_composition_and_reference_benchmarks_offline():
    from raman_bench.plotting import overview

    ref = overview.reference_benchmarks()
    assert ref.groupby("benchmark").size().to_dict() == {"TALENT": 300, "TabArena": 51, "UCR": 138, "UEA": 36}
    ov = pd.DataFrame({
        "dataset": ["a", "b", "c"], "name": ["A", "B", "C"],
        "domain": ["Medical", "Chemical", "Chemical"], "task_type": ["Classification", "Regression", "Regression"],
        "n_spectra": [100, 50, 25], "n_features": [1000, 500, 800], "n_targets": [1, 2, 1],
        "source": ["Zenodo", "Kaggle", "Zenodo"], "new": [True, False, False],
    })
    donuts = {title: dict(zip(labels, values)) for title, labels, values, _ in overview._composition(ov)}
    assert donuts["Spectra by domain"] == {"Medical & Clinical": 100, "Chemical & Industrial": 75}
    assert donuts["Datasets by task"] == {"Classification": 1, "Regression": 2}
    assert donuts["Data sources"] == {"Kaggle": 1, "Zenodo": 2}
    assert donuts["New vs. existing"] == {"First published with RamanBench": 1, "Previously published": 2}


def test_known_contamination_is_marked_and_explained(tmp_path):
    assert model_info.display_name("RAMANPFN") == "RamanPFN" + model_info.CONTAMINATION_MARK
    assert model_info.display_name("RAMANPFN", mark=False) == "RamanPFN"
    assert model_info.contamination("RF") is None

    import matplotlib.pyplot as plt

    from raman_bench.plotting import static

    fig, ax = plt.subplots()
    ax.set_title(model_info.display_name("RAMANPFN"))
    static.save(fig, tmp_path, "flagged", formats=("png",))
    assert (tmp_path / "flagged.png").exists()

    pytest.importorskip("plotly")
    import plotly.graph_objects as go

    from raman_bench.plotting import interactive

    page = interactive.write(go.Figure(go.Scatter(x=[1], y=[1], name=model_info.display_name("RAMANPFN"))),
                             tmp_path, "flagged")
    assert "arXiv:2608.02157" in page.read_text(encoding="utf-8")
    clean = interactive.write(go.Figure(go.Scatter(x=[1], y=[1], name="Random Forest")), tmp_path, "clean")
    assert "arXiv:2608.02157" not in clean.read_text(encoding="utf-8")


def test_figure_notes_explain_marks():
    assert model_info.figure_notes("Random Forest, XGBoost") == []
    notes = model_info.figure_notes("RamanPFN† and OrionMSP (clf. only)")
    assert len(notes) == 2
    assert notes[1] == model_info.SUBSET_NOTE
