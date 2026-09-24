"""Tests for the v1 "not learnable" filter (raman_bench.filters)."""

import pandas as pd

from raman_bench.filters import (
    LearnabilityFilterConfig,
    compute_unlearnable_keys,
    filter_trivial_keys,
    get_unlearnable_keys,
    get_unlearnable_keys_from_dir,
)


def _row(dataset, model, error, method_subtype="default"):
    return {
        "dataset": dataset,
        "ta_name": model,
        "method_subtype": method_subtype,
        "metric_error": error,
    }


def _df(rows):
    return pd.DataFrame(rows)


def test_flagged_when_best_model_does_not_beat_dummy_margin():
    rows = [
        _row("ds_a__0", "DUMMY", 0.5),
        _row("ds_a__0", "RF", 0.48),  # only 0.02 better than Dummy -- below default 0.05 margin
    ]
    flagged = compute_unlearnable_keys(_df(rows))
    assert "ds_a__0" in flagged
    assert "best_error=0.48" in flagged["ds_a__0"]


def test_not_flagged_when_best_model_beats_dummy_margin():
    rows = [
        _row("ds_a__0", "DUMMY", 0.5),
        _row("ds_a__0", "RF", 0.1),  # well below 0.05-margin threshold
    ]
    flagged = compute_unlearnable_keys(_df(rows))
    assert "ds_a__0" not in flagged


def test_averages_over_folds_not_and_across_folds():
    # Two "folds" worth of rows for the same (key, model) -- averaged, not
    # AND-ed, unlike the trivial filter's criteria.
    rows = [
        _row("ds_a__0", "DUMMY", 0.5),
        _row("ds_a__0", "DUMMY", 0.5),
        _row("ds_a__0", "RF", 0.0),   # great fold
        _row("ds_a__0", "RF", 0.45),  # bad fold, but mean is still well below margin
    ]
    flagged = compute_unlearnable_keys(_df(rows))
    assert "ds_a__0" not in flagged  # mean RF error = 0.225, mean Dummy = 0.5, beats margin


def test_missing_dummy_baseline_flagged_conservatively():
    rows = [_row("ds_a__0", "RF", 0.1)]
    flagged = compute_unlearnable_keys(_df(rows))
    assert flagged.get("ds_a__0") == "no_dummy_baseline"


def test_autogluon_excluded_from_best_by_default():
    rows = [
        _row("ds_a__0", "DUMMY", 0.5),
        _row("ds_a__0", "AUTOGLUON", 0.1),  # would pass alone, but is excluded from "best"
        _row("ds_a__0", "RF", 0.48),  # every individual model still fails
    ]
    flagged = compute_unlearnable_keys(_df(rows))
    assert "ds_a__0" in flagged


def test_method_subtype_restricts_to_default_variant():
    rows = [
        _row("ds_a__0", "DUMMY", 0.5, method_subtype="default"),
        _row("ds_a__0", "RF", 0.48, method_subtype="default"),
        _row("ds_a__0", "RF", 0.0, method_subtype="tuned"),  # ignored by default subtype filter
    ]
    flagged = compute_unlearnable_keys(_df(rows))
    assert "ds_a__0" in flagged  # only the "default" rows count


def test_empty_input_returns_empty():
    assert compute_unlearnable_keys(pd.DataFrame()) == {}
    assert compute_unlearnable_keys(None) == {}


def test_config_gating_off_by_default():
    rows = [
        _row("ds_a__0", "DUMMY", 0.5),
        _row("ds_a__0", "RF", 0.48),
    ]
    assert get_unlearnable_keys(_df(rows), {"enabled": False}) == set()
    assert get_unlearnable_keys(_df(rows), None) == set()
    assert get_unlearnable_keys(_df(rows), {"enabled": True}) == {"ds_a__0"}


def test_config_from_nested_learnability_filter_block():
    cfg = LearnabilityFilterConfig.from_dict(
        {"learnability_filter": {"enabled": True, "min_dummy_margin": 0.1}}
    )
    assert cfg.enabled is True
    assert cfg.min_dummy_margin == 0.1


def test_get_unlearnable_keys_from_dir_missing_file_returns_empty(tmp_path):
    result = get_unlearnable_keys_from_dir(str(tmp_path), {"enabled": True})
    assert result == set()


def test_get_unlearnable_keys_from_dir_reads_hpo_results(tmp_path):
    rows = [
        _row("ds_a__0", "DUMMY", 0.5),
        _row("ds_a__0", "RF", 0.48),
    ]
    _df(rows).to_csv(tmp_path / "hpo_results.csv", index=False)
    result = get_unlearnable_keys_from_dir(str(tmp_path), {"enabled": True})
    assert result == {"ds_a__0"}


def test_filter_trivial_keys_also_works_for_unlearnable_keys():
    df = pd.DataFrame({"dataset": ["ds_a__0", "ds_b__0"], "value": [1, 2]})
    out = filter_trivial_keys(df, {"ds_a__0"})
    assert list(out["dataset"]) == ["ds_b__0"]
