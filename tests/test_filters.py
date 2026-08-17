"""Tests for the v1 trivial-dataset filter (raman_bench.filters)."""

import pandas as pd
import pytest

from raman_bench.filters import (
    TrivialFilterConfig,
    compute_trivial_keys,
    filter_trivial_keys,
    get_trivial_keys,
    get_trivial_keys_from_dir,
)


def _row(dataset, fold, model, error, problem_type="regression", method_subtype="default"):
    return {
        "dataset": dataset,
        "fold": fold,
        "ta_name": model,
        "method": f"{model} ({method_subtype})",
        "method_subtype": method_subtype,
        "metric_error": error,
        "problem_type": problem_type,
    }


def _df(rows):
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# compute_trivial_keys -- criterion 1 (perfect score)
# ---------------------------------------------------------------------------


def test_perfect_regression_flagged_when_zero_error_every_fold():
    rows = [
        *[_row("ds_a__0", f, "PLS", 0.0, "regression") for f in range(3)],
        *[_row("ds_a__0", f, "RF", 0.5, "regression") for f in range(3)],
    ]
    flagged = compute_trivial_keys(_df(rows))
    assert "ds_a__0" in flagged
    assert flagged["ds_a__0"] == "perfect:PLS"


def test_perfect_not_flagged_unless_every_fold_is_perfect():
    rows = [
        _row("ds_a__0", 0, "PLS", 0.0, "regression"),
        _row("ds_a__0", 1, "PLS", 0.2, "regression"),  # not perfect on this fold
        _row("ds_a__0", 0, "RF", 0.5, "regression"),
        _row("ds_a__0", 1, "RF", 0.5, "regression"),
    ]
    flagged = compute_trivial_keys(_df(rows))
    assert "ds_a__0" not in flagged


def test_perfect_classification_uses_perfect_clf_threshold():
    rows = [
        *[_row("ds_c__0", f, "RF", 0.0, "binary") for f in range(2)],
        *[_row("ds_c__0", f, "PLS", 0.3, "binary") for f in range(2)],
    ]
    # A loose classification threshold should catch RF's error of 0.0 as "perfect".
    flagged = compute_trivial_keys(_df(rows), perfect_clf=0.01, perfect_reg=0.0)
    assert flagged.get("ds_c__0") == "perfect:RF"

    # A stricter (impossible) threshold should not flag anything via criterion 1.
    rows2 = [
        *[_row("ds_c__0", f, "RF", 0.02, "binary") for f in range(2)],
        *[_row("ds_c__0", f, "PLS", 0.3, "binary") for f in range(2)],
    ]
    flagged2 = compute_trivial_keys(_df(rows2), perfect_clf=0.01, perfect_reg=0.0)
    assert "ds_c__0" not in flagged2


def test_perfect_clf_and_perfect_reg_are_independent_thresholds():
    # A key with regression error 0.0 should NOT be flagged by perfect_clf.
    rows = [_row("ds_r__0", f, "RF", 0.0, "regression") for f in range(2)]
    flagged = compute_trivial_keys(_df(rows), perfect_clf=-1.0, perfect_reg=0.0)
    assert "ds_r__0" in flagged  # perfect_reg=0.0 still catches it


# ---------------------------------------------------------------------------
# compute_trivial_keys -- criterion 2 (tie at the top)
# ---------------------------------------------------------------------------


def test_tie_flagged_when_min_tie_models_tied_every_fold():
    rows = [
        *[_row("ds_b__0", f, "RF", 0.30, "regression") for f in range(3)],
        *[_row("ds_b__0", f, "XT", 0.30, "regression") for f in range(3)],
        *[_row("ds_b__0", f, "PLS", 0.50, "regression") for f in range(3)],
    ]
    flagged = compute_trivial_keys(_df(rows), min_tie_models=2)
    assert flagged.get("ds_b__0") == "tie:2"


def test_tie_not_flagged_when_tie_breaks_on_one_fold():
    rows = [
        _row("ds_b__0", 0, "RF", 0.30, "regression"),
        _row("ds_b__0", 0, "XT", 0.30, "regression"),
        _row("ds_b__0", 1, "RF", 0.30, "regression"),
        _row("ds_b__0", 1, "XT", 0.40, "regression"),  # tie breaks here
    ]
    flagged = compute_trivial_keys(_df(rows), min_tie_models=2)
    assert "ds_b__0" not in flagged


def test_tie_respects_min_tie_models_threshold():
    # Only 2 models tie; requiring 3 should not flag it.
    rows = [
        *[_row("ds_b__0", f, "RF", 0.30, "regression") for f in range(2)],
        *[_row("ds_b__0", f, "XT", 0.30, "regression") for f in range(2)],
    ]
    flagged = compute_trivial_keys(_df(rows), min_tie_models=3)
    assert "ds_b__0" not in flagged


def test_tie_decimals_absorbs_floating_point_noise():
    rows = [
        *[_row("ds_b__0", f, "RF", 0.30000001, "regression") for f in range(2)],
        *[_row("ds_b__0", f, "XT", 0.29999999, "regression") for f in range(2)],
    ]
    flagged = compute_trivial_keys(_df(rows), min_tie_models=2, tie_decimals=4)
    assert flagged.get("ds_b__0") == "tie:2"

    # With no rounding (many decimals), the two values are distinct -> no tie.
    flagged_strict = compute_trivial_keys(_df(rows), min_tie_models=2, tie_decimals=10)
    assert "ds_b__0" not in flagged_strict


def test_single_model_key_never_flagged_by_tie_criterion():
    rows = [_row("ds_solo__0", f, "PLS", 0.4, "regression") for f in range(3)]
    flagged = compute_trivial_keys(_df(rows), min_tie_models=2)
    assert "ds_solo__0" not in flagged


# ---------------------------------------------------------------------------
# compute_trivial_keys -- gating / edge cases
# ---------------------------------------------------------------------------


def test_empty_dataframe_returns_empty_dict():
    assert compute_trivial_keys(pd.DataFrame()) == {}


def test_none_input_returns_empty_dict():
    assert compute_trivial_keys(None) == {}


def test_missing_required_columns_raises():
    df = pd.DataFrame({"dataset": ["a"], "fold": [0]})
    with pytest.raises(ValueError, match="missing required column"):
        compute_trivial_keys(df)


def test_exclude_models_removes_perfect_dummy_baseline():
    rows = [
        *[_row("ds_a__0", f, "DUMMY", 0.0, "regression") for f in range(2)],
        *[_row("ds_a__0", f, "RF", 0.4, "regression") for f in range(2)],
    ]
    flagged = compute_trivial_keys(_df(rows), exclude_models=["DUMMY"])
    assert "ds_a__0" not in flagged


def test_method_subtype_default_ignores_tuned_variants_for_tie_count():
    # RF's own (default) and (tuned) rows tie with each other -- since both
    # are the *same* model, this must NOT count as a 2-model tie.
    rows = [
        *[_row("ds_a__0", f, "RF", 0.30, "regression", method_subtype="default") for f in range(2)],
        *[_row("ds_a__0", f, "RF", 0.30, "regression", method_subtype="tuned") for f in range(2)],
    ]
    flagged = compute_trivial_keys(_df(rows), min_tie_models=2, method_subtype="default")
    assert "ds_a__0" not in flagged

    # Passing method_subtype=None (no filtering) reproduces the double count.
    flagged_unfiltered = compute_trivial_keys(_df(rows), min_tie_models=2, method_subtype=None)
    assert flagged_unfiltered.get("ds_a__0") == "tie:2"


def test_mixed_problem_type_within_one_model_skips_criterion_one_without_crashing():
    # A single model's own rows disagree on problem_type for the same key --
    # genuinely malformed input (a key should be consistently classification
    # or regression). Criterion 1 must skip this (key, model) group rather
    # than guess a threshold, and must not raise.
    rows = [
        _row("ds_weird__0", 0, "RF", 0.0, "regression"),
        _row("ds_weird__0", 1, "RF", 0.0, "binary"),
    ]
    flagged = compute_trivial_keys(_df(rows), min_tie_models=2)
    assert "ds_weird__0" not in flagged


# ---------------------------------------------------------------------------
# TrivialFilterConfig
# ---------------------------------------------------------------------------


def test_config_defaults_are_off():
    cfg = TrivialFilterConfig()
    assert cfg.enabled is False
    assert cfg.perfect_clf == 0.0
    assert cfg.perfect_reg == 0.0
    assert cfg.min_tie_models == 2
    assert cfg.tie_decimals == 4
    assert cfg.method_subtype == "default"


def test_config_from_dict_nested_block():
    raw = {"trivial_filter": {"enabled": True, "perfect_clf": 0.02, "min_tie_models": 5}}
    cfg = TrivialFilterConfig.from_dict(raw)
    assert cfg.enabled is True
    assert cfg.perfect_clf == 0.02
    assert cfg.min_tie_models == 5
    assert cfg.tie_decimals == 4  # default retained


def test_config_from_dict_flat_block():
    raw = {"enabled": True, "tie_decimals": 6}
    cfg = TrivialFilterConfig.from_dict(raw)
    assert cfg.enabled is True
    assert cfg.tie_decimals == 6


def test_config_from_dict_none_or_empty():
    assert TrivialFilterConfig.from_dict(None).enabled is False
    assert TrivialFilterConfig.from_dict({}).enabled is False


# ---------------------------------------------------------------------------
# get_trivial_keys -- off-by-default gate
# ---------------------------------------------------------------------------


def test_get_trivial_keys_disabled_by_default_even_with_flaggable_data():
    rows = [_row("ds_a__0", f, "PLS", 0.0, "regression") for f in range(3)]
    assert get_trivial_keys(_df(rows), config=None) == set()
    assert get_trivial_keys(_df(rows), config={}) == set()
    assert get_trivial_keys(_df(rows), config={"trivial_filter": {"enabled": False}}) == set()


def test_get_trivial_keys_enabled_returns_flagged_set():
    rows = [
        *[_row("ds_a__0", f, "PLS", 0.0, "regression") for f in range(3)],
        *[_row("ds_b__0", f, "RF", 0.4, "regression") for f in range(3)],
    ]
    keys = get_trivial_keys(_df(rows), config={"trivial_filter": {"enabled": True}})
    assert keys == {"ds_a__0"}


def test_get_trivial_keys_accepts_config_object_directly():
    rows = [_row("ds_a__0", f, "PLS", 0.0, "regression") for f in range(2)]
    cfg = TrivialFilterConfig(enabled=True)
    assert get_trivial_keys(_df(rows), cfg) == {"ds_a__0"}


# ---------------------------------------------------------------------------
# get_trivial_keys_from_dir
# ---------------------------------------------------------------------------


def test_get_trivial_keys_from_dir_disabled_skips_file_read(tmp_path):
    # No hpo_results.csv written at all -- must not raise since disabled.
    keys = get_trivial_keys_from_dir(str(tmp_path), config={"trivial_filter": {"enabled": False}})
    assert keys == set()


def test_get_trivial_keys_from_dir_missing_file_returns_empty(tmp_path):
    keys = get_trivial_keys_from_dir(str(tmp_path), config={"trivial_filter": {"enabled": True}})
    assert keys == set()


def test_get_trivial_keys_from_dir_reads_real_csv(tmp_path):
    rows = [_row("ds_a__0", f, "PLS", 0.0, "regression") for f in range(2)]
    _df(rows).to_csv(tmp_path / "hpo_results.csv", index=False)
    keys = get_trivial_keys_from_dir(str(tmp_path), config={"trivial_filter": {"enabled": True}})
    assert keys == {"ds_a__0"}


# ---------------------------------------------------------------------------
# filter_trivial_keys
# ---------------------------------------------------------------------------


def test_filter_trivial_keys_drops_matching_rows():
    df = pd.DataFrame({"dataset": ["ds_a__0", "ds_a__0", "ds_b__0"], "value": [1, 2, 3]})
    out = filter_trivial_keys(df, {"ds_a__0"})
    assert out["dataset"].tolist() == ["ds_b__0"]


def test_filter_trivial_keys_noop_when_empty_set():
    df = pd.DataFrame({"dataset": ["ds_a__0"], "value": [1]})
    out = filter_trivial_keys(df, set())
    assert out is df


def test_filter_trivial_keys_noop_when_none_df():
    assert filter_trivial_keys(None, {"ds_a__0"}) is None


def test_filter_trivial_keys_noop_when_key_col_missing():
    df = pd.DataFrame({"other_col": ["ds_a__0"]})
    out = filter_trivial_keys(df, {"ds_a__0"})
    assert out is df
