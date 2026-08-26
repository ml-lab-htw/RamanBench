"""Tests for raman_bench.dataset_categorization.

Tier boundaries are checked exactly at their edges (upper-exclusive), and a
handful of real datasets from the v1_default roster are used as sanity
anchors so a future threshold change has to consciously update these too.
"""

import pytest

from raman_bench.dataset_categorization import (
    categorize_dataset,
    dimensionality_tier,
    imbalance_tier,
    scale_tier,
    split_type,
    target_arity,
    wavenumber_coverage,
)

# ---------------------------------------------------------------------------
# scale_tier
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n_samples,expected",
    [
        (1, "micro"),
        (99, "micro"),
        (100, "tiny"),
        (999, "tiny"),
        (1_000, "small"),
        (9_999, "small"),
        (10_000, "medium"),
        (99_999, "medium"),
        (100_000, "large"),
        (130_061, "large"),  # mlrod
    ],
)
def test_scale_tier_boundaries(n_samples, expected):
    assert scale_tier(n_samples) == expected


def test_scale_tier_real_anchors():
    assert scale_tier(90) == "micro"  # amino_acids_glycine
    assert scale_tier(1151) == "small"  # alzheimer
    assert scale_tier(53134) == "medium"  # wheat_lines


# ---------------------------------------------------------------------------
# dimensionality_tier
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n_features,expected",
    [
        (1, "low"),
        (999, "low"),
        (1_000, "medium"),
        (2_999, "medium"),
        (3_000, "high"),
        (11_689, "high"),  # itaconic_acid_species
    ],
)
def test_dimensionality_tier_boundaries(n_features, expected):
    assert dimensionality_tier(n_features) == expected


def test_dimensionality_tier_real_anchors():
    assert dimensionality_tier(114) == "low"  # tg_ecoli_fermentation
    assert dimensionality_tier(885) == "low"  # alzheimer
    assert dimensionality_tier(1870) == "medium"  # bioprocess_substrates


# ---------------------------------------------------------------------------
# split_type
# ---------------------------------------------------------------------------


def test_split_type():
    assert split_type(True) == "grouped"
    assert split_type(False) == "iid"
    assert split_type(None) == "unknown"


# ---------------------------------------------------------------------------
# imbalance_tier
# ---------------------------------------------------------------------------


def test_imbalance_tier_balanced():
    assert imbalance_tier([50, 50]) == "balanced"
    assert imbalance_tier([100, 50]) == "balanced"  # IR=2 < 3


def test_imbalance_tier_moderate():
    assert imbalance_tier([100, 20]) == "moderate"  # IR=5


def test_imbalance_tier_severe():
    assert imbalance_tier([1000, 20]) == "severe"  # IR=50


def test_imbalance_tier_multiclass_uses_extremes():
    assert imbalance_tier([500, 100, 90, 5]) == "severe"  # IR=100


def test_imbalance_tier_requires_at_least_two_classes():
    with pytest.raises(ValueError):
        imbalance_tier([100])


# ---------------------------------------------------------------------------
# target_arity
# ---------------------------------------------------------------------------


def test_target_arity():
    assert target_arity(1) == "single-target"
    assert target_arity(2) == "multi-target"
    assert target_arity(12) == "multi-target"


# ---------------------------------------------------------------------------
# wavenumber_coverage
# ---------------------------------------------------------------------------


def test_wavenumber_coverage():
    assert wavenumber_coverage(1800) == "fingerprint"
    assert wavenumber_coverage(1799) == "fingerprint"
    assert wavenumber_coverage(1801) == "extended"
    assert wavenumber_coverage(3000) == "extended"


# ---------------------------------------------------------------------------
# categorize_dataset (aggregator)
# ---------------------------------------------------------------------------


def test_categorize_dataset_regression_no_class_counts():
    result = categorize_dataset(
        n_samples=6960, n_features=1870, n_targets=1, is_grouped=False, freq_max=1900.0
    )
    assert result == {
        "scale_tier": "small",
        "dimensionality_tier": "medium",
        "split_type": "iid",
        "target_arity": "single-target",
        "wavenumber_coverage": "extended",
    }
    assert "imbalance_tier" not in result


def test_categorize_dataset_classification_with_class_counts():
    result = categorize_dataset(
        n_samples=1151,
        n_features=885,
        n_targets=1,
        is_grouped=None,
        class_counts=[600, 551],
    )
    assert result["scale_tier"] == "small"
    assert result["dimensionality_tier"] == "low"
    assert result["split_type"] == "unknown"
    assert result["imbalance_tier"] == "balanced"
    assert "wavenumber_coverage" not in result
