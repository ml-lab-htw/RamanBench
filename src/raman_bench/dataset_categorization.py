"""Dataset categorization axes for RamanBench, following the sub-benchmark
scheme in the BeyondArena paper (arXiv:2606.30410) -- split type (IID vs.
grouped), scale (sample-count tier), and dimensionality (feature-count tier)
-- plus additional axes that matter specifically for Raman spectroscopy
benchmarking (class imbalance, single- vs. multi-target regression, and
wavenumber coverage).

Every threshold below is a plain module-level constant, calibrated against
the actual 77-dataset ``v1_default`` roster (see the tuple docstrings) rather
than copied verbatim from BeyondArena, because two of its thresholds are
degenerate for Raman spectra:

- BeyondArena's dimensionality split (low: <=100 columns, high: >100) puts
  *every* RamanBench dataset in "high" -- the narrowest spectrum here still
  has 114 wavenumber points. Recalibrated to three tiers spanning the real
  114-11,689 feature range instead.
- BeyondArena's smallest scale tier ("tiny") starts at 100 rows, but 25 of
  RamanBench's 77 datasets have fewer than 100 samples (as few as 12) --
  small-cohort spectroscopy studies are common in this domain in a way they
  aren't in large-scale tabular benchmarking. Added a "micro" tier below
  "tiny" to cover them, rather than leaving them uncategorized.

BeyondArena's "text"/"high cardinality" feature-type axes have no Raman
analogue (every feature is a continuous wavenumber intensity, never
categorical/text), so they're intentionally not reproduced here. BeyondArena's
"temporal" task type is also omitted -- RamanBench has no time-series-across-
samples structure, only the IID/grouped distinction.

This module is pure (no I/O, no dependency on ``raman_data`` types) so it can
be unit-tested directly and reused anywhere a plain scalar dataset summary is
available -- ``raman_bench_paper``'s dataset-stats cache calls
:func:`categorize_dataset` once per dataset and merges the result in.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Scale (sample count) -- BeyondArena's tiny/small/medium/large thresholds,
# with an added "micro" floor (see module docstring).
# ---------------------------------------------------------------------------

SCALE_TIERS = ("micro", "tiny", "small", "medium", "large")
_SCALE_THRESHOLDS = (
    100,
    1_000,
    10_000,
    100_000,
)  # upper-exclusive bounds for micro/tiny/small/medium


def scale_tier(n_samples: int) -> str:
    """micro (<100) / tiny (100-999) / small (1e3-1e4) / medium (1e4-1e5) / large (>=1e5)."""
    for tier, bound in zip(SCALE_TIERS, _SCALE_THRESHOLDS):
        if n_samples < bound:
            return tier
    return "large"


# ---------------------------------------------------------------------------
# Dimensionality (feature count) -- recalibrated for Raman spectra's real
# range (114-11,689 wavenumber points across the v1_default roster).
# ---------------------------------------------------------------------------

DIMENSIONALITY_TIERS = ("low", "medium", "high")
_DIM_THRESHOLDS = (1_000, 3_000)  # upper-exclusive bounds for low/medium


def dimensionality_tier(n_features: int) -> str:
    """low (<1000) / medium (1000-2999) / high (>=3000) wavenumber points."""
    for tier, bound in zip(DIMENSIONALITY_TIERS, _DIM_THRESHOLDS):
        if n_features < bound:
            return tier
    return "high"


# ---------------------------------------------------------------------------
# Split type -- IID vs. grouped, mirroring ``DatasetInfo.is_grouped``.
# ---------------------------------------------------------------------------


def split_type(is_grouped: bool | None) -> str:
    """ "grouped" / "iid" / "unknown" (``is_grouped`` not yet determined)."""
    if is_grouped is True:
        return "grouped"
    if is_grouped is False:
        return "iid"
    return "unknown"


# ---------------------------------------------------------------------------
# Class imbalance (classification only) -- standard imbalance-ratio bins
# (majority-class count / minority-class count).
# ---------------------------------------------------------------------------

IMBALANCE_TIERS = ("balanced", "moderate", "severe")
_IMBALANCE_THRESHOLDS = (3.0, 10.0)  # upper-exclusive bounds for balanced/moderate


def imbalance_tier(class_counts: list[int]) -> str:
    """balanced (IR<3) / moderate (3<=IR<10) / severe (IR>=10).

    IR (imbalance ratio) = majority-class count / minority-class count,
    the standard measure in the imbalanced-classification literature.
    """
    if not class_counts or len(class_counts) < 2:
        raise ValueError("imbalance_tier needs at least 2 classes' counts")
    counts = [c for c in class_counts if c > 0]
    ir = max(counts) / min(counts)
    for tier, bound in zip(IMBALANCE_TIERS, _IMBALANCE_THRESHOLDS):
        if ir < bound:
            return tier
    return "severe"


# ---------------------------------------------------------------------------
# Target arity -- single- vs. multi-target regression panels.
# ---------------------------------------------------------------------------


def target_arity(n_targets: int) -> str:
    """ "single-target" (1 target column) vs. "multi-target" (>1)."""
    return "single-target" if n_targets <= 1 else "multi-target"


# ---------------------------------------------------------------------------
# Wavenumber coverage -- informational only, not a difficulty axis. Whether
# the measured range extends into the high-wavenumber C-H/O-H stretch region
# (roughly >1800 cm^-1) or stays within the classic fingerprint region.
# ---------------------------------------------------------------------------

_FINGERPRINT_MAX = 1_800.0


def wavenumber_coverage(freq_max: float) -> str:
    """ "fingerprint" (max shift <=1800 cm^-1) vs. "extended" (>1800 cm^-1).

    A rough heuristic, not a precise spectroscopic boundary -- informational
    grouping only (which part of the spectrum was measured), not a claim
    about task difficulty.
    """
    return "fingerprint" if freq_max <= _FINGERPRINT_MAX else "extended"


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


def categorize_dataset(
    *,
    n_samples: int,
    n_features: int,
    n_targets: int,
    is_grouped: bool | None = None,
    freq_max: float | None = None,
    class_counts: list[int] | None = None,
) -> dict[str, str]:
    """Compute every categorization axis for one dataset from its summary stats.

    Parameters mirror the fields already collected by
    ``raman_bench_paper/scripts/dataset_stats.py``'s per-dataset stats dict,
    so callers can typically do ``categorize_dataset(**stats_subset)``.

    ``class_counts`` (per-class sample counts) is optional -- pass it for
    classification datasets to get ``imbalance_tier``; omitted (or ``None``)
    for regression datasets, where it doesn't apply.
    """
    categories = {
        "scale_tier": scale_tier(n_samples),
        "dimensionality_tier": dimensionality_tier(n_features),
        "split_type": split_type(is_grouped),
        "target_arity": target_arity(n_targets),
    }
    if freq_max is not None:
        categories["wavenumber_coverage"] = wavenumber_coverage(freq_max)
    if class_counts is not None:
        categories["imbalance_tier"] = imbalance_tier(class_counts)
    return categories
