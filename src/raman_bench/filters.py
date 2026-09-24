"""Trivial-dataset filter for RamanBench v1 results.

Flags a (dataset, target) *key* as "trivial" -- carrying little discriminative
signal between models -- using the same two-criterion definition TabArena
(NeurIPS 2025, arXiv:2506.16791) uses to curate which datasets enter its own
benchmark suite. Appendix B.1, "Dataset Selection Criteria", quoted verbatim::

    "We exclude datasets that are trivial to solve... We define trivial
    datasets as datasets where one of the following criteria applies: (1) at
    least one of the models in our scope is consistently able to achieve
    perfect performance; (2) multiple models achieve exactly the same highest
    performance."

TabArena applies this once, at dataset-curation time -- deciding which
datasets enter its fixed suite at all, before any leaderboard numbers exist.
RamanBench applies it differently: post-hoc, per (dataset, target) key,
against already-computed ``hpo_results`` (see :mod:`scripts.aggregate_results`),
opt-in and off by default (:attr:`TrivialFilterConfig.enabled`). Nothing here
excludes a dataset from being *run* -- it only flags keys whose results a
downstream leaderboard/plot/table may want to drop, exactly like
``raman_bench_paper.filters`` did for the v0.1 pipeline's per-seed metrics
CSVs. This module is the v1 port of that reference implementation, rebuilt
against v1's actual data shape (see "Data shape" below) rather than copied.

Data shape
----------
v1 caches one ``results.pkl`` per ``(model, dataset, target, repeat, fold)``
(written by ``scripts/run_experiment.py``), and ``scripts/aggregate_results.py``
turns every cached result for a run into two tidy tables via TabArena's own
``EndToEnd.from_raw``: ``model_results`` (one row per raw hyperparameter
config) and ``hpo_results`` (one row per model, recycled into
default/tuned/tuned+ensemble variants). This module's functions expect
``hpo_results`` (or an equivalent DataFrame) as input, for two reasons:

1. **Granularity.** ``raman_bench_paper.filters`` iterated per-*seed* rows,
   one seed being one independent holdout evaluation. v1 has no single
   "seed" column -- ``run_experiment.py`` runs real repeated k-fold CV, one
   job per ``(repeat, fold)`` pair. But TabArena's own pipeline already
   flattens that pair into a single scalar before it ever reaches
   ``hpo_results``/``model_results``: ``BaselineResult.split_idx`` (see
   ``tabarena.benchmark.task.utils.get_split_idx``,
   ``split_idx = n_folds * n_samples * repeat + n_samples * fold + sample``)
   is written out as that table's ``fold`` column
   (``BaselineResult.compute_df_result``: ``"fold": self.split_idx``).
   So ``hpo_results``'s ``fold`` column is already the right granularity --
   one row per independent evaluation, exactly analogous to the old
   per-seed row -- with no extra reconstruction needed. This was confirmed
   by reading the installed ``tabarena`` package directly (not assumed) and
   independently corroborated by ``scripts/plot_v1_results.py``'s own
   comment: "Each (dataset, method) has one row per (repeat, fold)".
2. **Model identity.** ``hpo_results`` carries one row per (key, fold) per
   *model*, already recycled into default/tuned/tuned+ensemble; the more
   granular ``model_results`` instead has one row per raw hyperparameter
   config, which would conflate "how many distinct configs were tried" with
   "how many distinct models tie" if used directly for criterion 2 below.
   ``method_subtype="default"`` (this module's own default) additionally
   restricts to one row per *architecture* per key/fold, so that e.g. a
   model's own "(default)" and "(tuned)" variants are never counted as two
   separately-tied "models" in criterion 2 -- matching what "the models in
   our scope" means in the TabArena quote above (distinct architectures, not
   HPO variants of the same one).

Unlike the v0.1-era metrics (F1 for classification, RMSE for regression --
opposite directions, "higher is better" vs "lower is better"), v1's
``metric_error`` column is *always* an error: TabArena/AutoGluon's own
problem-type-appropriate metric (ROC AUC for binary classification, log loss
for multiclass, RMSE for regression), converted to a uniformly
lower-is-better, zero-is-perfect error value (``Scorer.error()``, e.g.
``1 - roc_auc`` for an AUC-based binary scorer). ``problem_type`` itself is
stored using AutoGluon's own convention (``"binary"`` / ``"multiclass"`` /
``"regression"``, confirmed by reading
``tabarena.benchmark.task.wrapper.RamanBenchTaskWrapper``'s docstring and
``self.problem_type = metadata.problem_type`` assignment directly) -- not the
``"classification"``/``"regression"`` strings RamanBench's own
``run_experiment.py`` uses internally before task construction.

Because of this, ``perfect_clf``/``perfect_reg`` here are **both**
"at-or-below" error thresholds (default ``0.0`` each, meaning zero error --
the theoretical floor for any of these metrics), not the
opposite-direction ``perfect_clf=1.0`` (an F1 ceiling) /
``perfect_reg=0.0`` (an RMSE floor) pair ``raman_bench_paper.filters`` used.
Same config *keys*, deliberately different default *semantics* -- copying
the old ``1.0`` default for ``perfect_clf`` here would silently never match
anything, since a v1 classification error essentially never sits at or above
``1.0``. Kept as two separate keys anyway (rather than one shared
``perfect_error``) so a caller can still tune classification- and
regression-key thresholds independently, and for straightforward config-file
compatibility with the shape used in ``raman_bench_paper/configs/*.json``.

Tie detection (criterion 2) rounds ``metric_error`` to ``tie_decimals``
places before comparing, the same floating-point-noise-tolerant convention
added upstream in ``bencheval.winrate_utils.compute_winrate_matrix``/
``compute_winrate`` by ``autogluon/tabarena#311`` -- a PR by this project's
own maintainer, motivated (per that PR's own description) by exactly this
porting work: "While porting the trivial-dataset criterion to our benchmark,
we noticed [strict equality was too conservative for floating-point noise]".
This module does **not**, however, call ``bencheval``'s
``compute_winrate``/``compute_winrate_matrix`` directly -- see "Why not
bencheval" below -- it just reuses the same round-before-compare technique
and parameter name, independently.

Why not bencheval
------------------
``bencheval.winrate_utils.compute_winrate_matrix``/``compute_winrate`` (the
function ``tie_decimals`` actually landed on) computes a pairwise win-rate
matrix -- or its per-method average -- aggregated (seed-weighted) *across*
every task in the input, i.e. a single leaderboard-style summary over many
datasets at once. That is a different shape of question from what this
module needs: an independent True/False decision *per key*, gated on the
condition holding on *every single fold* for that key specifically (an AND
across folds, not an average across folds/tasks). Reusing it would mean
calling it once per key just to read an aggregate win-rate back out and
reverse-engineer "was every fold tied" from a single blended number -- more
indirection than the direct groupby-round-compare below, for the same
result. More importantly, criterion 1 (a model reaching a *perfect absolute
score*) has no equivalent in ``compute_winrate_matrix`` at all -- win-rate is
relative (who beat whom), never compared against an absolute
threshold -- so at least half of this filter would need an independent
implementation regardless of the tie-detection choice. Given that, this
module keeps a small, self-contained, directly testable implementation
(mirroring ``raman_bench_paper.filters.compute_trivial_keys``'s own
structure) rather than a partial, awkward-fit dependency on ``bencheval``'s
leaderboard-shaped utilities.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)

# AutoGluon's own problem_type strings, as actually written into v1's
# hpo_results/model_results ``problem_type`` column (see module docstring).
CLASSIFICATION_PROBLEM_TYPES = frozenset({"binary", "multiclass"})
REGRESSION_PROBLEM_TYPES = frozenset({"regression"})


@dataclass
class TrivialFilterConfig:
    """Config-driven settings for the trivial-dataset filter. Off by default.

    Mirrors the ``trivial_filter`` block shape from
    ``raman_bench_paper/configs/v0_default.json`` (``enabled``, ``perfect_clf``,
    ``perfect_reg``, ``min_tie_models``, ``tie_decimals``) for continuity, plus
    one v1-specific addition (``method_subtype``, see :func:`compute_trivial_keys`).
    See the module docstring for why ``perfect_clf``/``perfect_reg`` default to
    ``0.0`` here rather than the paper config's ``1.0``/``0.0`` pair.
    """

    enabled: bool = False
    perfect_clf: float = 0.0
    perfect_reg: float = 0.0
    min_tie_models: int = 2
    tie_decimals: int = 4
    method_subtype: str | None = "default"

    @classmethod
    def from_dict(cls, raw: dict | None) -> TrivialFilterConfig:
        """Build from a plain dict, either the ``trivial_filter`` block itself
        or a parent config containing one under the ``"trivial_filter"`` key
        (matching how ``raman_bench_paper.filters.get_trivial_keys`` reads its
        config)."""
        raw = raw or {}
        cfg = raw.get("trivial_filter", raw) if isinstance(raw, dict) else {}
        cfg = cfg or {}
        return cls(
            enabled=bool(cfg.get("enabled", False)),
            perfect_clf=float(cfg.get("perfect_clf", 0.0)),
            perfect_reg=float(cfg.get("perfect_reg", 0.0)),
            min_tie_models=int(cfg.get("min_tie_models", 2)),
            tie_decimals=int(cfg.get("tie_decimals", 4)),
            method_subtype=cfg.get("method_subtype", "default"),
        )


def compute_trivial_keys(
    hpo_results: pd.DataFrame,
    *,
    perfect_clf: float = 0.0,
    perfect_reg: float = 0.0,
    min_tie_models: int = 2,
    tie_decimals: int = 4,
    method_subtype: str | None = "default",
    key_col: str = "dataset",
    model_col: str = "ta_name",
    fold_col: str = "fold",
    error_col: str = "metric_error",
    problem_type_col: str = "problem_type",
    exclude_models: Iterable[str] = (),
) -> dict[str, str]:
    """Return ``{key: reason}`` for every key flagged trivial.

    Pure function: no config gating, no I/O -- see :func:`get_trivial_keys`
    for the config-driven entry point. Operates on a long-form DataFrame
    shaped like ``scripts/aggregate_results.py``'s ``hpo_results`` output
    (one row per (key, model, fold), see the module docstring for why).

    ``reason`` is ``"perfect:<model>"`` (criterion 1) or ``"tie:<n>"``
    (criterion 2, ``n`` = the largest tied-model count observed across the
    key's folds).

    Parameters
    ----------
    hpo_results : pd.DataFrame
        Must contain ``key_col``, ``model_col``, ``fold_col``, ``error_col``,
        ``problem_type_col``. Extra columns (``method``, ``method_subtype``,
        ``time_train_s``, ...) are ignored except ``method_subtype`` when
        ``method_subtype`` (the filter parameter) is not ``None``.
    perfect_clf, perfect_reg : float
        Criterion-1 thresholds: a model is "perfect" on a key if its
        ``error_col`` value is at-or-below this threshold on *every* fold for
        that key. ``perfect_clf`` applies when the key's ``problem_type`` is
        one of :data:`CLASSIFICATION_PROBLEM_TYPES`; ``perfect_reg`` when it
        is :data:`REGRESSION_PROBLEM_TYPES`. Both are error thresholds
        (lower = stricter), not raw-metric thresholds -- see the module
        docstring.
    min_tie_models : int
        Criterion-2 threshold: how many models must share the lowest
        (rounded) error on a fold for it to count as a tie. TabArena's own
        wording is "multiple models" (>= 2); ``raman_bench_paper``'s own
        applied config raised this to 5 for its own purposes -- default here
        matches the TabArena definition itself (2), not that one repo's
        tuning.
    tie_decimals : int
        Round ``error_col`` to this many decimals before tie comparison, to
        absorb floating-point noise -- same convention as
        ``bencheval.winrate_utils.compute_winrate_matrix``'s own
        ``tie_decimals`` (``autogluon/tabarena#311``); see "Why not
        bencheval" in the module docstring for why that function isn't
        called directly.
    method_subtype : str | None
        If not ``None`` and a ``method_subtype`` column is present, restrict
        to rows with this subtype first (default ``"default"``) -- keeps one
        row per (key, model, fold), so a model's own tuned/tuned+ensemble
        variants are never double-counted as separate "models" for
        criterion 2. Pass ``None`` to use every row as-is (e.g. against
        ``model_results``, where there is no ``method_subtype`` column).
    key_col, model_col, fold_col, error_col, problem_type_col : str
        Column names, defaulting to ``scripts/aggregate_results.py``'s own
        output schema. ``model_col`` defaults to ``"ta_name"`` (not
        ``"config_type"``) to match the model-identity column
        ``scripts/plot_v1_results.py`` already groups by, for consistency
        within this codebase; the two columns are equivalent modulo casing.
    exclude_models : Iterable[str]
        Model names to drop before evaluating either criterion (e.g. a
        trivially-perfect ``DUMMY``/constant-predictor baseline that would
        otherwise flag every key).

    Returns
    -------
    dict[str, str]
        Empty if ``hpo_results`` is empty/None or nothing is flagged.
    """
    if hpo_results is None or len(hpo_results) == 0:
        return {}

    required = {key_col, model_col, fold_col, error_col, problem_type_col}
    missing = required - set(hpo_results.columns)
    if missing:
        raise ValueError(
            f"hpo_results is missing required column(s): {sorted(missing)}. "
            f"Available columns: {sorted(hpo_results.columns)}"
        )

    df = hpo_results.copy()
    if method_subtype is not None and "method_subtype" in df.columns:
        df = df[df["method_subtype"] == method_subtype]
    if exclude_models:
        df = df[~df[model_col].isin(set(exclude_models))]
    df = df.dropna(subset=[error_col])
    if df.empty:
        return {}

    flagged: dict[str, str] = {}

    # Criterion 1: a model's error is at/below the problem-type-appropriate
    # "perfect" threshold on EVERY fold evaluated for that key.
    for (key, model), grp in df.groupby([key_col, model_col]):
        if key in flagged:
            continue
        ptypes = set(grp[problem_type_col].unique())
        if ptypes <= REGRESSION_PROBLEM_TYPES:
            threshold = perfect_reg
        elif ptypes <= CLASSIFICATION_PROBLEM_TYPES:
            threshold = perfect_clf
        else:
            # A single key should never mix problem types; if the input is
            # malformed enough that it does, skip rather than guess which
            # threshold applies.
            logger.warning(
                "Key %r has mixed/unrecognized problem_type values %s -- skipping "
                "criterion-1 check for it.",
                key,
                sorted(ptypes),
            )
            continue
        if (grp[error_col] <= threshold).all():
            flagged[key] = f"perfect:{model}"

    # Criterion 2: >= min_tie_models models tied for the lowest (best,
    # rounded) error on EVERY fold evaluated for that key.
    for key, grp in df.groupby(key_col):
        if key in flagged:
            continue
        folds = grp[fold_col].unique()
        if len(folds) == 0:
            continue
        tied_every_fold = True
        max_n_tied = 0
        for fold in folds:
            sub = grp[grp[fold_col] == fold]
            if sub.empty:
                tied_every_fold = False
                break
            errs = sub[error_col].round(tie_decimals)
            best = errs.min()
            n_tied = int((errs == best).sum())
            if n_tied < min_tie_models:
                tied_every_fold = False
                break
            max_n_tied = max(max_n_tied, n_tied)
        if tied_every_fold:
            flagged[key] = f"tie:{max_n_tied}"

    return flagged


def get_trivial_keys(
    hpo_results: pd.DataFrame,
    config: dict | TrivialFilterConfig | None,
    exclude_models: Iterable[str] = (),
) -> set[str]:
    """Config-gated entry point. Returns the set of keys to drop.

    Returns the empty set (no computation performed) unless
    ``config``/``config["trivial_filter"]`` has ``"enabled": true`` (or
    ``config`` is already a :class:`TrivialFilterConfig` with
    ``enabled=True``). Logs each flagged key and its reason when any are
    found, matching ``raman_bench_paper.filters.get_trivial_keys``'s own
    console output.
    """
    cfg = (
        config if isinstance(config, TrivialFilterConfig) else TrivialFilterConfig.from_dict(config)
    )
    if not cfg.enabled:
        return set()

    flagged = compute_trivial_keys(
        hpo_results,
        perfect_clf=cfg.perfect_clf,
        perfect_reg=cfg.perfect_reg,
        min_tie_models=cfg.min_tie_models,
        tie_decimals=cfg.tie_decimals,
        method_subtype=cfg.method_subtype,
        exclude_models=exclude_models,
    )
    if flagged:
        logger.info("[trivial-filter] excluding %d dataset key(s):", len(flagged))
        for key, reason in sorted(flagged.items()):
            logger.info("  - %s  (%s)", key, reason)
    return set(flagged.keys())


def get_trivial_keys_from_dir(
    aggregated_dir: str,
    config: dict | TrivialFilterConfig | None,
    exclude_models: Iterable[str] = (),
    filename: str = "hpo_results.csv",
) -> set[str]:
    """Convenience wrapper: load ``hpo_results.csv`` from an aggregated results
    directory (``scripts/aggregate_results.py``'s ``--output-dir``) and call
    :func:`get_trivial_keys` on it. Returns the empty set (without reading the
    file) if the filter is disabled or the file doesn't exist.
    """
    cfg = (
        config if isinstance(config, TrivialFilterConfig) else TrivialFilterConfig.from_dict(config)
    )
    if not cfg.enabled:
        return set()
    path = os.path.join(aggregated_dir, filename)
    if not os.path.isfile(path):
        logger.warning("[trivial-filter] %s not found -- nothing to filter.", path)
        return set()
    hpo_results = pd.read_csv(path)
    return get_trivial_keys(hpo_results, cfg, exclude_models=exclude_models)


def filter_trivial_keys(
    df: pd.DataFrame,
    trivial_keys: Iterable[str],
    key_col: str = "dataset",
) -> pd.DataFrame:
    """Drop rows whose ``key_col`` value is in ``trivial_keys``.

    Returns ``df`` unchanged (not a copy) if ``df`` is ``None``,
    ``trivial_keys`` is empty, or ``key_col`` is not a column of ``df`` --
    mirroring ``raman_bench_paper.filters.filter_keys``'s permissive
    no-op-on-mismatch behavior, so callers can pass this both
    ``model_results`` and ``hpo_results`` (or any other frame keyed the same
    way) without checking shapes first.

    Generic despite the name (it just drops rows by ``key_col`` membership) --
    also the right function to call with :func:`get_unlearnable_keys`'s output,
    rather than duplicating this for the "not learnable" criterion.
    """
    trivial_keys = set(trivial_keys)
    if df is None or not trivial_keys or key_col not in df.columns:
        return df
    return df[~df[key_col].isin(trivial_keys)].copy()


# ── "Not learnable" filter (v1-native port of ablation_baseline_check.py) ──────

DEFAULT_DUMMY_MODEL = "DUMMY"
DEFAULT_EXCLUDE_FROM_BEST = frozenset({"DUMMY", "AUTOGLUON"})


@dataclass
class LearnabilityFilterConfig:
    """Config-driven settings for the "not learnable" filter. Off by default.

    v1-native reimplementation of ``raman_bench_paper/scripts/ablation_baseline_check.py``'s
    ablation, generalized to a single metric-agnostic criterion -- see
    :func:`compute_unlearnable_keys`'s docstring for why the paper's two separate
    formulas (classification F1 margin, regression R² floor) collapse into one
    "beats Dummy's error by more than a margin" check under v1's ``metric_error``
    convention. Off by default, same as :class:`TrivialFilterConfig` and for the
    same reason: this flags keys a downstream leaderboard/plot/table may want to
    drop, it never excludes a key from being *run*.

    Unlike the trivial filter, a key flagged here is not permanently trivial --
    a new/better model can beat Dummy by more than ``min_dummy_margin`` on a key
    that previously failed. See ``configs/v1/EXCLUDED_TARGETS.md``'s periodic
    "learnability sweep" for the intended re-check process.
    """

    enabled: bool = False
    min_dummy_margin: float = 0.05
    method_subtype: str | None = "default"
    dummy_model: str = DEFAULT_DUMMY_MODEL

    @classmethod
    def from_dict(cls, raw: dict | None) -> LearnabilityFilterConfig:
        """Build from a plain dict, either the ``learnability_filter`` block
        itself or a parent config containing one under that key (mirrors
        :meth:`TrivialFilterConfig.from_dict`)."""
        raw = raw or {}
        cfg = raw.get("learnability_filter", raw) if isinstance(raw, dict) else {}
        cfg = cfg or {}
        return cls(
            enabled=bool(cfg.get("enabled", False)),
            min_dummy_margin=float(cfg.get("min_dummy_margin", 0.05)),
            method_subtype=cfg.get("method_subtype", "default"),
            dummy_model=cfg.get("dummy_model", DEFAULT_DUMMY_MODEL),
        )


def compute_unlearnable_keys(
    hpo_results: pd.DataFrame,
    *,
    min_dummy_margin: float = 0.05,
    method_subtype: str | None = "default",
    key_col: str = "dataset",
    model_col: str = "ta_name",
    error_col: str = "metric_error",
    dummy_model: str = DEFAULT_DUMMY_MODEL,
    exclude_from_best: Iterable[str] = DEFAULT_EXCLUDE_FROM_BEST,
) -> dict[str, str]:
    """Return ``{key: reason}`` for every key flagged "not learnable".

    Pure function, same shape/conventions as :func:`compute_trivial_keys`
    (see that docstring for the ``hpo_results``/column-name background).

    Port of ``raman_bench_paper.filters``' ablation-check criterion
    (``scripts/ablation_baseline_check.py``): a key fails if no in-scope model
    meaningfully beats the ``Dummy`` (mean/majority-class) baseline, *averaged
    over folds* -- not an AND-across-folds like :func:`compute_trivial_keys`'s
    criteria, matching the paper script's own ``groupby(...).mean()`` before
    thresholding.

    Why one criterion instead of the paper's two
    -----------------------------------------------
    The paper script used two *differently-shaped* formulas: classification
    passed if ``best F1 - Dummy F1 > 0.05``; regression passed if
    ``best R² > 0`` (no Dummy comparison at all -- R²>0 is, by definition,
    "beats the mean predictor"). Porting both literally would need R² in
    ``hpo_results``, which v1's schema doesn't carry (only ``metric_error``,
    TabArena/AutoGluon's own problem-type-appropriate *error* -- see the
    module docstring's "Unlike the v0.1-era metrics" section).

    These two formulas are the same underlying idea in different clothes,
    though: RamanBench's own ``DUMMY`` model predicts the training mean for
    regression (``raman_bench/models/generate/dummy.py``), so "beats the mean
    predictor" (R²>0) *is* "beats Dummy" -- for regression, Dummy's RMSE and
    the mean-predictor RMSE are the same number. So both problem types reduce
    to one general rule: "does the best in-scope model beat Dummy's error by
    more than ``min_dummy_margin``?" -- computed directly against
    ``metric_error`` (always lower-is-better, always zero-is-perfect, per the
    module docstring), with no problem-type branching needed at all.

    This is *not* claimed to be numerically identical to the paper's original
    regression check (an error-margin threshold vs. an R²-floor threshold are
    different units), so re-deriving ``configs/v1/quality_exclusions.json``'s
    ``not_learnable`` entries with this function against real v1 data may not
    reproduce the exact v0.1-era list byte-for-byte -- expected, and fine:
    the intent (flag keys nothing beats Dummy on) is preserved, the specific
    keys flagged may legitimately shift once real v1-model coverage exists.

    Parameters
    ----------
    hpo_results : pd.DataFrame
        Must contain ``key_col``, ``model_col``, ``error_col``. Extra columns
        ignored except ``method_subtype`` when that filter parameter is not
        ``None`` (same restriction-to-one-row-per-architecture-per-key
        rationale as :func:`compute_trivial_keys`).
    min_dummy_margin : float
        How much lower (better) the best in-scope model's mean error must be
        than Dummy's mean error to count as "learnable". Default ``0.05``,
        carried over from the paper's classification threshold -- an
        ``error_col`` unit, not an F1 unit, so this is a reasonable starting
        point rather than a like-for-like port; tune per deployment.
    dummy_model : str
        Model name treated as the baseline (default ``"DUMMY"``).
    exclude_from_best : Iterable[str]
        Models dropped before picking the "best" (default ``DUMMY`` +
        ``AUTOGLUON``, matching ``ablation_baseline_check.py``'s own
        ``other_scores.drop(index=[DUMMY_MODEL, "AUTOGLUON"])`` -- AUTOGLUON
        is TabArena's own bagged-ensemble-of-everything meta-model, excluded
        so a key can't look "learnable" purely because the ensemble
        arithmetically outperforms Dummy while every individual architecture
        still fails to).

    Returns
    -------
    dict[str, str]
        Empty if ``hpo_results`` is empty/None or nothing is flagged. Reason
        is ``"best_error=<value> (Dummy not beaten by >= <margin>)"``, or
        ``"no_dummy_baseline"`` if the key has no Dummy row to compare
        against (flagged conservatively rather than silently passed).
    """
    if hpo_results is None or len(hpo_results) == 0:
        return {}

    required = {key_col, model_col, error_col}
    missing = required - set(hpo_results.columns)
    if missing:
        raise ValueError(
            f"hpo_results is missing required column(s): {sorted(missing)}. "
            f"Available columns: {sorted(hpo_results.columns)}"
        )

    df = hpo_results.copy()
    if method_subtype is not None and "method_subtype" in df.columns:
        df = df[df["method_subtype"] == method_subtype]
    df = df.dropna(subset=[error_col])
    if df.empty:
        return {}

    flagged: dict[str, str] = {}
    exclude_from_best = set(exclude_from_best)
    agg = df.groupby([key_col, model_col])[error_col].mean().reset_index()

    for key, grp in agg.groupby(key_col):
        scores = grp.set_index(model_col)[error_col]
        dummy_error = scores.get(dummy_model)
        if dummy_error is None or pd.isna(dummy_error):
            flagged[key] = "no_dummy_baseline"
            continue
        candidates = scores.drop(index=exclude_from_best, errors="ignore")
        if candidates.empty:
            flagged[key] = "no_dummy_baseline"
            continue
        best_error = candidates.min()
        if (dummy_error - best_error) <= min_dummy_margin:
            flagged[key] = f"best_error={best_error:.4g} (Dummy not beaten by >= {min_dummy_margin})"

    return flagged


def get_unlearnable_keys(
    hpo_results: pd.DataFrame,
    config: dict | LearnabilityFilterConfig | None,
) -> set[str]:
    """Config-gated entry point, mirroring :func:`get_trivial_keys`.

    Returns the empty set (no computation performed) unless
    ``config``/``config["learnability_filter"]`` has ``"enabled": true``.
    """
    cfg = (
        config
        if isinstance(config, LearnabilityFilterConfig)
        else LearnabilityFilterConfig.from_dict(config)
    )
    if not cfg.enabled:
        return set()

    flagged = compute_unlearnable_keys(
        hpo_results,
        min_dummy_margin=cfg.min_dummy_margin,
        method_subtype=cfg.method_subtype,
        dummy_model=cfg.dummy_model,
    )
    if flagged:
        logger.info("[learnability-filter] excluding %d dataset key(s):", len(flagged))
        for key, reason in sorted(flagged.items()):
            logger.info("  - %s  (%s)", key, reason)
    return set(flagged.keys())


def get_unlearnable_keys_from_dir(
    aggregated_dir: str,
    config: dict | LearnabilityFilterConfig | None,
    filename: str = "hpo_results.csv",
) -> set[str]:
    """Convenience wrapper, mirroring :func:`get_trivial_keys_from_dir`: load
    ``hpo_results.csv`` from an aggregated results directory and call
    :func:`get_unlearnable_keys` on it. Returns the empty set (without
    reading the file) if the filter is disabled or the file doesn't exist.
    """
    cfg = (
        config
        if isinstance(config, LearnabilityFilterConfig)
        else LearnabilityFilterConfig.from_dict(config)
    )
    if not cfg.enabled:
        return set()
    path = os.path.join(aggregated_dir, filename)
    if not os.path.isfile(path):
        logger.warning("[learnability-filter] %s not found -- nothing to filter.", path)
        return set()
    hpo_results = pd.read_csv(path)
    return get_unlearnable_keys(hpo_results, cfg)
