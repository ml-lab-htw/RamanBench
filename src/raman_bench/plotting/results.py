"""Load aggregated v1 results and score them the way TabArena does.

Input is ``hpo_results.csv`` as written by ``scripts/aggregate_results.py``
(one row per task, fold and model variant). Scoring uses TabArena's own
evaluator, ``bencheval.evaluator.BenchmarkEvaluator``: Elo with bootstrap
confidence intervals calibrated so that Random Forest scores 1000, win rate,
improvability and the pairwise win-rate matrix. Nothing here re-derives a
metric TabArena already defines.

Missing runs are handled as in TabArena: a model that has no result for a
task gets the reference model's (Random Forest's) result there and the row is
flagged as imputed; ``imputed_pct`` carries how much of a model's score is
imputed. A model with no result at all for one task type (e.g. ROCKET on
regression) is left out of that task type instead of being imputed, since it
cannot run there. Times are only ever taken from real, non-imputed runs.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from raman_bench.plotting import models as model_info

logger = logging.getLogger(__name__)

TASK_GROUPS = ("all", "classification", "regression")
VARIANTS = ("default", "tuned", "tuned_ensemble")

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SCOPE = REPO_ROOT / "configs" / "v1" / "scope_default.json"
DEFAULT_TARGET_LIST = REPO_ROOT / "configs" / "v1" / "target_list.json"


@dataclass
class GroupScores:
    """Scores of every model on one task group (all / classification / regression)."""

    name: str
    leaderboard: pd.DataFrame
    """One row per model (index), sorted by Elo. See :func:`score_group` for the columns."""
    winrate_matrix: pd.DataFrame
    """Square matrix: entry (i, j) is the fraction of tasks where model i beats model j."""
    results: pd.DataFrame
    """Per (dataset, fold, model) rows the scores were computed from, imputation included."""
    n_tasks: int


def load_results(
    hpo_results: str | Path,
    *,
    variant: str = "default",
    scope: str | Path | None = DEFAULT_SCOPE,
    target_list: str | Path | None = DEFAULT_TARGET_LIST,
    exclude_models: tuple[str, ...] = ("DUMMY",),
) -> pd.DataFrame:
    """Read ``hpo_results.csv`` into one tidy row per (dataset, fold, model).

    Parameters
    ----------
    variant
        ``"default"`` (each model's default configuration), ``"tuned"`` or
        ``"tuned_ensemble"``. Models that were only run with their default
        configuration have no tuned rows; for them the default row is used.
    scope
        Scope config whose ``models`` list restricts which models are plotted
        (``None`` keeps every model in the file).
    target_list
        ``configs/v1/target_list.json``; keys marked ``excluded`` there are
        dropped, and so are folds beyond the key's ``n_repeats`` x the scope's
        ``n_splits`` (``None`` keeps every key and fold).
    exclude_models
        Model keys left out entirely (default: the ``DUMMY`` baseline).

    AutoGluon-extreme runs (``scripts/run_autogluon_baseline.py``; baseline rows
    without a ``config_type``) are mapped to their ``AUTOGLUON-EXTREME-*`` keys and
    flagged ``reference``: they are scored like every model but plotted as lines.
    """
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")

    df = pd.read_csv(hpo_results, low_memory=False)
    dir_to_key = {d: k for k, (_, d) in model_info.REFERENCE_MODELS.items()}
    is_ref = df["config_type"].isna() & df["method"].isin(dir_to_key)
    df.loc[is_ref, "config_type"] = df.loc[is_ref, "method"].map(dir_to_key)
    df.loc[is_ref, "method_subtype"] = "default"
    df = df.dropna(subset=["config_type"])
    df = df[~df["config_type"].isin(set(exclude_models))]
    df = df[df["method_subtype"].isin({"default", variant})]
    if variant != "default":
        # Prefer the requested variant; fall back to the default row where a model
        # has no tuned run for that task.
        df = df.assign(_pref=(df["method_subtype"] == variant).astype(int))
        df = df.sort_values("_pref").drop_duplicates(["dataset", "fold", "config_type"], keep="last")
        df = df.drop(columns="_pref")

    # Two result directories can map to one model key: e.g. TabArena's own TA-LimiX-2
    # smoke-test runs next to RamanBench-LimiX2 (both config_type LIMIX2). Keep the
    # RamanBench run, which is what the sweep produces.
    ta_name = df["ta_name"].astype(str) if "ta_name" in df.columns else pd.Series("", index=df.index)
    df = df.assign(ta_name=ta_name, _ours=ta_name.str.startswith("RamanBench-").astype(int))
    df = df.sort_values("_ours")
    dup = df.duplicated(["dataset", "fold", "config_type"], keep="last")
    if dup.any():
        logger.warning(
            "Dropping %d duplicate row(s) for the same (task, fold, model): %s",
            int(dup.sum()), df.loc[dup].groupby(["config_type", "ta_name"]).size().to_dict(),
        )
    df = df[~dup].drop(columns="_ours")

    out = pd.DataFrame(
        {
            "dataset": df["dataset"].astype(str),
            "fold": df["fold"].astype(int),
            "model": df["config_type"].astype(str),
            "metric_error": df["metric_error"].astype(float).clip(lower=0),
            "metric": df["metric"].astype(str),
            "task": np.where(df["problem_type"] == "regression", "regression", "classification"),
            "time_train_s": df["time_train_s"].astype(float),
            "time_infer_s": df["time_infer_s"].astype(float),
        }
    )
    out["reference"] = out["model"].isin(model_info.REFERENCE_MODELS)
    out = out.dropna(subset=["metric_error"])

    n_splits = 3
    if scope is not None:
        with open(scope) as f:
            scope_cfg = json.load(f)
        in_scope = set(scope_cfg["models"])
        n_splits = int(scope_cfg.get("n_splits", n_splits))
        dropped = sorted(set(out["model"]) - in_scope)
        if dropped:
            logger.info("Dropping %d model(s) outside the scope: %s", len(dropped), dropped)
        out = out[out["model"].isin(in_scope)]

    if target_list is not None:
        with open(target_list) as f:
            targets = json.load(f)
        kept = [t for t in targets if not t["excluded"]]
        key = [f"{t['dataset']}__{t['target_idx']}" for t in kept]
        n_rows = dict(zip(key, (t["num_instances"] for t in kept)))
        # Folds the sweep runs per task: n_repeats x n_splits. Older sweeps ran more
        # repeats for some models; scoring those extra folds would impute every other model.
        n_folds = dict(zip(key, (t["n_repeats"] * n_splits for t in kept)))
        n_before = out["dataset"].nunique()
        out = out[out["dataset"].isin(n_folds)]
        out = out[out["fold"] < out["dataset"].map(n_folds)]
        logger.info("Kept %d of %d task(s) (excluded targets dropped)", out["dataset"].nunique(), n_before)
        out["num_instances"] = out["dataset"].map(n_rows)

    return out.reset_index(drop=True)


def impute_missing(
    results: pd.DataFrame,
    *,
    reference_model: str = "RF",
    max_imputed_pct: float = 50.0,
) -> pd.DataFrame:
    """Fill each model's missing (dataset, fold) cells with *reference_model*'s result.

    The task set is the reference model's: tasks it has no result for are
    dropped for everyone. Models above *max_imputed_pct* percent imputed are
    dropped (their score would mostly be the reference model's). Imputed rows
    carry ``imputed=True`` and NaN times.
    """
    ref = results[results["model"] == reference_model].set_index(["dataset", "fold"])
    if ref.empty:
        raise ValueError(f"Reference model {reference_model!r} has no results")

    parts = []
    dropped = {}
    for model, rows in results.groupby("model"):
        rows = rows.set_index(["dataset", "fold"])
        rows = rows[rows.index.isin(ref.index)]
        missing = ref.index.difference(rows.index)
        pct = 100.0 * len(missing) / len(ref)
        if pct > max_imputed_pct:
            dropped[model] = round(pct, 1)
            continue
        filled = ref.loc[missing].copy()
        filled["model"] = model
        filled[["time_train_s", "time_infer_s"]] = np.nan
        parts.append(pd.concat([rows.assign(imputed=False), filled.assign(imputed=True)]))
    if dropped:
        logger.warning(
            "Dropped %d model(s) with more than %.0f%% imputed results: %s",
            len(dropped), max_imputed_pct, dropped,
        )
    return pd.concat(parts).reset_index()


def normalized_score(results_per_task: pd.DataFrame) -> pd.Series:
    """TabRepo normalised score per model: best -> 1, median -> 0, clipped at 0.

    Computed per dataset on fold-averaged error, then averaged over datasets
    (Salinas & Erickson, TabRepo), so every dataset weighs the same.
    """
    err = results_per_task.groupby(["dataset", "model"])["metric_error"].mean().unstack("model")
    best = err.min(axis=1)
    median = err.median(axis=1)
    denom = (median - best).where(lambda s: s > 1e-12)
    norm = err.rsub(median, axis=0).div(denom, axis=0).clip(lower=0)
    # Datasets where median == best: every model at least as good as the median scores 1.
    ties = denom.isna()
    norm.loc[ties] = err.loc[ties].le(median[ties], axis=0).astype(float)
    return norm.mean(axis=0)


def score_group(
    results: pd.DataFrame,
    name: str,
    *,
    reference_model: str = "RF",
    max_imputed_pct: float = 50.0,
    bootstrap_rounds: int = 200,
    n_splits: int = 3,
) -> GroupScores:
    """Score the models on one task group.

    Leaderboard columns: ``elo``, ``elo+``/``elo-`` (95% bootstrap CI widths),
    ``rank``, ``winrate``, ``improvability`` (+ CI widths), ``normalized_score``,
    ``median_time_train_s``, ``median_time_infer_s``, ``median_time_total_s``,
    ``median_infer_per_1k_s``, ``median_time_total_per_1k_s``, ``imputed_pct``, ``n_tasks`` and the model's
    ``display_name``, ``category``, ``release_date``, ``contamination`` (why its scores may be
    optimistic, empty for most models).
    """
    from bencheval.evaluator import BenchmarkEvaluator

    if name != "all":
        results = results[results["task"] == name]
    else:
        # A model that never ran one task type belongs on that type's leaderboard
        # only; imputing a whole task type would rank it on the reference's results.
        task_types = results.groupby("model")["task"].nunique()
        single = sorted(task_types.index[task_types < results["task"].nunique()])
        if single:
            logger.info("Left out of the all-tasks ranking (one task type only): %s", single)
            results = results[~results["model"].isin(single)]
    filled = impute_missing(results, reference_model=reference_model, max_imputed_pct=max_imputed_pct)
    data = filled[["dataset", "fold", "model", "metric_error"]]

    evaluator = BenchmarkEvaluator(
        method_col="model", task_col="dataset", seed_column="fold", columns_to_agg_extra=[]
    )
    lb = evaluator.leaderboard(
        data,
        elo_kwargs={
            "calibration_framework": reference_model,
            "calibration_elo": 1000,
            "BOOTSTRAP_ROUNDS": bootstrap_rounds,
        },
        sort_by=None,
    )
    per_task = evaluator.compute_results_per_task(data)
    winrate_matrix = evaluator.compute_winrate_matrix(per_task)

    real = filled[~filled["imputed"]].copy()
    real["time_total_s"] = real["time_train_s"] + real["time_infer_s"]
    if "num_instances" in real.columns:
        # Test rows per outer fold: num_instances / n_splits (the scope's outer CV folds).
        real["infer_per_1k_s"] = real["time_infer_s"] / (real["num_instances"] / n_splits) * 1000.0
        # Train + predict on one fold covers every spectrum of the task once, so
        # this is the time per 1K spectra and comparable across dataset sizes.
        real["time_total_per_1k_s"] = real["time_total_s"] / real["num_instances"] * 1000.0
    times = real.groupby("model")[
        [c for c in ("time_train_s", "time_infer_s", "time_total_s", "infer_per_1k_s", "time_total_per_1k_s")
         if c in real]
    ].median()
    lb = lb.join(times.add_prefix("median_"))
    lb["normalized_score"] = normalized_score(filled)
    lb["imputed_pct"] = filled.groupby("model")["imputed"].mean() * 100.0
    lb["n_tasks"] = filled.groupby("model")["dataset"].nunique()
    lb["is_reference"] = lb.index.isin(model_info.REFERENCE_MODELS)
    lb["display_name"] = [model_info.display_name(m) for m in lb.index]
    lb["contamination"] = [model_info.contamination(m) or "" for m in lb.index]
    lb["category"] = [model_info.category(m) for m in lb.index]
    lb["release_date"] = [model_info.release_date(m) for m in lb.index]
    lb = lb.sort_values("elo", ascending=False)

    return GroupScores(
        name=name,
        leaderboard=lb,
        winrate_matrix=winrate_matrix.loc[lb.index, lb.index],
        results=filled,
        n_tasks=filled["dataset"].nunique(),
    )


def score_all(results: pd.DataFrame, **kwargs) -> dict[str, GroupScores]:
    """:func:`score_group` for every task group present in *results*."""
    groups = ["all"] + [g for g in ("classification", "regression") if (results["task"] == g).any()]
    return {g: score_group(results, g, **kwargs) for g in groups}


def split_references(leaderboard: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(ranked models, reference systems)`` of a leaderboard."""
    ref = leaderboard["is_reference"] if "is_reference" in leaderboard else pd.Series(False, leaderboard.index)
    return leaderboard[~ref], leaderboard[ref]


def select_focus(leaderboard: pd.DataFrame, top_k: int | None, always: tuple[str, ...] = ()) -> set[str]:
    """Models drawn in colour: the *top_k* best by Elo in each category.

    ``top_k=None`` (or ``0``) highlights every model. Models in *always* are
    highlighted whenever they are present.
    """
    leaderboard, _ = split_references(leaderboard)
    if not top_k:
        return set(leaderboard.index)
    ranked = leaderboard.sort_values("elo", ascending=False)
    focus = set(ranked.groupby("category", sort=False).head(top_k).index)
    return focus | (set(always) & set(leaderboard.index))


#: A model this close to the Pareto front (in normalized score) also counts as a
#: trade-off answer. The paper used 0.05; 0.1 keeps strong but slower models such
#: as RamanPFN (0.053 behind the regression front) in the win-rate matrix.
NEAR_PARETO_TOLERANCE = 0.1
TIME_COL = "median_time_total_per_1k_s"


def pareto_front(df: pd.DataFrame, x: str, y: str, higher_is_better: bool) -> pd.DataFrame:
    """Rows on the Pareto front: lower *x* and better *y* than every row before them."""
    best = -np.inf if higher_is_better else np.inf
    keep = []
    for idx, row in df.sort_values(x).iterrows():
        val = row[y]
        if (val > best) if higher_is_better else (val < best):
            keep.append(idx)
            best = val
    return df.loc[keep]


def near_pareto_models(
    lb: pd.DataFrame,
    metric: str = "normalized_score",
    higher_is_better: bool = True,
    tolerance: float = NEAR_PARETO_TOLERANCE,
) -> set[str]:
    """Pareto-optimal models of *metric* vs. median train+predict time per 1K spectra, plus every model
    within *tolerance* of the best score the front reaches at the same or lower cost.

    Models cheaper than the front's cheapest point are compared against that point.
    """
    df = split_references(lb)[0].dropna(subset=[TIME_COL, metric])
    front = pareto_front(df, TIME_COL, metric, higher_is_better)
    if front.empty:
        return set()
    times = front[TIME_COL].to_numpy()
    scores = front[metric].to_numpy()  # monotonic in time by construction
    near = set(front.index)
    for model, row in df.iterrows():
        idx = int(np.searchsorted(times, row[TIME_COL], side="right")) - 1
        ref = scores[max(idx, 0)]
        gap = (ref - row[metric]) if higher_is_better else (row[metric] - ref)
        if gap <= tolerance:
            near.add(model)
    return near


def pareto_selection(scores: dict[str, GroupScores], tolerance: float = NEAR_PARETO_TOLERANCE) -> set[str]:
    """Pareto- and near-Pareto-optimal models (normalized score vs. time) in any task type."""
    groups = [g for g in ("classification", "regression") if g in scores] or ["all"]
    selected = set()
    for g in groups:
        selected |= near_pareto_models(scores[g].leaderboard, tolerance=tolerance)
    return selected
