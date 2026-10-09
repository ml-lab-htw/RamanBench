"""Compare a new model against the RamanBench v1 leaderboard.

The package ships the per-fold results of every leaderboard model
(``data/precomputed/v1/reference_results.parquet``) and the protocol they were produced
with (``protocol.json``). Run your model on the same tasks and folds, either with
:func:`raman_bench.evaluate.evaluate_estimator` (any scikit-learn estimator) or with
``scripts/run_experiment.py`` (a model registered in RamanBench), then::

    from raman_bench.compare import compare

    scores = compare("results/my_model")          # directory with the cached results.pkl files
    scores["all"].leaderboard.head(10)            # Elo, win rate, rank, times, ...

Scoring is the leaderboard's own (:func:`raman_bench.plotting.results.score_all`, i.e.
TabArena's ``BenchmarkEvaluator``): Elo anchored at Random Forest = 1000, win rate,
improvability. A task your model has no result for gets Random Forest's result and counts
as imputed; a model more than ``max_imputed_pct`` percent imputed is left out. To compare on
the tasks you ran only, pass ``tasks="own"`` (the scores are then not comparable with the
published leaderboard).

Ranking by another metric: ``classification_metric="accuracy"``, ``regression_metric="mae"``
and so on (:func:`raman_bench.fold_metrics.metric_names`). They are computed per fold from
the stored test predictions; the default is the protocol's own metric (ROC AUC for binary,
log loss for multiclass, RMSE for regression).

Needs the ``plots`` extra (``pip install "raman-bench[plots]"``) for scoring and figures.
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
from functools import lru_cache
from importlib import resources
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

REFERENCE_FILE = "reference_results.parquet"
PROTOCOL_FILE = "protocol.json"
TIDY_COLUMNS = ["dataset", "fold", "model", "metric_error", "metric", "task", "time_train_s", "time_infer_s"]


def _reference_dir():
    return resources.files("raman_bench").joinpath("data", "precomputed", "v1")


@lru_cache(maxsize=1)
def load_protocol() -> dict:
    """The v1 evaluation protocol (folds, bagging, time budget, row caps, task list)."""
    return json.loads(_reference_dir().joinpath(PROTOCOL_FILE).read_text())


def protocol_tasks(task_type: str | None = None, tasks: list[str] | None = None) -> list[dict]:
    """The tasks of the protocol, optionally only one *task_type* or the named *tasks*.

    *task_type* is ``"classification"`` or ``"regression"``; *tasks* are task keys
    (``"<dataset>__<target_idx>"``) or bare dataset names (all targets of that dataset).
    """
    selected = load_protocol()["tasks"]
    if task_type is not None:
        if task_type not in ("classification", "regression"):
            raise ValueError(f"task_type must be 'classification' or 'regression', got {task_type!r}")
        selected = [t for t in selected if t["problem_type"] == task_type]
    if tasks is not None:
        wanted = set(tasks)
        unknown = wanted - {t["task"] for t in selected} - {t["dataset"] for t in selected}
        if unknown:
            raise ValueError(f"Not in the protocol (or not of the requested task type): {sorted(unknown)}")
        selected = [t for t in selected if t["task"] in wanted or t["dataset"] in wanted]
    return selected


def load_reference(
    *,
    models: list[str] | None = None,
    exclude_models: tuple[str, ...] = ("DUMMY",),
) -> pd.DataFrame:
    """Per-fold results of the leaderboard models, one row per (task, fold, model).

    Columns: ``dataset`` (the task key ``<dataset>__<target_idx>``), ``fold``, ``model``,
    ``metric_error`` (lower is better: 1 - ROC AUC for binary, log loss for multiclass,
    RMSE for regression), ``metric``, ``task`` (``classification``/``regression``),
    ``time_train_s``, ``time_infer_s``, ``reference`` (AutoGluon reference systems, drawn
    as lines and not ranked), ``num_instances``, ``ta_name`` (the result directory) and one
    column per metric of :data:`raman_bench.fold_metrics.METRICS` (NaN where it doesn't
    apply, e.g. ``accuracy`` on a regression task).
    """
    with resources.as_file(_reference_dir().joinpath(REFERENCE_FILE)) as path:
        df = pd.read_parquet(path)
    if models is not None:
        unknown = set(models) - set(df["model"])
        if unknown:
            raise ValueError(f"No reference results for {sorted(unknown)}")
        df = df[df["model"].isin(models)]
    return df[~df["model"].isin(exclude_models)].reset_index(drop=True)


def load_own_results(source, *, variant: str = "default") -> pd.DataFrame:
    """Your results as tidy rows, restricted to the protocol's tasks and folds.

    *source* is one of

    - a results directory with cached ``results.pkl`` files (what
      :func:`raman_bench.evaluate.evaluate_estimator` and ``scripts/run_experiment.py``
      write), aggregated here with :func:`raman_bench.aggregation.aggregate`;
    - an ``hpo_results.csv`` written by ``scripts/aggregate_results.py``, or that table
      as a DataFrame;
    - a DataFrame already in tidy form (columns :data:`TIDY_COLUMNS`), e.g. from your own
      evaluation loop. It must use the protocol's splits to be comparable.

    A ``preprocessing`` column carries the RamanBench preprocessing steps each model was
    fit with (read from the cached results; missing for CSV input unless your tidy
    DataFrame has the column). From a results directory, every metric of
    :data:`raman_bench.fold_metrics.METRICS` is added per fold; a tidy DataFrame may carry
    those columns itself.
    """
    from raman_bench.plotting.results import load_results

    preprocessing: dict[str, str] = {}
    per_fold = None
    if isinstance(source, pd.DataFrame) and "model" in source.columns:
        missing = set(TIDY_COLUMNS) - set(source.columns)
        if missing:
            raise ValueError(f"Tidy results are missing columns {sorted(missing)}")
        df = source.copy()
        if "reference" not in df.columns:
            df["reference"] = False
    else:
        if not isinstance(source, pd.DataFrame):
            path = Path(source)
            if path.is_dir():
                from raman_bench.aggregation import aggregate, preprocessing_by_model
                from raman_bench.fold_metrics import fold_metrics

                with contextlib.redirect_stdout(io.StringIO()):  # TabArena prints progress
                    _, source = aggregate(str(path))
                if source.empty:
                    raise ValueError(f"No cached results.pkl files under {path}")
                preprocessing = preprocessing_by_model(str(path))
                per_fold = fold_metrics(path)
            else:
                source = path
        df = load_results(source, variant=variant, scope=None, target_list=None, exclude_models=())
        df["preprocessing"] = df["model"].map(preprocessing) if preprocessing else None
        if per_fold is not None:
            df = attach_fold_metrics(df, per_fold)

    tasks = {t["task"]: t for t in load_protocol()["tasks"]}
    outside = sorted(set(df["dataset"]) - set(tasks))
    if outside:
        logger.warning("Ignoring %d task(s) that are not in the protocol: %s", len(outside), outside)
    df = df[df["dataset"].isin(tasks)]
    n_folds = df["dataset"].map(lambda k: tasks[k]["n_folds"])
    df = df[df["fold"] < n_folds]
    df["num_instances"] = df["dataset"].map(lambda k: tasks[k]["num_instances"])
    return df.reset_index(drop=True)


def attach_fold_metrics(results: pd.DataFrame, per_fold: pd.DataFrame) -> pd.DataFrame:
    """*results* with the metric columns of *per_fold* (:func:`raman_bench.fold_metrics.fold_metrics`)
    joined on result directory, task and fold."""
    keys = ["ta_name", "dataset", "fold"]
    per_fold = per_fold.drop_duplicates(keys, keep="last")
    results = results.drop(columns=[c for c in per_fold.columns if c not in keys and c in results.columns])
    return results.merge(per_fold, on=keys, how="left")


def select_metric(
    results: pd.DataFrame,
    *,
    classification_metric: str | None = None,
    regression_metric: str | None = None,
) -> pd.DataFrame:
    """*results* with ``metric_error``/``metric`` taken from another metric column.

    ``None`` keeps the protocol's metric for that task type. Folds where the chosen metric
    is undefined (NaN, e.g. ROC AUC with a class missing from the test fold) are dropped.
    """
    from raman_bench.fold_metrics import METRICS, metric_names, to_error

    results = results.copy()
    for task, metric in (("classification", classification_metric), ("regression", regression_metric)):
        if metric is None:
            continue
        if metric not in METRICS or METRICS[metric].task != task:
            raise ValueError(f"{task}_metric must be one of {metric_names(task)}, got {metric!r}")
        if metric not in results.columns:
            raise ValueError(
                f"No {metric!r} column in these results: per-fold metrics come from results.pkl "
                "files (raman_bench.fold_metrics), not from an hpo_results.csv"
            )
        rows = results["task"] == task
        values = results.loc[rows, metric]
        missing = results.loc[rows & results[metric].isna(), "model"].value_counts()
        if not missing.empty:
            logger.info("%s: %d fold(s) without a value are left out: %s", metric, int(missing.sum()), missing.to_dict())
        results.loc[rows, "metric_error"] = to_error(values, metric)
        results.loc[rows, "metric"] = metric
    return results.dropna(subset=["metric_error"]).reset_index(drop=True)


def coverage(own: pd.DataFrame) -> pd.DataFrame:
    """Per model and task type: how many protocol tasks and folds *own* has results for."""
    rows = []
    for task_type in ("classification", "regression"):
        expected = {t["task"]: t["n_folds"] for t in protocol_tasks(task_type)}
        for model, g in own[own["task"] == task_type].groupby("model"):
            folds = g.groupby("dataset")["fold"].nunique()
            rows.append(
                {
                    "model": model,
                    "task": task_type,
                    "tasks_run": len(folds),
                    "tasks_total": len(expected),
                    "tasks_complete": int(sum(folds.get(k, 0) >= n for k, n in expected.items())),
                    "missing_tasks": sorted(set(expected) - set(folds.index)),
                }
            )
    return pd.DataFrame(rows)


def _score(results, own_preprocessing, reference_model, max_imputed_pct, bootstrap_rounds):
    """:func:`score_all` plus the ``preprocessing`` column on every leaderboard."""
    from raman_bench.plotting.results import score_all

    scores = score_all(
        results,
        reference_model=reference_model,
        max_imputed_pct=max_imputed_pct,
        bootstrap_rounds=bootstrap_rounds,
    )
    preprocessing = {**load_protocol().get("preprocessing", {}), **own_preprocessing}
    for s in scores.values():
        s.leaderboard["preprocessing"] = s.leaderboard.index.map(lambda m: preprocessing.get(m, "unknown"))
    return scores


def leaderboard(
    *,
    classification_metric: str | None = None,
    regression_metric: str | None = None,
    reference_model: str = "RF",
    bootstrap_rounds: int = 200,
):
    """The v1 leaderboard, scored from the bundled reference results.

    Same scores as the published leaderboard (up to bootstrap noise in the Elo
    confidence intervals), or ranked by another metric (see :func:`select_metric`).
    Returns ``{"all" | "classification" | "regression": GroupScores}``.
    """
    results = select_metric(
        load_reference(), classification_metric=classification_metric, regression_metric=regression_metric
    )
    return _score(results, {}, reference_model, 50.0, bootstrap_rounds)


def compare(
    own,
    *,
    variant: str = "default",
    tasks: str = "all",
    reference_models: list[str] | None = None,
    replace_reference: bool = False,
    out_dir: str | Path | None = None,
    figures: bool = True,
    classification_metric: str | None = None,
    regression_metric: str | None = None,
    reference_model: str = "RF",
    max_imputed_pct: float = 50.0,
    bootstrap_rounds: int = 200,
    **figure_kwargs,
):
    """Score your model(s) together with the leaderboard models.

    Parameters
    ----------
    own
        Your results; anything :func:`load_own_results` accepts.
    tasks
        ``"all"`` scores on every protocol task, like the published leaderboard (tasks you
        did not run are imputed, see the module docstring). ``"own"`` scores every model on
        only the tasks you ran.
    reference_models
        Leaderboard models to compare against (default: all of them).
    replace_reference
        Your results replace a leaderboard model's results when the model key is the same
        (e.g. a re-run of ``RF``). Without it, a clash raises ``ValueError``.
    out_dir
        Writes ``leaderboards/leaderboard_{all,classification,regression}.csv`` there and,
        with *figures*, the leaderboard figures (:mod:`raman_bench.plotting`);
        *figure_kwargs* go to :func:`raman_bench.plotting.generate_from_results`.
    classification_metric, regression_metric
        Rank by another metric (:func:`raman_bench.fold_metrics.metric_names`), e.g.
        ``"accuracy"`` or ``"mae"``; ``None`` is the protocol's metric. Needs your results
        as a results directory, or as a tidy DataFrame with that metric's column.

    Returns ``{"all" | "classification" | "regression": GroupScores}``; each
    ``.leaderboard`` has one row per model, sorted by Elo, with a ``preprocessing`` column:
    the RamanBench preprocessing steps the model was fit with (``"none"``: raw spectra;
    steps inside your own scikit-learn ``Pipeline`` are not visible here).
    """
    if tasks not in ("all", "own"):
        raise ValueError(f"tasks must be 'all' or 'own', got {tasks!r}")
    own_df = load_own_results(own, variant=variant)
    if own_df.empty:
        raise ValueError("None of your results are on a protocol task")
    reference = load_reference(models=reference_models)

    clash = sorted(set(own_df["model"]) & set(reference["model"]))
    if clash and not replace_reference:
        raise ValueError(
            f"{clash} already on the leaderboard; give your model another name or pass replace_reference=True"
        )
    reference = reference[~reference["model"].isin(clash)]

    for row in coverage(own_df).itertuples():
        if row.tasks_run < row.tasks_total:
            logger.warning(
                "%s ran %d of %d %s tasks; the other %d are %s",
                row.model, row.tasks_run, row.tasks_total, row.task, row.tasks_total - row.tasks_run,
                "left out for every model" if tasks == "own" else "imputed with Random Forest's result",
            )
    if tasks == "own":
        reference = reference[reference["dataset"].isin(set(own_df["dataset"]))]

    results = pd.concat([reference, own_df.reindex(columns=reference.columns)], ignore_index=True)
    results = select_metric(
        results, classification_metric=classification_metric, regression_metric=regression_metric
    )
    own_prep = {}
    if "preprocessing" in own_df.columns:
        own_prep = own_df.dropna(subset=["preprocessing"]).groupby("model")["preprocessing"].first().to_dict()
    scores = _score(results, own_prep, reference_model, max_imputed_pct, bootstrap_rounds)
    for g, s in scores.items():
        ran = own_df if g == "all" else own_df[own_df["task"] == g]
        dropped = sorted(set(ran["model"]) - set(s.leaderboard.index))
        if dropped:
            logger.warning(
                "Not ranked on %s: %s (more than %.0f%% imputed or no results there)", g, dropped, max_imputed_pct
            )

    if out_dir is not None:
        if figures:
            from raman_bench.plotting.pipeline import generate_from_results

            generate_from_results(
                results,
                out_dir,
                variant=variant,
                scores=scores,
                reference_model=reference_model,
                **figure_kwargs,
            )
        else:
            from raman_bench.plotting.pipeline import write_leaderboards

            write_leaderboards(scores, Path(out_dir) / "leaderboards")
    return scores
