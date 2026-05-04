"""Leaderboard utilities for ranking new models against precomputed baselines.

The :class:`Leaderboard` class provides the primary interface for researchers
who want to evaluate a new model against the 28 baseline models in the
RamanBench v0.1 results — **without re-running all baselines**.

Typical workflow
----------------
::

    from raman_bench import Leaderboard
    from sklearn.cross_decomposition import PLSRegression

    # 1. Load precomputed v0.1 results
    lb = Leaderboard.from_precomputed()

    # 2. Print current ranking
    print(lb.rank())

    # 3. Evaluate your model and add it to the leaderboard
    results = lb.evaluate_and_add("My-PLS", PLSRegression(n_components=10))
    print(lb.rank())

    # 4. Visualise
    lb.plot()

See also
--------
- raman-data: https://github.com/ml-lab-htw/raman_data
- Source:     https://github.com/ml-lab-htw/RamanBench
- Leaderboard: https://huggingface.co/spaces/HTW-KI-Werkstatt/RamanBench
- Paper:       https://arxiv.org/abs/TBD
"""

from __future__ import annotations

import importlib.resources
import logging
import os
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_RANK_COL = "Rank"

# Columns from display metadata kept verbatim (not recomputed from raw metrics)
_META_COLS = ["Model", "Category", "Elo", "Train Time s", "Infer. s/1K"]


def _load_bundled_csv(filename: str) -> pd.DataFrame:
    """Load a CSV bundled in ``data/precomputed/`` inside this package."""
    try:
        ref = importlib.resources.files("raman_bench") / "data" / "precomputed" / filename
        with importlib.resources.as_file(ref) as path:
            return pd.read_csv(path)
    except Exception:
        pkg_root = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(pkg_root, "data", "precomputed", filename)
        return pd.read_csv(csv_path)


class Leaderboard:
    """Manage and extend the RamanBench leaderboard.

    Parameters
    ----------
    reg_metrics : pd.DataFrame
        Raw per-(seed, key, model) regression metrics.  Must contain columns
        ``seed``, ``key``, ``model``, ``rmse`` (and optionally ``mse``,
        ``mae``, ``r2``, …).
    clf_metrics : pd.DataFrame
        Raw per-(seed, key, model) classification metrics.  Must contain
        columns ``seed``, ``key``, ``model``, ``f1_score`` (and optionally
        ``accuracy``, ``precision``, ``recall``, …).
    display_meta : pd.DataFrame
        Per-model display metadata indexed by ``model_id``.  Columns:
        ``model_id``, ``Model``, ``Category``, ``Elo``,
        ``Train Time s``, ``Infer. s/1K``.  Missing models (e.g. newly
        added) are filled with sensible defaults.

    Notes
    -----
    Use the class methods :meth:`from_precomputed` or :meth:`from_results_dir`
    to construct instances — do not call ``__init__`` directly.
    """

    def __init__(
        self,
        reg_metrics: pd.DataFrame,
        clf_metrics: pd.DataFrame,
        display_meta: pd.DataFrame | None = None,
    ):
        self._reg_metrics = reg_metrics.copy()
        self._clf_metrics = clf_metrics.copy()
        self._display_meta = display_meta.copy() if display_meta is not None else pd.DataFrame()
        self._added_models: list[str] = []

        # Populated by _rebuild()
        self._overall: pd.DataFrame = pd.DataFrame()
        self._clf: pd.DataFrame = pd.DataFrame()
        self._reg: pd.DataFrame = pd.DataFrame()
        self._rebuild()

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_precomputed(cls) -> Leaderboard:
        """Load the bundled v0.1 precomputed results.

        Returns the leaderboard as published alongside the paper, containing
        28 baseline models evaluated on 74 datasets (163 targets).

        Returns
        -------
        Leaderboard
        """
        reg_metrics = _load_bundled_csv("regression_metrics.csv")
        clf_metrics = _load_bundled_csv("classification_metrics.csv")

        # Load display metadata (Model name, Category, Elo, timing) from the
        # pre-built overall leaderboard CSV — these columns are not derivable
        # from raw metrics alone.
        meta_df: pd.DataFrame = pd.DataFrame()
        try:
            raw_lb = _load_bundled_csv("leaderboard_overall.csv")
            meta_cols = ["model_id"] + [c for c in _META_COLS if c in raw_lb.columns]
            meta_df = raw_lb[meta_cols].copy()
        except Exception:
            pass

        logger.info(
            "Loaded precomputed leaderboard: %d regression rows, %d classification rows",
            len(reg_metrics),
            len(clf_metrics),
        )
        return cls(reg_metrics, clf_metrics, meta_df)

    @classmethod
    def from_results_dir(cls, results_dir: str) -> Leaderboard:
        """Load leaderboard from a local results directory.

        Expects ``metrics/classification_metrics.csv`` and
        ``metrics/regression_metrics.csv`` inside *results_dir*.

        Parameters
        ----------
        results_dir : str
            Path produced by running the benchmark pipeline
            (``scripts/run_benchmark.py``).

        Returns
        -------
        Leaderboard
        """
        metrics_dir = os.path.join(results_dir, "metrics")
        clf_path = os.path.join(metrics_dir, "classification_metrics.csv")
        reg_path = os.path.join(metrics_dir, "regression_metrics.csv")

        reg_df = pd.read_csv(reg_path) if os.path.exists(reg_path) else pd.DataFrame()
        clf_df = pd.read_csv(clf_path) if os.path.exists(clf_path) else pd.DataFrame()

        return cls(reg_df, clf_df)

    # ------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------

    def rank(self, task: str = "overall") -> pd.DataFrame:
        """Return a ranked leaderboard DataFrame.

        Parameters
        ----------
        task : {"overall", "classification", "regression"}
            Which leaderboard to return.

        Returns
        -------
        pd.DataFrame
            Sorted by Score (descending) with a ``Rank`` column prepended.
        """
        df = self._select_leaderboard(task).copy()
        if "Score" in df.columns:
            df = df.sort_values("Score", ascending=False).reset_index(drop=True)
        df[_RANK_COL] = df.index + 1
        cols = [_RANK_COL] + [c for c in df.columns if c != _RANK_COL]
        return df[cols]

    def summary(self) -> str:
        """Return a human-readable summary of the current leaderboard."""
        df = self.rank()
        lines = ["RamanBench Leaderboard (v0.1)", "=" * 40]
        for _, row in df.iterrows():
            model = row.get("Model", row.get("model_id", "?"))
            score = row.get("Score", float("nan"))
            elo = row.get("Elo", float("nan"))
            elo_str = f"  Elo={elo:.0f}" if not (isinstance(elo, float) and np.isnan(elo)) else ""
            lines.append(f"  #{int(row[_RANK_COL]):2d}  {model:<28}  Score={score:.3f}{elo_str}")
        if self._added_models:
            lines.append(f"\nAdded models: {', '.join(self._added_models)}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Adding new models
    # ------------------------------------------------------------------

    def add_results(
        self,
        model_name: str,
        metrics_df: pd.DataFrame,
    ) -> None:
        """Add raw per-(seed, key) metrics for a new model to the leaderboard.

        The metrics are stored in the same format as the paper's CSV files and
        the leaderboard scores are recomputed from all models combined
        (including the new one), so the normalization is updated automatically.

        Parameters
        ----------
        model_name : str
            Display name and internal identifier for the new model.
        metrics_df : pd.DataFrame
            DataFrame with one row per (seed, dataset_key) containing metric
            columns.  Must include ``seed`` and ``key``.  Regression rows need
            ``rmse``; classification rows need ``f1_score``.  A ``model``
            column is added automatically.
        """
        df = metrics_df.copy()
        df["model"] = model_name

        reg_cols = {"rmse", "mse", "mae", "r2"}
        clf_cols = {"f1_score", "accuracy", "precision", "recall"}

        if reg_cols & set(df.columns):
            keep = ["seed", "key", "model"] + [c for c in df.columns if c in reg_cols]
            self._reg_metrics = pd.concat(
                [self._reg_metrics, df[keep].dropna(subset=["rmse"])],
                ignore_index=True,
            )

        if clf_cols & set(df.columns):
            keep = ["seed", "key", "model"] + [c for c in df.columns if c in clf_cols]
            self._clf_metrics = pd.concat(
                [self._clf_metrics, df[keep].dropna(subset=["f1_score"])],
                ignore_index=True,
            )

        if model_name not in self._added_models:
            self._added_models.append(model_name)

        self._rebuild()
        logger.info("Added model '%s' to leaderboard.", model_name)

    def evaluate_and_add(
        self,
        model_name: str,
        model: Any,
        config_path: str | None = None,
        seeds: int = 3,
        task: str = "overall",
    ) -> pd.DataFrame:
        """Run a model through the full benchmark and add it to the leaderboard.

        This is a convenience wrapper that:

        1. Loads all benchmark datasets via :class:`~raman_bench.benchmark.RamanBenchmark`.
        2. Fits and evaluates *model* on each train/test split.
        3. Computes metrics.
        4. Calls :meth:`add_results` to insert the model into the leaderboard.

        *model* must expose a scikit-learn-compatible API:
        ``fit(X, y)`` and ``predict(X)``.

        Parameters
        ----------
        model_name : str
            Display name for the leaderboard.
        model : object
            A scikit-learn-compatible estimator.
        config_path : str | None
            Path to a benchmark config JSON.  Defaults to the bundled
            ``configs/benchmark_v0.1.json``.
        seeds : int
            Number of random seeds to average over (default 3).
        task : str
            Filter to only regression or classification datasets when set to
            ``"regression"`` or ``"classification"``; default ``"overall"``
            runs both.

        Returns
        -------
        pd.DataFrame
            Per-(seed, key) metrics for the newly evaluated model, in the same
            format as the paper's raw metrics CSVs.
        """
        import time

        from raman_data import TASK_TYPE

        from raman_bench.benchmark import configure_benchmark
        from raman_bench.config import load_config
        from raman_bench.metrics import compute_metrics

        if config_path is None:
            pkg_root = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(pkg_root, "..", "..", "configs", "benchmark_v0.1.json")

        config = load_config(config_path)
        records = []
        fit_times: list[float] = []
        infer_us_per_sample: list[float] = []

        for seed in range(seeds):
            config["random_state"] = seed
            bench = configure_benchmark(config)
            for train_df, test_df, key, task_type in bench:
                if train_df is None:
                    continue
                if task == "regression" and task_type != TASK_TYPE.Regression:
                    continue
                if task == "classification" and task_type != TASK_TYPE.Classification:
                    continue
                label_col = train_df.columns[-1]
                X_train = train_df.drop(columns=[label_col]).values
                y_train = train_df[label_col].values
                X_test = test_df.drop(columns=[label_col]).values
                y_test = test_df[label_col].values

                try:
                    t0 = time.perf_counter()
                    model.fit(X_train, y_train)
                    fit_times.append(time.perf_counter() - t0)

                    t1 = time.perf_counter()
                    y_pred = model.predict(X_test)
                    infer_s = time.perf_counter() - t1
                    infer_us_per_sample.append((infer_s / len(X_test)) * 1e6)

                    metrics = compute_metrics(y_test, y_pred, task_type=task_type)
                    records.append(
                        {
                            "seed": seed,
                            "key": key,
                            "task_type": task_type.name,
                            **metrics,
                        }
                    )
                except Exception as e:
                    logger.warning("Model %s failed on %s (seed %d): %s", model_name, key, seed, e)

        metrics_df = pd.DataFrame(records)
        if not metrics_df.empty:
            # µs/sample == ms/1K — same unit as the precomputed "Infer. s/1K" column
            mean_train_s = float(np.mean(fit_times)) if fit_times else float("nan")
            mean_infer_ms_per_1k = (
                float(np.mean(infer_us_per_sample)) if infer_us_per_sample else float("nan")
            )
            self._upsert_display_meta(
                model_name,
                {
                    "Train Time s": mean_train_s,
                    "Infer. s/1K": mean_infer_ms_per_1k,
                },
            )
            self.add_results(model_name, metrics_df)
        return metrics_df

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot(
        self,
        task: str = "overall",
        n_top: int = 30,
        figsize: tuple = (10, 8),
    ):
        """Plot a horizontal bar chart of model scores.

        Parameters
        ----------
        task : {"overall", "classification", "regression"}
            Which leaderboard to visualise.
        n_top : int
            Show only the top *n_top* models.
        figsize : tuple
            Matplotlib figure size.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        df = self.rank(task).head(n_top)
        model_col = "Model" if "Model" in df.columns else "model_id"
        models = df[model_col].tolist()
        scores = df["Score"].tolist() if "Score" in df.columns else [0] * len(df)
        colors = ["#e74c3c" if m in self._added_models else "#3498db" for m in models]

        fig, ax = plt.subplots(figsize=figsize)
        ax.barh(range(len(models)), scores[::-1], color=colors[::-1])
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models[::-1])
        ax.set_xlabel("Score")
        ax.set_title(f"RamanBench — {task.title()} Leaderboard")
        ax.axvline(0, color="black", linewidth=0.5)

        if self._added_models:
            from matplotlib.patches import Patch

            legend = [
                Patch(color="#3498db", label="Baseline (v0.1)"),
                Patch(color="#e74c3c", label="New model"),
            ]
            ax.legend(handles=legend, loc="lower right")

        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Internal: rebuild leaderboard from raw metrics
    # ------------------------------------------------------------------

    def _upsert_display_meta(self, model_id: str, values: dict) -> None:
        """Insert or update display metadata columns for *model_id*."""
        if self._display_meta.empty or "model_id" not in self._display_meta.columns:
            self._display_meta = pd.DataFrame([{"model_id": model_id, **values}])
            return
        mask = self._display_meta["model_id"] == model_id
        if mask.any():
            for col, val in values.items():
                self._display_meta.loc[mask, col] = val
        else:
            new_row = pd.DataFrame([{"model_id": model_id, **values}])
            self._display_meta = pd.concat([self._display_meta, new_row], ignore_index=True)

    def _rebuild(self) -> None:
        """Recompute Score, Avg Rank, Improvability from current raw metrics."""
        reg_scores = _per_dataset_scores(self._reg_metrics, "rmse", higher_is_better=False)
        clf_scores = _per_dataset_scores(self._clf_metrics, "f1_score", higher_is_better=True)

        self._reg = self._merge_meta(_aggregate_leaderboard(reg_scores))
        self._clf = self._merge_meta(_aggregate_leaderboard(clf_scores))

        non_empty = [df for df in [reg_scores, clf_scores] if not df.empty]
        all_scores = pd.concat(non_empty, ignore_index=True) if non_empty else reg_scores
        self._overall = self._merge_meta(_aggregate_leaderboard(all_scores))

    def _merge_meta(self, lb: pd.DataFrame) -> pd.DataFrame:
        """Merge display metadata into a computed leaderboard DataFrame."""
        if lb.empty:
            return lb
        if self._display_meta.empty or "model_id" not in self._display_meta.columns:
            lb["Model"] = lb["model_id"]
            return lb
        available = [c for c in _META_COLS if c in self._display_meta.columns]
        meta = self._display_meta[["model_id"] + available]
        merged = lb.merge(meta, on="model_id", how="left")
        if "Model" in merged.columns:
            merged["Model"] = merged["Model"].fillna(merged["model_id"])
        else:
            merged["Model"] = merged["model_id"]
        col_order = ["model_id", "Model"] + [
            c
            for c in [
                "Category",
                "Elo",
                "Score",
                "Avg Rank",
                "Improvability",
                "Train Time s",
                "Infer. s/1K",
            ]
            if c in merged.columns
        ]
        extra = [c for c in merged.columns if c not in col_order]
        return merged[col_order + extra]

    def _select_leaderboard(self, task: str) -> pd.DataFrame:
        if task == "overall":
            return self._overall
        if task == "classification":
            return self._clf
        if task == "regression":
            return self._reg
        raise ValueError(
            f"Unknown task {task!r}. Use 'overall', 'classification', or 'regression'."
        )


# ------------------------------------------------------------------
# Helpers for building leaderboard from raw metrics
# ------------------------------------------------------------------


def _per_dataset_scores(
    metrics_df: pd.DataFrame,
    metric_col: str,
    higher_is_better: bool,
) -> pd.DataFrame:
    """Compute per-(model, dataset) normalized scores and ranks.

    For each dataset key, models are ranked and a normalized score is computed:
    - best model in that dataset  → 1.0
    - median model in that dataset → 0.0
    - scores clipped to [0, 1]

    Returns a DataFrame with columns ``model``, ``key``, ``norm_score``,
    ``rank``.  Only datasets with at least two models are included.
    """
    if metrics_df.empty or metric_col not in metrics_df.columns:
        return pd.DataFrame(columns=["model", "key", "norm_score", "rank"])

    # Average the metric across seeds for each (model, dataset)
    per_ds = (
        metrics_df.groupby(["model", "key"])[metric_col]
        .mean()
        .reset_index()
        .dropna(subset=[metric_col])
    )

    results = []
    for key, group in per_ds.groupby("key"):
        if len(group) < 2:
            continue
        vals = group[metric_col].values
        e_vals = vals if higher_is_better else -vals

        median_e = float(np.median(e_vals))
        best_e = float(np.max(e_vals))
        denom = best_e - median_e

        if denom > 0:
            norm = np.clip((e_vals - median_e) / denom, 0.0, 1.0)
        else:
            norm = np.zeros(len(e_vals))

        ranks = pd.Series(e_vals).rank(ascending=False, method="min").values

        for i, row in enumerate(group.itertuples(index=False)):
            results.append(
                {
                    "model": row.model,
                    "key": key,
                    "norm_score": float(norm[i]),
                    "rank": float(ranks[i]),
                }
            )

    return pd.DataFrame(results)


def _aggregate_leaderboard(per_ds: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-dataset scores into a leaderboard row per model.

    Models that are missing from some datasets receive norm_score=0 and
    rank=(n_models_on_that_key + 1) for each absent (model, key) pair.
    This prevents partial submissions from inflating overall rank.
    """
    if per_ds.empty:
        return pd.DataFrame(columns=["model_id", "Score", "Avg Rank", "Improvability"])

    all_models = per_ds["model"].unique()
    all_keys = per_ds["key"].unique()
    worst_rank_per_key = per_ds.groupby("key")["rank"].max().add(1).to_dict()

    have = set(zip(per_ds["model"], per_ds["key"]))
    penalty_rows = [
        {"model": m, "key": k, "norm_score": 0.0, "rank": float(worst_rank_per_key[k])}
        for m in all_models
        for k in all_keys
        if (m, k) not in have
    ]
    if penalty_rows:
        per_ds = pd.concat([per_ds, pd.DataFrame(penalty_rows)], ignore_index=True)

    agg = (
        per_ds.groupby("model")
        .agg(Score=("norm_score", "mean"), avg_rank=("rank", "mean"))
        .reset_index()
        .rename(columns={"model": "model_id", "avg_rank": "Avg Rank"})
    )
    agg["Improvability"] = ((1.0 - agg["Score"]) * 100).round(1)
    agg["Score"] = agg["Score"].round(4)
    agg["Avg Rank"] = agg["Avg Rank"].round(1)
    return agg


def _build_leaderboard_from_metrics(
    clf_df: pd.DataFrame,
    reg_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build overall/clf/reg leaderboard DataFrames from raw metrics CSVs.

    Kept for backwards compatibility with :meth:`Leaderboard.from_results_dir`.
    """
    lb = Leaderboard(reg_df, clf_df)
    return lb._overall, lb._clf, lb._reg
