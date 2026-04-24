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
- Leaderboard: https://huggingface.co/spaces/ml-lab-htw/RamanBench
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

# Column used as the primary ranking score (higher is better)
_PRIMARY_SCORE = "Score"
_ELO_COL = "Elo"
_RANK_COL = "Rank"


def _load_bundled_csv(filename: str) -> pd.DataFrame:
    """Load a CSV bundled in ``data/precomputed/`` inside this package."""
    try:
        # Python 3.9+ path
        ref = importlib.resources.files("raman_bench") / "data" / "precomputed" / filename
        with importlib.resources.as_file(ref) as path:
            return pd.read_csv(path)
    except Exception:
        # Fallback: walk up from __file__ to find the data directory
        pkg_root = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(pkg_root, "data", "precomputed", filename)
        return pd.read_csv(csv_path)


class Leaderboard:
    """Manage and extend the RamanBench leaderboard.

    Parameters
    ----------
    overall : pd.DataFrame
        Combined leaderboard (all task types).
    clf : pd.DataFrame
        Classification-only leaderboard.
    reg : pd.DataFrame
        Regression-only leaderboard.
    dataset_stats : dict
        Dataset metadata loaded from ``dataset_stats.json``.

    Notes
    -----
    Use the class methods :meth:`from_precomputed` or :meth:`from_results_dir`
    to construct instances — do not call ``__init__`` directly.
    """

    def __init__(
        self,
        overall: pd.DataFrame,
        clf: pd.DataFrame,
        reg: pd.DataFrame,
        dataset_stats: dict | None = None,
    ):
        self._overall = overall.copy()
        self._clf = clf.copy()
        self._reg = reg.copy()
        self._dataset_stats = dataset_stats or {}
        self._added_models: list[str] = []

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_precomputed(cls) -> "Leaderboard":
        """Load the bundled v0.1 precomputed results.

        Returns the leaderboard as published alongside the NeurIPS paper,
        containing 28 baseline models evaluated on 74 datasets (163 targets).

        Returns
        -------
        Leaderboard
        """
        overall = _load_bundled_csv("leaderboard_overall.csv")
        clf = _load_bundled_csv("leaderboard_clf.csv")
        reg = _load_bundled_csv("leaderboard_reg.csv")

        dataset_stats: dict = {}
        try:
            import json
            pkg_root = os.path.dirname(os.path.abspath(__file__))
            stats_path = os.path.join(pkg_root, "data", "precomputed", "dataset_stats.json")
            if os.path.exists(stats_path):
                with open(stats_path) as f:
                    dataset_stats = json.load(f)
        except Exception:
            pass

        logger.info(
            "Loaded precomputed leaderboard: %d models, %d datasets",
            len(overall),
            len(dataset_stats),
        )
        return cls(overall, clf, reg, dataset_stats)

    @classmethod
    def from_results_dir(cls, results_dir: str) -> "Leaderboard":
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

        clf_df = pd.read_csv(clf_path) if os.path.exists(clf_path) else pd.DataFrame()
        reg_df = pd.read_csv(reg_path) if os.path.exists(reg_path) else pd.DataFrame()

        overall, clf_lb, reg_lb = _build_leaderboard_from_metrics(clf_df, reg_df)
        return cls(overall, clf_lb, reg_lb)

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
            Sorted by primary score (descending).  A ``Rank`` column is added.
        """
        df = self._select_leaderboard(task).copy()
        if _PRIMARY_SCORE in df.columns:
            df = df.sort_values(_PRIMARY_SCORE, ascending=False).reset_index(drop=True)
        df[_RANK_COL] = df.index + 1
        cols = [_RANK_COL] + [c for c in df.columns if c != _RANK_COL]
        return df[cols]

    def summary(self) -> str:
        """Return a human-readable summary of the current leaderboard."""
        df = self.rank()
        lines = ["RamanBench Leaderboard (v0.1)", "=" * 40]
        for _, row in df.iterrows():
            model = row.get("Model", row.get("model", "?"))
            score = row.get(_PRIMARY_SCORE, float("nan"))
            elo = row.get(_ELO_COL, float("nan"))
            lines.append(f"  #{int(row[_RANK_COL]):2d}  {model:<28}  Score={score:.3f}  Elo={elo:.0f}")
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
        task: str = "overall",
    ) -> None:
        """Add pre-computed metrics for a new model to the leaderboard.

        Parameters
        ----------
        model_name : str
            Display name of the new model.
        metrics_df : pd.DataFrame
            DataFrame with one row per (dataset_key, seed) and metric columns.
            Must contain columns ``key``, ``seed``, and the relevant metrics
            (e.g. ``rmse``, ``r2`` for regression; ``f1_score``, ``accuracy``
            for classification).
        task : {"overall", "classification", "regression"}
            Which leaderboard to update.
        """
        lb = self._select_leaderboard(task)
        new_row = _summarise_model_metrics(model_name, metrics_df, task)
        updated = pd.concat([lb, pd.DataFrame([new_row])], ignore_index=True)
        self._set_leaderboard(task, updated)
        if model_name not in self._added_models:
            self._added_models.append(model_name)
        logger.info("Added model '%s' to %s leaderboard.", model_name, task)

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
            Leaderboard to update (default ``"overall"``).

        Returns
        -------
        pd.DataFrame
            Per-(key, seed) metrics for the newly evaluated model.
        """
        from raman_bench.benchmark import configure_benchmark
        from raman_bench.config import load_config
        from raman_bench.metrics import compute_metrics
        from raman_data import TASK_TYPE

        if config_path is None:
            pkg_root = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(
                pkg_root, "..", "..", "configs", "benchmark_v0.1.json"
            )

        config = load_config(config_path)
        records = []

        for seed in range(seeds):
            config["random_state"] = seed
            bench = configure_benchmark(config)
            for train_df, test_df, key, task_type in bench:
                if train_df is None:
                    continue
                label_col = train_df.columns[-1]
                X_train = train_df.drop(columns=[label_col]).values
                y_train = train_df[label_col].values
                X_test = test_df.drop(columns=[label_col]).values
                y_test = test_df[label_col].values

                try:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    metrics = compute_metrics(y_test, y_pred, task_type=task_type)
                    records.append({"seed": seed, "key": key, "task_type": task_type, **metrics})
                except Exception as e:
                    logger.warning("Model %s failed on %s (seed %d): %s", model_name, key, seed, e)
                    records.append({"seed": seed, "key": key, "status": "fail", "error": str(e)})

        metrics_df = pd.DataFrame(records)
        self.add_results(model_name, metrics_df, task=task)
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
        model_col = "Model" if "Model" in df.columns else "model"
        models = df[model_col].tolist()
        scores = df[_PRIMARY_SCORE].tolist() if _PRIMARY_SCORE in df.columns else [0] * len(df)
        colors = [
            "#e74c3c" if m in self._added_models else "#3498db"
            for m in models
        ]

        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.barh(range(len(models)), scores[::-1], color=colors[::-1])
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
    # Internal helpers
    # ------------------------------------------------------------------

    def _select_leaderboard(self, task: str) -> pd.DataFrame:
        if task == "overall":
            return self._overall
        if task == "classification":
            return self._clf
        if task == "regression":
            return self._reg
        raise ValueError(f"Unknown task {task!r}. Use 'overall', 'classification', or 'regression'.")

    def _set_leaderboard(self, task: str, df: pd.DataFrame):
        if task == "overall":
            self._overall = df
        elif task == "classification":
            self._clf = df
        elif task == "regression":
            self._reg = df


# ------------------------------------------------------------------
# Helpers for building leaderboard from raw metrics
# ------------------------------------------------------------------

def _summarise_model_metrics(
    model_name: str,
    metrics_df: pd.DataFrame,
    task: str,
) -> dict:
    """Aggregate per-(key, seed) metrics into a single leaderboard row."""
    row: dict = {"Model": model_name}

    numeric = metrics_df.select_dtypes(include=[np.number])
    for col in numeric.columns:
        if col not in ("seed",):
            row[col] = float(numeric[col].mean())

    return row


def _build_leaderboard_from_metrics(
    clf_df: pd.DataFrame,
    reg_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build overall/clf/reg leaderboard DataFrames from raw metrics CSVs."""
    def _agg(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()
        group_cols = [c for c in ["model"] if c in df.columns]
        if not group_cols:
            return pd.DataFrame()
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        metric_cols = [c for c in numeric if c not in ("seed", "target_idx")]
        result = df.groupby(group_cols)[metric_cols].mean().reset_index()
        result.rename(columns={"model": "Model"}, inplace=True)
        return result

    clf_lb = _agg(clf_df)
    reg_lb = _agg(reg_df)

    if not clf_lb.empty and not reg_lb.empty:
        overall = pd.concat([clf_lb, reg_lb], ignore_index=True)
        overall = overall.groupby("Model").mean(numeric_only=True).reset_index()
    elif not clf_lb.empty:
        overall = clf_lb
    else:
        overall = reg_lb

    return overall, clf_lb, reg_lb
