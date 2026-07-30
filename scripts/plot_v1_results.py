#!/usr/bin/env python
"""Plot per-dataset performance from an aggregated v1 results CSV.

Consumes ``hpo_results.csv`` (written by ``scripts/aggregate_results.py``) and
produces one bar chart per model, showing ``metric_error`` across every
(dataset,target) task -- grouped by default/tuned/tuned+ensemble when more
than the default config was actually run for that model.

Not a full multi-model leaderboard (Elo/rank/win-rate need >=2 models on the
same tasks to be meaningful -- see ``bencheval.tabarena.TabArena`` for that,
once there's more than one model's worth of v1 results to compare). This is
the per-model performance view that's meaningful even for a single model.

Usage:
    python scripts/plot_v1_results.py --input results/v1/aggregated/hpo_results.csv --output-dir results/v1/plots
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_model(df: pd.DataFrame, model: str, output_dir: str) -> str:
    sub = df[df["ta_name"] == model].copy()
    sub = sub.sort_values(["dataset", "method"])

    # Each (dataset, method) has one row per seed/fold -- TabArena's "fold"
    # column is the split axis, which for RamanBench is really the repeated
    # seed (see the v1 refactor: repeat=seed, fold is always 0 upstream).
    # Aggregate across seeds (mean +/- std) rather than showing just one.
    agg = (
        sub.groupby(["dataset", "method"])["metric_error"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    datasets = sorted(agg["dataset"].unique())
    method_types = sorted(agg["method"].str.extract(r"\((.+)\)")[0].unique())

    fig, ax = plt.subplots(figsize=(max(8, len(datasets) * 0.4), 5))
    width = 0.8 / max(len(method_types), 1)
    x = range(len(datasets))

    for i, method_type in enumerate(method_types):
        means, stds = [], []
        for ds in datasets:
            row = agg[(agg["dataset"] == ds) & (agg["method"] == f"{model} ({method_type})")]
            means.append(row["mean"].iloc[0] if len(row) else float("nan"))
            stds.append(row["std"].iloc[0] if len(row) and row["count"].iloc[0] > 1 else 0.0)
        offsets = [xi + i * width for xi in x]
        ax.bar(offsets, means, width=width, yerr=stds, capsize=2, label=method_type)

    ax.set_xticks([xi + width * (len(method_types) - 1) / 2 for xi in x])
    ax.set_xticklabels(datasets, rotation=90, fontsize=7)
    ax.set_ylabel("mean metric_error across seeds (lower is better)")
    ax.set_title(f"{model}: per-task performance", loc="left", fontweight="bold")
    ax.legend()
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{model}_per_task.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default="results/v1/aggregated/hpo_results.csv")
    parser.add_argument("--output-dir", default="results/v1/plots")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if df.empty:
        print(f"No rows in {args.input} -- nothing to plot.")
        return

    for model in sorted(df["ta_name"].unique()):
        path = plot_model(df, model, args.output_dir)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
