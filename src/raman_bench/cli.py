"""Command-line interface for RamanBench.

Usage
-----
::

    # Show the precomputed (v0.1) leaderboard
    raman-bench leaderboard

    # Rank your v1 results against the leaderboard models (raman_bench.compare)
    raman-bench compare results/my_model --output-dir results/my_model_figures

    # ... ranked by another metric (raman-bench metrics lists them)
    raman-bench compare results/my_model --classification-metric accuracy --regression-metric mae

    # scripts/run_experiment.py calls that run a registered model on the v1 protocol
    raman-bench protocol MY_MODEL --task-type regression

    # The v1 outer CV folds as plain spectrum ids, for running the protocol elsewhere
    raman-bench folds raman_bench_v1_folds.parquet --task-type classification

    # Package and ecosystem info
    raman-bench info

See Also
--------
- Source:      https://github.com/ml-lab-htw/RamanBench
- Leaderboard: https://huggingface.co/spaces/HTW-KI-Werkstatt/RamanBench
- raman-data:  https://github.com/ml-lab-htw/raman_data
- Paper:       https://arxiv.org/abs/2605.02003
"""

import argparse
import logging
import sys
import warnings

warnings.filterwarnings("ignore", message="'force_all_finite' was renamed")

from raman_bench.logging_utils import LOG_FORMAT  # noqa: E402


def cmd_leaderboard(args):
    """Print the precomputed leaderboard."""
    from raman_bench.leaderboard import Leaderboard

    lb = Leaderboard.from_precomputed()
    task = getattr(args, "task", "overall")
    print(lb.summary())

    if getattr(args, "plot", False):
        fig = lb.plot(task=task)
        fig.savefig("leaderboard.png", dpi=150, bbox_inches="tight")
        print("Saved leaderboard.png")


def cmd_compare(args):
    """Score results against the bundled v1 reference results and print the leaderboard."""
    from raman_bench.compare import compare, load_protocol

    scores = compare(
        args.results,
        tasks=args.tasks,
        out_dir=args.output_dir,
        figures=not args.no_figures,
        bootstrap_rounds=args.bootstrap_rounds,
        classification_metric=args.classification_metric,
        regression_metric=args.regression_metric,
    )
    reference = set(load_protocol()["models"])
    cols = ["preprocessing", "elo", "rank", "winrate", "imputed_pct", "median_time_total_per_1k_s"]
    for group in ("all", "classification", "regression"):
        if group not in scores:
            continue
        lb = scores[group].leaderboard
        lb = lb[[c for c in cols if c in lb.columns]].rename(columns={"median_time_total_per_1k_s": "s_per_1k"})
        lb.insert(0, "pos", range(1, len(lb) + 1))
        lb.index = [m if m in reference else f"* {m}" for m in lb.index]
        if args.top:
            lb = lb[(lb["pos"] <= args.top) | ~lb.index.isin(list(reference))]
        print(f"\n== {group}: {scores[group].n_tasks} tasks (* = yours) ==")
        print(lb.to_string(float_format=lambda x: f"{x:.3g}" if abs(x) < 10 else f"{x:.0f}"))
    if args.output_dir:
        print(f"\nWrote leaderboards{' and figures' if not args.no_figures else ''} to {args.output_dir}")


def cmd_protocol(args):
    """Print the run_experiment.py calls for a registered model on the v1 protocol."""
    from raman_bench.evaluate import protocol_commands

    for line in protocol_commands(
        args.model, task_type=args.task_type, tasks=args.tasks, results_dir=args.results_dir
    ):
        print(line)


def cmd_metrics(_args):
    """List the metrics compare can rank by."""
    from raman_bench.fold_metrics import METRICS

    for task in ("classification", "regression"):
        print(f"{task}:")
        for name, m in METRICS.items():
            if m.task == task:
                print(f"  {name:<20} {'higher' if m.higher_is_better else 'lower'} is better  {m.description}")


def cmd_folds(args):
    """Write the v1 outer cross-validation folds (raman_bench.folds) to a file."""
    from raman_bench.folds import export_folds

    folds = export_folds(
        args.output, task_type=args.task_type, tasks=args.tasks, cache_dir=args.cache_dir
    )
    print(f"Wrote {len(folds)} rows ({folds['task'].nunique()} tasks) to {args.output}")


def cmd_info(_args):
    """Print package and ecosystem info."""
    import raman_bench

    print(f"raman-bench {raman_bench.__version__}")
    print()
    print("Ecosystem:")
    print("  raman-data (datasets)")
    print("    PyPI:   pip install raman-data")
    print("    Source: https://github.com/ml-lab-htw/raman_data")
    print()
    print("  raman-bench (this package)")
    print("    PyPI:   pip install raman-bench")
    print("    Source: https://github.com/ml-lab-htw/RamanBench")
    print()
    print("  Live Leaderboard")
    print("    https://huggingface.co/spaces/HTW-KI-Werkstatt/RamanBench")
    print()
    print("  Paper (under review)")
    print("    https://arxiv.org/abs/2605.02003")


def main():
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")

    parser = argparse.ArgumentParser(
        prog="raman-bench",
        description="RamanBench — large-scale ML benchmark for Raman spectroscopy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    # ---- leaderboard ----
    lb_p = sub.add_parser("leaderboard", help="Show the precomputed leaderboard")
    lb_p.add_argument(
        "--task",
        choices=["overall", "classification", "regression"],
        default="overall",
    )
    lb_p.add_argument("--plot", action="store_true", help="Save leaderboard.png")
    lb_p.set_defaults(func=cmd_leaderboard)

    # ---- compare ----
    cmp_p = sub.add_parser("compare", help="Rank your v1 results against the leaderboard models")
    cmp_p.add_argument("results", help="Results directory (results.pkl files) or hpo_results.csv")
    cmp_p.add_argument(
        "--tasks", choices=["all", "own"], default="all",
        help="'all': every protocol task, like the leaderboard (default); 'own': only the tasks you ran",
    )
    cmp_p.add_argument("--output-dir", help="Write leaderboard CSVs and figures here")
    cmp_p.add_argument("--no-figures", action="store_true", help="With --output-dir: leaderboard CSVs only")
    cmp_p.add_argument("--top", type=int, default=0, help="Print only the top N models per group")
    cmp_p.add_argument("--bootstrap-rounds", type=int, default=200, help="Elo bootstrap rounds for the CIs")
    from raman_bench.fold_metrics import metric_names

    cmp_p.add_argument(
        "--classification-metric", choices=metric_names("classification"),
        help="Rank classification by this metric (default: ROC AUC for binary, log loss for multiclass)",
    )
    cmp_p.add_argument(
        "--regression-metric", choices=metric_names("regression"),
        help="Rank regression by this metric (default: RMSE)",
    )
    cmp_p.set_defaults(func=cmd_compare)

    # ---- protocol ----
    pr_p = sub.add_parser("protocol", help="run_experiment.py calls for a registered model on the v1 protocol")
    pr_p.add_argument("model", help="Model key, e.g. PLS")
    pr_p.add_argument("--task-type", choices=["classification", "regression"])
    pr_p.add_argument("--tasks", nargs="*", help="Task keys or dataset names (default: all)")
    pr_p.add_argument("--results-dir", default="results/v1/user")
    pr_p.set_defaults(func=cmd_protocol)

    # ---- metrics ----
    me_p = sub.add_parser("metrics", help="List the metrics compare can rank by")
    me_p.set_defaults(func=cmd_metrics)

    # ---- folds ----
    fo_p = sub.add_parser("folds", help="Write the v1 outer CV folds (spectrum ids per task and fold)")
    fo_p.add_argument("output", help="Output file: .parquet, or .csv")
    fo_p.add_argument("--task-type", choices=["classification", "regression"])
    fo_p.add_argument("--tasks", nargs="*", help="Task keys or dataset names (default: all)")
    fo_p.add_argument("--cache-dir", default=".cache_v1", help="Dataset download cache")
    fo_p.set_defaults(func=cmd_folds)

    # ---- info ----
    info_p = sub.add_parser("info", help="Show package and ecosystem info")
    info_p.set_defaults(func=cmd_info)

    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
