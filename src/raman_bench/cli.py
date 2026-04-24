"""Command-line interface for RamanBench.

Usage
-----
::

    # Run the full benchmark pipeline
    raman-bench run --config configs/benchmark_v0.1.json

    # Run individual steps
    raman-bench run --step predictions
    raman-bench run --step metrics
    raman-bench run --step plots

    # Show the precomputed leaderboard
    raman-bench leaderboard

    # Evaluate a single model (requires a Python entry point)
    raman-bench info

See Also
--------
- Source:      https://github.com/ml-lab-htw/RamanBench
- Leaderboard: https://huggingface.co/spaces/ml-lab-htw/RamanBench
- raman-data:  https://github.com/ml-lab-htw/raman_data
- Paper:       https://arxiv.org/abs/TBD
"""

import argparse
import logging
import os
import sys
import warnings

warnings.filterwarnings("ignore", message="'force_all_finite' was renamed")

from raman_bench.logging_utils import LOG_FORMAT  # noqa: E402


def _default_config() -> str:
    pkg_root = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(pkg_root, "..", "..", "configs", "benchmark_v0.1.json")


def cmd_run(args):
    """Run the benchmark pipeline (predictions → metrics → plots)."""
    from raman_bench.config import load_config

    config = load_config(args.config)

    if args.output:
        config["output_dir"] = args.output

    if args.model:
        config["models"] = [args.model]

    step = args.step

    if step in ("all", "predictions"):
        from raman_bench.predictions import compute_predictions
        compute_predictions(
            config,
            seed_index=args.seed_index,
            overwrite=args.overwrite,
            reverse=args.reverse,
        )

    if step in ("all", "metrics"):
        from raman_bench.evaluation import compute_metrics_from_predictions
        compute_metrics_from_predictions(config)

    if step in ("all", "plots"):
        try:
            from raman_bench.plotting import PlotPipeline
            PlotPipeline(config).run_all()
        except ImportError:
            logging.warning("Plotting requires additional dependencies. Run: pip install raman-bench[full]")


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
    print("    https://huggingface.co/spaces/ml-lab-htw/RamanBench")
    print()
    print("  Paper (NeurIPS 2026)")
    print("    https://arxiv.org/abs/TBD")


def main():
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")

    parser = argparse.ArgumentParser(
        prog="raman-bench",
        description="RamanBench — large-scale ML benchmark for Raman spectroscopy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    # ---- run ----
    run_p = sub.add_parser("run", help="Run the benchmark pipeline")
    run_p.add_argument("--config", default=_default_config(), help="Path to config JSON")
    run_p.add_argument(
        "--step",
        choices=["all", "predictions", "metrics", "plots"],
        default="all",
        help="Pipeline step (default: all)",
    )
    run_p.add_argument("--output", default=None, help="Override output directory")
    run_p.add_argument("--seed-index", type=int, default=None, help="Run only this seed index")
    run_p.add_argument("--model", default=None, help="Run only this model")
    run_p.add_argument("--overwrite", action="store_true", help="Overwrite existing predictions")
    run_p.add_argument("--reverse", action="store_true", help="Iterate datasets in reverse order")
    run_p.set_defaults(func=cmd_run)

    # ---- leaderboard ----
    lb_p = sub.add_parser("leaderboard", help="Show the precomputed leaderboard")
    lb_p.add_argument(
        "--task",
        choices=["overall", "classification", "regression"],
        default="overall",
    )
    lb_p.add_argument("--plot", action="store_true", help="Save leaderboard.png")
    lb_p.set_defaults(func=cmd_leaderboard)

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
