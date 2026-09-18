"""Command-line interface for RamanBench.

Usage
-----
::

    # Show the precomputed leaderboard
    raman-bench leaderboard

    # Evaluate a single model (requires a Python entry point)
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
