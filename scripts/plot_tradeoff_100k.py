"""Elo vs. median time per 100K spectra for a chosen set of models (e.g. the newest ones), log x-axis.

Reads a ``leaderboard_*.csv`` written by ``scripts/plot_results.py`` and draws one figure per time
measure: train + predict, and inference only. Only the models in ``--highlight`` are coloured; the
labels get a white background so the reference lines do not run through the names, and the footnote
under the figure is left out.

    python scripts/plot_tradeoff_100k.py \
        --leaderboard results/v1/figures_X/leaderboards/leaderboard_all.csv \
        --highlight TABSTAR_V1P1 TABH2O SELDON SELDON-MINI TABABAX MLP-A TABM-A \
        --output-dir results/v1/figures_X/tradeoff_100k --title "RamanBench v1"
"""
import argparse
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from raman_bench.plotting import static

# (leaderboard column per 1K spectra, file stem, x-axis label)
MEASURES = (
    ("median_time_total_per_1k_s", "elo_vs_time_100k_train_infer", "Median train + predict time per 100K spectra (s)"),
    ("median_infer_per_1k_s", "elo_vs_time_100k_infer", "Median inference time per 100K spectra (s)"),
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--leaderboard", type=Path, required=True)
    ap.add_argument("--highlight", nargs="+", required=True, help="Model keys to colour")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--title", default="RamanBench v1")
    ap.add_argument("--formats", nargs="+", default=["png"])
    args = ap.parse_args()

    lb = pd.read_csv(args.leaderboard, index_col=0)
    missing = set(args.highlight) - set(lb.index)
    assert not missing, f"not in the leaderboard: {sorted(missing)}"

    static.TASK_TITLES = {**static.TASK_TITLES, "all": args.title}
    static._figure_footnotes = lambda fig: None
    plain_save = static.save

    for column, stem, xlabel in MEASURES:
        d = lb.copy()
        d["time_100k_s"] = d[column] * 100.0
        static.TIME_COL = "time_100k_s"

        def save(fig, *a, _xlabel=xlabel, **k):
            ax = fig.axes[0]
            ax.set_xlabel(_xlabel)
            for text in ax.texts:
                if text.get_text() != "better":
                    text.set_bbox(dict(facecolor="white", edgecolor="none", pad=1.2))
            for coll in ax.collections:  # coloured markers stay above the label boxes
                if coll.get_zorder() == 3:
                    coll.set_zorder(5)
            return plain_save(fig, *a, **k)

        static.save = save
        scores = {"all": SimpleNamespace(leaderboard=d)}
        paths = static.plot_tradeoff(
            scores, {"all": set(args.highlight)}, args.output_dir, args.formats, "elo", stem, groups=["all"], strict=True
        )
        print("wrote", *paths, flush=True)
    static.save = plain_save
    print(f"done: {len(MEASURES)} figures in {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
