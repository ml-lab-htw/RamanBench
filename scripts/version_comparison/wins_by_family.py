"""Who wins in benchmark version A vs. B? Share of tasks won per model category, each bar stacked by family.

Writes two figures: wins_by_family_full (all tasks and models of each version) and wins_by_family_shared
(only tasks and models present in both). See README.md for the inputs.

usage: python scripts/version_comparison/wins_by_family.py --old OLD --new NEW --out OUT_DIR
           [--old-label v0.1] [--new-label v1] [--no-footnote]
"""
import argparse
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import FancyBboxPatch, Patch

from common import CATS, FAMILY_COLOURS, load_version, restrict, shared, winners

# Category labels for the x axis (two lines).
XLABELS = {"Tabular Foundation": "Tabular\nfoundation models", "Raman-Specific": "Raman-specific\nnetworks",
           "Deep Learning": "Generic\ndeep learning", "Traditional ML": "Traditional ML\n(PLS, LR, KNN)",
           "Trees": "Trees and\nboosting", "TS Classification": "Time-series\nclassifiers"}
GROUPS = [(k, XLABELS[k], c) for k, _, c in CATS]


def panel(ax, a, b, labels, title, colours, groups, ymax):
    na, nb = len(a), len(b)
    step, w = 3.2, 1.15
    u = ymax / 100  # text offsets scale with the axis
    for gi, (g, label, base) in enumerate(groups):
        x0 = gi * step
        ax.axvspan(x0 - 1.45, x0 + 1.45, color=base, alpha=0.06, lw=0, zorder=0)
        tops = []
        for off, d, n, ver, new in ((-0.62, a, na, labels[0], False), (0.62, b, nb, labels[1], True)):
            fam = d[d["cat"] == g]["family"].value_counts()
            order = [f for f in colours[g] if f in fam.index]
            y = 0.0
            for f in order:
                h = 100 * fam[f] / n
                ax.add_patch(FancyBboxPatch((x0 + off - w / 2, y), w, h, boxstyle="round,pad=0,rounding_size=0.12",
                                            fc=colours[g][f], ec="white", lw=1.2, zorder=3, mutation_aspect=u / 6))
                if h >= 3.2 * u:
                    dark = sum(to_rgb(colours[g][f])) < 1.9
                    txt = f"{f}\n({fam[f]})" if len(f) > 8 and h >= 5.5 * u else f"{f} ({fam[f]})"
                    ax.text(x0 + off, y + h / 2, txt, linespacing=0.95, ha="center", va="center", fontsize=6.4,
                            zorder=4, color="white" if dark else "#222222",
                            fontweight="bold" if h >= 6 * u else None)
                y += h
            tops.append(y)
            ax.text(x0 + off, y + 1.0 * u, f"{y:.0f}%", ha="center", va="bottom", fontsize=10, fontweight="bold",
                    color=base if new else "#777777", zorder=4)
            ax.text(x0 + off, y + 4.4 * u, f"{fam.sum()} tasks", ha="center", va="bottom", fontsize=6.8,
                    color="#777777", zorder=4)
            ax.text(x0 + off, -1.6 * u, ver, ha="center", va="top", fontsize=8,
                    color=base if new else "#555555", fontweight="bold" if new else None)
        delta = tops[1] - tops[0]
        sign = "+" if delta >= 0 else "−"
        badge = "#2E7D32" if delta >= 0.5 else "#B23B3B" if delta <= -0.5 else "#8A8A8A"
        ax.text(x0, max(tops) + 9.5 * u, f"{sign}{abs(delta):.0f} pp", ha="center", va="bottom", fontsize=8.5,
                fontweight="bold", color="white", zorder=5,
                bbox=dict(boxstyle="round,pad=0.28,rounding_size=0.6", fc=badge, ec="none"))
        ax.text(x0, -6.5 * u, label, ha="center", va="top", fontsize=9.5, fontweight="bold", color=base)
    ax.set_xlim(-1.6, (len(groups) - 1) * step + 1.6)
    ax.set_ylim(0, ymax)
    ax.set_ylabel("share of tasks won (%)")
    ax.set_xticks([])
    ax.yaxis.grid(True, color="#E6E6E6", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=8, color="#333333")


def colour_map(*ws) -> dict:
    """Per category, the winning families ordered by total wins (largest at the bottom), in their fixed colours."""
    allw = pd.concat(ws)
    return {g: {f: FAMILY_COLOURS.get(f, base) for f in allw[allw["cat"] == g]["family"].value_counts().index}
            for g, _, base in GROUPS}


def figure(a, b, args, stem, subtitle, metric_note):
    labels = (args.old_label, args.new_label)
    colours = colour_map(a, b)
    fig, (ax, az) = plt.subplots(1, 2, figsize=(16, 7.8), gridspec_kw={"width_ratios": [1.05, 4]})
    panel(ax, a, b, labels, CATS[0][1].capitalize(), colours, GROUPS[:1], 100)
    top = max(100 * (x["cat"] == g).sum() / len(x) for x in (a, b) for g, *_ in GROUPS[1:])
    panel(az, a, b, labels, "All other model categories (y-axis zoomed)", colours, GROUPS[1:],
          max(5, -(-(top + 6) // 5) * 5))
    az.set_ylabel("")
    fig.suptitle(f"Who wins in RamanBench {labels[0]} vs. {labels[1]}? Share of tasks won, by category and model "
                 f"family — {subtitle}", x=0.01, ha="left", fontsize=13.5, fontweight="bold")
    handles = []
    for g, label, _ in GROUPS:
        if colours[g]:
            handles.append(Patch(fc="none", ec="none", label=label.replace("\n", " ")))
            handles += [Patch(fc=c, ec="white", label=f"   {f}") for f, c in colours[g].items()]
    leg = az.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.005, 1.02), frameon=False, fontsize=7.6,
                    handlelength=1.1, handleheight=0.9, labelspacing=0.28)
    for t in leg.get_texts():
        if not t.get_text().startswith("   "):
            t.set_fontweight("bold")
    if not args.no_footnote:
        fig.text(0.01, 0.012,
                 f"Winner per task = best mean score on each version's own metric ({metric_note}) among ranked "
                 f"models; DUMMY and AutoGluon excluded. {labels[0]}: {len(a)} tasks, {labels[1]}: {len(b)} tasks. "
                 f"Each family has the same colour in both versions. Families merge variants (all TabPFN versions, "
                 f"Kumo-Tabular S/M/L, ...).\nCategories as on the {labels[1]} leaderboard; RamanPFN is excluded "
                 f"(developed on the RamanBench datasets). Badge = change in percentage points.",
                 fontsize=7.8, color="#666666")
    fig.tight_layout(rect=(0, 0.01 if args.no_footnote else 0.05, 1, 0.97))
    for ext in ("png", "pdf"):
        fig.savefig(args.out / f"{stem}.{ext}", dpi=220)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--old", type=Path, required=True,
                   help="older version: reference-results parquet or v0.1 precomputed_metrics dir")
    p.add_argument("--new", type=Path, required=True, help="newer version: reference-results parquet")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--old-label", default="v0.1")
    p.add_argument("--new-label", default="v1")
    p.add_argument("--no-footnote", action="store_true", help="leave out the footnote (clean version for posts)")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    old, new = load_version(args.old, args.old_label), load_version(args.new, args.new_label)
    tasks, models = shared(old, new)
    note = f"{args.old_label}: {old.metric_desc}; {args.new_label}: {new.metric_desc}"
    n_models = len(set(models))
    for (x, y), stem, sub in (((old.scores, new.scores), "wins_by_family_full", "full benchmarks"),
                              ((restrict(old.scores, tasks, models), restrict(new.scores, tasks, models)),
                               "wins_by_family_shared", f"same {len(tasks)} tasks, same {n_models} models")):
        a, b = winners(x), winners(y)
        figure(a, b, args, stem, sub, note)
        print(stem, len(a), len(b))
        print(pd.concat({args.old_label: a.groupby("cat").size(), args.new_label: b.groupby("cat").size()},
                        axis=1).fillna(0).astype(int))
        missing = sorted({f for w in (a, b) for f in w["family"]} - set(FAMILY_COLOURS))
        if missing:
            print("no fixed colour (drawn in their category's colour; add to common.FAMILY_COLOURS):", missing)


if __name__ == "__main__":
    main()
