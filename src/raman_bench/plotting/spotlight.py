"""Spotlight figures: one or two models in the foreground, the rest of the field kept quiet.

Two figures share one look (soft canvas, glowing frontier over a shaded dominated region, halo markers, a callout
per spotlighted model, the other front/record models named in grey):

* :func:`plot_spotlight_tradeoff`: Elo against median time, Pareto front.
* :func:`plot_spotlight_release`: Elo against model release date, running best.

They are meant for slides and posts, where the standard figures (:mod:`raman_bench.plotting.static`) are too
dense. The models to spotlight are chosen by the caller (``--highlight`` of ``scripts/plot_results.py``).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox

from raman_bench.plotting import models as model_info
from raman_bench.plotting.results import (
    GroupScores,
    by_release,
    pareto_front,
    record_holders,
    split_references,
)
from raman_bench.plotting.static import save

INK, GREY, GREY_DARK, LIGHT = "#0B2545", "#6B7280", "#4B5563", "#C9CED6"
CANVAS, PANEL = "#F7F9FC", "white"
#: Spotlight colours: vermillion, then a lighter amber for the second model.
SPOTLIGHT_COLORS = ("#D55E00", "#F2A03D")
HF_URL = "huggingface.co/spaces/HTW-KI-Werkstatt/RamanBench"

_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 13,
    "axes.edgecolor": "#B8BEC8",
    "axes.linewidth": 1.0,
    "pdf.fonttype": 42,
}

# Candidate label offsets (dx, dy in points, ha, va), tried in order.
_GREY_OFFSETS = [
    (9, 7, "left", "bottom"), (9, -9, "left", "top"), (-9, 7, "right", "bottom"), (-9, -9, "right", "top"),
    (0, 12, "center", "bottom"), (0, -13, "center", "top"), (14, 0, "left", "center"), (-14, 0, "right", "center"),
    (24, 18, "left", "bottom"), (-24, 18, "right", "bottom"), (24, -20, "left", "top"), (-24, -20, "right", "top"),
    (40, 40, "left", "bottom"), (-40, 40, "right", "bottom"), (40, -42, "left", "top"), (-40, -42, "right", "top"),
    (60, 70, "left", "bottom"), (-60, 70, "right", "bottom"), (60, -72, "left", "top"), (-60, -72, "right", "top"),
    (90, 20, "left", "center"), (-90, 20, "right", "center"), (90, -25, "left", "center"), (-90, -25, "right", "center"),
]
_CALLOUT_OFFSETS = [
    (0, 45, "center", "bottom"), (-40, 45, "right", "bottom"), (40, 45, "left", "bottom"), (0, -55, "center", "top"),
    (-40, -50, "right", "top"), (40, -50, "left", "top"), (70, 0, "left", "center"), (-70, 0, "right", "center"),
    (0, 95, "center", "bottom"), (0, -95, "center", "top"), (-110, 70, "right", "bottom"), (-110, -70, "right", "top"),
]


# --------------------------------------------------------------------------- building blocks


def _canvas(title: str, subtitle: str, xlabel: str):
    """Figure with the soft background, the accent bar, title and subtitle; returns ``(fig, ax)``."""
    fig = plt.figure(figsize=(11.5, 7.0), facecolor=CANVAS)
    ax = fig.add_axes([0.075, 0.175, 0.905, 0.64], facecolor=PANEL)
    fig.patches.append(matplotlib.patches.Rectangle(
        (0.075, 0.955), 0.055, 0.007, transform=fig.transFigure, color=SPOTLIGHT_COLORS[0], lw=0))
    fig.text(0.075, 0.91, title, fontsize=24, fontweight="bold", color=INK, ha="left", va="center")
    fig.text(0.075, 0.86, subtitle, fontsize=13.5, color=GREY, ha="left", va="center")
    ax.set_xlabel(xlabel, fontsize=13, color=GREY_DARK, labelpad=6)
    ax.set_ylabel("Elo  (Random Forest = 1000)", fontsize=13, color=GREY_DARK, labelpad=10)
    ax.tick_params(colors=GREY_DARK, labelsize=12, length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return fig, ax


def _backdrop(ax, xlim, ylim) -> None:
    """Vertical gradient and grid; fixes the limits so later artists cannot move them."""
    grad = np.linspace(0, 1, 256).reshape(-1, 1)
    ax.imshow(grad, extent=[*xlim, *ylim], origin="lower", aspect="auto", zorder=0, alpha=0.9,
              cmap=LinearSegmentedColormap.from_list("spotlight", ["#FFFFFF", "#EEF2F7"]))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid(True, which="major", color="#E3E7EE", lw=0.9, zorder=0.6)
    ax.set_axisbelow(True)


def _frontier(ax, xs, ys, ymin) -> None:
    """Shaded region under a staircase (already extended to the axes) with a glowing dashed line on top."""
    ax.fill_between(xs, ymin, ys, step="post", color="#E7ECF3", alpha=0.65, zorder=0.5, lw=0)
    ax.step(xs, ys, where="post", color="#AEB6C4", lw=7, alpha=0.18, zorder=1.4, solid_capstyle="round")
    ax.step(xs, ys, where="post", color="#7C8696", lw=1.7, ls=(0, (5, 3)), zorder=1.5)


def _markers(ax, crowd: pd.DataFrame, named: pd.DataFrame, spotlight: pd.DataFrame, colors) -> None:
    """The field in light grey, the named (front/record) models in darker grey, the spotlight models with halos."""
    ax.scatter(crowd["x"], crowd["elo"], s=46, color=LIGHT, alpha=0.85, edgecolors="white", linewidths=0.6, zorder=2)
    ax.scatter(named["x"], named["elo"], s=120, color=GREY, edgecolors="white", linewidths=1.6, zorder=3.2)
    for (_, row), color in zip(spotlight.iterrows(), colors):
        for size, alpha in ((1500, 0.07), (800, 0.12), (400, 0.2)):
            ax.scatter([row["x"]], [row["elo"]], s=size, color=color, alpha=alpha, linewidths=0, zorder=3.5)
        ax.scatter([row["x"]], [row["elo"]], s=230, color=color, edgecolors="white", linewidths=2.4, zorder=5)


class _Labels:
    """Collision-free label placement in display space.

    A label takes the first candidate offset whose box lies inside the axes and covers no earlier label and no
    other marker; if none fits, the first offset is used.
    """

    def __init__(self, ax, points: pd.DataFrame):
        self.ax = ax
        ax.figure.canvas.draw()  # fixes the transforms
        self.renderer = ax.figure.canvas.get_renderer()
        self.bounds = ax.get_window_extent(self.renderer)
        self.points = {}
        for key, row in points.iterrows():
            px, py = ax.transData.transform((row["x"], row["elo"]))
            self.points[key] = Bbox.from_extents(px - 8, py - 8, px + 8, py + 8)
        self.taken: list[Bbox] = []

    def place(self, key, xy, offsets, margin=3, **text_kwargs):
        """Pass 1: free of labels and markers; pass 2: free of labels only; else the first offset.

        A label further than 30 points from its marker gets a thin leader line.
        """
        def draw(dx, dy, ha, va):
            leader = dict(arrowstyle="-", color="#9AA3B2", lw=0.9, shrinkA=0, shrinkB=6) if max(abs(dx), abs(dy)) > 30 else None
            kwargs = {"arrowprops": leader, **text_kwargs} if "arrowprops" not in text_kwargs else text_kwargs
            return self.ax.annotate(
                xy=xy, xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, annotation_clip=False, **kwargs
            )

        b = self.bounds
        for allow_markers in (False, True):
            for offset in offsets:
                note = draw(*offset)
                bb = note.get_window_extent(self.renderer)
                bb = Bbox.from_extents(bb.x0 - margin, bb.y0 - margin, bb.x1 + margin, bb.y1 + margin)
                inside = bb.x0 > b.x0 + 4 and bb.x1 < b.x1 - 4 and bb.y0 > b.y0 + 4 and bb.y1 < b.y1 - 4
                if inside and not any(bb.overlaps(o) for o in self.taken) and (
                    allow_markers or not any(bb.overlaps(p) for k, p in self.points.items() if k != key)
                ):
                    self.taken.append(bb)
                    return note
                note.remove()
        note = draw(*offsets[0])
        self.taken.append(note.get_window_extent(self.renderer))
        return note


def _label_all(labels: _Labels, ax, spotlight: pd.DataFrame, colors, named: pd.DataFrame, rank, n_models) -> None:
    """Callouts for the spotlight models first, so the grey names work around them."""
    for (key, row), color in zip(spotlight.iterrows(), colors):
        text = f"{row['display_name']}\nElo {row['elo']:.0f}  ·  #{rank[key]} of {n_models}"
        note = labels.place(
            key, (row["x"], row["elo"]), _CALLOUT_OFFSETS, margin=8, text=text, fontsize=14, color=INK,
            fontweight="bold", linespacing=1.4, zorder=8,
            bbox=dict(boxstyle="round,pad=0.55,rounding_size=0.8", fc="white", ec=color, lw=2.6),
            arrowprops=dict(arrowstyle="-", color=color, lw=2.2, shrinkA=0, shrinkB=12),
        )
        note.get_bbox_patch().set_path_effects(
            [pe.SimplePatchShadow(offset=(2.5, -2.5), shadow_rgbFace=INK, alpha=0.16), pe.Normal()]
        )
    halo = [pe.withStroke(linewidth=3.4, foreground="white")]
    for key, row in named.iterrows():
        labels.place(
            key, (row["x"], row["elo"]), _GREY_OFFSETS, text=row["display_name"], fontsize=12, color=GREY_DARK,
            fontweight="bold", path_effects=halo, zorder=6,
        )


def _legend(fig, spotlight: pd.DataFrame, colors, named_label: str) -> None:
    items = [
        Line2D([], [], marker="o", ls="", ms=13, mfc=c, mec="white", mew=2, label=row["display_name"])
        for (_, row), c in zip(spotlight.iterrows(), colors)
    ]
    items += [
        Line2D([], [], marker="o", ls="", ms=10, mfc=GREY, mec="white", mew=1.5, label=named_label),
        Line2D([], [], marker="o", ls="", ms=8, mfc=LIGHT, mec="white", mew=1, label="All other models"),
    ]
    fig.legend(handles=items, loc="lower left", bbox_to_anchor=(0.065, -0.005), ncol=len(items), frameon=False,
               fontsize=11.5, labelcolor=GREY_DARK, handletextpad=0.3, columnspacing=1.6)
    fig.text(0.98, 0.075, HF_URL, fontsize=10.5, color="#9AA3B2", ha="right")


def _spotlight_keys(lb: pd.DataFrame, highlight: tuple[str, ...] | list[str]) -> list[str]:
    """The leaderboard keys of *highlight* (case-insensitive), in the order given; at most two."""
    by_upper = {k.upper(): k for k in lb.index}
    keys = [by_upper[h.upper()] for h in highlight if h.upper() in by_upper]
    if not 1 <= len(keys) <= len(SPOTLIGHT_COLORS):
        raise ValueError(f"spotlight needs 1-{len(SPOTLIGHT_COLORS)} models present in the results, got {keys}")
    return keys


def _rank_of(lb_all: pd.DataFrame) -> tuple[pd.Series, int]:
    models = split_references(lb_all)[0]
    return models["elo"].rank(ascending=False, method="min").astype(int), len(models)


# --------------------------------------------------------------------------- Elo vs. time


def plot_spotlight_tradeoff(
    scores: GroupScores,
    highlight,
    out_dir: Path,
    formats,
    stem: str = "spotlight_elo_vs_time",
    *,
    per_spectra: int = 1000,
    time_col: str = "median_time_total_per_1k_s",
):
    """Elo vs. median time per *per_spectra* spectra, the Pareto front in grey, *highlight* (1-2 keys) in colour.

    *time_col* is a per-1K-spectra column of the leaderboard (``median_infer_per_1k_s`` for inference only).
    """
    lb, refs = split_references(scores.leaderboard)
    keys = _spotlight_keys(lb, highlight)
    colors = SPOTLIGHT_COLORS[: len(keys)]
    df = lb.dropna(subset=[time_col, "elo"]).copy()
    df["x"] = df[time_col] * per_spectra / 1000.0
    front = pareto_front(df, "x", "elo", True)
    rank, n_models = _rank_of(scores.leaderboard)

    what = "inference" if time_col == "median_infer_per_1k_s" else "train + predict"
    title = " and ".join(df.loc[k, "display_name"] for k in keys)
    with plt.rc_context(_RC):
        fig, ax = _canvas(
            title,
            f"Elo vs. median {what} time per {per_spectra:,} spectra  ·  {scores.n_tasks} Raman tasks  ·  "
            f"{n_models} models",
            f"Median {what} time per {per_spectra:,} spectra (s, log scale)",
        )
        ax.set_xscale("log")
        xmin, xmax = df["x"].min() / 2.2, df["x"].max() * 6
        ymin, ymax = df["elo"].min() - 40, df["elo"].max() + 330
        _backdrop(ax, (xmin, xmax), (ymin, ymax))

        fx, fy = front["x"].to_numpy(), front["elo"].to_numpy()
        order = np.argsort(fx)
        fx, fy = fx[order], fy[order]
        _frontier(ax, np.r_[fx, xmax], np.r_[fy, fy[-1]], ymin)

        for _, ref in refs.iterrows():
            ax.axhline(ref["elo"], color="#8A93A3", lw=1.1, ls=(0, (2, 3)), zorder=1.2)
            ax.text(xmax / 1.05, ref["elo"] + 4, ref["display_name"], ha="right", va="bottom", fontsize=10.5,
                    color="#7A8394", style="italic", zorder=1.3)

        spotlight = df.loc[keys]
        named = front[~front.index.isin(keys)]
        crowd = df[~df.index.isin(front.index) & ~df.index.isin(keys)]
        _markers(ax, crowd, named, spotlight, colors)
        if len(keys) == 2:
            ax.plot(spotlight["x"], spotlight["elo"], color="#E08A3C", lw=2.2, ls=(0, (1, 2)), alpha=0.9,
                    zorder=4.2, solid_capstyle="round")
        _label_all(_Labels(ax, df), ax, spotlight, colors, named, rank, n_models)
        _legend(fig, spotlight, colors, "Other Pareto-optimal models")
        return save(fig, out_dir, stem, formats)


# --------------------------------------------------------------------------- Elo vs. release date


def _release_x(year: float) -> float:
    """Piecewise time axis: the decades before 2018 are squeezed hard, 2018-2024 mildly, 2025 on gets the room."""
    if year < 2018:
        return 2018 - (2018 - year) * 0.05
    if year < 2025:
        return 2018 + (year - 2018) * 0.55
    return 2018 + 7 * 0.55 + (year - 2025)


def plot_spotlight_release(
    scores: GroupScores,
    highlight,
    out_dir: Path,
    formats,
    stem: str = "spotlight_elo_vs_release_date",
    *,
    hide=(),
):
    """Elo over model release date: the running best as a staircase, *highlight* (1-2 keys) in colour.

    *hide* models are not drawn (e.g. a second model released the same day); the Elo scores and ranks still
    include them.
    """
    lb_all = scores.leaderboard
    rank, n_models = _rank_of(lb_all)
    lb = by_release(lb_all)
    lb = lb.assign(x=[_release_x(d) for d in lb["release_date"]])
    record = record_holders(lb)
    keys = _spotlight_keys(lb, highlight)
    colors = SPOTLIGHT_COLORS[: len(keys)]
    hidden = {h.upper() for h in hide}
    lb = lb[~lb.index.str.upper().isin(hidden)]

    with plt.rc_context(_RC):
        fig, ax = _canvas(
            "Model progress on RamanBench",
            f"Elo by model release date  ·  {scores.n_tasks} Raman tasks  ·  {n_models} models  ·  "
            "dashed line: best model so far",
            "Model release (axis compressed before 2025)",
        )
        x_first, x_last = _release_x(1950), _release_x(lb["release_date"].max() + 0.55)
        ymin, ymax = lb["elo"].min() - 50, lb["elo"].max() + 330
        _backdrop(ax, (x_first, x_last), (ymin, ymax))

        _frontier(ax, [x_first, *record["x"], x_last], [record["elo"].iloc[0], *record["elo"], record["elo"].iloc[-1]],
                  ymin)
        ax.axhline(1000, color="#8A93A3", lw=1.1, ls=(0, (2, 3)), zorder=1.2)
        ax.text(x_first + 0.08, 1006, "Random Forest = 1000", fontsize=10.5, color="#7A8394", style="italic",
                va="bottom", zorder=1.3)
        ax.axvline(_release_x(2018.0), color="#E3E7EE", lw=6, zorder=0.7)  # the break in the axis
        years = [1980, 2000, 2018, 2020, 2022, 2024, 2025, 2026, 2027]
        ax.set_xticks([_release_x(y) for y in years])
        ax.set_xticklabels([str(y) for y in years])

        spotlight = lb.loc[keys]
        named = record[~record.index.isin(keys) & record.index.isin(lb.index)]
        crowd = lb[~lb.index.isin(record.index) & ~lb.index.isin(keys)]
        _markers(ax, crowd, named, spotlight, colors)
        _label_all(_Labels(ax, lb), ax, spotlight, colors, named, rank, n_models)
        _legend(fig, spotlight, colors, "Earlier best model")
        return save(fig, out_dir, stem, formats)
