"""Static (PNG/PDF) leaderboard figures, drawn with matplotlib.

Each figure takes the :class:`~raman_bench.plotting.results.GroupScores` of the
task groups it shows plus a *focus* set: models in the focus set are drawn in
their category colour and labelled, every other model recedes into a grey
field. That is TabArena's focus design (``tabarena.plot.plot_pareto_focus``),
needed once the field grows past a few dozen models. An empty focus means
"highlight everything".
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from raman_bench.plotting import models as model_info  # noqa: E402
from raman_bench.plotting.results import GroupScores, pareto_front, split_references  # noqa: E402

TASK_TITLES = {"all": "All tasks", "classification": "Classification", "regression": "Regression"}
FRONT_COLOR = "#9A9A9A"


def apply_style() -> None:
    """Shared look for every static figure."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )


def save(fig, out_dir: Path, stem: str, formats=("png", "pdf"), dpi: int = 200) -> list[Path]:
    """Write *fig* as ``<out_dir>/<stem>.<fmt>`` for every format and close it."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        path = out_dir / f"{stem}.{fmt}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def _fill(model: str, focus: set[str]) -> str:
    return model_info.color(model) if model in focus else model_info.MUTED_COLOR


def _text(model: str, focus: set[str]) -> str:
    return model_info.label_color(model) if model in focus else model_info.MUTED_LABEL_COLOR


def _category_legend(fig, models, y: float = -0.02, muted: bool = False) -> None:
    cats = sorted({model_info.category(m) for m in models}, key=model_info.category_rank)
    handles = [Patch(facecolor=model_info.CATEGORY_COLORS[c], label=c) for c in cats]
    if muted:
        handles.append(Patch(facecolor=model_info.MUTED_COLOR, label="Other models"))
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, y),
        ncol=min(len(handles), 5),
        frameon=False,
        handlelength=1.2,
        columnspacing=1.2,
    )


def _reference_lines(ax, refs: pd.DataFrame, col: str, scale: float = 1.0) -> None:
    """A dashed horizontal line per reference system, labelled at the right edge."""
    # Highest line labelled above, the others below, so close lines keep readable labels.
    for i, (_, row) in enumerate(refs.sort_values(col, ascending=False).iterrows()):
        y = row[col] * scale
        ax.axhline(y, color=model_info.REFERENCE_COLOR, ls="--", lw=1.2, zorder=1.5)
        ax.annotate(
            row["display_name"], (1.0, y), xycoords=("axes fraction", "data"), xytext=(-4, 3 if i == 0 else -3),
            textcoords="offset points", ha="right", va="bottom" if i == 0 else "top", fontsize=9,
            color=model_info.REFERENCE_COLOR, style="italic",
        )


# --------------------------------------------------------------------------- Elo ranking


def _elo_bars(ax, lb: pd.DataFrame, focus: set[str], title: str, show_labels: bool = True) -> None:
    lb, refs = split_references(lb)
    x = np.arange(len(lb))
    for i, (model, row) in enumerate(lb.iterrows()):
        imputed = row["imputed_pct"] > 0
        ax.bar(
            i,
            row["elo"],
            width=0.75,
            color=_fill(model, focus),
            hatch="///" if imputed else None,
            edgecolor="white" if imputed else "none",
            linewidth=0,
            zorder=2,
        )
    ax.errorbar(
        x,
        lb["elo"],
        yerr=[lb["elo-"], lb["elo+"]],
        fmt="none",
        ecolor="#444444",
        elinewidth=0.8,
        capsize=2,
        zorder=3,
    )
    ax.axhline(1000, color=FRONT_COLOR, lw=0.8, ls="--", zorder=1)
    _reference_lines(ax, refs, "elo")
    low = (lb["elo"] - lb["elo-"]).min()
    high = max((lb["elo"] + lb["elo+"]).max(), refs["elo"].max() if len(refs) else -np.inf)
    ax.set_ylim(max(0, np.floor((low - 50) / 100) * 100), high + 60)
    ax.set_xlim(-0.6, len(lb) - 0.4)
    ax.set_ylabel("Elo")
    ax.set_title(f"{title}  ({int(lb['n_tasks'].max())} tasks)", loc="left", fontweight="bold")
    ax.grid(axis="x", visible=False)
    ax.set_xticks(x)
    if show_labels:
        ax.set_xticklabels(lb["display_name"], rotation=60, ha="right", rotation_mode="anchor")
        for tick, model in zip(ax.get_xticklabels(), lb.index):
            tick.set_color("#222222" if model in focus else model_info.MUTED_LABEL_COLOR)
    else:
        ax.set_xticklabels([])


def plot_elo_ranking(scores: dict[str, GroupScores], focus: dict[str, set[str]], out_dir: Path, formats):
    """``elo_ranking`` (all tasks) and ``elo_ranking_combined`` (classification / regression panels)."""
    paths = []
    lb_all = scores["all"].leaderboard
    lb, _ = split_references(lb_all)
    fig, ax = plt.subplots(figsize=(max(8, 0.24 * len(lb) + 2), 4.2))
    _elo_bars(ax, lb_all, focus["all"], TASK_TITLES["all"])
    _category_legend(fig, lb.index, y=-0.18, muted=len(focus["all"]) < len(lb))
    _imputed_note(fig, lb)
    paths += save(fig, out_dir, "elo_ranking", formats)

    groups = [g for g in ("classification", "regression") if g in scores]
    if groups:
        n_max = max(len(split_references(scores[g].leaderboard)[0]) for g in groups)
        fig, axes = plt.subplots(len(groups), 1, figsize=(max(8, 0.24 * n_max + 2), 4.0 * len(groups)))
        axes = np.atleast_1d(axes)
        models = set()
        for ax, g in zip(axes, groups):
            _elo_bars(ax, scores[g].leaderboard, focus[g], TASK_TITLES[g])
            lb_g, _ = split_references(scores[g].leaderboard)
            ax.set_xlim(-0.6, n_max - 0.4)
            models |= set(lb_g.index)
        fig.tight_layout(h_pad=1.5)
        muted = any(len(focus[g]) < len(scores[g].leaderboard) for g in groups)
        _category_legend(fig, models, y=0.0, muted=muted)
        paths += save(fig, out_dir, "elo_ranking_combined", formats)
    return paths


def _imputed_note(fig, lb: pd.DataFrame) -> None:
    if (lb["imputed_pct"] > 0).any():
        fig.text(
            0.99, 0.99, "hatched: some results imputed with Random Forest",
            ha="right", va="top", fontsize=8, color="#666666",
        )


# --------------------------------------------------------------------------- trade-off scatter


#: Label offsets (points) tried in order until a label overlaps nothing placed before it.
_LABEL_OFFSETS = [(6, 3), (6, -11), (-6, 3), (-6, -11), (0, 9), (0, -15), (12, 12), (12, -20), (-12, 12), (-12, -20)]


def _label_points(ax, df: pd.DataFrame, x: str, y: str, focus: set[str], avoid: pd.DataFrame | None = None) -> None:
    """Label *df*'s points next to them, trying :data:`_LABEL_OFFSETS` to avoid overlaps.

    Deterministic and scale-agnostic (works in display space, so log axes are fine).
    *avoid* holds every plotted point; a label may not cover one of those markers either.
    """
    fig = ax.figure
    fig.canvas.draw()  # fixes the transforms
    renderer = fig.canvas.get_renderer()
    avoid = df if avoid is None else avoid
    marker_px = ax.transData.transform(avoid[[x, y]].to_numpy(dtype=float))
    placed = []
    # Highest points first: their labels claim space above, the rest settle around them.
    for model, row in df.sort_values(y, ascending=False).iterrows():
        own = ax.transData.transform([[row[x], row[y]]])[0]
        # Pass 1: free of labels and markers. Pass 2: free of labels only. Else: first offset.
        for allow_markers in (False, True):
            for dx, dy in _LABEL_OFFSETS:
                ha = "center" if dx == 0 else ("left" if dx > 0 else "right")
                text = ax.annotate(
                    row["display_name"], (row[x], row[y]), xytext=(dx, dy), textcoords="offset points",
                    ha=ha, fontsize=8.5, color=_text(model, focus), zorder=4,
                )
                text.update_positions(renderer)  # apply the offset before measuring
                box = text.get_window_extent(renderer).expanded(1.02, 1.1)
                hits_marker = not allow_markers and any(
                    box.contains(px, py) and not np.allclose((px, py), own) for px, py in marker_px
                )
                if not hits_marker and not any(box.overlaps(o) for o in placed):
                    placed.append(box)
                    break
                text.remove()
            else:
                continue
            break
        else:
            text = ax.annotate(
                row["display_name"], (row[x], row[y]), xytext=_LABEL_OFFSETS[0], textcoords="offset points",
                fontsize=8.5, color=_text(model, focus), zorder=4,
            )
            placed.append(text.get_window_extent(renderer))


def _tradeoff_ax(ax, lb, metric, label, higher_is_better, focus, title):
    x = "median_time_total_s"
    lb, refs = split_references(lb)
    df = lb.dropna(subset=[x, metric])
    front = pareto_front(df, x, metric, higher_is_better)
    _reference_lines(ax, refs, metric)
    # Pareto-optimal models always keep their colour: they are the trade-off's answer.
    highlight = focus | set(front.index)
    muted = df[~df.index.isin(highlight)]
    shown = df[df.index.isin(highlight)]
    ax.scatter(muted[x], muted[metric], s=28, color=model_info.MUTED_COLOR, zorder=2, linewidths=0)
    ax.scatter(
        shown[x], shown[metric], s=60, zorder=3, linewidths=0.6, edgecolors="white",
        color=[model_info.color(m) for m in shown.index],
    )
    ax.step(front[x], front[metric], where="post", color=FRONT_COLOR, ls="--", lw=1, zorder=1)
    ax.set_xscale("log")
    ax.set_xlabel("Median time per task: train + predict (s)")
    ax.set_ylabel(label)
    ax.set_title(f"{title}  ({'upper' if higher_is_better else 'lower'} left is better)")
    # Labels are placed in display space, so the caller adds them once the layout is final.
    return lambda: _label_points(ax, shown, x, metric, highlight, avoid=df)


def plot_tradeoff(scores, focus, out_dir: Path, formats, metric: str, stem: str):
    """Score vs time, one panel per task type, with the Pareto front."""
    label, higher = {
        "normalized_score": ("Normalized score", True),
        "improvability": ("Improvability", False),
        "elo": ("Elo", True),
    }[metric]
    groups = [g for g in ("classification", "regression") if g in scores] or ["all"]
    fig, axes = plt.subplots(1, len(groups), figsize=(7.5 * len(groups), 5.2))
    axes = np.atleast_1d(axes)
    models, labelers = set(), []
    for ax, g in zip(axes, groups):
        lb = scores[g].leaderboard
        models |= set(split_references(lb)[0].index)
        labelers.append(_tradeoff_ax(ax, lb, metric, label, higher, focus[g], TASK_TITLES[g]))
        if metric == "improvability":
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    fig.tight_layout()
    for add_labels in labelers:
        add_labels()
    _category_legend(fig, models, y=0.0, muted=True)
    return save(fig, out_dir, stem, formats)


# --------------------------------------------------------------------------- Elo vs release date

_BREAK_YEAR = 2014.0
_SQUEEZE = 0.12


def _release_x(year: float) -> float:
    """Squeeze the decades before :data:`_BREAK_YEAR` so recent releases get the room."""
    return year if year >= _BREAK_YEAR else _BREAK_YEAR - (_BREAK_YEAR - year) * _SQUEEZE


def plot_elo_vs_release_date(scores: GroupScores, focus: set[str], out_dir: Path, formats, stem: str):
    """Elo over model release date with the running best ("state of the art") as a staircase."""
    lb = split_references(scores.leaderboard)[0].dropna(subset=["release_date"]).sort_values("release_date")
    lb = lb.assign(x=[_release_x(d) for d in lb["release_date"]])
    record = lb[lb["elo"] > lb["elo"].cummax().shift(fill_value=-np.inf)]

    fig, ax = plt.subplots(figsize=(9, 5))
    stairs_x = list(record["x"]) + [_release_x(lb["release_date"].max() + 0.3)]
    stairs_y = list(record["elo"]) + [record["elo"].iloc[-1]]
    ax.step(stairs_x, stairs_y, where="post", color=FRONT_COLOR, lw=1.5, zorder=1)
    for model, row in lb.iterrows():
        highlighted = model in focus or model in record.index
        ax.scatter(
            row["x"], row["elo"], s=70 if highlighted else 30, zorder=3 if highlighted else 2,
            color=model_info.color(model) if highlighted else model_info.MUTED_COLOR,
            edgecolors="white", linewidths=0.6,
        )
    _label_points(ax, lb[lb.index.isin(focus | set(record.index))], "x", "elo", focus | set(record.index), avoid=lb)
    ax.axhline(1000, color=FRONT_COLOR, lw=0.8, ls=":")

    major = [1960, 1980, 2000] + list(range(2016, int(lb["release_date"].max()) + 2, 2))
    ax.set_xticks([_release_x(y) for y in major])
    ax.set_xticklabels([str(y) for y in major])
    bx = _release_x(_BREAK_YEAR)
    ax.axvline(bx, color="#DDDDDD", lw=6, zorder=0)
    ax.set_xlabel("Model release")
    ax.set_ylabel("Elo")
    ax.set_title(f"Model progress over time ({TASK_TITLES[scores.name].lower()})")
    _category_legend(fig, lb.index, y=-0.02, muted=True)
    return save(fig, out_dir, stem, formats)


# --------------------------------------------------------------------------- win-rate matrix


def plot_winrate_matrix(
    scores: GroupScores, models: list[str], out_dir: Path, formats, stem: str, caption: str | None = None
):
    """Pairwise win rates of *models* (rows beat columns), ordered by Elo; *caption* goes under the matrix."""
    wr = scores.winrate_matrix.loc[models, models] * 100
    names = scores.leaderboard.loc[models, "display_name"]
    n = len(models)
    size = max(6, 0.32 * n + 2)
    fig, ax = plt.subplots(figsize=(size + 1, size))
    data = wr.to_numpy(dtype=float)
    np.fill_diagonal(data, np.nan)
    im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=100, aspect="equal")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=60, ha="left", rotation_mode="anchor")
    ax.set_yticklabels(names)
    ax.xaxis.tick_top()
    for labels in (ax.get_xticklabels(), ax.get_yticklabels()):
        for tick, model in zip(labels, models):
            tick.set_color(model_info.label_color(model))
    ax.grid(False)
    if n <= 32:
        for i in range(n):
            for j in range(n):
                if i != j:
                    ax.text(j, i, f"{data[i, j]:.0f}", ha="center", va="center", fontsize=6.5, color="#222222")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Win rate of row vs. column (%)")
    ax.set_title(f"Pairwise win rates ({TASK_TITLES[scores.name].lower()}, {scores.n_tasks} tasks)", pad=12)
    if caption:
        fig.text(0.5, 0.0, caption, ha="center", va="top", fontsize=9, color="#555555", wrap=True)
    return save(fig, out_dir, stem, formats)


# --------------------------------------------------------------------------- efficiency


def plot_efficiency_overview(scores: GroupScores, focus: set[str], out_dir: Path, formats, stem: str):
    """Median train time and median inference time per 1K samples, one bar per model."""
    lb = split_references(scores.leaderboard)[0]
    lb = lb.dropna(subset=["median_time_train_s"]).sort_values("median_time_train_s")
    panels = [("median_time_train_s", "Median train time (s)")]
    if "median_infer_per_1k_s" in lb:
        panels.append(("median_infer_per_1k_s", "Median inference (s / 1K samples)"))
    else:
        panels.append(("median_time_infer_s", "Median inference time (s)"))
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), max(5, 0.24 * len(lb) + 1.5)), sharey=True)
    y = np.arange(len(lb))
    for ax, (col, title) in zip(np.atleast_1d(axes), panels):
        ax.barh(y, lb[col], color=[_fill(m, focus) for m in lb.index], height=0.75, zorder=2)
        ax.set_xscale("log")
        ax.set_title(title, fontweight="bold")
        ax.grid(axis="y", visible=False)
    axes = np.atleast_1d(axes)
    axes[0].invert_yaxis()  # shared y: inverting once flips every panel
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(lb["display_name"])
    for tick, model in zip(axes[0].get_yticklabels(), lb.index):
        tick.set_color("#222222" if model in focus else model_info.MUTED_LABEL_COLOR)
    fig.tight_layout()
    _category_legend(fig, lb.index, y=0.0, muted=len(focus) < len(lb))
    return save(fig, out_dir, stem, formats)
