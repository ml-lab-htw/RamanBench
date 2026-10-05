"""Interactive (HTML) twins of the static leaderboard figures, drawn with Plotly.

TabArena ships an interactive twin next to every static leaderboard figure
(``tabarena.plot.interactive``). Its explorers hardcode TabArena's five model
families, so RamanBench builds its own with Plotly, keeping its categories
(Raman-specific, time-series classifiers, ...) and colours. Each page offers:

* hover details for every model (Elo with CI, rank, win rate, times, imputed share),
* the category legend: click to hide/show a category, double-click to isolate it,
* a toggle between "top models per category in colour" and "all models in colour".
"""

from __future__ import annotations

import html
from pathlib import Path

import numpy as np
import pandas as pd

from raman_bench.plotting import models as model_info
from raman_bench.plotting.results import TIME_COL, GroupScores, pareto_front, split_references
from raman_bench.plotting.static import TASK_TITLES

PLOTLY_CONFIG = {
    "displaylogo": False,
    "toImageButtonOptions": {"format": "png", "scale": 2},
}
FONT = "system-ui, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"


def _go():
    import plotly.graph_objects as go

    return go


def _hover(lb: pd.DataFrame) -> list[str]:
    def fmt(v, spec):
        return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else format(v, spec)

    out = []
    for _, r in lb.iterrows():
        out.append(
            f"<b>{r['display_name']}</b> · {r['category']}<br>"
            f"Elo {fmt(r['elo'], '.0f')} (−{fmt(r['elo-'], '.0f')} / +{fmt(r['elo+'], '.0f')})<br>"
            f"Rank {fmt(r['rank'], '.1f')} · win rate {fmt(r['winrate'] * 100, '.1f')}%<br>"
            f"Improvability {fmt(r['improvability'] * 100, '.1f')}% · "
            f"normalized score {fmt(r['normalized_score'], '.3f')}<br>"
            f"Median train {fmt(r.get('median_time_train_s'), '.3g')} s · "
            f"predict {fmt(r.get('median_time_infer_s'), '.3g')} s<br>"
            f"Imputed {fmt(r['imputed_pct'], '.1f')}% of {int(r['n_tasks'])} tasks"
        )
    return out


def _colors(models, focus: set[str]) -> tuple[list[str], list[str]]:
    full = [model_info.color(m) for m in models]
    focused = [model_info.color(m) if m in focus else model_info.MUTED_COLOR for m in models]
    return focused, full


def _layout(fig, title: str, height: int, focus_traces: list, top_k: int | None) -> None:
    """Shared layout + the focus/all toggle restyling every trace in *focus_traces*.

    *focus_traces* holds ``(trace_index, focused_colors, full_colors)``.
    """
    fig.update_layout(
        title={"text": title, "x": 0.01, "xanchor": "left"},
        height=height,
        template="plotly_white",
        font={"family": FONT, "size": 13},
        legend={"orientation": "h", "y": -0.18, "x": 0.5, "xanchor": "center", "title": None},
        margin={"t": 90, "l": 70, "r": 30, "b": 60},
        hoverlabel={"align": "left"},
    )
    if not focus_traces or not top_k:
        return
    idx = [t[0] for t in focus_traces]
    fig.update_layout(
        updatemenus=[
            {
                "type": "buttons",
                "direction": "right",
                "x": 1.0,
                "xanchor": "right",
                "y": 1.12,
                "yanchor": "bottom",
                "showactive": True,
                "buttons": [
                    {
                        "label": f"Top {top_k} per category",
                        "method": "restyle",
                        "args": [{"marker.color": [t[1] for t in focus_traces]}, idx],
                    },
                    {
                        "label": "All in colour",
                        "method": "restyle",
                        "args": [{"marker.color": [t[2] for t in focus_traces]}, idx],
                    },
                ],
            }
        ]
    )


def _reference_lines(fig, refs: pd.DataFrame, metric: str, scale: float = 1.0, **pos) -> None:
    """Dashed, labelled horizontal line per reference system (e.g. AutoGluon)."""
    for i, (_, row) in enumerate(refs.sort_values(metric, ascending=False).iterrows()):
        fig.add_hline(
            y=row[metric] * scale, line={"dash": "dash", "color": model_info.REFERENCE_COLOR, "width": 1.5},
            annotation_text=f"<i>{row['display_name']}</i>",
            annotation_position="top right" if i == 0 else "bottom right",
            annotation_font={"color": model_info.REFERENCE_COLOR, "size": 11}, **pos,
        )


def _by_category(lb: pd.DataFrame):
    for cat in sorted(lb["category"].unique(), key=model_info.category_rank):
        yield cat, lb[lb["category"] == cat]


def write(fig, out_dir: Path, stem: str, include_plotlyjs: str | bool = "cdn") -> Path:
    """Write *fig* as a self-contained ``<stem>.html``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.html"
    fig.write_html(path, include_plotlyjs=include_plotlyjs, full_html=True, config=PLOTLY_CONFIG)
    # Explain the contamination mark under the figure (see models.KNOWN_CONTAMINATION).
    spec = fig.to_json()
    notes = [
        f"{model_info.CONTAMINATION_MARK} {html.escape(model_info.display_name(m, mark=False))}: {html.escape(note)}"
        for m, note in model_info.KNOWN_CONTAMINATION.items()
        if model_info.display_name(m) in spec
    ]
    if notes:
        footer = "".join(
            f'<p style="font:12px Inter,Arial,sans-serif;color:#555;margin:4px 12px">{n}</p>' for n in notes
        )
        page = path.read_text(encoding="utf-8")
        path.write_text(page.replace("</body>", footer + "</body>", 1), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- Elo ranking


def elo_ranking(scores: dict[str, GroupScores], focus, top_k, groups: list[str]):
    """Elo bars with 95% CIs, one stacked panel per task group in *groups*."""
    from plotly.subplots import make_subplots

    go = _go()
    fig = make_subplots(
        rows=len(groups), cols=1, vertical_spacing=0.22 if len(groups) > 1 else 0.1,
        subplot_titles=[f"{TASK_TITLES[g]} ({scores[g].n_tasks} tasks)" for g in groups],
    )
    focus_traces, seen = [], set()
    for row, g in enumerate(groups, start=1):
        lb, refs = split_references(scores[g].leaderboard)
        for cat, sub in _by_category(lb):
            focused, full = _colors(sub.index, focus[g])
            fig.add_trace(
                go.Bar(
                    x=sub["display_name"], y=sub["elo"], name=cat, legendgroup=cat,
                    showlegend=cat not in seen,
                    marker={"color": focused, "line": {"width": 0},
                            "pattern": {"shape": ["/" if p > 0 else "" for p in sub["imputed_pct"]]}},
                    error_y={"type": "data", "symmetric": False, "array": sub["elo+"],
                             "arrayminus": sub["elo-"], "color": "#444444", "thickness": 1, "width": 2},
                    hovertext=_hover(sub), hoverinfo="text",
                ),
                row=row, col=1,
            )
            seen.add(cat)
            focus_traces.append((len(fig.data) - 1, focused, full))
        fig.update_xaxes(
            categoryorder="array", categoryarray=list(lb["display_name"]), tickangle=-55, row=row, col=1
        )
        # add_hline skips subplots without traces, so the lines go in after the bars.
        _reference_lines(fig, refs, "elo", row=row, col=1)
        low = (lb["elo"] - lb["elo-"]).min()
        high = max((lb["elo"] + lb["elo+"]).max(), refs["elo"].max() if len(refs) else -np.inf)
        fig.update_yaxes(title_text="Elo", range=[max(0, np.floor((low - 50) / 100) * 100), high + 60],
                         row=row, col=1)
        fig.add_hline(y=1000, line={"dash": "dash", "color": "#9A9A9A", "width": 1}, row=row, col=1)
    fig.update_layout(bargap=0.2)
    _layout(fig, "RamanBench Elo ranking (Random Forest = 1000)", 520 * len(groups), focus_traces, top_k)
    return fig


# --------------------------------------------------------------------------- trade-off


def tradeoff(scores: dict[str, GroupScores], focus, top_k, metric: str):
    """Score vs median train+predict time, one panel per task type, with the Pareto front."""
    from plotly.subplots import make_subplots

    go = _go()
    label, higher = {
        "normalized_score": ("Normalized score", True),
        "improvability": ("Improvability (%)", False),
        "elo": ("Elo", True),
    }[metric]
    groups = [g for g in ("classification", "regression") if g in scores] or ["all"]
    fig = make_subplots(rows=1, cols=len(groups), subplot_titles=[TASK_TITLES[g] for g in groups],
                        horizontal_spacing=0.08)
    focus_traces, seen = [], set()
    x = TIME_COL
    for col, g in enumerate(groups, start=1):
        lb, refs = split_references(scores[g].leaderboard)
        lb = lb.dropna(subset=[x, metric]).copy()
        scale = 100.0 if metric == "improvability" else 1.0
        front = pareto_front(lb, x, metric, higher)
        fig.add_trace(
            go.Scatter(x=front[x], y=front[metric] * scale, mode="lines", line_shape="hv",
                       line={"dash": "dash", "color": "#9A9A9A", "width": 1},
                       hoverinfo="skip", showlegend=False),
            row=1, col=col,
        )
        labelled = focus[g] | set(front.index)
        for cat, sub in _by_category(lb):
            focused, full = _colors(sub.index, labelled)  # Pareto-optimal models stay coloured
            fig.add_trace(
                go.Scatter(
                    x=sub[x], y=sub[metric] * scale, mode="markers+text", name=cat, legendgroup=cat,
                    showlegend=cat not in seen,
                    text=[n if m in labelled else "" for m, n in zip(sub.index, sub["display_name"])],
                    textposition="top center", textfont={"size": 10, "color": "#444444"},
                    marker={"color": focused, "size": 11, "line": {"width": 1, "color": "white"}},
                    hovertext=_hover(sub), hoverinfo="text",
                ),
                row=1, col=col,
            )
            seen.add(cat)
            focus_traces.append((len(fig.data) - 1, focused, full))
        _reference_lines(fig, refs, metric, scale, row=1, col=col)  # after the traces (see elo_ranking)
        fig.update_xaxes(type="log", title_text="Median train + predict time per 1K spectra (s)", row=1, col=col)
        fig.update_yaxes(title_text=label, row=1, col=col)
    arrow = "↖ better" if higher else "↙ better"
    _layout(fig, f"{label} vs. time  ({arrow})", 620, focus_traces, top_k)
    return fig


# --------------------------------------------------------------------------- Elo vs release date


def elo_vs_release_date(scores: GroupScores, focus: set[str], top_k):
    """Elo over release date; the staircase is the best Elo released so far."""
    go = _go()
    lb = split_references(scores.leaderboard)[0].dropna(subset=["release_date"]).sort_values("release_date")
    record = lb[lb["elo"] > lb["elo"].cummax().shift(fill_value=-np.inf)]
    fig = go.Figure()
    end = lb["release_date"].max() + 0.3
    fig.add_trace(
        go.Scatter(x=list(record["release_date"]) + [end], y=list(record["elo"]) + [record["elo"].iloc[-1]],
                   mode="lines", line_shape="hv", line={"color": "#9A9A9A", "width": 2},
                   name="State of the art", hoverinfo="skip")
    )
    highlight = focus | set(record.index)
    focus_traces = []
    for cat, sub in _by_category(lb):
        focused, full = _colors(sub.index, highlight)
        fig.add_trace(
            go.Scatter(
                x=sub["release_date"], y=sub["elo"], mode="markers+text", name=cat, legendgroup=cat,
                text=[n if m in highlight else "" for m, n in zip(sub.index, sub["display_name"])],
                textposition="top center", textfont={"size": 10, "color": "#444444"},
                marker={"color": focused, "size": 11, "line": {"width": 1, "color": "white"}},
                hovertext=_hover(sub), hoverinfo="text",
            )
        )
        focus_traces.append((len(fig.data) - 1, focused, full))
    fig.add_hline(y=1000, line={"dash": "dot", "color": "#9A9A9A", "width": 1})
    fig.update_xaxes(title_text="Model release", range=[2013.5, end + 0.3])
    fig.update_yaxes(title_text="Elo")
    _layout(fig, f"Model progress over time ({TASK_TITLES[scores.name].lower()}; drag to zoom, older models left)",
            600, focus_traces, top_k)
    return fig


# --------------------------------------------------------------------------- win rates


def winrate_matrix(scores: GroupScores, models: list[str] | None = None, caption: str | None = None):
    """Pairwise win-rate heatmap of *models* (default: all), ordered by Elo."""
    go = _go()
    lb = split_references(scores.leaderboard)[0]
    if models is not None:
        lb = lb.loc[[m for m in lb.index if m in set(models)]]
    names = list(lb["display_name"])
    wr = scores.winrate_matrix.loc[lb.index, lb.index].to_numpy(dtype=float) * 100
    np.fill_diagonal(wr, np.nan)
    hover = [[f"<b>{names[i]}</b> beats <b>{names[j]}</b><br>on {wr[i, j]:.1f}% of tasks" if i != j else ""
              for j in range(len(names))] for i in range(len(names))]
    fig = go.Figure(
        go.Heatmap(z=wr, x=names, y=names, colorscale="RdYlGn", zmin=0, zmax=100,
                   hovertext=hover, hoverinfo="text", colorbar={"title": "Win rate (%)"},
                   xgap=1, ygap=1)
    )
    size = max(600, 18 * len(names) + 250)
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(side="top", tickangle=-55)
    title = f"Pairwise win rates: row vs. column ({TASK_TITLES[scores.name].lower()}, {scores.n_tasks} tasks)"
    if caption:
        title += f"<br><sup>{caption}</sup>"
    _layout(fig, title, size, [], None)
    fig.update_layout(width=size + 100, margin={"t": 220})
    return fig


# --------------------------------------------------------------------------- efficiency


def efficiency_overview(scores: GroupScores, focus: set[str], top_k):
    """Median train time and inference time per 1K samples."""
    from plotly.subplots import make_subplots

    go = _go()
    lb = split_references(scores.leaderboard)[0]
    lb = lb.dropna(subset=["median_time_train_s"]).sort_values("median_time_train_s")
    infer = "median_infer_per_1k_s" if "median_infer_per_1k_s" in lb else "median_time_infer_s"
    panels = [("median_time_train_s", "Median train time (s)"),
              (infer, "Median inference (s / 1K samples)" if infer.endswith("1k_s") else "Median inference (s)")]
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, subplot_titles=[p[1] for p in panels],
                        horizontal_spacing=0.03)
    focus_traces, seen = [], set()
    for col, (metric, _) in enumerate(panels, start=1):
        for cat, sub in _by_category(lb):
            focused, full = _colors(sub.index, focus)
            fig.add_trace(
                go.Bar(y=sub["display_name"], x=sub[metric], orientation="h", name=cat, legendgroup=cat,
                       showlegend=cat not in seen, marker={"color": focused},
                       hovertext=_hover(sub), hoverinfo="text"),
                row=1, col=col,
            )
            seen.add(cat)
            focus_traces.append((len(fig.data) - 1, focused, full))
        fig.update_xaxes(type="log", row=1, col=col)
    fig.update_yaxes(categoryorder="array", categoryarray=list(lb["display_name"])[::-1], row=1, col=1)
    _layout(fig, "Efficiency (lower is faster)", max(500, 22 * len(lb) + 200), focus_traces, top_k)
    return fig
