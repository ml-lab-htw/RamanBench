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
import json
import re
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

from raman_bench.plotting import models as model_info
from raman_bench.plotting.results import (
    TIME_COL,
    CriticalDifference,
    GroupScores,
    pareto_front,
    split_references,
)
from raman_bench.plotting.static import CD_CAPTION, TASK_TITLES, cd_geometry, cd_title

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


#: Frames narrower than this (CSS px, i.e. phones) get a figure's phone layout.
MOBILE_BREAKPOINT = 700

#: Applies ``M`` (see :func:`mobile_spec`) once Plotly has drawn, if the frame is narrow,
#: then fits the embedding iframe (same-origin ``srcdoc``) to the new page height.
_MOBILE_JS = """<script>
(function () {
  var M = %s, BP = %d;
  function go() {
    var gd = document.querySelector(".plotly-graph-div");
    if (!gd || !window.Plotly || !gd._fullLayout) { setTimeout(go, 100); return; }
    if (document.documentElement.clientWidth >= BP) return;
    var lay = Object.assign({}, M.layout || {});
    if (M.min_width) {
      lay.width = M.min_width; lay.autosize = false; document.body.style.overflowX = "auto";
      var hint = document.createElement("p");
      hint.textContent = "Swipe sideways to see the whole figure.";
      hint.style.cssText = "font:12px Inter,Arial,sans-serif;color:#777;margin:6px 12px;position:sticky;left:0";
      document.body.insertBefore(hint, document.body.firstChild);
    }
    if (M.mobile_height) { lay.height = M.mobile_height; }
    var p = Plotly.relayout(gd, lay);
    if (M.restyle) { p = p.then(function () { return Plotly.restyle(gd, M.restyle); }); }
    p.then(function () {
      var f = window.frameElement;
      if (f) { f.style.height = (document.documentElement.scrollHeight + 8) + "px"; }
    });
  }
  if (document.readyState === "complete") { go(); } else { window.addEventListener("load", go); }
})();
</script>"""


def set_mobile(fig, **spec) -> None:
    """Give *fig* an explicit phone layout instead of the automatic one (see :func:`mobile_spec`).

    ``min_width`` keeps the desktop layout at that width and lets the page scroll sideways
    (figures listing every model by name); ``layout``/``restyle``/``mobile_height`` are
    applied with ``Plotly.relayout``/``Plotly.restyle``.
    """
    fig.layout.meta = {**(fig.layout.meta or {}), "mobile": spec}


def _stack_columns(lay: dict, plot_px: float) -> tuple[dict, int]:
    """Relayout stacking a one-row grid of xy subplots into one column (``{}`` if not one)."""
    xs = [k for k in lay if re.fullmatch(r"xaxis\d*", k)]
    doms = {k: tuple(lay[k].get("domain", (0, 1))) for k in xs}
    cols = sorted(set(doms.values()))
    ys = {k: "yaxis" + k[5:] for k in xs}
    ydoms = {tuple(lay.get(ys[k], {}).get("domain", (0, 1))) for k in xs}
    if len(cols) < 2 or len(ydoms) != 1:
        return {}, 1
    n, ytop = len(cols), next(iter(ydoms))[1]
    row = plot_px / n
    top = [1 - (j * row + 34) / plot_px for j in range(n)]  # room for the panel title
    bottom = [1 - ((j + 1) * row - 72) / plot_px for j in range(n)]  # room for x ticks and title
    upd = {}
    for k in xs:
        j = cols.index(doms[k])
        upd[f"{k}.domain"] = [0, 1]
        upd[f"{ys[k]}.domain"] = [bottom[j], top[j]]
    for i, a in enumerate(lay.get("annotations", [])):  # subplot titles sit on top of their column
        if a.get("xref") == "paper" and a.get("yref") == "paper" and abs(a.get("y", -9) - ytop) < 0.03:
            j = next((c for c, (lo, hi) in enumerate(cols) if lo - 0.01 <= a.get("x", -9) <= hi + 0.01), None)
            if j is not None:
                upd[f"annotations[{i}].x"] = 0.5
                upd[f"annotations[{i}].y"] = top[j]
    return upd, n


def mobile_spec(fig, panel_px: int = 400, legend_px: int = 190) -> dict:
    """The phone layout :func:`write` embeds: *fig*'s own (:func:`set_mobile`) or an automatic one.

    Automatic: tighter margins, smaller fonts, a horizontal legend, and a one-row grid of
    subplots stacked into one column, *panel_px* per panel.
    """
    spec = dict((fig.layout.meta or {}).get("mobile", {}))
    if "min_width" in spec or "layout" in spec:
        return spec
    lay = fig.layout.to_plotly_json()
    layout = {"margin.l": 48, "margin.r": 12, "font.size": 11, "title.font.size": 14,
              "legend.orientation": "h", "legend.x": 0, "legend.xanchor": "left"}
    n = _stack_columns(lay, 1.0)[1]
    if n > 1:
        height = n * panel_px + legend_px + 90
        plot_px = height - 90 - legend_px
        layout.update(_stack_columns(lay, plot_px)[0])
        layout.update({"margin.t": 90, "margin.b": legend_px, "legend.y": -50 / plot_px, "legend.yanchor": "top"})
        spec["mobile_height"] = height
    spec["layout"] = layout
    return spec


def write(fig, out_dir: Path, stem: str, include_plotlyjs: str | bool = "cdn") -> Path:
    """Write *fig* as a self-contained ``<stem>.html`` that switches to its phone layout on narrow screens."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.html"
    fig.write_html(path, include_plotlyjs=include_plotlyjs, full_html=True, config=PLOTLY_CONFIG)
    extra = _MOBILE_JS % (json.dumps(mobile_spec(fig)), MOBILE_BREAKPOINT)
    # Explain model marks under the figure (see models.figure_notes). Decoded, so the
    # check holds whether or not plotly escapes non-ASCII ("\u2020").
    notes = model_info.figure_notes(json.dumps(json.loads(fig.to_json()), ensure_ascii=False))
    extra = "".join(
        f'<p style="font:12px Inter,Arial,sans-serif;color:#555;margin:4px 12px">{html.escape(n)}</p>'
        for n in notes
    ) + extra
    page = path.read_text(encoding="utf-8")
    path.write_text(page.replace("</body>", extra + "</body>", 1), encoding="utf-8")
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
    set_mobile(fig, min_width=max(760, 14 * max(len(split_references(scores[g].leaderboard)[0]) for g in groups)))
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
    set_mobile(fig, min_width=size + 100)
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
    set_mobile(fig, min_width=760)
    return fig


# --------------------------------------------------------------------------- critical difference


def _lines(segments) -> tuple[list, list]:
    """One trace's x/y for many polylines, separated by ``None``."""
    xs, ys = [], []
    for seg in segments:
        xs += [p[0] for p in seg] + [None]
        ys += [p[1] for p in seg] + [None]
    return xs, ys


def critical_difference(cds: list[CriticalDifference], scores: dict[str, GroupScores]):
    """Critical difference diagrams, one panel per task type, in the static figure's layout.

    Hover a model's dot or name for its mean rank, Elo and the models it is not
    significantly different from; hover a bar for its members.
    """
    from plotly.subplots import make_subplots

    go = _go()
    geoms = [cd_geometry(c) for c in cds]
    depth = [-g["bottom"] + 3.2 for g in geoms]
    fig = make_subplots(rows=len(cds), cols=1, row_heights=depth, vertical_spacing=0.06,
                        subplot_titles=[cd_title(c) for c in cds])
    seen = set()
    for row, (c, g) in enumerate(zip(cds, geoms), start=1):
        lb = scores[c.name].leaderboard
        lo, hi = g["edge"]
        k = g["k"]
        ranks = c.mean_rank
        names = lb.loc[ranks.index, "display_name"]
        # Pad the x range so the names fit beside the scale.
        pad = 0.3 * k

        def add(trace):
            fig.add_trace(trace, row=row, col=1)

        ticks = [((t, 0), (t, 0.25 if t == 1 or t % 5 == 0 else 0.12)) for t in range(1, k + 1)]
        xs, ys = _lines([((1, 0), (k, 0)), *ticks, ((1, 1.6), (1 + c.cd, 1.6)),
                         ((1, 1.45), (1, 1.75)), ((1 + c.cd, 1.45), (1 + c.cd, 1.75))])
        add(go.Scatter(x=xs, y=ys, mode="lines", line={"color": "#222222", "width": 1},
                       hoverinfo="skip", showlegend=False))
        major = [t for t in range(1, k + 1) if t == 1 or t % 5 == 0]
        add(go.Scatter(x=major + [1 + c.cd + 0.4], y=[0.55] * len(major) + [1.6], mode="text",
                       text=[str(t) for t in major] + [f"CD = {c.cd:.1f}"],
                       textposition=["top center"] * len(major) + ["middle right"],
                       textfont={"size": 11}, hoverinfo="skip", showlegend=False))
        elbows = [((r, 0), (r, y), (lo if left else hi, y)) for r, y, left in zip(ranks, g["y"], g["left"])]
        xs, ys = _lines(elbows)
        add(go.Scatter(x=xs, y=ys, mode="lines", line={"color": "#B5B5B5", "width": 1},
                       hoverinfo="skip", showlegend=False))
        for (a, b), (ra, rb, y) in zip(c.groups, g["groups"]):
            members = ", ".join(names.iloc[a:b + 1])
            add(go.Scatter(x=list(ranks.iloc[a:b + 1]), y=[y] * (b - a + 1), mode="lines+markers",
                           line={"color": "#222222", "width": 5}, marker={"size": 4, "color": "#222222"},
                           hovertext=f"<b>Not significantly different</b> ({b - a + 1} models):<br>"
                                     + "<br>".join(textwrap.wrap(members, 90)),
                           hoverinfo="text", showlegend=False))
        tied = {m: set() for m in ranks.index}
        for a, b in c.groups:
            members = list(ranks.index[a:b + 1])
            for m in members:
                tied[m] |= set(members) - {m}
        hover = {
            m: (f"<b>{names[m]}</b> · {lb.loc[m, 'category']}<br>Mean rank {ranks[m]:.2f} · "
                f"Elo {lb.loc[m, 'elo']:.0f}<br>Not significantly different from {len(tied[m])} model(s)")
            for m in ranks.index
        }
        sub = pd.DataFrame({"rank": ranks, "y": g["y"], "left": g["left"], "name": names,
                            "category": lb.loc[ranks.index, "category"]})
        for cat in sorted(sub["category"].unique(), key=model_info.category_rank):
            part = sub[sub["category"] == cat]
            color = model_info.color(part.index[0])
            add(go.Scatter(x=part["rank"], y=[0] * len(part), mode="markers", name=cat, legendgroup=cat,
                           showlegend=cat not in seen, marker={"color": color, "size": 9},
                           hovertext=[hover[m] for m in part.index], hoverinfo="text"))
            seen.add(cat)
            label_x = [lo - 0.3 if left else hi + 0.3 for left in part["left"]]
            text = [f"{n} ({r:.1f})" if left else f"({r:.1f}) {n}"
                    for n, r, left in zip(part["name"], part["rank"], part["left"])]
            add(go.Scatter(x=label_x, y=part["y"], mode="text", text=text, legendgroup=cat, showlegend=False,
                           textposition=["middle left" if left else "middle right" for left in part["left"]],
                           textfont={"color": model_info.label_color(part.index[0]), "size": 11},
                           hovertext=[hover[m] for m in part.index], hoverinfo="text"))
        fig.update_xaxes(range=[lo - pad, hi + pad], visible=False, row=row, col=1)
        fig.update_yaxes(range=[g["bottom"] - 0.8, 2.4], visible=False, row=row, col=1)
    height = int(15 * sum(depth) + 260)
    _layout(fig, "Critical difference diagrams (mean rank, 1 = best)", height, [], None)
    fig.update_layout(legend={"y": -0.02}, margin={"b": 170},
                      annotations=list(fig.layout.annotations) + [
                          {"text": "<br>".join(textwrap.wrap(CD_CAPTION, 150)), "xref": "paper", "yref": "paper",
                           "x": 0, "y": -0.075, "xanchor": "left", "yanchor": "top", "showarrow": False,
                           "font": {"size": 11, "color": "#555555"}, "align": "left"}])
    for a in fig.layout.annotations[:len(cds)]:
        a.update(x=0, xanchor="left", font={"size": 14})
    set_mobile(fig, min_width=900)  # names on both sides of the scale need the width
    return fig
