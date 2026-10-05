"""Dataset-level overview figures: what RamanBench contains, rather than how models score.

* Figure 1, :func:`plot_overview` / :func:`overview_interactive`: samples vs. features of
  every RamanBench dataset next to TabArena, TALENT, UCR and UEA, beside the Elo of each
  model over its release date.
* Figure 2, :func:`plot_raman_examples`: example spectra from the four application
  domains (static only; hundreds of overlaid spectra gain little from interactivity).
* Figure 3, :func:`plot_composition` / :func:`composition_interactive`: six donuts with
  datasets and spectra by domain and task, hosting platform, and how many datasets were
  first published with RamanBench.

The dataset table behind them (:func:`dataset_overview`) comes from raman_data's
dataset registry plus the RamanBench mirror on the Hugging Face Hub.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from raman_bench.plotting import models as model_info

logger = logging.getLogger(__name__)

MIRROR_REPO = "HTW-KI-Werkstatt/RamanBench"
REFERENCE_BENCHMARKS = Path(__file__).parent / "data" / "reference_benchmarks.csv"

DOMAIN_ORDER = ["MaterialScience", "Biological", "Medical", "Chemical"]
DOMAIN_LABELS = {
    "MaterialScience": "Material Science",
    "Biological": "Biological & Biotechnological",
    "Medical": "Medical & Clinical",
    "Chemical": "Chemical & Industrial",
}
DOMAIN_COLORS = {
    "MaterialScience": model_info.OKABE_ITO["blue"],
    "Biological": model_info.OKABE_ITO["bluish_green"],
    "Medical": model_info.OKABE_ITO["reddish_purple"],
    "Chemical": model_info.OKABE_ITO["orange"],
}
TASK_COLORS = {"Classification": model_info.OKABE_ITO["sky_blue"], "Regression": model_info.OKABE_ITO["vermillion"]}

SOURCE_ORDER = ["HuggingFace", "Kaggle", "Zenodo", "RWTH", "GitHub", "Mendeley", "GoogleDrive", "Figshare", "Other"]
SOURCE_COLORS = {
    "HuggingFace": "#FFD21E",
    "Kaggle": "#20BEFF",
    "Zenodo": "#1B91FF",
    "RWTH": "#00843D",
    "GitHub": "#24292E",
    "Mendeley": "#C62C2C",
    "GoogleDrive": "#4285F4",
    "Figshare": "#E87D0D",
    "Other": "#888888",
}
NEW_COLORS = {"First published with RamanBench": "#CC79A7", "Previously published": "#BBBBBB"}

BENCHMARK_STYLE = {
    # name: (colour, matplotlib marker, plotly marker)
    "TabArena": ("#E69F00", "^", "triangle-up"),
    "TALENT": ("#D55E00", "s", "square"),
    "UCR": ("#0072B2", "D", "diamond"),
    "UEA": ("#56B4E9", "P", "cross"),
}
RAMAN_COLOR = "#CC79A7"

#: Datasets whose data was first made public together with RamanBench.
FIRST_PUBLISHED_WITH_RAMANBENCH = frozenset({
    "adenine_colloidal_gold", "adenine_colloidal_silver", "adenine_solid_gold", "adenine_solid_silver",
    "ecoli_metabolites", "ecoli_metabolites_dig4bio", "fuel_benchtop", "fuel_handheld",
    "ht_raman_bio_catalysis_axp", "kaiser_ecoli_fermentation", "kaiser_ecoli_fermentation_supernatant",
    "ralstonia_fermentations", "streptococcus_thermophilus_fermentation_kaiser",
    "streptococcus_thermophilus_fermentation_timegate", "tg_ecoli_fermentation",
    "tg_ecoli_fermentation_supernatant", "yeast_fermentation",
})

#: Figure 2 panels: one dataset per domain.
EXAMPLE_PANELS = [
    {"dataset": "mlrod", "domain": "MaterialScience", "title": "Mars Analogue Minerals (MLROD)"},
    {"dataset": "yeast_fermentation", "domain": "Biological", "title": "Yeast Fermentation",
     "target": "Ethanol [mol / L]", "unit": "mol/L"},
    {"dataset": "head_neck_cancer", "domain": "Medical", "title": "Head & Neck Cancer"},
    {"dataset": "sugar_mixtures_low_snr", "domain": "Chemical", "title": "Sugar Mixtures (Low SNR)",
     "target": "Sucrose", "scale": 100, "unit": "%"},
]


# --------------------------------------------------------------------------- data


@lru_cache(maxsize=1)
def _registry() -> dict[str, tuple[object, str]]:
    """raman_data dataset id -> (DatasetInfo, hosting platform)."""
    from raman_data.loaders.FigshareLoader import FigshareLoader
    from raman_data.loaders.GitHubLoader import GitHubLoader
    from raman_data.loaders.GoogleDriveLoader import GoogleDriveLoader
    from raman_data.loaders.HuggingFaceLoader import HuggingFaceLoader
    from raman_data.loaders.KaggleLoader import KaggleLoader
    from raman_data.loaders.MendeleyLoader import MendeleyLoader
    from raman_data.loaders.MiscLoader import MiscLoader
    from raman_data.loaders.RWTHLoader import RWTHLoader
    from raman_data.loaders.ZenodoLoader import ZenodoLoader

    out = {}
    for source, loader in [
        ("HuggingFace", HuggingFaceLoader), ("Kaggle", KaggleLoader), ("Zenodo", ZenodoLoader),
        ("RWTH", RWTHLoader), ("GitHub", GitHubLoader), ("Mendeley", MendeleyLoader),
        ("GoogleDrive", GoogleDriveLoader), ("Figshare", FigshareLoader), ("Other", MiscLoader),
    ]:
        for key, info in loader.DATASETS.items():
            out.setdefault(key, (info, source))
    return out


def _is_shift(col: str) -> bool:
    try:
        float(col)
        return True
    except ValueError:
        return False


def mirror_frame(dataset: str, mirror_repo: str = MIRROR_REPO, columns: list[str] | None = None) -> pd.DataFrame:
    """The dataset's wide Parquet table from the RamanBench mirror (cached by huggingface_hub)."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(mirror_repo, f"{dataset}/data/train-00000-of-00001.parquet", repo_type="dataset")
    return pd.read_parquet(path, columns=columns)


def _n_shifts(dataset: str, mirror_repo: str) -> int:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(mirror_repo, f"{dataset}/data/train-00000-of-00001.parquet", repo_type="dataset")
    return sum(_is_shift(name) for name in pq.read_schema(path).names)


def dataset_overview(tasks, target_list: str | Path, mirror_repo: str = MIRROR_REPO) -> pd.DataFrame:
    """One row per benchmarked dataset.

    *tasks* are the scored task keys (``<dataset>__<target_idx>``). Columns:
    ``dataset``, ``name``, ``domain``, ``task_type``, ``n_spectra``, ``n_features``,
    ``n_targets``, ``source``, ``new``.
    """
    with open(target_list) as f:
        targets = pd.DataFrame(json.load(f))
    targets["key"] = targets["dataset"] + "__" + targets["target_idx"].astype(str)
    targets = targets[targets["key"].isin(set(tasks))]
    registry = _registry()

    rows = []
    for dataset, group in targets.groupby("dataset"):
        info, source = registry.get(dataset, (None, "Other"))
        rows.append({
            "dataset": dataset,
            "name": getattr(info, "name", dataset),
            "domain": info.application_type.name if info is not None else "Unknown",
            "task_type": info.task_type.name if info is not None else "Unknown",
            "n_spectra": int(group["num_instances"].max()),
            "n_features": _n_shifts(dataset, mirror_repo),
            "n_targets": len(group),
            "source": source,
            "new": dataset in FIRST_PUBLISHED_WITH_RAMANBENCH,
        })
    return pd.DataFrame(rows)


def reference_benchmarks() -> pd.DataFrame:
    """Dataset sizes of TabArena (v0.1), TALENT (basic), UCR and UEA.

    Columns ``benchmark``, ``dataset``, ``n_samples``, ``n_features``. Sources: the TabArena
    dataset-curation metadata (arXiv:2506.16791), the TALENT basic benchmark, and
    timeseriesclassification.com's dataset table (UCR = univariate, UEA = multivariate).
    """
    return pd.read_csv(REFERENCE_BENCHMARKS)


def _composition(ov: pd.DataFrame) -> list[tuple[str, list[str], list[float], list[str]]]:
    """The six donuts as ``(title, labels, values, colours)``."""
    def by(col, order, colors, labels=None, weight=None):
        counts = ov.groupby(col)[weight].sum() if weight else ov.groupby(col).size()
        keys = [k for k in order if k in counts.index]
        return [(labels or {}).get(k, k) for k in keys], [float(counts[k]) for k in keys], [colors[k] for k in keys]

    tasks = ["Classification", "Regression"]
    ov = ov.assign(provenance=ov["new"].map({True: "First published with RamanBench",
                                             False: "Previously published"}))
    return [
        ("Datasets by domain", *by("domain", DOMAIN_ORDER, DOMAIN_COLORS, DOMAIN_LABELS)),
        ("Spectra by domain", *by("domain", DOMAIN_ORDER, DOMAIN_COLORS, DOMAIN_LABELS, weight="n_spectra")),
        ("Datasets by task", *by("task_type", tasks, TASK_COLORS)),
        ("Spectra by task", *by("task_type", tasks, TASK_COLORS, weight="n_spectra")),
        ("Data sources", *by("source", SOURCE_ORDER, SOURCE_COLORS)),
        ("New vs. existing", *by("provenance", list(NEW_COLORS), NEW_COLORS)),
    ]


# --------------------------------------------------------------------------- static


def plot_overview(ov: pd.DataFrame, scores, out_dir: Path, formats, stem: str = "overview"):
    """Figure 1: (a) samples vs. features against other benchmarks, (b) Elo over release date."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from raman_bench.plotting import static

    ref = reference_benchmarks()
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"wspace": 0.25})
    handles = []
    for bench, (color, marker, _) in BENCHMARK_STYLE.items():
        sub = ref[ref["benchmark"] == bench]
        ax_a.scatter(sub["n_features"], sub["n_samples"], c=color, marker=marker, s=22, alpha=0.5, edgecolors="none")
        handles.append(Line2D([0], [0], marker=marker, color="w", markerfacecolor=color, markersize=8,
                              label=f"{bench} ({len(sub)})"))
    ax_a.scatter(ov["n_features"], ov["n_spectra"], c=RAMAN_COLOR, s=38, alpha=0.9, edgecolors="white", linewidths=0.4)
    handles.append(Line2D([0], [0], marker="o", color="w", markerfacecolor=RAMAN_COLOR, markersize=8,
                          label=f"RamanBench ({len(ov)})"))
    ax_a.axvline(ov["n_features"].min(), color="gray", lw=0.8, ls=":")
    ax_a.set_xscale("log")
    ax_a.set_yscale("log")
    ax_a.set_xlabel("Number of features")
    ax_a.set_ylabel("Number of samples")
    ax_a.set_title("(a) Samples vs. features", fontweight="bold")
    ax_a.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=False)

    static.release_ax(ax_b, scores, focus=set())
    ax_b.set_title("(b) Model progress over time", fontweight="bold")
    return static.save(fig, out_dir, stem, formats)


def plot_composition(ov: pd.DataFrame, out_dir: Path, formats, stem: str = "composition"):
    """Figure 3: six donuts with counts in the wedges and one legend per pair."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    from raman_bench.plotting import static

    donuts = _composition(ov)
    fig, axes = plt.subplots(1, len(donuts), figsize=(2.6 * len(donuts), 3.4))
    for ax, (title, labels, values, colors) in zip(axes, donuts):
        wedges, _ = ax.pie(values, colors=colors, startangle=90, counterclock=False,
                           wedgeprops={"width": 0.5, "edgecolor": "white"})
        total = sum(values)
        for wedge, value in zip(wedges, values):
            if value / total > 0.06:
                angle = np.deg2rad((wedge.theta1 + wedge.theta2) / 2)
                ax.text(0.75 * np.cos(angle), 0.75 * np.sin(angle), _short(value),
                        ha="center", va="center", fontsize=8)
        ax.set_title(title, fontsize=10, fontweight="bold")
    # One legend under each pair of donuts that share a colour scheme (domain, task),
    # and one under each of the last two.
    groups = [(0, 1), (2, 3)] + [(i, i) for i in range(4, len(donuts))]
    for first, last in groups:
        _, labels, _, colors = donuts[first]
        x = (axes[first].get_position().x0 + axes[last].get_position().x1) / 2
        fig.legend(handles=[Patch(facecolor=c, label=lab) for lab, c in zip(labels, colors)],
                   loc="upper center", bbox_to_anchor=(x, 0.24), ncol=2 if len(labels) > 4 else 1,
                   frameon=False, fontsize=8, handlelength=1.0)
    fig.subplots_adjust(bottom=0.25, wspace=0.05)
    return static.save(fig, out_dir, stem, formats)


def _short(value: float) -> str:
    """Wedge label: ``26`` or ``136k``."""
    return f"{value / 1000:.0f}k" if value >= 10_000 else f"{int(value):,}"


def _class_palette(base: str, n: int) -> list:
    import matplotlib.colors as mc

    h, s, v = mc.rgb_to_hsv(mc.to_rgb(base))
    return [mc.hsv_to_rgb((h, s * (1 - 0.5 * f), v * (0.55 + 0.45 * f))) for f in np.linspace(0, 1, max(n, 1))]


def plot_raman_examples(out_dir: Path, formats, stem: str = "raman_examples", mirror_repo: str = MIRROR_REPO):
    """Figure 2: mean ± std spectra per class (classification) or per target quartile (regression)."""
    import matplotlib.colors as mc
    import matplotlib.pyplot as plt

    from raman_bench.plotting import static

    fig, axes = plt.subplots(1, len(EXAMPLE_PANELS), figsize=(16, 3.2), gridspec_kw={"wspace": 0.12})
    for ax, panel in zip(axes, EXAMPLE_PANELS):
        df = mirror_frame(panel["dataset"], mirror_repo)
        shift_cols = sorted([c for c in df.columns if _is_shift(c)], key=float)
        shifts = np.array([float(c) for c in shift_cols])
        spectra = df[shift_cols].to_numpy(float)
        color = DOMAIN_COLORS[panel["domain"]]
        ax.set_facecolor(mc.to_rgba(color, 0.05))

        if "target" not in panel:
            labels = df["target"].astype(str).to_numpy()
            names = _class_names(panel["dataset"], mirror_repo)
            counts = pd.Series(labels).value_counts()
            shown = [c for c in counts.index if counts[c] >= 2][:4]
            curves = []
            for c, cls in zip(_class_palette(color, len(shown)), shown):
                block = spectra[labels == cls]
                mean, std = block.mean(0), block.std(0)
                ax.fill_between(shifts, mean - 0.5 * std, mean + 0.5 * std, color=c, alpha=0.15, lw=0)
                ax.plot(shifts, mean, color=c, lw=1.4, label=names.get(cls, cls)[:26])
                curves.append(mean)
            ax.legend(fontsize=7, loc="upper right", framealpha=0.75, edgecolor="none", handlelength=1.0)
        else:
            y = df[panel["target"]].astype(float).to_numpy() * panel.get("scale", 1)
            ok = ~np.isnan(y)
            order = np.argsort(y[ok])
            block, y = spectra[ok][order], y[ok][order]
            norm = mc.Normalize(y.min(), y.max())
            cmap = mc.LinearSegmentedColormap.from_list("d", [mc.to_rgba(color, 0.25), color, "#333333"])
            curves = []
            for idx in np.array_split(np.arange(len(y)), 4):
                mean, std = block[idx].mean(0), block[idx].std(0)
                curves.append(mean)
                c = cmap(norm(y[idx].mean()))
                ax.fill_between(shifts, mean - 0.5 * std, mean + 0.5 * std, color=c, alpha=0.15, lw=0)
                ax.plot(shifts, mean, color=c, lw=1.1)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            cax = ax.inset_axes([0.55, 0.86, 0.4, 0.05])
            bar = fig.colorbar(sm, cax=cax, orientation="horizontal")
            bar.set_ticks([y.min(), y.max()])
            bar.ax.tick_params(labelsize=7, length=2)
            bar.set_label(f"{panel['target'].split(' [')[0]} ({panel['unit']})", fontsize=8, labelpad=1)
            bar.outline.set_visible(False)

        # Scale to the plotted means (not single spectra), leaving headroom for legend/colour bar.
        lo, hi = np.min(curves), np.max(curves)
        ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.45 * (hi - lo))
        ax.set_xlim(shifts.min(), shifts.max())
        ax.set_yticks([])
        ax.grid(False)
        ax.set_title(f"{panel['title']}\n", fontsize=10, fontweight="bold")
        ax.text(0.5, 1.02, DOMAIN_LABELS[panel["domain"]], transform=ax.transAxes, ha="center", fontsize=9,
                color=color)
        ax.set_xlabel(r"Raman shift (cm$^{-1}$)", fontsize=9)
        ax.tick_params(axis="x", labelsize=8)
    axes[0].set_ylabel("Intensity (a.u.)")
    return static.save(fig, out_dir, stem, formats)


def _class_names(dataset: str, mirror_repo: str) -> dict[str, str]:
    """Label -> class name from the mirror's metadata.json, when it lists them."""
    from huggingface_hub import hf_hub_download

    try:
        meta = json.load(open(hf_hub_download(mirror_repo, f"{dataset}/metadata.json", repo_type="dataset")))
    except Exception:  # noqa: BLE001 -- names are cosmetic; fall back to the raw labels
        return {}
    names = meta.get("target_names")
    if isinstance(names, str):
        try:
            import ast

            names = ast.literal_eval(names)
        except (ValueError, SyntaxError):
            names = None
    return {str(i): str(n) for i, n in enumerate(names or [])}


# --------------------------------------------------------------------------- interactive


def overview_interactive(ov: pd.DataFrame, scores):
    """Interactive Figure 1."""
    from plotly.subplots import make_subplots

    from raman_bench.plotting import interactive
    from raman_bench.plotting.results import split_references

    go = interactive._go()
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.09,
                        subplot_titles=("(a) Samples vs. features", "(b) Model progress over time"))
    ref = reference_benchmarks()
    for bench, (color, _, symbol) in BENCHMARK_STYLE.items():
        sub = ref[ref["benchmark"] == bench]
        fig.add_trace(go.Scatter(
            x=sub["n_features"], y=sub["n_samples"], mode="markers", name=f"{bench} ({len(sub)})",
            marker={"color": color, "symbol": symbol, "size": 7, "opacity": 0.55},
            hovertext=[f"<b>{d}</b> ({bench})<br>{n:,} samples · {f:,} features"
                       for d, n, f in zip(sub["dataset"], sub["n_samples"], sub["n_features"])],
            hoverinfo="text",
        ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=ov["n_features"], y=ov["n_spectra"], mode="markers", name=f"RamanBench ({len(ov)})",
        marker={"color": RAMAN_COLOR, "size": 9, "line": {"width": 1, "color": "white"}},
        hovertext=[f"<b>{n}</b><br>{DOMAIN_LABELS.get(d, d)} · {t}<br>{s:,} spectra · {f:,} features"
                   for n, d, t, s, f in zip(ov["name"], ov["domain"], ov["task_type"], ov["n_spectra"],
                                            ov["n_features"])],
        hoverinfo="text", legendgroup="benchmarks",
    ), row=1, col=1)
    fig.update_xaxes(type="log", title_text="Number of features", row=1, col=1)
    fig.update_yaxes(type="log", title_text="Number of samples", row=1, col=1)

    lb = split_references(scores.leaderboard)[0].dropna(subset=["release_date"]).sort_values("release_date")
    record = lb[lb["elo"] > lb["elo"].cummax().shift(fill_value=-np.inf)]
    end = lb["release_date"].max() + 0.3
    fig.add_trace(go.Scatter(
        x=list(record["release_date"]) + [end], y=list(record["elo"]) + [record["elo"].iloc[-1]],
        mode="lines", line_shape="hv", line={"color": "#9A9A9A", "width": 2}, name="State of the art",
        hoverinfo="skip", showlegend=False,
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=lb["release_date"], y=lb["elo"], mode="markers+text", showlegend=False,
        text=[n if m in record.index else "" for m, n in zip(lb.index, lb["display_name"])],
        textposition="top left", textfont={"size": 10, "color": "#444444"},
        marker={"size": [11 if m in record.index else 7 for m in lb.index],
                "color": [model_info.color(m) if m in record.index else model_info.MUTED_COLOR for m in lb.index],
                "line": {"width": 1, "color": "white"}},
        hovertext=interactive._hover(lb), hoverinfo="text",
    ), row=1, col=2)
    fig.add_hline(y=1000, line={"dash": "dot", "color": "#9A9A9A", "width": 1}, row=1, col=2)
    fig.update_xaxes(title_text="Model release (drag to zoom; older models left)", range=[2013.5, end + 0.3],
                     row=1, col=2)
    fig.update_yaxes(title_text="Elo", row=1, col=2)
    fig.update_layout(
        template="plotly_white", height=520, margin={"l": 60, "r": 20, "t": 60, "b": 110},
        legend={"orientation": "h", "x": 0.0, "y": -0.2, "xanchor": "left"},
        font={"family": "Inter, Arial, sans-serif"},
    )
    return fig


def composition_interactive(ov: pd.DataFrame):
    """Interactive Figure 3: one legend under each pair of donuts sharing a colour scheme."""
    from plotly.subplots import make_subplots

    from raman_bench.plotting import interactive

    go = interactive._go()
    donuts = _composition(ov)
    n = len(donuts)
    fig = make_subplots(rows=1, cols=n, specs=[[{"type": "domain"}] * n], subplot_titles=[d[0] for d in donuts],
                        horizontal_spacing=0.02)
    # Donut -> legend: domain pair, task pair, sources, provenance.
    legend_of = ["legend", "legend", "legend2", "legend2", "legend3", "legend4"]
    for i, (title, labels, values, colors) in enumerate(donuts):
        unit = "spectra" if title.startswith("Spectra") else "datasets"
        fig.add_trace(go.Pie(
            labels=labels, values=values, hole=0.5, sort=False, direction="clockwise",
            marker={"colors": colors, "line": {"color": "white", "width": 1}},
            text=[_short(v) for v in values], textinfo="text", textposition="inside",
            insidetextorientation="horizontal",
            hovertemplate=f"<b>%{{label}}</b><br>%{{value:,}} {unit} (%{{percent}})<extra>{title}</extra>",
            name=title, legend=legend_of[i], showlegend=i % 2 == 0 or i >= 4,
        ), row=1, col=i + 1)
    centres = {"legend": 1 / n, "legend2": 3 / n, "legend3": 4.5 / n, "legend4": 5.5 / n}
    legends = {
        key: {"orientation": "v", "x": x, "xanchor": "center", "y": -0.02, "yanchor": "top",
              "font": {"size": 11}, "bgcolor": "rgba(0,0,0,0)"}
        for key, x in centres.items()
    }
    fig.update_layout(
        template="plotly_white", height=470, margin={"l": 10, "r": 10, "t": 50, "b": 150},
        font={"family": "Inter, Arial, sans-serif"}, uniformtext={"minsize": 9, "mode": "hide"},
        **legends,
    )
    interactive.set_mobile(fig, **_composition_mobile(n))
    return fig


def _composition_mobile(n: int, row_px: int = 370, top_px: int = 40) -> dict:
    """Phone layout for Figure 3: donuts in pairs, three rows, each legend under its row."""
    rows = (n + 1) // 2
    plot_px = rows * row_px
    domains, layout = [], {"margin.t": top_px, "margin.b": 10, "margin.l": 6, "margin.r": 6}
    for i in range(n):
        r, c = divmod(i, 2)
        top = 1 - (r * row_px + 30) / plot_px
        bottom = top - 165 / plot_px
        domains.append({"x": [0.03, 0.47] if c == 0 else [0.53, 0.97], "y": [bottom, top]})
        layout[f"annotations[{i}].x"] = 0.25 if c == 0 else 0.75
        layout[f"annotations[{i}].y"] = top
    # legend: domain pair (row 0), legend2: task pair (row 1), legend3/legend4: sources, provenance (row 2)
    for key, r, x, orient in (("legend", 0, 0.5, "h"), ("legend2", 1, 0.5, "h"),
                              ("legend3", 2, 0.25, "v"), ("legend4", 2, 0.75, "v")):
        layout.update({f"{key}.x": x, f"{key}.xanchor": "center", f"{key}.orientation": orient,
                       f"{key}.y": 1 - (r * row_px + 205) / plot_px, f"{key}.yanchor": "top",
                       f"{key}.font.size": 10})
    return {"layout": layout, "restyle": {"domain": domains}, "mobile_height": plot_px + top_px + 10}
