"""Generate every leaderboard figure from an aggregated ``hpo_results.csv``.

Output layout under *out_dir*::

    leaderboards/leaderboard_{all,classification,regression}.csv
    leaderboards/datasets.csv  # one row per benchmarked dataset (Figures 1 and 3)
    static/<figure>.{png,pdf}
    interactive/<figure>.html
    index.html               # links every figure, static and interactive
"""

from __future__ import annotations

import html
import logging
from pathlib import Path

from raman_bench.plotting import interactive, overview, static
from raman_bench.plotting import models as model_info
from raman_bench.plotting.results import (
    DEFAULT_SCOPE,
    DEFAULT_TARGET_LIST,
    MIN_CD_TASKS,
    NEAR_PARETO_TOLERANCE,
    critical_difference,
    load_results,
    pareto_selection,
    score_all,
    select_focus,
    split_references,
)

logger = logging.getLogger(__name__)

FIGURES = {
    "overview": "Figure 1: RamanBench overview",
    "raman_examples": "Figure 2: Raman spectra examples",
    "composition": "Figure 3: Benchmark composition",
    "elo_ranking_combined": "Elo ranking, classification and regression",
    "elo_ranking": "Elo ranking, all tasks",
    "metrics_vs_time": "Normalized score vs. time",
    "improvability_vs_time": "Improvability vs. time",
    "elo_vs_time": "Elo vs. time",
    "elo_vs_release_date": "Model progress over time",
    "pairwise_win_rates": "Pairwise win rates",
    "efficiency_overview": "Efficiency overview",
    "critical_difference": "Critical difference diagrams",
}

LEADERBOARD_COLUMNS = [
    "display_name", "category", "elo", "elo-", "elo+", "rank", "winrate", "improvability",
    "normalized_score", "median_time_train_s", "median_time_infer_s", "median_infer_per_1k_s",
    "median_time_total_per_1k_s",
    "imputed_pct", "n_tasks", "evaluated_on", "release_date", "is_reference", "contamination",
]


def generate_all(
    hpo_results: str | Path,
    out_dir: str | Path,
    *,
    variant: str = "default",
    focus_top_k: int | None = 2,
    formats: tuple[str, ...] = ("png", "pdf"),
    make_interactive: bool = True,
    include_plotlyjs: str | bool = "cdn",
    scope: str | Path | None = DEFAULT_SCOPE,
    target_list: str | Path | None = DEFAULT_TARGET_LIST,
    reference_model: str = "RF",
    max_imputed_pct: float = 50.0,
    bootstrap_rounds: int = 200,
    exclude_models: tuple[str, ...] = ("DUMMY",),
    dataset_figures: bool = True,
) -> dict[str, list[Path]]:
    """Score *hpo_results* and write leaderboards, static and interactive figures.

    ``focus_top_k`` models per category (by Elo) are drawn in colour, the rest
    in grey; ``None``/``0`` colours every model. ``dataset_figures`` adds Figures 1-3
    (:mod:`raman_bench.plotting.overview`), which read dataset metadata from raman_data
    and the RamanBench mirror. Returns ``{figure: [paths]}``.
    """
    out_dir = Path(out_dir)
    results = load_results(
        hpo_results, variant=variant, scope=scope, target_list=target_list, exclude_models=exclude_models
    )
    scores = score_all(
        results,
        reference_model=reference_model,
        max_imputed_pct=max_imputed_pct,
        bootstrap_rounds=bootstrap_rounds,
    )
    focus = {g: select_focus(s.leaderboard, focus_top_k) for g, s in scores.items()}
    # The Elo ranking and the efficiency overview list every model by name, so nothing is greyed out.
    everyone = {g: set(s.leaderboard.index) for g, s in scores.items()}
    written: dict[str, list[Path]] = {}

    lb_dir = out_dir / "leaderboards"
    lb_dir.mkdir(parents=True, exist_ok=True)
    for g, s in scores.items():
        path = lb_dir / f"leaderboard_{g}.csv"
        cols = [c for c in LEADERBOARD_COLUMNS if c in s.leaderboard.columns]
        s.leaderboard[cols].rename_axis("model").to_csv(path, float_format="%.6g")
        written.setdefault("leaderboards", []).append(path)

    static.apply_style()
    st = out_dir / "static"
    datasets = None
    if dataset_figures and target_list is not None:
        datasets = overview.dataset_overview(scores["all"].results["dataset"].unique(), target_list)
        datasets.to_csv(lb_dir / "datasets.csv", index=False)
        written["leaderboards"].append(lb_dir / "datasets.csv")
        written["overview"] = overview.plot_overview(datasets, scores["all"], st, formats)
        written["raman_examples"] = overview.plot_raman_examples(st, formats)
        written["composition"] = overview.plot_composition(datasets, st, formats)
    written["elo_ranking"] = static.plot_elo_ranking(scores, everyone, st, formats)
    for metric, stem in (
        ("normalized_score", "metrics_vs_time"),
        ("improvability", "improvability_vs_time"),
        ("elo", "elo_vs_time"),
    ):
        written[stem] = static.plot_tradeoff(scores, focus, st, formats, metric, stem)
    written["elo_vs_release_date"] = static.plot_elo_vs_release_date(
        scores["all"], focus["all"], st, formats, "elo_vs_release_date"
    )
    all_models = list(split_references(scores["all"].leaderboard)[0].index)
    # Pareto/near-Pareto models plus the top-k per category by Elo over all tasks.
    top_k = focus_top_k or 2
    winrate_set = pareto_selection(scores) | select_focus(scores["all"].leaderboard, top_k)
    pareto_models = [m for m in all_models if m in winrate_set]
    captions = {"pairwise_win_rates": _winrate_caption(len(pareto_models), len(all_models), top_k)}
    written["pairwise_win_rates"] = static.plot_winrate_matrix(
        scores["all"], pareto_models, st, formats, "pairwise_win_rates", captions["pairwise_win_rates"]
    )
    written["pairwise_win_rates"] += static.plot_winrate_matrix(
        scores["all"], all_models, st, formats, "pairwise_win_rates_all"
    )
    written["efficiency_overview"] = static.plot_efficiency_overview(
        scores["all"], everyone["all"], st, formats, "efficiency_overview"
    )
    cds = [critical_difference(scores[g]) for g in ("classification", "regression")
           if g in scores and scores[g].n_tasks >= MIN_CD_TASKS]
    if cds:
        written["critical_difference"] = static.plot_critical_difference(cds, scores, st, formats)

    if make_interactive:
        it = out_dir / "interactive"
        groups = [g for g in ("classification", "regression") if g in scores]
        figs = {
            "elo_ranking_combined": interactive.elo_ranking(scores, everyone, None, groups or ["all"]),
            "elo_ranking": interactive.elo_ranking(scores, everyone, None, ["all"]),
            "metrics_vs_time": interactive.tradeoff(scores, focus, focus_top_k, "normalized_score"),
            "improvability_vs_time": interactive.tradeoff(scores, focus, focus_top_k, "improvability"),
            "elo_vs_time": interactive.tradeoff(scores, focus, focus_top_k, "elo"),
            "elo_vs_release_date": interactive.elo_vs_release_date(scores["all"], focus["all"], focus_top_k),
            "pairwise_win_rates": interactive.winrate_matrix(
                scores["all"], pareto_models, captions["pairwise_win_rates"]
            ),
            "pairwise_win_rates_all": interactive.winrate_matrix(scores["all"]),
            "efficiency_overview": interactive.efficiency_overview(scores["all"], everyone["all"], None),
        }
        if cds:
            figs["critical_difference"] = interactive.critical_difference(cds, scores)
        if datasets is not None:
            figs["overview"] = overview.overview_interactive(datasets, scores["all"])
            figs["composition"] = overview.composition_interactive(datasets)
        for stem, fig in figs.items():
            written.setdefault(stem, []).append(interactive.write(fig, it, stem, include_plotlyjs))

    index = _write_index(out_dir, scores, variant, captions)
    written["index"] = [index]
    logger.info("Wrote %d file(s) under %s", sum(len(v) for v in written.values()), out_dir)
    return written


def _winrate_caption(n_shown: int, n_total: int, top_k: int) -> str:
    return (
        f"Shown: {n_shown} of {n_total} models. These are the top {top_k} per model category by Elo "
        f"over all tasks, plus every model that is Pareto-optimal or within {NEAR_PARETO_TOLERANCE:g} "
        "normalized score of the Pareto front in normalized score vs. median train + predict time per 1K spectra "
        "(classification or regression). pairwise_win_rates_all has every model."
    )


def _contamination_notes(scores) -> str:
    """Footnotes for flagged models present in any leaderboard."""
    flagged = sorted({m for s in scores.values() for m in s.leaderboard.index if model_info.contamination(m)})
    return "".join(
        f'<p class="caption">{model_info.CONTAMINATION_MARK} {html.escape(model_info.display_name(m, mark=False))}: '
        f"{html.escape(model_info.contamination(m))}</p>"
        for m in flagged
    )


def _write_index(out_dir: Path, scores, variant: str, captions: dict[str, str] | None = None) -> Path:
    """A plain gallery page linking every figure in all its formats."""
    captions = captions or {}
    rows = []
    for stem, title in FIGURES.items():
        links = []
        for rel in (f"interactive/{stem}.html", f"static/{stem}.png", f"static/{stem}.pdf",
                    f"interactive/{stem}_all.html", f"static/{stem}_all.png", f"static/{stem}_all.pdf"):
            if (out_dir / rel).exists():
                links.append(f'<a href="{rel}">{html.escape(Path(rel).name)}</a>')
        if not links:
            continue
        thumb = f"static/{stem}.png"
        img = f'<img src="{thumb}" alt="">' if (out_dir / thumb).exists() else ""
        caption = f'<p class="caption">{html.escape(captions[stem])}</p>' if stem in captions else ""
        rows.append(f"<section><h2>{html.escape(title)}</h2><p>{' · '.join(links)}</p>{caption}{img}</section>")
    tasks = ", ".join(f"{g}: {s.n_tasks} tasks, {len(s.leaderboard)} models" for g, s in scores.items())
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>RamanBench figures</title>
<style>
:root {{ color-scheme: light dark; }}
body {{ font-family: system-ui, -apple-system, "Segoe UI", sans-serif; max-width: 1100px;
       margin: 0 auto; padding: 16px; background: #fff; color: #222; }}
@media (prefers-color-scheme: dark) {{ body {{ background: #161616; color: #ddd; }} a {{ color: #8ab4f8; }} }}
img {{ max-width: 100%; border: 1px solid #ddd; background: #fff; }}
section {{ margin: 2em 0; }}
.caption {{ color: #666; font-size: 0.92em; }}
</style></head><body>
<h1>RamanBench figures</h1>
<p>Variant: <b>{html.escape(variant)}</b> · {html.escape(tasks)} ·
<a href="leaderboards/">leaderboard CSVs</a></p>
{_contamination_notes(scores)}
{"".join(rows)}
</body></html>
"""
    path = out_dir / "index.html"
    path.write_text(page, encoding="utf-8")
    return path
