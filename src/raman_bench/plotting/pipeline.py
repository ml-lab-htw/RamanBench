"""Generate every leaderboard figure from an aggregated ``hpo_results.csv``.

Output layout under *out_dir*::

    leaderboards/leaderboard_{all,classification,regression}.csv
    static/<figure>.{png,pdf}
    interactive/<figure>.html
    index.html               # links every figure, static and interactive
"""

from __future__ import annotations

import html
import logging
from pathlib import Path

from raman_bench.plotting import interactive, static
from raman_bench.plotting.results import (
    DEFAULT_SCOPE,
    DEFAULT_TARGET_LIST,
    load_results,
    score_all,
    select_focus,
    split_references,
)

logger = logging.getLogger(__name__)

FIGURES = {
    "elo_ranking_combined": "Elo ranking, classification and regression",
    "elo_ranking": "Elo ranking, all tasks",
    "metrics_vs_time": "Normalized score vs. time",
    "improvability_vs_time": "Improvability vs. time",
    "elo_vs_time": "Elo vs. time",
    "elo_vs_release_date": "Model progress over time",
    "pairwise_win_rates": "Pairwise win rates",
    "efficiency_overview": "Efficiency overview",
}

LEADERBOARD_COLUMNS = [
    "display_name", "category", "elo", "elo-", "elo+", "rank", "winrate", "improvability",
    "normalized_score", "median_time_train_s", "median_time_infer_s", "median_infer_per_1k_s",
    "imputed_pct", "n_tasks", "release_date", "is_reference",
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
) -> dict[str, list[Path]]:
    """Score *hpo_results* and write leaderboards, static and interactive figures.

    ``focus_top_k`` models per category (by Elo) are drawn in colour, the rest
    in grey; ``None``/``0`` colours every model. Returns ``{figure: [paths]}``.
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
    focus_models = [m for m in all_models if m in focus["all"]]
    written["pairwise_win_rates"] = static.plot_winrate_matrix(
        scores["all"], focus_models, st, formats, "pairwise_win_rates"
    )
    if len(focus_models) < len(all_models):
        written["pairwise_win_rates"] += static.plot_winrate_matrix(
            scores["all"], all_models, st, formats, "pairwise_win_rates_all"
        )
    written["efficiency_overview"] = static.plot_efficiency_overview(
        scores["all"], everyone["all"], st, formats, "efficiency_overview"
    )

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
            "pairwise_win_rates": interactive.winrate_matrix(scores["all"]),
            "efficiency_overview": interactive.efficiency_overview(scores["all"], everyone["all"], None),
        }
        for stem, fig in figs.items():
            written.setdefault(stem, []).append(interactive.write(fig, it, stem, include_plotlyjs))

    index = _write_index(out_dir, scores, variant)
    written["index"] = [index]
    logger.info("Wrote %d file(s) under %s", sum(len(v) for v in written.values()), out_dir)
    return written


def _write_index(out_dir: Path, scores, variant: str) -> Path:
    """A plain gallery page linking every figure in all its formats."""
    rows = []
    for stem, title in FIGURES.items():
        links = []
        for rel in (f"interactive/{stem}.html", f"static/{stem}.png", f"static/{stem}.pdf",
                    f"static/{stem}_all.png", f"static/{stem}_all.pdf"):
            if (out_dir / rel).exists():
                links.append(f'<a href="{rel}">{html.escape(Path(rel).name)}</a>')
        if not links:
            continue
        thumb = f"static/{stem}.png"
        img = f'<img src="{thumb}" alt="">' if (out_dir / thumb).exists() else ""
        rows.append(f"<section><h2>{html.escape(title)}</h2><p>{' · '.join(links)}</p>{img}</section>")
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
</style></head><body>
<h1>RamanBench figures</h1>
<p>Variant: <b>{html.escape(variant)}</b> · {html.escape(tasks)} ·
<a href="leaderboards/">leaderboard CSVs</a></p>
{"".join(rows)}
</body></html>
"""
    path = out_dir / "index.html"
    path.write_text(page, encoding="utf-8")
    return path
