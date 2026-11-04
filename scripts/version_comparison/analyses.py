"""Eleven looks at two RamanBench versions (figures 01-10, plus 01b).

01/01b winner flow, 02 Elo rank changes, 03 progress over model release dates, 04 headroom per task,
05 winner by dataset shape, 06 winner by domain, 07 ties within noise, 08 cost of the best result,
09 regret of always using one model, 10 licences. Figures 01, 01b and 02 compare the two versions; the
others look at the newer one. See README.md for the inputs.

usage: python scripts/version_comparison/analyses.py --old OLD --new NEW --leaderboard-dir LB_DIR --out OUT_DIR
           [--old-leaderboard CSV] [--old-label v0.1] [--new-label v1] [--only 01,03] [--no-footnote]
"""
import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, PathPatch
from matplotlib.path import Path as MPath

from raman_bench.plotting import models as info

from common import (CATS, CCOL, CLAB, EXCLUDE, HOSTED, METRIC_NAMES, category, colour, dataset_of,
                    licence, load_version, name_of, normalized, restrict, shared, winners)

FIGURES = ["01", "01b", "02", "03", "04", "05", "06", "07", "08", "09", "10"]
p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
p.add_argument("--old", type=Path, required=True,
               help="older version: reference-results parquet or v0.1 precomputed_metrics dir")
p.add_argument("--new", type=Path, required=True, help="newer version: reference-results parquet")
p.add_argument("--leaderboard-dir", type=Path, required=True,
               help="the newer version's leaderboard_all.csv and datasets.csv (scripts/plot_results.py output)")
p.add_argument("--old-leaderboard", type=Path,
               help="the older version's leaderboard CSV with Elo, for figure 02 (default: "
                    "OLD/leaderboard_overall.csv for a v0.1 dir; figure 02 is skipped when there is none)")
p.add_argument("--out", type=Path, required=True)
p.add_argument("--old-label", default="v0.1")
p.add_argument("--new-label", default="v1")
p.add_argument("--only", help=f"comma-separated subset of {','.join(FIGURES)}")
p.add_argument("--no-footnote", action="store_true", help="leave out the footnotes (clean versions for posts)")
p.add_argument("--since", type=float, default=2016.5, help="figure 03: first release date shown")
p.add_argument("--regret-models", nargs="+", default=["TABPFN-V3.5", "TABFM", "TABICLV2", "CAT", "RF", "PLS"],
               help="figure 09: models compared with the leaderboard's #1 (missing ones are skipped)")
ARGS = p.parse_args()
OUT, OLD, NEW, SINCE = ARGS.out, ARGS.old_label, ARGS.new_label, ARGS.since
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titleweight": "bold", "axes.titlesize": 12,
                     "axes.spines.top": False, "axes.spines.right": False})


def save(fig, stem):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("wrote", stem)


def footnote(fig, text, y=-0.02):
    if not ARGS.no_footnote:
        fig.text(0.01, y, text, fontsize=7.8, color="#666666", ha="left", va="top", wrap=True)


def cat_legend(ax, cats=None, **kw):
    cats = cats or [k for k, *_ in CATS]
    ax.legend(handles=[Patch(fc=CCOL[c], label=CLAB[c]) for c in cats], frameon=False, **kw)


def read_leaderboard(path):
    """Elo leaderboard CSV of either format (v0.1: model_id, Elo; v1+: model, elo)."""
    lb_ = pd.read_csv(path).rename(columns={"model_id": "model", "Elo": "elo"})
    if "is_reference" in lb_:
        lb_ = lb_[~lb_["is_reference"].astype(bool)]
    return lb_[~lb_["model"].isin(EXCLUDE)]


# ---------------------------------------------------------------- data
OLD_V, NEW_V = load_version(ARGS.old, OLD), load_version(ARGS.new, NEW)
if NEW_V.metric is None:
    p.error("--new must be a reference-results parquet (it needs the leaderboard's metric_error)")
old, new = OLD_V.scores, NEW_V.scores
DUMMY_ERR, METRIC = NEW_V.dummy_error, NEW_V.metric
lb = read_leaderboard(ARGS.leaderboard_dir / "leaderboard_all.csv")
old_lb_path = ARGS.old_leaderboard or (ARGS.old / "leaderboard_overall.csv" if ARGS.old.is_dir() else None)
OLD_LB = read_leaderboard(old_lb_path) if old_lb_path and old_lb_path.exists() else None
ds = pd.read_csv(ARGS.leaderboard_dir / "datasets.csv").set_index("dataset")
DOMAIN = {"MaterialScience": "Materials", "Biological": "Biological", "Chemical": "Chemical", "Medical": "Medical"}

SHARED, SHARED_MODELS = shared(OLD_V, NEW_V)
W_OLD, W_NEW = winners(old), winners(new)
S_OLD, S_NEW = winners(restrict(old, SHARED, SHARED_MODELS)), winners(restrict(new, SHARED, SHARED_MODELS))
P_NEW = new.pivot_table(index="key", columns="model", values="score")
ERR = -P_NEW                                # fold-mean metric_error per task x model
NS_NEW = normalized(ERR)




# ---------------------------------------------------------------- 01 winner flow (alluvial)
def fig01(left=None, right=None, stem="01_winner_flow", title=None):
    left = S_OLD if left is None else left
    right = S_NEW if right is None else right
    a, b = left["family"], right["family"]
    tot = pd.concat([a, b]).value_counts()
    cat_of = pd.concat([left, right]).drop_duplicates("family").set_index("family")["cat"]
    keep = lambda f: f if tot[f] >= 3 else f"other {CLAB[cat_of[f]].split()[0].lower()}"
    cat_of = pd.concat([cat_of, pd.Series({keep(f): cat_of[f] for f in tot.index})])
    cat_of = cat_of[~cat_of.index.duplicated()]
    a, b = a.map(keep), b.map(keep)
    # Each family in its fixed colour (common.FAMILY_COLOURS, as in the who-wins figure); grouped
    # "other ..." nodes in their category's colour.
    fcol = dict(zip(pd.concat([left["family"], right["family"]]),
                    pd.concat([left["model"], right["model"]]).map(colour)))
    col = lambda f: fcol.get(f, CCOL[cat_of[f]])
    order_cat = {k: i for i, (k, *_) in enumerate(CATS)}
    def nodes(s):
        c = s.value_counts()
        return sorted(c.index, key=lambda f: (order_cat[cat_of[f]], -c[f])), c
    la, ca = nodes(a)
    rb, cb = nodes(b)
    n, gap = len(a), 1.6
    fig, ax = plt.subplots(figsize=(11, 9))
    def place(order, counts):
        y, pos = 0.0, {}
        for f in order:
            pos[f] = [y, y + counts[f]]
            y += counts[f] + gap
        return pos, y - gap
    pa, ha = place(la, ca)
    pb, hb = place(rb, cb)
    H = max(ha, hb)
    pa = {f: [H - v[1], H - v[0]] for f, v in pa.items()}
    off = (H - hb) / 2
    pb = {f: [H - v[1] - off, H - v[0] - off] for f, v in pb.items()}
    x0, x1, w = 0.0, 1.0, 0.035
    fa = {f: pa[f][1] for f in pa}     # running top of each node (fill top-down)
    fb = {f: pb[f][1] for f in pb}
    flows = pd.crosstab(a, b)
    for f in la:
        for g in sorted(flows.columns, key=lambda g: -pb[g][1]):
            k = flows.loc[f, g] if g in flows.columns else 0
            if not k:
                continue
            ya1, ya0 = fa[f], fa[f] - k
            yb1, yb0 = fb[g], fb[g] - k
            fa[f], fb[g] = ya0, yb0
            xm = (x0 + x1) / 2
            verts = [(x0 + w, ya1), (xm, ya1), (xm, yb1), (x1 - w, yb1), (x1 - w, yb0), (xm, yb0), (xm, ya0),
                     (x0 + w, ya0), (x0 + w, ya1)]
            codes = [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.LINETO, MPath.CURVE4,
                     MPath.CURVE4, MPath.CURVE4, MPath.CLOSEPOLY]
            stay = f == g
            ax.add_patch(PathPatch(MPath(verts, codes), fc=col(g), ec="none",
                                   alpha=0.55 if stay else 0.3, zorder=1))
    for pos, x, ha_, counts, ver in ((pa, x0, "right", ca, OLD), (pb, x1 - w, "left", cb, NEW)):
        for f, (y0, y1) in pos.items():
            ax.add_patch(plt.Rectangle((x, y0), w, y1 - y0, fc=col(f), ec="white", lw=0.8, zorder=3))
            tx = x - 0.012 if ha_ == "right" else x + w + 0.012
            ax.text(tx, (y0 + y1) / 2, f"{f}  {counts[f]}" if ha_ == "left" else f"{counts[f]}  {f}",
                    ha=ha_, va="center", fontsize=9, fontweight="bold" if counts[f] >= 10 else None)
    stay = (a == b).sum()
    ax.text(x0 + w / 2, H + 2.5, f"{OLD} winner", ha="center", fontsize=11, fontweight="bold", color="#555555")
    ax.text(x1 - w / 2, H + 2.5, f"{NEW} winner", ha="center", fontsize=11, fontweight="bold", color="#D55E00")
    ax.set_xlim(-0.32, 1.32); ax.set_ylim(-2, H + 6); ax.axis("off")
    ax.set_title(title or f"Who took over which task? The same {n} tasks and {len(SHARED_MODELS)} models, "
                          f"{OLD} winner → {NEW} winner", loc="left", fontsize=13)
    footnote(fig, f"{stay} of {n} tasks keep their winning family; {n - stay} change hands, only because the "
                  f"protocol changed (same tasks, same models). Darker bands = same family in both versions. "
                  f"Families with fewer than 3 wins are grouped per category.", y=0.035)
    # No category legend: nodes carry their family names, in the fixed family colours.
    save(fig, stem)


def fig01b():
    """Same shared tasks; old winners among the old models, new winners among all new models (new ones included)."""
    left = winners(old[old["key"].isin(SHARED)])
    right = winners(new[new["key"].isin(SHARED)])
    fig01(left, right, "01b_winner_flow_new_models",
          f"Who took over which task? The same {len(SHARED)} tasks, {OLD} winner → {NEW} winner with all new models")


# ---------------------------------------------------------------- 02 rank changes (slope chart)
def fig02():
    models = [m for m in SHARED_MODELS if m in set(OLD_LB["model"]) and m in set(lb["model"])]
    ro = OLD_LB.set_index("model").loc[models, "elo"].rank(ascending=False).astype(int)
    rn = lb.set_index("model").loc[models, "elo"].rank(ascending=False).astype(int)
    fig, ax = plt.subplots(figsize=(9, 10))
    for m in models:
        d = ro[m] - rn[m]
        big = abs(d) >= 5
        col = CCOL[category(m)]
        ax.plot([0, 1], [ro[m], rn[m]], color=col, lw=2.6 if big else 1.2, alpha=1 if big else 0.45,
                solid_capstyle="round", zorder=3 if big else 2)
        ax.scatter([0, 1], [ro[m], rn[m]], color=col, s=36, zorder=4)
        ax.text(-0.03, ro[m], name_of(m), ha="right", va="center", fontsize=8.6,
                fontweight="bold" if big else None)
        arrow = f"▲{d}" if d > 0 else f"▼{-d}" if d < 0 else "="
        ax.text(1.03, rn[m], f"{name_of(m)}  ", ha="left", va="center", fontsize=8.6,
                fontweight="bold" if big else None)
        ax.text(1.42, rn[m], arrow, ha="left", va="center", fontsize=8.6, fontweight="bold",
                color="#2E7D32" if d > 0 else "#B23B3B" if d < 0 else "#888888")
    ax.set_ylim(len(models) + 0.7, -0.6); ax.set_xlim(-0.55, 1.6); ax.axis("off")
    ax.text(0, -0.25, f"{OLD} rank", ha="center", fontweight="bold", color="#555555")
    ax.text(1, -0.25, f"{NEW} rank", ha="center", fontweight="bold", color="#D55E00")
    ax.set_title(f"Who rose, who fell? Elo rank among the {len(models)} models in both versions",
                 loc="left", fontsize=13)
    cat_legend(ax, loc="lower center", bbox_to_anchor=(0.5, -0.07), ncol=3, fontsize=8.5)
    footnote(fig, "Each version's official Elo, re-ranked among the models present in both. Bold: moved 5 places "
                  f"or more. {NEW} Elo covers all {NEW} tasks.", y=0.03)
    save(fig, "02_rank_changes")


# ---------------------------------------------------------------- 03 state of the art over time
def fig03():
    rel = {m: info.release_date(m) for m in P_NEW.columns}
    rel = {m: d for m, d in rel.items() if d is not None}
    pls = P_NEW["PLS"]
    dates = sorted(set(rel.values()))
    err = ERR  # fold-mean metric_error (1 - ROC AUC binary, log loss multiclass, RMSE regression)
    # "Beats PLS" on the same metric: at least 5% lower error than PLS (scale-free, so it means the
    # same for ROC AUC, log loss and RMSE).
    BEAT = 0.95
    err_pls = err["PLS"]
    # Improvability (TabArena): how much lower today's best error is, as a share of the best error
    # so far, (err_so_far - err_today) / err_so_far, averaged over tasks; 0 = today's best is out.
    best_today = err[list(rel)].min(axis=1)
    beat, imp, events = [], [], []
    prev = set()
    for t in dates:
        avail = [m for m, d in rel.items() if d <= t]
        best = err[avail].min(axis=1)
        now = set(best.index[best < BEAT * err_pls])
        beat.append(len(now)); imp.append(100 * ((best - best_today) / best.where(best > 0)).fillna(0).clip(lower=0).mean())
        released = [m for m, d in rel.items() if d == t]
        if len(now - prev) >= 3:
            gains = {m: ((err[m] < BEAT * err_pls) & ~best.index.isin(list(prev))).sum() for m in released}
            events.append((t, len(now), max(gains, key=gains.get), len(now - prev)))
        prev = now
    x = np.array(dates)
    fig, ax = plt.subplots(figsize=(12, 6.2))
    ax.step(x, beat, where="post", color="#D55E00", lw=2.6)
    ax.fill_between(x, beat, step="post", color="#D55E00", alpha=0.12)
    for i, (t, y, m, d) in enumerate(events):
        ax.annotate(f"{name_of(m)}\n+{d} tasks", (t, y), xytext=(-22 if i == len(events) - 2 else 0,
                                                                 18 + 30 * (i % 2 if t > max(dates) - 1 else 0)), textcoords="offset points", ha="center",
                    fontsize=8, fontweight="bold", color="#8C2D04",
                    arrowprops=dict(arrowstyle="-", color="#8C2D04", lw=0.8))
    ax.set_xlim(SINCE, max(dates) + 0.35)
    ax.set_ylim(0, len(P_NEW) * 1.12)
    ax.axhline(len(P_NEW), color="#999999", lw=0.8, ls=":")
    ax.text(SINCE + 0.1, len(P_NEW) - 2, f"all {len(P_NEW)} tasks", va="top", fontsize=8, color="#777777")
    ax.set_ylabel("tasks where the best model released so far\nhas at least 5% lower error than PLS", color="#D55E00")
    ax2 = ax.twinx()
    ax2.step(x, imp, where="post", color="#0072B2", lw=1.8, ls="--")
    ax2.set_ylabel("improvability of the best model released so far:\nmean % of its error that today's best removes",
                   color="#0072B2")
    ax2.spines["right"].set_visible(True); ax2.set_ylim(0, max(imp[np.searchsorted(x, SINCE) - 1:]) * 1.12)
    print("fig03 events:", [(round(float(t), 2), m, d, y) for t, y, m, d in events], "beat:", beat[0], beat[-1], f"before {SINCE:.4g}:", beat[np.searchsorted(x, SINCE) - 1])
    print("fig03 improvability:", {round(float(t), 2): round(float(v), 1) for t, v in zip(x, imp) if t >= SINCE - 0.2})
    ax.set_xlabel("model release date")
    ax.set_title(f"When did deep learning move past PLS? RamanBench {NEW} tasks, by model release date", loc="left")
    footnote(fig, f"Error = {NEW} leaderboard metric ({NEW_V.metric_desc}, mean over folds). "
                  f"{beat[np.searchsorted(x, SINCE) - 1]} tasks were already beaten by models released before {SINCE:.4g} "
                  f"(LR, KNN, trees, boosting)."
                  f" Labels: releases that added "
                  f"3 or more tasks, with the model that added the most.", y=0.0)
    save(fig, "03_state_of_the_art_over_time")


# ---------------------------------------------------------------- 04 headroom
def fig04():
    # Headroom = the best model's error as a share of the trivial baseline's (DUMMY) error, on the
    # task's own metric: 0 = perfect, 1 = no better than predicting the majority class / the mean.
    best = W_NEW.copy()
    best["headroom"] = (-best["score"]) / DUMMY_ERR.reindex(best.index)
    best["domain"] = [DOMAIN.get(ds.loc[dataset_of(k), "domain"], "?") for k in best.index]
    best = best.sort_values("headroom", ascending=False)
    dcol = {"Medical": "#B23B3B", "Biological": "#2E7D32", "Chemical": "#4E79A7", "Materials": "#9C755F"}
    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(best))
    h = best["headroom"].clip(upper=1.0)
    ax.bar(x, h, color=[dcol[d] for d in best["domain"]], width=0.85)
    over = best["headroom"] > 1
    ax.bar(x[over.values], h[over], color="none", ec="white", hatch="///", width=0.85, lw=0)
    lines = []
    for i, (k, r) in enumerate(best.head(10).iterrows()):
        nm = ds.loc[dataset_of(k), "name"] + (f" (target {int(k[-1]) + 1})" if ds.loc[dataset_of(k), "n_targets"] > 1 else "")
        ax.text(i, h.iloc[i] + 0.015, str(i + 1), ha="center", va="bottom", fontsize=7.5, fontweight="bold")
        mt = METRIC[k]
        val = 1 + r["score"] if mt == "roc_auc" else -r["score"]
        lines.append(f"{i + 1:>2}  {nm}: {100 * r['headroom']:.0f}% of baseline error, best {METRIC_NAMES[mt]} "
                     f"{val:.3g} ({name_of(r['model'])})")
    ax.text(0.2, 0.97, "hardest tasks\n" + "\n".join(lines), transform=ax.transAxes, va="top", fontsize=7.8,
            family="DejaVu Sans Mono", bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#DDDDDD"))
    solved = (best["headroom"] <= 0.05).sum()
    ax.axhline(0.05, color="#999999", ls=":", lw=0.9)
    ax.annotate(f"{solved} tasks ({100 * solved / len(best):.0f}%): at most 5% of the baseline's error left",
                (len(best) - 1, 0.05), xytext=(len(best) - 1, 0.17), ha="right", fontsize=8.5, color="#555555",
                arrowprops=dict(arrowstyle="-", color="#999999", lw=0.8),
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none"))
    ax.set_xlim(-1, len(best)); ax.set_xticks([])
    ax.set_ylabel("headroom = best model's error / baseline's error")
    ax.set_title(f"How much is left to learn? Headroom of every RamanBench {NEW} task (best model, all models)",
                 loc="left")
    med = best.groupby("domain")["headroom"].median().sort_values(ascending=False)
    ax.legend(handles=[Patch(fc=dcol[d], label=f"{d} (median {med[d]:.2f}, {int((best['domain'] == d).sum())} tasks)")
                       for d in med.index], frameon=False, loc="center right", fontsize=8.5, title="domain",
              title_fontsize=8.5)
    footnote(fig, f"Error per task = fold-mean metric of the {NEW} leaderboard (1 − ROC AUC binary, log loss multiclass, "
                  "RMSE regression); baseline = DUMMY (majority class / mean). Tasks where no model beats the "
                  "baseline are capped at 1 and hatched.", y=0.0)
    ax.set_ylim(0, 1.1)
    save(fig, "04_headroom")


# ---------------------------------------------------------------- 05 winner by dataset shape
def fig05():
    b = W_NEW.copy()
    b["n"] = [ds.loc[dataset_of(k), "n_spectra"] for k in b.index]
    b["p"] = [ds.loc[dataset_of(k), "n_features"] for k in b.index]
    rng = np.random.default_rng(0)
    jit = lambda s: s * np.exp(rng.normal(0, 0.06, len(s)))
    fig, ax = plt.subplots(figsize=(11, 7.5))
    for c, *_ in CATS:
        for task, mk in (("classification", "o"), ("regression", "s")):
            s = b[(b["cat"] == c) & (b["task"] == task)]
            ax.scatter(jit(s["n"]), jit(s["p"]), s=70, marker=mk, color=CCOL[c], ec="white", lw=0.7, alpha=0.9,
                       zorder=3)
    ax.set_xscale("log"); ax.set_yscale("log")
    nx, py = 500, 2000
    ax.axvline(nx, color="#BBBBBB", lw=0.9, ls="--"); ax.axhline(py, color="#BBBBBB", lw=0.9, ls="--")
    xl, yl = ax.get_xlim(), ax.get_ylim()
    quads = []
    for (xa, xb), (ya, yb) in [((xl[0], nx), (yl[0], py)), ((nx, xl[1]), (yl[0], py)),
                               ((xl[0], nx), (py, yl[1])), ((nx, xl[1]), (py, yl[1]))]:
        q = b[(b["n"] >= xa) & (b["n"] < xb) & (b["p"] >= ya) & (b["p"] < yb)]
        if len(q):
            tfm = (q["cat"] == "Tabular Foundation").mean()
            quads.append(f"{'few' if xb <= nx else 'many'} spectra, {'narrow' if yb <= py else 'wide'}: "
                         f"{len(q)} tasks, {100 * tfm:.0f}% to foundation models")
    ax.set_xlim(xl); ax.set_ylim(yl)
    ax.set_xlabel("number of spectra (log)"); ax.set_ylabel("number of wavenumbers / features (log)")
    ax.set_title("Does the shape of the data decide the winner? Hardly: foundation models win everywhere",
                 loc="left")
    handles = [Patch(fc=CCOL[c], label=CLAB[c]) for c, *_ in CATS if (b["cat"] == c).any()]
    handles += [Line2D([], [], marker="o", ls="", color="#777777", label="classification"),
                Line2D([], [], marker="s", ls="", color="#777777", label="regression")]
    ax.legend(handles=handles, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8.5)
    ax.text(1.02, 0.42, f"quadrants (dashed lines)\n" + "\n".join(quads), transform=ax.transAxes, va="top",
            fontsize=8.3, color="#333333", bbox=dict(boxstyle="round,pad=0.5", fc="#FAFAFA", ec="#DDDDDD"))
    footnote(fig, f"Spectra and features per dataset (multi-target datasets: one point per target, slightly "
                  f"jittered). Dashed lines at {nx} spectra and {py} features.", y=0.0)
    save(fig, "05_winner_by_dataset_shape")


# ---------------------------------------------------------------- 06 winner by domain
def fig06():
    b = W_NEW.copy()
    b["domain"] = [DOMAIN.get(ds.loc[dataset_of(k), "domain"], "?") for k in b.index]
    tab = pd.crosstab(b["domain"], b["cat"])
    tab.loc["All tasks"] = tab.sum()
    order = ["All tasks"] + tab.drop("All tasks").sum(axis=1).sort_values().index.tolist()[::-1]
    tab = tab.loc[order[::-1]]
    share = tab.div(tab.sum(axis=1), axis=0) * 100
    fig, ax = plt.subplots(figsize=(12, 4.8))
    left = np.zeros(len(share))
    for c, *_ in CATS:
        if c not in share:
            continue
        v = share[c].values
        ax.barh(range(len(share)), v, left=left, color=CCOL[c], ec="white", lw=1.2, height=0.7, label=CLAB[c])
        for i, (l, w) in enumerate(zip(left, v)):
            if w >= 5:
                ax.text(l + w / 2, i, f"{w:.0f}%", ha="center", va="center", fontsize=8.5,
                        color="white", fontweight="bold")
        left += v
    ax.set_yticks(range(len(share)))
    ax.set_yticklabels([f"{d}  ({int(tab.loc[d].sum())} tasks)" for d in share.index], fontsize=10)
    for t in ax.get_yticklabels():
        if t.get_text().startswith("All"):
            t.set_fontweight("bold")
    ax.set_xlim(0, 100); ax.set_xlabel("share of tasks won (%)")
    # best family per domain
    for i, d in enumerate(share.index):
        s = b if d == "All tasks" else b[b["domain"] == d]
        top = s["family"].value_counts()
        ax.text(101, i, f"top: {top.index[0]} ({top.iloc[0]})", va="center", fontsize=8.5, color="#444444")
    ax.set_title(f"Which model to try first in your field? Winners of the {NEW} tasks, by application domain",
                 loc="left")
    ax.legend(frameon=False, ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.16), fontsize=8.5)
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    save(fig, "06_winner_by_domain")


# ---------------------------------------------------------------- 07 ties
def fig07():
    rows = []
    for k, g in new.groupby("key"):
        g = g.sort_values("score", ascending=False)
        w = g.iloc[0]
        noise = w["sd"]
        tied = (g["score"].iloc[1:] >= w["score"] - noise).sum()
        scale = max(-w["score"], 1e-12)          # winner's error: margins in % of it, any metric
        rows.append({"key": k, "tied": tied, "runner_tied": g["score"].iloc[1] >= w["score"] - noise,
                     "margin": (w["score"] - g["score"].iloc[1]) / scale, "noise": noise / scale})
    t = pd.DataFrame(rows)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw={"width_ratios": [1.25, 1]})
    bins = [0, 1, 2, 3, 5, 10, 20, 100]
    labels = ["0", "1", "2", "3–4", "5–9", "10–19", "20+"]
    cnt = pd.cut(t["tied"], bins=bins, right=False, labels=labels).value_counts().reindex(labels)
    cols = plt.cm.Oranges(np.linspace(0.35, 0.95, len(labels)))
    ax.bar(labels, cnt.values, color=cols, ec="white")
    for i, v in enumerate(cnt.values):
        ax.text(i, v + 0.6, str(v), ha="center", fontsize=9, fontweight="bold")
    ax.set_xlabel("models within one fold-to-fold standard deviation of the winner")
    ax.set_ylabel("tasks")
    ax.set_title(f"How clear is the win? Median: {t['tied'].median():.0f} models tie with the winner", loc="left")
    bx.scatter(t["noise"].clip(1e-4), t["margin"].clip(1e-4), s=26, color=np.where(t["runner_tied"], "#D55E00",
                                                                                     "#0072B2"), alpha=0.75)
    lim = [1e-4, 10]
    bx.plot(lim, lim, color="#888888", lw=0.9, ls="--")
    bx.set_xscale("log"); bx.set_yscale("log"); bx.set_xlim(*lim); bx.set_ylim(*lim)
    bx.set_xlabel("winner's fold-to-fold standard deviation\n(relative to the winner's error)")
    bx.set_ylabel("margin over the runner-up\n(relative to the winner's error)")
    share = 100 * t["runner_tied"].mean()
    bx.text(1.5e-4, 5, f"below the line: runner-up within noise\n({share:.0f}% of tasks)", fontsize=8.5,
            color="#D55E00", va="top")
    bx.set_title("Winning margin vs. noise, per task", loc="left")
    fig.suptitle(f"Read the winner counts with care: most {NEW} wins are within fold-to-fold noise", x=0.01, ha="left",
                 fontsize=13.5, fontweight="bold")
    footnote(fig, f"Per task: error = {NEW} leaderboard metric ({NEW_V.metric_desc}), mean over the outer "
                  "folds; noise = standard deviation of the winner's fold errors. A model 'ties' when its mean "
                  "error is within that standard deviation of the winner's.",
             y=0.0)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    save(fig, "07_ties")


# ---------------------------------------------------------------- 08 cost of the best result
def fig08(tol=0.05):
    d = new[~new["model"].isin(HOSTED)]          # hosted API times are server-side, not comparable
    rows = []
    for k, g in d.groupby("key"):
        w = g.loc[g["score"].idxmax()]
        near = g[-g["score"] <= (1 + tol) * -w["score"]]   # at most tol more error than the winner
        c = near.loc[near["time"].idxmin()]
        rows.append({"key": k, "w_time": w["time"], "c_time": c["time"], "c_model": c["model"],
                     "c_cat": category(c["model"]), "same": c["model"] == w["model"]})
    t = pd.DataFrame(rows)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.2, 1]})
    for c, *_ in CATS:
        s = t[t["c_cat"] == c]
        ax.scatter(s["w_time"], s["c_time"], s=34, color=CCOL[c], alpha=0.85, ec="white", lw=0.5, label=CLAB[c])
    lo, hi = 0.05, max(t["w_time"].max(), t["c_time"].max()) * 2
    ax.plot([lo, hi], [lo, hi], color="#888888", ls="--", lw=0.9)
    for f in (10, 100):
        ax.plot([lo, hi], [lo / f, hi / f], color="#CCCCCC", ls=":", lw=0.9)
        ax.text(hi / 1.3, hi / f / 1.3, f"{f}× faster", fontsize=7.5, color="#999999", ha="right", va="top")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo / 20, hi)
    ax.set_xlabel("winner: time per fold (s, train + predict)")
    ax.set_ylabel(f"cheapest model within {tol:.0%} of the winner's error (s)")
    diff = t[~t["same"]]
    speed = (diff["w_time"] / diff["c_time"]).median()
    fast = 100 * (t["c_time"] < 10).mean()
    ax.set_title(f"On {len(diff)} tasks a cheaper model is within {tol:.0%}: median {speed:.0f}× faster", loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    top = t["c_model"].map(name_of).value_counts().head(12)[::-1]
    bx.barh(top.index, top.values, color=[CCOL[category(m)] for m in
                                          t.drop_duplicates("c_model").set_index(t["c_model"].map(name_of)
                                                                                 .drop_duplicates())
                                          .loc[top.index, "c_model"]])
    for i, v in enumerate(top.values):
        bx.text(v + 0.3, i, str(v), va="center", fontsize=9)
    bx.set_xlabel("tasks where it is the cheapest near-best model")
    bx.set_title("The budget champions", loc="left")
    fig.suptitle(f"What does the best result cost? On {fast:.0f}% of tasks a model under 10 s per fold is within "
                 f"{tol:.0%} of the winner's error", x=0.01, ha="left", fontsize=13.5, fontweight="bold")
    footnote(fig, "Time = mean over folds of fit + predict wall time on the benchmark hardware (GPU models on a GPU). "
                  "Chakra-Tab is left out: it runs on the provider's servers, so its times are not comparable. "
                  f"Error = {NEW} leaderboard metric ({NEW_V.metric_desc}).", y=0.0)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    save(fig, "08_cost_of_the_best")


# ---------------------------------------------------------------- 09 regret of a fixed choice
def fig09():
    top1 = lb.sort_values("elo", ascending=False)["model"].iloc[0]
    picks = [top1] + ARGS.regret_models
    picks = [m for m in dict.fromkeys(picks) if m in NS_NEW]
    pal = [colour(m) for m in picks]  # fixed family colours (common.py)
    fig, ax = plt.subplots(figsize=(11, 6.4))
    xs = np.linspace(0, 1, 400)
    for m, c in zip(picks, pal):
        r = (1 - NS_NEW[m]).dropna()                # regret = 1 - normalized score (best model = 0)
        ax.plot(xs, [(r <= x).mean() * 100 for x in xs], color=c, lw=3 if m == top1 else 1.8,
                label=f"always {name_of(m)}: median regret {r.median():.2f}, within 0.05 on {100 * (r <= 0.05).mean():.0f}%")
    ax.axvline(0.05, color="#BBBBBB", ls=":", lw=0.9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 101)
    ax.set_xlabel("regret = 1 − normalized score (0 = best model on the task, 1 = median or worse)")
    ax.set_ylabel("tasks with at most this regret (%)")
    ax.set_title(f"How much do you lose by always using one model? ({NEW}, {len(P_NEW)} tasks)", loc="left")
    ax.legend(frameon=False, fontsize=8.6, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    footnote(fig, f"Curves further up and left are better. {name_of(top1)} is #1 on the {NEW} leaderboard (Elo). "
                  f"Normalized score as on the {NEW} leaderboard (best = 1, median = 0, clipped), over every ranked model.",
             y=0.0)
    save(fig, "09_regret_of_a_fixed_choice")


# ---------------------------------------------------------------- 10 licences
def fig10(n=30):
    t = lb.sort_values("elo", ascending=False).head(n).iloc[::-1]
    lcol = {"open": "#2E7D32", "nc": "#B07AA1", "hosted": "#4E79A7"}
    llab = {"open": "open, commercial use allowed", "nc": "non-commercial or research-only weights", "hosted": "hosted closed API"}
    fig, ax = plt.subplots(figsize=(11, 10))
    y = np.arange(len(t))
    lic = t["model"].map(licence)
    ax.barh(y, t["elo"] - 1000, left=1000, color=[lcol[l] for l in lic], height=0.72, alpha=0.9)
    ax.errorbar(t["elo"], y, xerr=[t["elo-"], t["elo+"]], fmt="none", ecolor="#555555", elinewidth=0.8, capsize=2)
    firsts = {l: t[lic == l]["model"].iloc[-1] for l in lcol if (lic == l).any()}
    for yi, (m, e) in enumerate(zip(t["model"], t["elo"])):
        star = m in firsts.values()
        ax.text(1005, yi, name_of(m) + ("  ★ best " + {"open": "open", "nc": "non-commercial",
                                                           "hosted": "hosted"}[licence(m)] if star else ""),
                va="center", fontsize=8.6, color="white", fontweight="bold" if star else None)
        ax.text(e + t["elo+"].iloc[yi] + 4, yi, f"{e:.0f}", va="center", fontsize=8, color="#444444")
    ax.set_yticks([]); ax.set_xlim(1000, t["elo"].max() + 90)
    ax.set_xlabel(f"Elo ({NEW}, all tasks; whiskers = bootstrap interval)")
    best_open = firsts.get("open")
    ax.set_title(f"Can you ship it? Top {n} of the {NEW} leaderboard by licence — best commercially usable: "
                 f"{name_of(best_open)}", loc="left")
    ax.legend(handles=[Patch(fc=lcol[l], label=llab[l]) for l in lcol], frameon=False, loc="lower right",
              fontsize=9)
    ax.spines["left"].set_visible(False)
    footnote(fig, "Weights licences from the model cards (checked 2026-10-16). Kumo-Tabular: OpenMDW-1.1. TabPFN v2 and "
                  "TabPFN-Wide: Apache-2.0 plus a 'Built with PriorLabs-TabPFN' attribution. LimiX: commercial use only "
                  "with the licensor's authorization; SAP-RPT-OSS: research-only weights. RamanPFN excluded. Not legal advice: check each licence before use.", y=0.06)
    save(fig, "10_licences")



def main():
    only = ARGS.only.split(",") if ARGS.only else FIGURES
    for f in (fig01, fig01b, fig02, fig03, fig04, fig05, fig06, fig07, fig08, fig09, fig10):
        name = f.__name__[3:]
        if name not in only:
            continue
        if name == "02" and OLD_LB is None:
            print("skipped 02: no leaderboard of the older version (--old-leaderboard)")
            continue
        f()


if __name__ == "__main__":
    main()
