"""Shared pieces of the benchmark-version comparison figures (wins_by_family.py, analyses.py).

Loads the results of two benchmark versions into one format, and fixes the model categories, model
families and their colours, so that a family looks the same in every figure and in both versions.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from raman_bench.plotting import models as info

# Not ranked: baselines and ensembles of the ranked models. RamanPFN is left out because it was
# developed on the RamanBench datasets, so comparing it would not be fair; TabSTAR is not on the leaderboard.
EXCLUDE = {"RAMANPFN", "DUMMY", "AUTOGLUON", "AUTOGLUON-EXTREME-1H", "AUTOGLUON-EXTREME-5M",
           "AUTOGLUON-EXTREME-4H", "TABSTAR"}

# Figure categories: the leaderboard's categories, with tree ensembles and boosting merged.
# key, label, base colour (Okabe-Ito-like, colour-blind safe)
CATS = [
    ("Tabular Foundation", "Tabular foundation models", "#D55E00"),
    ("Raman-Specific", "Raman-specific networks", "#0072B2"),
    ("Deep Learning", "Generic deep learning", "#CC79A7"),
    ("Traditional ML", "Traditional ML", "#009E73"),
    ("Trees", "Trees and boosting", "#E69F00"),
    ("TS Classification", "Time-series classifiers", "#56B4E9"),
]
CCOL = {k: c for k, _, c in CATS}
CLAB = {k: lab for k, lab, _ in CATS}

# Model variants that count as one family (all TabPFN versions, Kumo-Tabular S/M/L, Chakra-Tab medium/full,
# ...); any other model is its own family, named by its leaderboard display name.
FAMILIES = [
    (r"^(REALTABPFN|TABPFN)", "TabPFN"), (r"^TABICL", "TabICL"), (r"^TABDPT", "TabDPT"),
    (r"MITRA", "Mitra"), (r"^LIMIX", "LimiX"), (r"^KUMO", "Kumo-Tabular"),
    (r"^CHAKRA", "Chakra-Tab"), (r"^TABFM", "TabFM"), (r"^RAMANPFN$", "RamanPFN"),
    (r"^SAP_RPT", "SAP-RPT"), (r"EXAONE", "EXAONE"), (r"^NORI", "Nori"),
]

# One colour per family. Tabular foundation models get distinct hues; the families of the other categories
# are shades of their category's colour. A family missing here is drawn in its category's colour, so add
# new winners when they appear.
FAMILY_COLOURS = {
    # Tabular foundation models
    "TabPFN": "#8c2d04", "Kumo-Tabular": "#d55e00", "TabICL": "#f28e2b", "TabFM": "#b07aa1",
    "LimiX": "#4e79a7", "TabDPT": "#59a14f", "Mitra": "#e15759", "EXAONE": "#76b7b2", "Chakra-Tab": "#edc948",
    "Nori": "#9c755f", "SAP-RPT": "#ff9da7", "TabSwift": "#bab0ac", "Causilo": "#86bcb6", "iLTM": "#d4a6c8",
    # Raman-specific networks
    "Deep CNN": "#005685", "RamanFormer": "#2d8bbf", "RamanNet": "#99c7e0",
    # Generic deep learning
    "ReZeroNet": "#995b7d", "RealMLP": "#ad6890", "CoAtNet": "#c474a1", "FCResNeXt": "#d590b6", "FastAI MLP": "#e0adc9",
    "Torch MLP": "#ebc9dc",
    # Traditional ML
    "PLS": "#007656", "Linear Model": "#2daf8c", "KNN": "#99d8c7",
    # Trees and boosting
    "XGBoost": "#ac7700", "CatBoost": "#c98c00", "LightGBM": "#e7a208", "Random Forest": "#eebd51",
    "Extra Trees": "#f5d999",
    # Time-series classifiers
    "ROCKET": "#4087af", "Arsenal": "#bbe1f6",
}

# Licence classes, checked 2026-10-16 against each model's weights licence (Hugging Face model cards and
# LICENSE files) and code licence. "nc" = weights not licensed for commercial use: non-commercial
# licences (TabPFN >= 2.5, TabFM/TabFM+, EXAONE, LimiX-2, Causilo), research-only weights (SAP-RPT-OSS),
# or commercial use only with the licensor's authorization (LimiX v1). RamanPFN runs on tabpfn's
# default checkpoint, TabPFN v3.5 for tabpfn >= 9 (non-commercial). Kumo-Tabular: OpenMDW-1.1 weights,
# Apache-2.0 code. TabPFN v2 / TabPFN-Wide: Prior Labs License 1.1 (Apache-2.0 + "Built with
# PriorLabs-TabPFN" attribution). RamanBench-native models are MIT / BSD (scikit-learn, aeon).
# Re-check before reusing: licences change.
NON_COMMERCIAL = {"REALTABPFN-V2.5", "REALTABPFN-V2.6", "TABPFN-V3", "TABPFN-V3.5", "TABPFN-V3.5-FAST", "TABFM",
                  "TABFM-PLUS", "TA-EXAONE-TABULAR", "LIMIX", "LIMIX2", "SAP_RPT_OSS", "CAUSILO", "RAMANPFN"}
HOSTED = {"CHAKRA-TAB", "CHAKRA-TAB-FULL"}  # closed APIs; their times are server-side

METRIC_NAMES = {"roc_auc": "ROC AUC", "log_loss": "log loss", "rmse": "RMSE"}


def name_of(m: str) -> str:
    return info.display_name(m, mark=False).replace(" (API)", "")


def family(m: str) -> str:
    for pat, name in FAMILIES:
        if re.search(pat, m):
            return name
    return name_of(m)


def category(m: str) -> str:
    if m == "RAMANPFN":
        return "Tabular Foundation"
    c = info.category(m)
    return "Trees" if c in ("Tree-based", "Gradient Boosting") else c


def colour(m: str) -> str:
    """Colour of model id *m*: its family's, else its category's."""
    return FAMILY_COLOURS.get(family(m), CCOL[category(m)])


def licence(m: str) -> str:
    return "hosted" if m in HOSTED else "nc" if m in NON_COMMERCIAL else "open"


@dataclass
class Version:
    """One benchmark version's results, per task x model.

    scores: columns task, key, model, score (higher = better), sd (fold/seed std of score), time (s per fold,
      NaN if unknown). Excluded models are dropped.
    metric: the task's metric name per task key (None for the v0.1 format).
    dummy_error: DUMMY's mean error per task key (None when the version has no error column).
    metric_desc: the metric in words, for footnotes.
    """
    label: str
    scores: pd.DataFrame
    metric: pd.Series | None
    dummy_error: pd.Series | None
    metric_desc: str


def load_version(path: Path, label: str) -> Version:
    """Load a version's results.

    *path* is either a reference-results parquet (benchmark v1 and later: one row per dataset x fold x model,
    with metric_error = 1 - ROC AUC binary, log loss multiclass, RMSE regression), or a v0.1
    ``precomputed_metrics`` directory (classification_metrics.csv with f1_score, regression_metrics.csv with
    rmse; v0.1 stored no probabilities, so it has no ROC AUC or log loss).
    """
    path = Path(path)
    if path.is_dir():
        c = pd.read_csv(path / "classification_metrics.csv").rename(columns={"f1_score": "score"})
        r = pd.read_csv(path / "regression_metrics.csv")
        r["score"] = -r["rmse"]
        c["task"], r["task"] = "classification", "regression"
        df = pd.concat([c, r])
        df = df[~df["imputed"].astype(bool)]
        df["key"] = df["key"].str.replace(r"_(\d+)$", r"__\1", regex=True)
        df["time"] = np.nan
        metric, dummy, desc = None, None, "F1 / RMSE"
    else:
        df = pd.read_parquet(path).rename(columns={"dataset": "key"})
        metric = df.groupby("key")["metric"].first()
        dummy = df[df["model"] == "DUMMY"].groupby("key")["metric_error"].mean()
        df["score"] = -df["metric_error"]
        df["time"] = df["time_train_s"] + df["time_infer_s"]
        desc = "1 − ROC AUC, log loss or RMSE"
    df = df[~df["model"].isin(EXCLUDE)]
    scores = df.groupby(["task", "key", "model"], as_index=False).agg(
        score=("score", "mean"), sd=("score", "std"), time=("time", "mean"))
    return Version(label, scores, metric, dummy, desc)


def shared(a: Version, b: Version) -> tuple[list[str], list[str]]:
    """Tasks with the same task type in both versions, and models in both."""
    ta, tb = a.scores.groupby("key")["task"].first(), b.scores.groupby("key")["task"].first()
    tasks = sorted(k for k in set(ta.index) & set(tb.index) if ta[k] == tb[k])
    models = sorted(set(a.scores["model"]) & set(b.scores["model"]))
    return tasks, models


def restrict(df: pd.DataFrame, tasks, models) -> pd.DataFrame:
    return df[df["key"].isin(tasks) & df["model"].isin(models)]


def winners(df: pd.DataFrame) -> pd.DataFrame:
    """Best model per task, with its family and category, indexed by task key."""
    best = df.loc[df.groupby("key")["score"].idxmax()].copy()
    best["family"], best["cat"] = best["model"].map(family), best["model"].map(category)
    return best.set_index("key")


def normalized(err: pd.DataFrame) -> pd.DataFrame:
    """The leaderboard's normalized score (TabRepo) of a task x model error table: best = 1, median = 0,
    clipped at 0; when the best equals the median, models at or below it score 1."""
    best_e, med_e = err.min(axis=1), err.median(axis=1)
    denom = (med_e - best_e).where(lambda d: d > 1e-12)
    ns = err.rsub(med_e, axis=0).div(denom, axis=0).clip(lower=0)
    ties = denom.isna()
    ns.loc[ties] = err.loc[ties].le(med_e[ties], axis=0).astype(float)
    return ns


def dataset_of(key: str) -> str:
    return re.sub(r"__\d+$", "", key)
