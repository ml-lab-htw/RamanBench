"""Model metadata for plots: display names, categories, colours, release dates.

Every RamanBench-specific plotting constant lives here, so no hex value or
display name is hardcoded in a figure module. Keys are the model keys used in
``configs/v1/scope_default.json`` and in the ``config_type`` column of
``hpo_results.csv``.

Colours follow the Okabe-Ito palette (colour-blind safe), one hue per
category, so a model's colour says what *kind* of model it is. TabArena colours
by model family the same way (``tabarena.plot.plot_pareto_focus.FAMILY_COLORS``);
RamanBench keeps its own finer categories because Raman-specific architectures
and time-series classifiers are a large part of what the benchmark compares.

Release dates come from each model's paper, or from TabArena's
``date_introduced`` metadata (``tabarena.models.<key>.info``) for the
TabArena-native models.
"""

from __future__ import annotations

OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
}

CATEGORY_ORDER = [
    "Baseline",
    "Traditional ML",
    "Tree-based",
    "Gradient Boosting",
    "Deep Learning",
    "Tabular Foundation",
    "TS Classification",
    "Raman-Specific",
    "Other",
]

CATEGORY_COLORS = {
    "Baseline": "#999999",
    "Traditional ML": OKABE_ITO["blue"],
    "Tree-based": OKABE_ITO["bluish_green"],
    "Gradient Boosting": OKABE_ITO["orange"],
    "Deep Learning": OKABE_ITO["reddish_purple"],
    "Tabular Foundation": OKABE_ITO["vermillion"],
    # Okabe-Ito yellow is too light for text on white: labels use a darker gold.
    "TS Classification": OKABE_ITO["yellow"],
    "Raman-Specific": OKABE_ITO["sky_blue"],
    "Other": "#555555",
}

#: Text colour for a category whose fill colour is too light to read as text.
LABEL_COLOR_OVERRIDES = {"TS Classification": "#A08C00"}

#: Reference systems: drawn as horizontal lines, not ranked as bars or points.
#: Key -> (display name, results directory name written by scripts/run_autogluon_baseline.py).
REFERENCE_MODELS = {
    "AUTOGLUON-EXTREME-5M": ("AutoGluon (extreme, 5 min)", "AutoGluon_extreme_5m"),
    "AUTOGLUON-EXTREME-1H": ("AutoGluon (extreme, 1 h)", "AutoGluon_extreme_1h"),
    "AUTOGLUON-EXTREME-4H": ("AutoGluon (extreme, 4 h)", "AutoGluon_extreme_4h"),
}
REFERENCE_COLOR = "#555555"

#: Colour of the greyed-out field in focus mode.
MUTED_COLOR = "#C8C8C8"
MUTED_LABEL_COLOR = "#9A9A9A"

# key: (display name, category, release year as a fractional year or None)
_MODELS: dict[str, tuple[str, str, float | None]] = {
    "DUMMY": ("Dummy", "Baseline", None),
    # Traditional ML
    "KNN": ("KNN", "Traditional ML", 1967.5),  # Cover & Hart, 1967
    "LR": ("Linear Model", "Traditional ML", 1958.5),
    "PLS": ("PLS", "Traditional ML", 1975.5),  # Wold (NIPALS), 1975
    "XRFM": ("xRFM", "Traditional ML", 2025 + 7 / 12),  # 2025-08
    # Tree-based
    "RF": ("Random Forest", "Tree-based", 2001 + 9 / 12),  # Breiman, 2001-10
    "XT": ("Extra Trees", "Tree-based", 2006.5),  # Geurts et al., 2006
    # Gradient boosting
    "XGB": ("XGBoost", "Gradient Boosting", 2014 + 2 / 12),  # 2014-03
    "GBM": ("LightGBM", "Gradient Boosting", 2016 + 8 / 12),  # 2016-09
    "CAT": ("CatBoost", "Gradient Boosting", 2017 + 5 / 12),  # 2017-06
    "EBM": ("EBM", "Gradient Boosting", 2019 + 8 / 12),  # 2019-09
    "PERPETUAL_BOOSTER": ("PerpetualBooster", "Gradient Boosting", 2024 + 4 / 12),  # 2024-05
    "APLR": ("APLR", "Gradient Boosting", 2023.5),
    "CTBOOST": ("CTBoost", "Gradient Boosting", 2026 + 3 / 12),  # 2026-04-10
    "CHIMERABOOST": ("ChimeraBoost", "Gradient Boosting", 2026 + 4 / 12),  # 2026-05-26
    # Deep learning (general-purpose architectures)
    "FASTAI": ("FastAI MLP", "Deep Learning", 2017 + 8 / 12),  # 2017-09
    "NN_TORCH": ("Torch MLP", "Deep Learning", 2019 + 11 / 12),  # 2019-12
    "COATNET": ("CoAtNet", "Deep Learning", 2021.5),  # Dai et al., 2021
    "REZERONET": ("ReZeroNet", "Deep Learning", 2021.5),  # Bachlechner et al., 2021
    "REALMLP": ("RealMLP", "Deep Learning", 2024 + 6 / 12),  # 2024-07
    "MODERNNCA": ("ModernNCA", "Deep Learning", 2024 + 6 / 12),  # 2024-07
    "TABM": ("TabM", "Deep Learning", 2024 + 9 / 12),  # 2024-10
    "FCRESNEXT": ("FCResNeXt", "Deep Learning", 2024.5),  # Zabergja et al., 2024
    # Tabular foundation models
    "REALTABPFN-V2": ("TabPFN v2", "Tabular Foundation", 2024.0),  # 2024-01
    "TABDPT": ("TabDPT", "Tabular Foundation", 2024 + 9 / 12),  # 2024-10
    # TABICL (AutoGluon's TabICLModel) defaults to the same TabICLv2 checkpoints as TABICLV2
    # since AutoGluon 1.6; it left the scope as a duplicate and is named for older result sets.
    "TABICL": ("TabICLv2 (AutoGluon)", "Tabular Foundation", 2026 + 1 / 12),  # 2026-02-12
    "SAP_RPT_OSS": ("SAP-RPT-OSS", "Tabular Foundation", 2025 + 5 / 12),  # 2025-06
    "TABSTAR": ("TabSTAR", "Tabular Foundation", 2025 + 4 / 12),  # 2025-05
    "MITRA": ("Mitra", "Tabular Foundation", 2025 + 6 / 12),  # 2025-07
    "LIMIX": ("LimiX", "Tabular Foundation", 2025 + 8 / 12),  # 2025-09
    "TABPFN-WIDE": ("TabPFN-Wide", "Tabular Foundation", 2025 + 9 / 12),  # 2025-10
    "REALTABPFN-V2.5": ("TabPFN v2.5", "Tabular Foundation", 2025 + 10 / 12),  # 2025-11
    "ORIONMSP": ("OrionMSP", "Tabular Foundation", 2025 + 10 / 12),  # 2025-11
    "ILTM": ("iLTM", "Tabular Foundation", 2025 + 10 / 12),  # 2025-11
    "TABICLV2": ("TabICLv2", "Tabular Foundation", 2026 + 1 / 12),  # 2026-02-12
    "REALTABPFN-V2.6": ("TabPFN v2.6", "Tabular Foundation", 2026 + 2 / 12),  # 2026-03
    "TABPFN-V3": ("TabPFN v3", "Tabular Foundation", 2026 + 4 / 12),  # 2026-05
    "TABSWIFT": ("TabSwift", "Tabular Foundation", 2026 + 5 / 12),  # 2026-06-05
    "NORI": ("Nori", "Tabular Foundation", 2026 + 5 / 12),  # 2026-06-12
    "TABFM": ("TabFM", "Tabular Foundation", 2026 + 5 / 12),  # 2026-06-30
    "TA-EXAONE-TABULAR": ("EXAONE-Tabular", "Tabular Foundation", 2026 + 6 / 12),  # 2026-07-31
    "TABLDM": ("Xiaomi-TabLDM", "Tabular Foundation", 2026 + 7 / 12),
    "TA-MITRA-V2": ("Mitra-v2", "Tabular Foundation", 2026 + 8 / 12),  # 2026-09-03
    "TABDPT-V1.3": ("TabDPT-1.3", "Tabular Foundation", 2026 + 8 / 12),  # 2026-09-08
    "LIMIX2": ("LimiX-2", "Tabular Foundation", 2026 + 8 / 12),  # 2026-09-15
    "TABPFN-V3.5": ("TabPFN v3.5", "Tabular Foundation", 2026 + 8 / 12),  # 2026-09-15
    "TABPFN-V3.5-FAST": ("TabPFN v3.5 Fast", "Tabular Foundation", 2026 + 8 / 12),  # 2026-09-15
    "KUMO-TABULAR": ("Kumo-Tabular", "Tabular Foundation", 2026 + 8 / 12),  # 2026-09-25
    "KUMO-TABULAR-MEDIUM": ("Kumo-Tabular (M)", "Tabular Foundation", 2026 + 8 / 12),
    "KUMO-TABULAR-SMALL": ("Kumo-Tabular (S)", "Tabular Foundation", 2026 + 8 / 12),
    # Time-series classifiers (classification only)
    "ROCKET": ("ROCKET", "TS Classification", 2019 + 9 / 12),  # Dempster et al., 2019-10
    "ARSENAL": ("Arsenal", "TS Classification", 2021 + 3 / 12),  # Middlehurst et al., 2021-04
    "HYDRA": ("Hydra", "TS Classification", 2022.5),  # Dempster et al., 2022
    # Raman-specific architectures
    "DEEPCNN": ("Deep CNN", "Raman-Specific", 2017.5),  # Liu et al., 2017
    "SANET": ("SANet", "Raman-Specific", 2021.5),  # Deng et al., 2021
    "RAMANNET": ("RamanNet", "Raman-Specific", 2023.5),  # Ibtehaz et al., 2023
    "RAMANTRANSFORMER": ("RamanTransformer", "Raman-Specific", 2023.5),  # Liu et al., 2023
    "RAMANFORMER": ("RamanFormer", "Raman-Specific", 2024.5),  # Koyun et al., 2024
    "RAMANPFN": ("RamanPFN", "Raman-Specific", 2026 + 7 / 12),  # arXiv:2608.02157
}


def display_name(model: str) -> str:
    """Human-readable name for *model*; the key itself when unknown."""
    if model.upper() in REFERENCE_MODELS:
        return REFERENCE_MODELS[model.upper()][0]
    entry = _MODELS.get(model.upper())
    return entry[0] if entry else model


def category(model: str) -> str:
    """Category of *model* (``"Other"`` when unknown, ``"Reference"`` for reference systems)."""
    if model.upper() in REFERENCE_MODELS:
        return "Reference"
    entry = _MODELS.get(model.upper())
    return entry[1] if entry else "Other"


def release_date(model: str) -> float | None:
    """Release date of *model* as a fractional year, or ``None`` when unknown."""
    entry = _MODELS.get(model.upper())
    return entry[2] if entry else None


def color(model: str) -> str:
    """Fill colour of *model*: its category's colour."""
    return CATEGORY_COLORS.get(category(model), REFERENCE_COLOR)


def label_color(model: str) -> str:
    """Text colour for *model*'s label (its colour, darkened where unreadable)."""
    cat = category(model)
    return LABEL_COLOR_OVERRIDES.get(cat, CATEGORY_COLORS.get(cat, REFERENCE_COLOR))


def category_rank(cat: str) -> int:
    """Position of *cat* in :data:`CATEGORY_ORDER` (unknown categories last)."""
    return CATEGORY_ORDER.index(cat) if cat in CATEGORY_ORDER else len(CATEGORY_ORDER)
