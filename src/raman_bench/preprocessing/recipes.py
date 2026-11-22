"""Named preprocessing recipes: the 26 recipes of the Raman preprocessing benchmark.

A recipe is a set of preprocessing steps in the recipe-config schema (``"preprocessing"``: step key ->
on, ``"preprocessing_params"``: ``prep_*`` overrides). The steps of a recipe always run in the fixed order
of :data:`STEP_ORDER`, whatever order its name lists them in. Each recipe can be used two ways:

- as a scikit-learn transformer, :func:`recipe` (a :class:`~raman_bench.preprocessing.RamanPreprocessor`)::

      from sklearn.pipeline import make_pipeline
      from sklearn.cross_decomposition import PLSRegression
      from raman_bench.preprocessing import recipe

      model = make_pipeline(recipe("bc_snv_avgpool"), PLSRegression(10, scale=False))

- inside RamanBench's benchmark models, :func:`recipe_hyperparameters` (the ``prep_*`` hyperparameters
  of the ``Prep_*`` wrappers, e.g. ``evaluate_estimator(..., hyperparameters=recipe_hyperparameters("snv"))``).

Both apply the same code, so a recipe gives the same output either way.
"""

from __future__ import annotations

import copy

from raman_bench.preprocessing.transformers import RamanPreprocessor, recipe_params

# The order in which RamanPreprocessingMixin applies the steps (``_spectrumwise``, then ``_preprocess_fit``):
# crop, cosmic rays, smoothing, baseline; scatter correction, derivative, normalisation; then pooling and
# standard scaling. GCU and LVSE run last, each on the same input; when both are on, their outputs are
# concatenated. ``tests/test_preprocessing_recipes.py`` checks this order against the code.
STEP_ORDER = (
    "crop_physical", "crop", "cosmic_ray_removal", "denoising", "wavelet_denoise", "baseline_correction",
    "airpls", "arpls", "rubberband", "msc", "emsc", "derivative", "snv", "vecnorm", "avgpool", "decimate",
    "standard_scaling", "gcu", "lvse",
)

STEP_NAMES = {
    "crop_physical": "crop (cm^-1)",
    "crop": "crop (fraction of channels)",
    "cosmic_ray_removal": "cosmic-ray removal",
    "denoising": "Savitzky-Golay smoothing",
    "wavelet_denoise": "wavelet denoising",
    "baseline_correction": "AsLS baseline",
    "airpls": "airPLS baseline",
    "arpls": "arPLS baseline",
    "rubberband": "rubberband baseline",
    "avgpool": "average pooling",
    "decimate": "decimation",
    "msc": "MSC",
    "emsc": "EMSC",
    "derivative": "Savitzky-Golay derivative",
    "snv": "SNV",
    "vecnorm": "vector normalisation",
    "standard_scaling": "standard scaling",
    "gcu": "GCU",
    "lvse": "LVSE",
}

RECIPES: dict[str, dict] = {
    "none": {"preprocessing": {}},
    "snv": {"preprocessing": {"snv": True}},
    "msc": {"preprocessing": {"msc": True}},
    "emsc": {"preprocessing": {"emsc": True}},
    "sg": {"preprocessing": {"denoising": True}},
    "bc": {"preprocessing": {"baseline_correction": True}},
    "stdscale": {"preprocessing": {"standard_scaling": True}},
    "bc_sg": {"preprocessing": {"baseline_correction": True, "denoising": True}},
    "bc_snv": {"preprocessing": {"baseline_correction": True, "snv": True}},
    "bc_sg_snv": {"preprocessing": {"baseline_correction": True, "denoising": True, "snv": True}},
    "bc_sg_msc": {"preprocessing": {"baseline_correction": True, "denoising": True, "msc": True}},
    "airpls_snv": {"preprocessing": {"airpls": True, "snv": True}},
    "arpls_snv": {"preprocessing": {"arpls": True, "snv": True}},
    "sg_rubberband_snv": {"preprocessing": {"denoising": True, "rubberband": True, "snv": True}},
    "wavelet_bc_snv": {"preprocessing": {"wavelet_denoise": True, "baseline_correction": True, "snv": True}},
    "crr_snv": {"preprocessing": {"cosmic_ray_removal": True, "snv": True}},
    "sg_d1_snv": {"preprocessing": {"derivative": True, "snv": True}},
    "sg_d2_snv": {"preprocessing": {"derivative": True, "snv": True},
                  "preprocessing_params": {"prep_deriv_order": 2, "prep_deriv_wl": 15}},
    "all": {"preprocessing": {"cosmic_ray_removal": True, "denoising": True, "baseline_correction": True,
                              "msc": True, "snv": True, "standard_scaling": True}},
    "avgpool": {"preprocessing": {"avgpool": True}},
    "decimate": {"preprocessing": {"decimate": True}},
    "bc_snv_avgpool": {"preprocessing": {"baseline_correction": True, "snv": True, "avgpool": True}},
    "gcu": {"preprocessing": {"gcu": True}},
    "lvse": {"preprocessing": {"lvse": True}},
    "gcu_lvse": {"preprocessing": {"gcu": True, "lvse": True}},
    # Four recipes fitted side by side, their outputs concatenated column-wise.
    "prep_ensemble": {"preprocessing": {},
                      "preprocessing_params": {"prep_ensemble_enabled": True, "prep_ensemble_blocks": [
                          {"snv": True}, {"emsc": True}, {"airpls": True, "snv": True}, {"snv": False}]}},
}


def recipe_names() -> list[str]:
    """Names of the named recipes, sorted."""
    return sorted(RECIPES)


def recipe_definition(name: str) -> dict:
    """``{"preprocessing": {...}, "preprocessing_params": {...}}`` of a named recipe (a copy)."""
    if name not in RECIPES:
        raise KeyError(f"unknown recipe {name!r}; the named recipes are {recipe_names()}")
    d = copy.deepcopy(RECIPES[name])
    d.setdefault("preprocessing_params", {})
    return d


def recipe_steps(name: str) -> list[str]:
    """The steps of a named recipe in the order they run (step keys of :data:`STEP_ORDER`).

    ``prep_ensemble`` has no single order: it fits its blocks side by side (see :data:`RECIPES`).
    """
    d = recipe_definition(name)
    if d["preprocessing_params"].get("prep_ensemble_enabled"):
        raise ValueError(f"{name} runs its blocks side by side: {d['preprocessing_params']['prep_ensemble_blocks']}")
    on = {k for k, v in d["preprocessing"].items() if v}
    return [s for s in STEP_ORDER if s in on]


def recipe_description(name: str) -> str:
    """A readable description, e.g. ``"AsLS baseline -> average pooling -> SNV"``."""
    if name == "prep_ensemble":
        return "ensemble of SNV, EMSC, airPLS baseline -> SNV, and no preprocessing, concatenated"
    steps = recipe_steps(name)
    if not steps:
        return "no preprocessing"
    names = [STEP_NAMES[s] for s in steps]
    if "derivative" in steps:
        order = recipe_hyperparameters(name)["prep_deriv_order"]
        names[steps.index("derivative")] = f"Savitzky-Golay {'first' if order == 1 else 'second'} derivative"
    if "gcu" in steps and "lvse" in steps:  # side by side, not chained
        names = names[:-2] + ["GCU + LVSE (concatenated)"]
    return " -> ".join(names)


def recipe(name: str, wavenumbers=None) -> RamanPreprocessor:
    """A named recipe as a fresh scikit-learn transformer."""
    d = recipe_definition(name)
    return RamanPreprocessor(d["preprocessing"], d["preprocessing_params"] or None, wavenumbers=wavenumbers)


def recipe_hyperparameters(name: str) -> dict:
    """The ``prep_*`` hyperparameters a benchmark model (``Prep_*`` wrapper) receives for a named recipe."""
    d = recipe_definition(name)
    return recipe_params(d["preprocessing"], d["preprocessing_params"])
