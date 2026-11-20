"""scikit-learn transformers of ``raman_bench.preprocessing.transformers``.

The central check is that a :class:`RamanPreprocessor` gives exactly what a benchmark model applies:
for each recipe, hyperparameters are built as ``scripts/run_experiment.py`` builds them, a real
``Prep_PLS`` (whose class switches steps on by default) and ``Prep_RF`` are fitted, and their
recipe parameters and their transform of held-out spectra must equal the transformer's.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline

from raman_bench.preprocessing import (
    MSC,
    SNV,
    AirPLSBaseline,
    AsLSBaseline,
    AveragePooling,
    Decimation,
    RamanPreprocessor,
    SavitzkyGolayDerivative,
    SavitzkyGolaySmoothing,
    recipe_params,
)
from raman_bench.preprocessing.raman_preprocessing import (
    avg_pool_spectra,
    baseline_correction_airpls,
    baseline_correction_asls,
    decimate_spectra,
    denoise_savgol,
    multiplicative_scatter_correction_transform,
    savgol_derivative,
    snv,
)

# Recipe forms as RamanBench recipe configs give them: (preprocessing, preprocessing_params).
RECIPES = {
    "none": ({}, None),
    "snv": ({"snv": True}, None),
    "bc_sg_snv": ({"baseline_correction": True, "denoising": True, "snv": True}, None),
    "bc_snv_avgpool": ({"baseline_correction": True, "snv": True, "avgpool": True}, None),
    "decimate": ({"decimate": True}, None),
    "sg_d2_snv": ({"derivative": True, "snv": True}, {"prep_deriv_order": 2, "prep_deriv_wl": 15}),
    "airpls_snv": ({"airpls": True, "snv": True}, None),
    "sg_rubberband_snv": ({"denoising": True, "rubberband": True, "snv": True}, None),
    "crr_snv": ({"cosmic_ray_removal": True, "snv": True}, None),
    "wavelet_bc_snv": ({"wavelet_denoise": True, "baseline_correction": True, "snv": True}, None),
    "msc_region": ({"msc": True}, {"prep_msc_start_frac": 0.2, "prep_msc_end_frac": 0.6}),
    "emsc": ({"emsc": True}, None),
    "stdscale": ({"standard_scaling": True}, None),
    "gcu_lvse": ({"gcu": True, "lvse": True}, None),
    "all": (
        {
            "cosmic_ray_removal": True,
            "baseline_correction": True,
            "denoising": True,
            "msc": True,
            "snv": True,
            "standard_scaling": True,
        },
        None,
    ),
    "prep_ensemble": (
        {},
        {
            "prep_ensemble_enabled": True,
            "prep_ensemble_blocks": [{"snv": True}, {"emsc": True}, {"airpls": True, "snv": True}, {"snv": False}],
        },
    ),
}


def _spectra(n=60, p=120, seed=0):
    """Positive spectra with a sloped baseline, two peaks, multiplicative scatter and noise."""
    rng = np.random.default_rng(seed)
    axis = np.linspace(400.0, 1800.0, p)
    peaks = np.exp(-((axis - 900) / 25) ** 2)[None, :] * rng.uniform(1, 3, (n, 1))
    peaks += np.exp(-((axis - 1450) / 40) ** 2)[None, :] * rng.uniform(0.5, 2, (n, 1))
    baseline = rng.uniform(0.5, 2, (n, 1)) * (1 + axis[None, :] / 1800)
    X = rng.uniform(0.8, 1.2, (n, 1)) * (peaks + baseline) + rng.normal(0, 0.01, (n, p))
    y = peaks.max(axis=1) + rng.normal(0, 0.05, n)
    return pd.DataFrame(X, columns=axis), pd.Series(y)


# --- scikit-learn behavior -------------------------------------------------------------------


def test_clone_and_params_round_trip():
    t = RamanPreprocessor({"snv": True}, {"prep_avgpool_size": 3})
    c = clone(t)
    assert c.get_params() == t.get_params()
    assert SavitzkyGolayDerivative(order=2).set_params(window_length=15).get_params() == {
        "order": 2,
        "window_length": 15,
        "polyorder": 2,
    }


def test_pipeline_and_grid_search():
    X, y = _spectra()
    pipe = make_pipeline(SavitzkyGolaySmoothing(), SNV(), PLSRegression(n_components=3))
    grid = GridSearchCV(pipe, {"savitzkygolaysmoothing__window_length": [7, 11]}, cv=3).fit(X, y)
    assert grid.best_params_["savitzkygolaysmoothing__window_length"] in (7, 11)
    assert grid.predict(X.iloc[:5]).shape[0] == 5


def test_feature_names_and_width():
    X, _ = _spectra(p=103)
    t = RamanPreprocessor({"snv": True, "avgpool": True}).fit(X)
    assert t.n_features_in_ == 103 and t.n_features_out_ == 20
    assert t.transform(X).shape == (len(X), 20)
    assert len(t.get_feature_names_out()) == 20
    same = SNV().fit(X)
    assert list(same.get_feature_names_out()) == list(X.columns)


def test_numpy_input_and_wrong_width():
    X, _ = _spectra()
    t = SNV().fit(X.to_numpy())
    with pytest.raises(ValueError, match="channels"):
        t.transform(X.to_numpy()[:, :50])


def test_override_of_a_decided_step_raises():
    with pytest.raises(ValueError, match="decided by `preprocessing`"):
        recipe_params({"snv": True}, {"prep_snv_enabled": False})


def test_recipe_params_resolution():
    p = recipe_params({"snv": True})
    assert p["prep_snv_enabled"] and not p["prep_bl_enabled"] and not p["prep_scaling_enabled"]
    assert not p["prep_aug_enabled"]
    assert recipe_params({"augmentation": True})["prep_aug_enabled"] is False
    assert recipe_params(None, {"prep_snv_enabled": True})["prep_snv_enabled"]


# --- single steps against the NumPy functions -----------------------------------------------


@pytest.mark.parametrize(
    "transformer, expected",
    [
        (SNV(), lambda X: snv(X)),
        (AveragePooling(size=4), lambda X: avg_pool_spectra(X, 4)),
        (Decimation(step=3), lambda X: decimate_spectra(X, 3)),
        (AsLSBaseline(lam=1e4, p=0.05), lambda X: baseline_correction_asls(X, lam=1e4, p=0.05)),
        (AirPLSBaseline(), lambda X: baseline_correction_airpls(X)),
        (SavitzkyGolaySmoothing(window_length=7, polyorder=2), lambda X: denoise_savgol(X, 7, 2)),
        (SavitzkyGolayDerivative(order=2, window_length=15), lambda X: savgol_derivative(X, 15, 2, 2)),
    ],
)
def test_stateless_steps_match_functions(transformer, expected):
    X, _ = _spectra()
    X_np = X.to_numpy(dtype=np.float64)
    np.testing.assert_allclose(transformer.fit_transform(X), expected(X_np.copy()), rtol=1e-12, atol=1e-12)


def test_msc_reference_comes_from_the_training_spectra():
    X, _ = _spectra()
    train, test = X.iloc[:40], X.iloc[40:]
    t = MSC(start_frac=0.1, end_frac=0.9).fit(train)
    expected = multiplicative_scatter_correction_transform(
        test.to_numpy(float), train.to_numpy(float).mean(axis=0), start=0.1, end=0.9
    )
    np.testing.assert_allclose(t.transform(test), expected, rtol=1e-12)


def test_single_step_equals_recipe_with_that_step():
    X, _ = _spectra()
    a = SavitzkyGolayDerivative(order=2, window_length=15).fit_transform(X)
    b = RamanPreprocessor({"derivative": True}, {"prep_deriv_order": 2, "prep_deriv_wl": 15}).fit_transform(X)
    np.testing.assert_array_equal(a, b)


def test_imports_and_runs_without_autogluon_or_tabarena():
    code = (
        "import sys; sys.modules['autogluon'] = None; sys.modules['tabarena'] = None\n"
        "import numpy as np\n"
        "from raman_bench.preprocessing import RamanPreprocessor\n"
        "X = np.random.default_rng(0).uniform(1, 2, (20, 60))\n"
        "out = RamanPreprocessor({'baseline_correction': True, 'snv': True, 'avgpool': True}).fit_transform(X)\n"
        "assert out.shape == (20, 12), out.shape\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


# --- identical to the benchmark models' own preprocessing -----------------------------------


@pytest.mark.parametrize("model_key", ["PLS", "RF"])
@pytest.mark.parametrize("recipe", sorted(RECIPES))
def test_equals_the_benchmark_models_preprocessing(model_key, recipe):
    pytest.importorskip("autogluon.tabular")
    from raman_bench.model import build_prep_model_hyperparameters
    from raman_bench.preprocessing.wrapped_models import PREPROCESSED_MODELS

    preprocessing, preprocessing_params = RECIPES[recipe]
    # as scripts/run_experiment.py: the recipe file's preprocessing dict, normalized (missing steps off)
    from raman_bench.config import _normalize_preprocessing_config

    cfg = {"_prep_restriction": _normalize_preprocessing_config({"preprocessing": preprocessing})["preprocessing_config"]}
    cls = PREPROCESSED_MODELS[model_key]
    hyperparameters = build_prep_model_hyperparameters(cls, cfg, preprocessing_params=preprocessing_params, optimize=False)

    X, y = _spectra()
    train, test = X.iloc[:45], X.iloc[45:]
    problem_type = "regression"
    model = cls(hyperparameters=hyperparameters, problem_type=problem_type)
    model.fit(X=train, y=y.iloc[:45])

    t = RamanPreprocessor(preprocessing, preprocessing_params).fit(train)
    recipe_used = getattr(model, "_prep_recipe", None)
    if recipe_used is None:  # nothing switched on: the model saw the raw spectra
        assert not any(v for k, v in t.params_.items() if k.endswith("_enabled"))
        np.testing.assert_array_equal(t.transform(test), test.to_numpy(float))
        return
    keys = [k for k in t.params_ if k != "prep_aug_enabled" and not k.startswith("prep_aug_")]
    defaults = recipe_params(None)  # the mixin reads a missing parameter as its default
    assert {k: recipe_used.get(k, defaults[k]) for k in keys} == {k: t.params_[k] for k in keys}

    test_np = test.to_numpy(np.float64)
    if t._ensemble:
        expected = model._preprocess_transform_ensemble(test_np)
    else:
        expected = model._preprocess_transform(test_np, params=recipe_used)
    np.testing.assert_allclose(t.transform(test), expected, rtol=1e-10, atol=1e-12)
