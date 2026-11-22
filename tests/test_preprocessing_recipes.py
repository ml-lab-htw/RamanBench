"""Named recipes (``raman_bench.preprocessing.recipes``): every recipe runs, its documented step order is
the order the code applies (the recipe equals the chain of its one-step transformers in that order), and
its benchmark hyperparameters are the transformer's parameters."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from raman_bench.preprocessing import recipes as R
from raman_bench.preprocessing import transformers as T


@pytest.fixture(scope="module")
def spectra() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    shifts = np.linspace(400, 1800, 400)
    peaks = sum(np.exp(-((shifts - c) / w) ** 2) for c, w in ((620, 12), (1000, 8), (1450, 20)))
    X = rng.uniform(0.5, 2, (60, 1)) * peaks + np.linspace(0, 1, 400) * rng.uniform(0, 2, (60, 1))
    X = X + rng.normal(scale=0.02, size=X.shape)
    X[3, 200] += 30  # a spike
    return pd.DataFrame(X, columns=shifts)


def _one_step(step: str, params: dict):
    """The one-step transformer of ``step`` with the recipe's overrides of its parameters."""
    if step == "standard_scaling":
        return StandardScaler()
    cls = next(c for c in T._OneStep.__subclasses__() if c._STEP == step)
    kwargs = {a: params[p] for a, p in cls._PARAMS.items() if p in params}
    return cls(**kwargs)


def test_the_benchmark_recipes():
    assert len(R.recipe_names()) == 26 and "none" in R.recipe_names()
    for name in R.recipe_names():
        assert set(R.recipe_definition(name)["preprocessing"]) <= set(R.STEP_ORDER)
        assert R.recipe_description(name)


def test_every_recipe_transforms(spectra):
    for name in R.recipe_names():
        Z = np.asarray(R.recipe(name).fit_transform(spectra), dtype=float)
        assert Z.shape[0] == len(spectra) and np.isfinite(Z).all(), name


@pytest.mark.parametrize("name", [n for n in R.recipe_names() if n not in ("prep_ensemble", "gcu_lvse", "none")])
def test_steps_run_in_the_documented_order(spectra, name):
    params = R.recipe_definition(name)["preprocessing_params"]
    chain = make_pipeline(*[_one_step(s, params) for s in R.recipe_steps(name)])
    np.testing.assert_allclose(np.asarray(R.recipe(name).fit_transform(spectra), float),
                               np.asarray(chain.fit_transform(spectra), float), rtol=1e-10, atol=1e-10)


def test_the_order_check_can_fail(spectra):
    """Pooling runs after SNV; the other order gives a different result."""
    assert R.recipe_steps("bc_snv_avgpool") == ["baseline_correction", "snv", "avgpool"]
    wrong = make_pipeline(T.AsLSBaseline(), T.AveragePooling(), T.SNV())
    assert not np.allclose(np.asarray(R.recipe("bc_snv_avgpool").fit_transform(spectra), float),
                           np.asarray(wrong.fit_transform(spectra), float))


def test_hyperparameters_are_the_transformers_parameters(spectra):
    for name in R.recipe_names():
        assert R.recipe_hyperparameters(name) == R.recipe(name).fit(spectra).params_, name
    assert R.recipe_hyperparameters("sg_d2_snv")["prep_deriv_order"] == 2
    assert R.recipe_hyperparameters("none")["prep_snv_enabled"] is False


def test_descriptions():
    assert R.recipe_description("sg_rubberband_snv") == "Savitzky-Golay smoothing -> rubberband baseline -> SNV"
    assert R.recipe_description("bc_snv_avgpool") == "AsLS baseline -> SNV -> average pooling"


def test_unknown_recipe():
    with pytest.raises(KeyError):
        R.recipe("snv_bc")


def test_gcu_is_seeded(spectra, monkeypatch):
    """GCU's NMF gets a fixed seed (default 0) in the mixin path and the transformer, so repeated fits
    agree; the seed is a parameter (``prep_gcu_random_state``, ``GCU(random_state=...)``)."""
    from raman_bench.preprocessing import mixin

    a = np.asarray(R.recipe("gcu").fit_transform(spectra), float)
    b = np.asarray(R.recipe("gcu").fit_transform(spectra), float)
    np.testing.assert_array_equal(a, b)
    assert R.recipe_hyperparameters("gcu")["prep_gcu_random_state"] == 0

    seeds = []
    real = mixin.gcu_fit
    monkeypatch.setattr(mixin, "gcu_fit", lambda X, **kw: seeds.append(kw["random_state"]) or real(X, **kw))
    T.GCU(random_state=7).fit(spectra)
    R.recipe("gcu_lvse").fit(spectra)
    assert seeds == [7, 0]


def test_ramanpfn_seeds_gcu_by_default():
    pytest.importorskip("autogluon.tabular")
    from raman_bench.models.custom.ramanpfn.model import RamanPFNModel

    assert RamanPFNModel().get_params()["random_state"] == 0
