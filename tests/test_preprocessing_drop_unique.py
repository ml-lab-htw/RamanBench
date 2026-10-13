"""Regression tests for the AutoGluon ``drop_unique`` / stateful-prep
fold-width mismatch bug, and for the scope-narrowing fix that followed it.

AutoGluon's ``drop_unique=True`` default drops columns that are constant
within the ``X`` a model's own ``_fit()`` receives, *before* our stateful
prep steps (scaler, MSC/EMSC, GCU/LVSE, and the ``_wavenumbers`` axis
``crop_physical`` indexes) fit on that same ``X`` -- desyncing fit-width
from the narrower transform-width AutoGluon later replays, e.g.::

    ValueError: X has 49 features, but StandardScaler is expecting 50
    features as input.

Fix: disable ``drop_unique`` only on model instances with a stateful prep
step enabled (``RamanPreprocessingMixin._has_stateful_prep_enabled`` /
``_get_default_auxiliary_params``), leaving no-recipe models' default
AutoGluon behavior untouched. See PR #142 for the investigation history
(confounded ``TabularPredictor``-level repro attempt, etc.).
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("autogluon")

from raman_bench.preprocessing.wrapped_models import PREPROCESSED_MODELS  # noqa: E402

N_SAMPLES = 40
N_FEATURES = 50


def _make_data_with_constant_column(n_samples=N_SAMPLES, n_features=N_FEATURES, seed=0):
    """40 rows x 50 cols, one genuinely constant column -- the deterministic,
    single-fit-call form of the synthetic repro from commit 85be2cf (see the
    module docstring for why this form, rather than a bagged
    ``TabularPredictor`` one, is the faithful regression test here).
    """
    rng = np.random.RandomState(seed)
    X = pd.DataFrame(
        rng.rand(n_samples, n_features),
        columns=[f"w{i:04d}" for i in range(n_features)],
    )
    X.iloc[:, 0] = 0.0
    y = pd.Series(rng.rand(n_samples))
    return X, y


def test_stateful_prep_survives_fit_with_constant_column():
    """The regression repro for the StandardScaler feature-count crash.

    Before any fix, AutoGluon's own ``drop_unique=True`` default drops the
    constant column from ``self._features_internal`` the moment
    ``initialize()`` runs, while ``RamanPreprocessingMixin._fit`` has
    already captured the full (undropped) width to fit ``self.scaler`` on --
    so the model's own ``_fit`` (which calls ``self.preprocess(X)``
    internally) feeds ``_preprocess_transform`` a narrower ``X`` than the
    scaler expects, raising ``ValueError: X has 49 features, but
    StandardScaler is expecting 50 features as input.``. Must fail on the
    pre-fix code and pass after it.
    """
    cls = PREPROCESSED_MODELS["KNN"]
    X, y = _make_data_with_constant_column()

    model = cls(hyperparameters={"prep_scaling_enabled": True})
    model.fit(X=X, y=y)

    preds = model.predict(X.iloc[:5])
    assert len(preds) == 5
    assert not pd.isna(preds).any()


def test_no_recipe_model_keeps_default_drop_unique_behavior():
    """Required scope-narrowing regression test: a ``Prep_*`` instance
    constructed with NO recipe hyperparameters at all must keep AutoGluon's
    unmodified default ``drop_unique=True`` behavior -- i.e. the fix must not
    change behavior for the (majority of the v1 leaderboard) no-recipe case.
    """
    cls = PREPROCESSED_MODELS["KNN"]
    X = pd.DataFrame(np.random.RandomState(0).rand(10, 5))
    y = pd.Series(np.random.RandomState(0).rand(10))

    model = cls(hyperparameters={})
    assert model._has_stateful_prep_enabled() is False

    model.initialize(X=X, y=y)
    assert model.aux_params.drop_unique is True


def test_recipe_model_disables_drop_unique():
    """Sanity check for the positive case: an enabled stateful recipe step
    flips the scope-narrowed override on for that model instance only.
    """
    cls = PREPROCESSED_MODELS["KNN"]
    X = pd.DataFrame(np.random.RandomState(0).rand(10, 5))
    y = pd.Series(np.random.RandomState(0).rand(10))

    model = cls(hyperparameters={"prep_scaling_enabled": True})
    assert model._has_stateful_prep_enabled() is True

    model.initialize(X=X, y=y)
    assert model.aux_params.drop_unique is False


def test_ensemble_block_stateful_step_also_disables_drop_unique():
    """A stateful step enabled only inside a ``prep_ensemble_blocks`` entry
    (no top-level ``prep_*_enabled`` flag) must also be detected -- the
    ensemble mechanism fits block-level MSC/EMSC/GCU/LVSE state the same way
    the base pipeline does (see ``_preprocess_fit_ensemble``), so it has the
    same fold-width-mismatch exposure.
    """
    cls = PREPROCESSED_MODELS["KNN"]
    model = cls(
        hyperparameters={
            "prep_ensemble_enabled": True,
            "prep_ensemble_blocks": [{"snv": True}, {"msc": True}],
        }
    )
    assert model._has_stateful_prep_enabled() is True


def test_crop_physical_survives_fit_with_constant_column():
    """Same bug class as the StandardScaler case, but for ``crop_physical``:
    ``_fit`` captures ``self._wavenumbers`` from the pre-drop ``X`` columns,
    then indexes the (potentially narrower, post-``drop_unique``) ``X`` with
    it. Without the fix, a constant column desyncs that axis from the
    columns actually being cropped -- either a length mismatch or a
    silently-shifted crop. Uses real float wavenumber column names (as
    RamanBench spectra actually have), with one genuinely constant channel.
    """
    cls = PREPROCESSED_MODELS["KNN"]
    wavenumbers = np.linspace(200.0, 2200.0, N_FEATURES)
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.rand(N_SAMPLES, N_FEATURES), columns=wavenumbers)
    X.iloc[:, 0] = 0.0  # constant column, within the crop_physical window below
    y = pd.Series(rng.rand(N_SAMPLES))

    model = cls(
        hyperparameters={
            "prep_crop_physical_enabled": True,
            "prep_crop_physical_start_cm": 200.0,
            "prep_crop_physical_end_cm": 1200.0,
        }
    )
    assert model._has_stateful_prep_enabled() is True
    model.fit(X=X, y=y)

    preds = model.predict(X.iloc[:5])
    assert len(preds) == 5
    assert not pd.isna(preds).any()
