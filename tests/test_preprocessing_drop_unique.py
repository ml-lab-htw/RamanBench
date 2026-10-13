"""Regression tests for the AutoGluon ``drop_unique`` / stateful-prep
fold-width mismatch bug, and for the scope-narrowing fix that followed it.

Root cause (see ``RamanPreprocessingMixin._has_stateful_prep_enabled``'s
docstring in ``mixin.py`` for the full trace): AutoGluon's own
``AbstractModel`` defaults ``aux_params.drop_unique=True``. ``_fit()``
(``RamanPreprocessingMixin._fit``) fits a stateful mixin transform
(``StandardScaler`` for ``prep_scaling_enabled``; MSC/EMSC's reference
spectrum; GCU/LVSE's fitted state) on ``X``'s *raw* width, captured directly
from the ``X`` argument AutoGluon's own ``fit()`` wrapper hands to ``_fit()``
-- i.e. *before* ``_preprocess_set_features`` has had a chance to drop any
column. If that same ``X`` has a column that is constant
(``nunique(dropna=False) < 2``), ``_preprocess_set_features`` (called once,
inside ``initialize()``, before ``_fit()`` even runs) drops it from
``self._features_internal`` when ``drop_unique=True``. The model's own
``_fit`` (e.g. ``KNNModel._fit``) then calls ``self.preprocess(X, y=y)``,
which re-applies that now-narrower feature alignment *before* calling our
``_preprocess`` hook -- so ``_preprocess_transform`` receives one column
fewer than ``self.scaler``/``self._msc_reference``/etc. was fit with,
raising e.g.::

    ValueError: X has 49 features, but StandardScaler is expecting 50
    features as input.

On real spectral data this is most often triggered by a literal
zero-variance channel within a bagged fold child's own small training
partition (a saturated/clipped detector pixel, a dead channel, a flat
baseline edge) -- fold-content-dependent, hence intermittent, in production.
The test below reproduces the same code-level mechanism deterministically
with a single (non-bagged) ``.fit()`` call on data with one genuinely
constant column: a bagged fold child's ``_fit()`` call sees exactly the same
``RamanPreprocessingMixin``/``AbstractModel`` interaction on its own
(smaller) partition, so this is a faithful minimal repro of the underlying
bug, not a different one. (A bagged, ``TabularPredictor``-level version of
this repro was attempted first but turned out to be confounded by two
unrelated AutoGluon behaviors that also happen to produce the identical
error message: (1) the dataset-level ``AutoMLPipelineFeatureGenerator`` a
``TabularPredictor`` applies before any model ever sees the data already
drops a column that is constant over the *entire* training set, regardless
of any model's own ``drop_unique`` setting -- sidestepping this bug
entirely; (2) a column with few enough distinct values can get tagged a
"bool" special feature type and excluded via ``ignored_type_group_special``,
again regardless of ``drop_unique``. Both would have made a
``TabularPredictor``-level test non-discriminating or misleading about which
mechanism it was actually exercising, so the direct, single-model-instance
repro below is used instead.)

First fix attempt (commit 85be2cf) set
``_default_auxiliary_params_extra = {"drop_unique": False}`` unconditionally
on ``RamanPreprocessingMixin`` -- which also silently changed
``drop_unique`` behavior for every ``Prep_*`` model with NO recipe enabled
at all (most of the v1 leaderboard, e.g. feature-counting/subsampling
foundation models), violating RamanBench's rule that v1 scores must not move
within the 1.x line for reasons unrelated to preprocessing. The fix was
narrowed to only disable ``drop_unique`` when a stateful prep step
(scaling, MSC, EMSC, GCU, LVSE) is actually enabled for that model instance
-- see ``RamanPreprocessingMixin._has_stateful_prep_enabled``/
``_get_default_auxiliary_params``.
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
