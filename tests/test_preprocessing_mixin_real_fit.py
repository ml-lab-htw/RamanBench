"""Regression tests for the AutoGluon ``self.features`` staleness bug.

``_output_feature_cols`` (see ``test_preprocessing_mixin_shape.py``) fixes
building a *correctly shaped* DataFrame after a shape-changing preprocessing
step (``crop``/``gcu``/``lvse``/the preprocessing-ensemble mechanism), but
that alone is not sufficient: AutoGluon's ``AbstractModel.fit()`` snapshots
``self.features``/``self.feature_metadata`` from the *original*,
pre-preprocessing ``X.columns`` before ``RamanPreprocessingMixin._fit()`` (our
override) ever runs. Many models' own ``_fit``/``_predict_proba`` then call
``self.preprocess(X)`` internally (e.g. ``KNNModel._fit`` does
``X = self.preprocess(X, y=y)``, and ``AbstractModel._predict_proba`` does the
same) — which does ``X[self.features]`` against our already reshaped ``X``,
using the *stale* original column list. That raised::

    KeyError: "None of [Index([0, 1, 2, ..., 1003], dtype='int64', length=1004)]
    are in the [columns]"

The tests below exercise the *real* AutoGluon ``AbstractModel.fit()`` ->
``_fit()`` -> (model-internal) ``self.preprocess()`` path — not the pure
``_preprocess_fit``/``_preprocess_transform`` functions in isolation, which
have their own tests that passed throughout even while this bug shipped,
since they never touch ``self.features``. See
``RamanPreprocessingMixin._resync_autogluon_features`` in ``mixin.py`` for
the fix.
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("autogluon")

from raman_bench.preprocessing.wrapped_models import PREPROCESSED_MODELS  # noqa: E402

N_SAMPLES = 40
N_FEATURES = 200


def _make_data(n_samples=N_SAMPLES, n_features=N_FEATURES, n_classes=3, seed=0):
    rng = np.random.RandomState(seed)
    X = pd.DataFrame(rng.rand(n_samples, n_features))
    y = pd.Series(rng.randint(0, n_classes, n_samples))
    return X, y


SHAPE_CHANGING_HYPERPARAMS = {
    "crop": {"prep_crop_enabled": True},
    "gcu": {"prep_gcu_enabled": True},
    "lvse": {"prep_lvse_enabled": True},
    "ensemble": {
        "prep_ensemble_enabled": True,
        "prep_ensemble_blocks": [{"snv": True}, {"crop": True}],
    },
}


@pytest.mark.parametrize("mechanism", list(SHAPE_CHANGING_HYPERPARAMS))
@pytest.mark.parametrize("model_key", ["KNN", "PLS"])
def test_shape_changing_preprocessing_survives_real_autogluon_fit(model_key, mechanism):
    """Fit + predict through the real AutoGluon ``AbstractModel.fit()`` path.

    Uses ``model.fit(X=..., y=...)`` (the public ``AbstractModel.fit``
    entrypoint, which is what actually sets ``self.features`` before
    ``_fit()`` runs) rather than calling ``_fit`` directly, so this exercises
    exactly the code path that raised the ``KeyError`` in production.
    """
    cls = PREPROCESSED_MODELS[model_key]
    X, y = _make_data()
    hyperparameters = dict(SHAPE_CHANGING_HYPERPARAMS[mechanism])

    model = cls(hyperparameters=hyperparameters)
    model.fit(X=X, y=y)

    preds = model.predict(X.iloc[:10])
    assert len(preds) == 10
    assert not pd.isna(preds).any()

    proba = model.predict_proba(X.iloc[:10])
    proba = np.asarray(proba)
    assert not np.isnan(proba).any()
    # Non-degenerate: predictions should not all collapse to a single class.
    assert len(set(np.asarray(preds))) > 1 or proba.shape[-1] == 1


@pytest.mark.parametrize("mechanism", list(SHAPE_CHANGING_HYPERPARAMS))
@pytest.mark.parametrize("model_key", ["KNN", "RF", "PLS"])
def test_shape_changing_preprocessing_survives_bagged_predict(model_key, mechanism):
    """The production scenario: a bagged model (``_c1_BAG_L1``) + a shape-changing
    recipe, then ``predictor.predict()``.

    Before fix option 1 (transform moved into ``_preprocess``, ``self.features``
    left alone) this raised ``KeyError: "None of [Index(['feature_0', ...])] are
    in the [columns]"`` on the bagged predict path —
    ``BaggedEnsembleModel._predict_proba_internal`` ->
    ``child.preprocess(X_raw, preprocess_stateful=False)`` ->
    ``_preprocess_nonadaptive`` -> ``X_raw[self.features]`` with ``self.features``
    mutated to the post-transform names. Fit + OOF scoring succeeded, so this only
    reproduces via the full ``TabularPredictor.fit(num_bag_folds=...)`` +
    ``predict`` path. See autogluon/autogluon#5898.
    """
    tabular = pytest.importorskip("autogluon.tabular")
    TabularPredictor = tabular.TabularPredictor

    rng = np.random.RandomState(0)
    n, p = 200, 200
    Xarr = rng.rand(n, p)
    yarr = Xarr[:, :20].sum(axis=1) + rng.normal(scale=0.1, size=n)
    df = pd.DataFrame(Xarr, columns=[f"w{i:04d}" for i in range(p)])
    df["target"] = yarr
    train, test = df.iloc[:160], df.iloc[160:].drop(columns=["target"])

    hp = dict(SHAPE_CHANGING_HYPERPARAMS[mechanism])
    hp["ag_args_ensemble"] = {"fold_fitting_strategy": "sequential_local"}

    predictor = TabularPredictor(label="target", problem_type="regression", verbosity=0)
    predictor.fit(
        train_data=train,
        hyperparameters={PREPROCESSED_MODELS[model_key]: hp},
        num_bag_folds=2,
        num_bag_sets=1,
        fit_weighted_ensemble=False,
    )
    preds = predictor.predict(test)
    assert len(preds) == len(test)
    assert not preds.isna().any()
