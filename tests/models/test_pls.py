import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from autogluon.common import space
from raman_bench.custom_models import PreprocessingPLS, FlexiblePipelinePLS
from raman_bench.models.custom.pls import PLSModel
from raman_bench.pre_pipeline import get_preprocessing_config


# ── Unit: _select_features ────────────────────────────────────────────────────

class TestPLSSelectFeatures:
    """PLSModel._select_features handles column count mismatches at predict time.

    AutoGluon injects an extra NaN column when tuning_data=training_data is used
    for datasets with < 20 samples (the 'tiny dataset' path in model.py).
    _select_features must silently drop extra columns and reorder to match the
    columns seen during _fit.
    """

    def _make_model_with_feature_names(self, feature_names):
        """Create a PLSModel stub with _feature_names set, bypassing __init__."""
        m = object.__new__(PLSModel)
        m._feature_names = list(feature_names)
        return m

    def test_extra_column_is_dropped(self):
        """Extra column in X is silently removed."""
        m = self._make_model_with_feature_names(["a", "b", "c"])
        X = pd.DataFrame({"a": [1.0], "b": [2.0], "c": [3.0], "extra": [np.nan]})
        result = m._select_features(X)
        assert list(result.columns) == ["a", "b", "c"]
        assert result.shape == (1, 3)

    def test_extra_nan_column_is_dropped(self):
        """NaN-only extra column (the AutoGluon injected column) is dropped."""
        m = self._make_model_with_feature_names(["f0", "f1"])
        X = pd.DataFrame({
            "f0": [0.5, 1.0],
            "f1": [0.3, 0.8],
            "autogluon_nan_injection": [np.nan, np.nan],
        })
        result = m._select_features(X)
        assert list(result.columns) == ["f0", "f1"]
        assert not result.isna().any().any()

    def test_column_reordering(self):
        """Columns are reordered to match training order even if X has them shuffled."""
        m = self._make_model_with_feature_names(["a", "b", "c"])
        X = pd.DataFrame({"c": [3.0], "a": [1.0], "b": [2.0]})
        result = m._select_features(X)
        assert list(result.columns) == ["a", "b", "c"]
        np.testing.assert_array_equal(result.values, [[1.0, 2.0, 3.0]])

    def test_missing_column_filled_with_zero(self):
        """A column present in training but absent in X is filled with 0."""
        m = self._make_model_with_feature_names(["a", "b", "c"])
        X = pd.DataFrame({"a": [1.0], "b": [2.0]})  # "c" is missing
        result = m._select_features(X)
        assert list(result.columns) == ["a", "b", "c"]
        assert result["c"].iloc[0] == 0.0

    def test_no_feature_names_passthrough(self):
        """Without _feature_names set, X is returned unchanged."""
        m = object.__new__(PLSModel)
        # _feature_names is not set
        X = pd.DataFrame({"a": [1.0], "extra": [np.nan]})
        result = m._select_features(X)
        assert list(result.columns) == ["a", "extra"]

    def test_non_dataframe_passthrough(self):
        """numpy arrays (no .columns) are returned unchanged."""
        m = self._make_model_with_feature_names(["a", "b"])
        X_np = np.array([[1.0, 2.0, 3.0]])  # 3 cols, model expects 2
        result = m._select_features(X_np)
        assert result is X_np


# ── Integration: tiny dataset triggers AutoGluon tuning_data path ─────────────

class TestPLSTinyDataset:
    """PLSModel survives AutoGluon's post-fit validation when n_samples < 20.

    With fewer than 20 training samples, model.py sets tuning_data=tabular_data.
    AutoGluon's feature generator then produces X_val with an extra column compared
    to X_train (the injected NaN column). Without _select_features, this causes:
        ValueError: X has N+1 features, but PLSRegression is expecting N features.
    """

    def _tiny_regression_df(self, n_samples=15):
        rng = np.random.RandomState(0)
        n_features = 30
        wavenumbers = np.linspace(400, 1800, n_features)
        X = np.array([
            sum(
                rng.uniform(0.5, 2.0) * np.exp(-0.5 * ((wavenumbers - c) / 30) ** 2)
                for c in rng.choice(wavenumbers, size=3, replace=False)
            ) + rng.normal(0, 0.02, n_features)
            for _ in range(n_samples)
        ])
        X = np.clip(X, 0.01, None)
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])
        df["target"] = rng.rand(n_samples) * 10
        return df

    def _tiny_classification_df(self, n_samples=15):
        df = self._tiny_regression_df(n_samples)
        df["target"] = ["A"] * 5 + ["B"] * 5 + ["C"] * 5
        return df

    def test_tiny_regression_does_not_raise(self, tmp_path):
        """Fitting on < 20 regression samples completes without feature-count error."""
        from raman_bench.model import AutoGluonModel
        from raman_data import TASK_TYPE

        model = AutoGluonModel(
            models=["PLS"],
            ensemble=False,
            optimize=False,
            task_type=TASK_TYPE.Regression,
            autogluon_path=str(tmp_path),
            model_extra_params={
                "prep_bl_enabled": False,
                "prep_denoise_enabled": False,
                "prep_snv_enabled": False,
            },
        )
        df = self._tiny_regression_df(n_samples=15)
        model.fit(df)  # must not raise ValueError about feature count
        preds = model.predict(df.drop(columns=["target"]))
        assert len(preds) == 15
        assert np.isfinite(preds).all()

    def test_tiny_classification_does_not_raise(self, tmp_path):
        """Fitting on < 20 classification samples completes without feature-count error."""
        from raman_bench.model import AutoGluonModel
        from raman_data import TASK_TYPE

        model = AutoGluonModel(
            models=["PLS"],
            ensemble=False,
            optimize=False,
            task_type=TASK_TYPE.Classification,
            autogluon_path=str(tmp_path),
            model_extra_params={
                "prep_bl_enabled": False,
                "prep_denoise_enabled": False,
                "prep_snv_enabled": False,
            },
        )
        df = self._tiny_classification_df(n_samples=15)
        model.fit(df)
        preds = model.predict(df.drop(columns=["target"]))
        assert len(preds) == 15
        assert set(preds.unique()).issubset({"A", "B", "C"})


def generate_synthetic_raman(n_samples=50, n_features=100):
    """Generates synthetic spectral data with a known relationship."""
    rng = np.random.default_rng(42)
    x = np.linspace(0, 10, n_features)

    # Create spectra: base peak + noise
    spectra = []
    targets = []
    for _ in range(n_samples):
        intensity = rng.uniform(0.5, 2.0)
        peak_pos = rng.uniform(4, 6)
        spectrum = intensity * np.exp(-(x - peak_pos) ** 2) + rng.normal(0, 0.05, n_features)
        spectra.append(spectrum)
        targets.append(intensity * 10)  # Target is directly proportional to intensity

    df = pd.DataFrame(spectra, columns=[f"w_{i}" for i in range(n_features)])
    df['target'] = targets
    return df


def test_pls_preprocessing_hpo_pipeline(tmp_path):
    """
    Test if the RamanPreprocessingModel integrates with AutoGluon
    and successfully completes a Hyperparameter Optimization (HPO) cycle.
    """
    # 1. Setup Data
    train = generate_synthetic_raman(n_samples=40)
    test = generate_synthetic_raman(n_samples=10)

    # 3. Define Search Space
    # We use very small ranges/trials to keep the test fast
    hyperparameters = {
        FlexiblePipelinePLS: {
            "proc_pipeline": get_preprocessing_config()[0],
            "sg_window_length": space.Int(lower=5, upper=11),
            "sg_deriv": space.Categorical(0, 1),
            "n_components": space.Int(lower=2, upper=5),
        }
    }

    # 4. Initialize and Fit
    save_path = str(tmp_path / "ag_test")
    predictor = TabularPredictor(
        path=save_path,
        problem_type='regression',
        label="target",
        verbosity=4,
    ).fit(
        train_data=train,
        hyperparameters=hyperparameters,
        hyperparameter_tune_kwargs={
            "scheduler": "local",
            "searcher": "random",
        },
    )

    # 5. Check if the model actually trained
    predictor.model_names()

    # Check if we can predict
    preds = predictor.predict(test.drop(columns=["target"]))
    assert len(preds) == len(test), "Prediction count mismatch"
    assert np.all(np.isfinite(preds)), "Predictions contain NaNs or Infs"

    # Check if the leaderboard works (validates internal metric calculation)
    leaderboard = predictor.leaderboard(test, silent=True)
    assert len(leaderboard) > 0

    # 6. Persistence Check: Can we load and still predict?
    del predictor
    loaded_predictor = TabularPredictor.load(save_path)
    loaded_preds = loaded_predictor.predict(test.drop(columns=["target"]))
    np.testing.assert_array_almost_equal(preds, loaded_preds)


def test_pipeline_pls(tmp_path):
    """
    Test if the RamanPreprocessingModel integrates with AutoGluon
    and successfully completes a Hyperparameter Optimization (HPO) cycle.
    """
    # 1. Setup Data
    train = generate_synthetic_raman(n_samples=40)
    test = generate_synthetic_raman(n_samples=10)

    pipeline, search_space = get_preprocessing_config()

    # 3. Define Search Space
    # We use very small ranges/trials to keep the test fast
    hyperparameters = {
        PreprocessingPLS: {"proc_pipeline": pipeline} | search_space,
    }

    # 4. Initialize and Fit
    save_path = str(tmp_path / "ag_test")
    predictor = TabularPredictor(
        path=save_path,
        problem_type='regression',
        label="target",
    ).fit(
        train_data=train,
        hyperparameters=hyperparameters,
        hyperparameter_tune_kwargs={
            "num_trials": 1,
            "scheduler": "local",
            "searcher": "random",
        },
        time_limit=3
    )

    # 5. Check if the model actually trained
    trained_models = predictor.model_names()
    assert any("PreprocessingPLS" in m for m in trained_models)

    # Check if we can predict
    preds = predictor.predict(test.drop(columns=["target"]))
    assert len(preds) == len(test), "Prediction count mismatch"
    assert np.all(np.isfinite(preds)), "Predictions contain NaNs or Infs"

    # Check if the leaderboard works (validates internal metric calculation)
    leaderboard = predictor.leaderboard(test, silent=True)
    assert len(leaderboard) > 0

    # 6. Persistence Check:
    del predictor
    loaded_predictor = TabularPredictor.load(save_path)
    loaded_preds = loaded_predictor.predict(test.drop(columns=["target"]))
    np.testing.assert_array_almost_equal(preds, loaded_preds)
