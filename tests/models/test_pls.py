import numpy as np
import pandas as pd
import pytest

pytest.importorskip("autogluon.tabular")

from raman_bench.models.custom.pls import PLSModel  # noqa: E402

# ── Unit: _select_features ────────────────────────────────────────────────────


class TestPLSSelectFeatures:
    """PLSModel._select_features handles column count mismatches at predict time.

    AutoGluon injects an extra NaN column when tuning_data=training_data is used
    for datasets with < 20 samples (the 'tiny dataset' path in model.py).
    _select_features must silently drop extra columns and reorder to match the
    columns seen during _fit.
    """

    def _make_model_with_feature_names(self, feature_names):
        m = object.__new__(PLSModel)
        m._feature_names = list(feature_names)
        return m

    def test_extra_column_is_dropped(self):
        m = self._make_model_with_feature_names(["a", "b", "c"])
        X = pd.DataFrame({"a": [1.0], "b": [2.0], "c": [3.0], "extra": [np.nan]})
        result = m._select_features(X)
        assert list(result.columns) == ["a", "b", "c"]
        assert result.shape == (1, 3)

    def test_extra_nan_column_is_dropped(self):
        m = self._make_model_with_feature_names(["f0", "f1"])
        X = pd.DataFrame(
            {
                "f0": [0.5, 1.0],
                "f1": [0.3, 0.8],
                "autogluon_nan_injection": [np.nan, np.nan],
            }
        )
        result = m._select_features(X)
        assert list(result.columns) == ["f0", "f1"]
        assert not result.isna().any().any()

    def test_column_reordering(self):
        m = self._make_model_with_feature_names(["a", "b", "c"])
        X = pd.DataFrame({"c": [3.0], "a": [1.0], "b": [2.0]})
        result = m._select_features(X)
        assert list(result.columns) == ["a", "b", "c"]
        np.testing.assert_array_equal(result.values, [[1.0, 2.0, 3.0]])

    def test_missing_column_filled_with_zero(self):
        m = self._make_model_with_feature_names(["a", "b", "c"])
        X = pd.DataFrame({"a": [1.0], "b": [2.0]})
        result = m._select_features(X)
        assert list(result.columns) == ["a", "b", "c"]
        assert result["c"].iloc[0] == 0.0

    def test_no_feature_names_passthrough(self):
        m = object.__new__(PLSModel)
        X = pd.DataFrame({"a": [1.0], "extra": [np.nan]})
        result = m._select_features(X)
        assert list(result.columns) == ["a", "extra"]

    def test_non_dataframe_passthrough(self):
        m = self._make_model_with_feature_names(["a", "b"])
        X_np = np.array([[1.0, 2.0, 3.0]])
        result = m._select_features(X_np)
        assert result is X_np


# ── Integration: tiny dataset triggers AutoGluon tuning_data path ─────────────


class TestPLSTinyDataset:
    """PLSModel survives AutoGluon's post-fit validation when n_samples < 20."""

    def _tiny_regression_df(self, n_samples=15):
        rng = np.random.RandomState(0)
        n_features = 30
        wavenumbers = np.linspace(400, 1800, n_features)
        X = np.array(
            [
                sum(
                    rng.uniform(0.5, 2.0) * np.exp(-0.5 * ((wavenumbers - c) / 30) ** 2)
                    for c in rng.choice(wavenumbers, size=3, replace=False)
                )
                + rng.normal(0, 0.02, n_features)
                for _ in range(n_samples)
            ]
        )
        X = np.clip(X, 0.01, None)
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])
        df["target"] = rng.rand(n_samples) * 10
        return df

    def _tiny_classification_df(self, n_samples=15):
        df = self._tiny_regression_df(n_samples)
        df["target"] = ["A"] * 5 + ["B"] * 5 + ["C"] * 5
        return df

    def test_tiny_regression_does_not_raise(self, tmp_path):
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
        model.fit(df)
        preds = model.predict(df.drop(columns=["target"]))
        assert len(preds) == 15
        assert np.isfinite(preds).all()

    def test_tiny_classification_does_not_raise(self, tmp_path):
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
