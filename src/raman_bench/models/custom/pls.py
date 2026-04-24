import logging

import numpy as np
from autogluon.common import space
from scipy.signal import savgol_filter
from sklearn import clone
from sklearn.cross_decomposition import PLSRegression

from raman_bench.models.custom.base import BaseCustomModel

logger = logging.getLogger(__name__)


class PLSModel(BaseCustomModel):
    """AutoGluon-compatible PLS model.

    For regression tasks, wraps sklearn PLSRegression directly.
    For classification tasks, implements PLS-DA (Discriminant Analysis) by
    one-hot encoding the target, fitting PLSRegression on the encoded targets,
    and using argmax of the predicted scores for classification.

    Reference:
        Wold, S., Ruhe, A., Wold, H., & Dunn III, W. J. (1984).
        The collinearity problem in linear regression. The Partial Least
        Squares (PLS) approach to generalized inverses.
        SIAM Journal on Scientific and Statistical Computing, 5(3), 735-743.
        https://doi.org/10.1137/0905052
    """

    def _set_default_params(self):
        default_params = {
            "n_components": 10,
            "max_iter": 500,
            "scale": True,
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    def _get_default_searchspace(self):
        return self._get_base_hp_searchspace()

    @staticmethod
    def _get_base_hp_searchspace():
        return {
            "n_components": space.Int(lower=2, upper=50),
            "scale": space.Categorical(True, False),
        }

    def _fit(self, X, y, **kwargs):
        params = self._get_model_params()
        n_components = params.get("n_components", 10)
        max_iter = params.get("max_iter", 500)
        scale = params.get("scale", True)

        # n_samples - 1: PLS deflation produces a constant (zero) residual after
        # extracting min(n_samples-1, rank) components; the next component then
        # divides by zero and produces NaN in X scores.
        n_components = min(n_components, X.shape[1], X.shape[0] - 1)

        self._feature_names = list(X.columns)

        if self.problem_type in ("multiclass", "binary"):
            self._classes = np.unique(y)
            self._n_classes = len(self._classes)
            # One-hot encode targets for PLS-DA
            self._class_to_idx = {c: i for i, c in enumerate(self._classes)}
            y_encoded = np.zeros((len(y), self._n_classes))
            for i, val in enumerate(y):
                y_encoded[i, self._class_to_idx[val]] = 1.0
            self.model = PLSRegression(n_components=n_components, max_iter=max_iter, scale=scale)
            self.model.fit(X.values, y_encoded)
        else:
            self.model = PLSRegression(
                n_components=n_components,
                max_iter=max_iter,
                scale=scale,
            ).fit(X.values, y.values)

    def _select_features(self, X):
        """Select only the columns seen during training, in training order.

        AutoGluon may pass X_val with extra columns (e.g. an injected NaN
        column when tuning_data=training_data for tiny datasets). Selecting
        by stored column names keeps the feature count consistent.
        """
        if hasattr(self, "_feature_names") and hasattr(X, "columns"):
            missing = [c for c in self._feature_names if c not in X.columns]
            if missing:
                logger.warning(
                    "PLSModel predict: %d training column(s) missing from X — "
                    "filling with 0: %s",
                    len(missing),
                    missing[:5],
                )
            X = X.reindex(columns=self._feature_names, fill_value=0.0)
        return X

    def _raw_predict(self, X):
        """Core regression prediction without going through the preprocessing mixin.

        Called by both _predict and _predict_proba (regression branch) so that
        _predict_proba never delegates to self._predict — which would re-enter
        RamanPreprocessingMixin._predict and apply preprocessing a second time.
        """
        X = self._select_features(X)
        return self.model.predict(X.values).ravel()

    def _predict(self, X, **kwargs):
        if self.problem_type in ("multiclass", "binary"):
            X = self._select_features(X)
            indices = np.argmax(self.model.predict(X.values), axis=1)
            return self._classes[indices]
        return self._raw_predict(X)

    def _predict_proba(self, X, **kwargs):
        if self.problem_type == "regression":
            return self._raw_predict(X)
        X = self._select_features(X)
        y_pred = self.model.predict(X.values)
        # Softmax to convert PLS scores to probabilities
        exp_scores = np.exp(y_pred - np.max(y_pred, axis=1, keepdims=True))
        proba = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        if self.problem_type == "binary":
            return proba[:, 1]
        return proba


class PreprocessingPLS(BaseCustomModel):
    """
    A wrapper model that applies Raman-specific preprocessing before
    fitting an inner base model (e.g., PLSModel).

    It allows AutoGluon to tune preprocessing parameters (like smoothing)
    simultaneously with the model's parameters (like n_components).
    """

    def _set_default_params(self):
        default_params = {
            "sg_window_length": 11,  # Must be odd
            "sg_polyorder": 2,
            "sg_deriv": 0,
            "n_components": 10,
            "max_iter": 500,
            "scale": True,
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    def _get_default_searchspace(self):
        # These are the default tunable ranges for preprocessing
        return {
            "sg_window_length": space.Int(lower=5, upper=51),  # Must be odd
            "sg_polyorder": space.Int(lower=1, upper=3),
            "sg_deriv": space.Categorical(0, 1, 2),
            "n_components": space.Int(lower=2, upper=50),
            "scale": space.Categorical(True, False),
        }

    def _preprocess_spectra(self, X, params):
        """Runs Raman specifc preprocessing steps to X."""
        # Work on a copy to avoid side effects
        X_out = X.copy()

        # --- 1. Savitzky-Golay Smoothing ---
        if "sg_window_length" in params:
            window_length = params.get("sg_window_length")
            polyorder = params.get("sg_polyorder", 2)
            deriv = params.get("sg_deriv", 0)

            # Constraint enforcement: window_length must be odd and > polyorder
            if window_length % 2 == 0:
                window_length += 1
            if window_length <= polyorder:
                window_length = polyorder + 2 if (polyorder + 2) % 2 != 0 else polyorder + 3

            # Check if X is DataFrame or Numpy for correct indexing
            if hasattr(X_out, "values"):
                data_values = X_out.values
            else:
                data_values = X_out

            # Apply filter row-wise (axis=1 is the spectral dimension)
            # mode='interp' handles the edges
            filtered_data = savgol_filter(
                data_values,
                window_length,
                polyorder,
                deriv=deriv,
                axis=1,
                mode="interp",
            )

            if hasattr(X_out, "values"):
                X_out.iloc[:, :] = filtered_data
            else:
                X_out = filtered_data

        return X_out

    def _fit(self, X, y, **kwargs):
        # 1. Get all hyperparameters (merged preprocessing + base model params)
        params = self._get_model_params()

        # 2. Separate preprocessing params from base model params
        # We identify prep params by specific keys
        prep_keys = ["sg_window_length", "sg_polyorder", "sg_deriv"]
        prep_params = {k: params[k] for k in prep_keys if k in params}

        # Anything that isn't a prep param is assumed to be for the base model (PLS)
        base_model_params = {k: v for k, v in params.items() if k not in prep_keys}

        # 3. Transform Data
        X_transformed = self._preprocess_spectra(X, prep_params)

        # 4. Instantiate and Fit the Base Model
        # We give it a unique name and path to avoid conflicts
        model_name = f"{self.name}_inner"
        self.pls_model = PLSModel(
            path=self.path + "_inner",
            name=model_name,
            problem_type=self.problem_type,
            eval_metric=self.eval_metric,
            hyperparameters=base_model_params,  # Pass stripped params to PLS
        )

        self.pls_model.fit(X=X_transformed, y=y, **kwargs)

        # Save params used so we can reproduce preprocessing in predict
        self.params.update(prep_params)

    def _predict(self, X, **kwargs):
        params = self._get_model_params()
        X_transformed = self._preprocess_spectra(X, params)
        return self.pls_model.predict(X_transformed, **kwargs)

    def _predict_proba(self, X, **kwargs):
        params = self._get_model_params()
        X_transformed = self._preprocess_spectra(X, params)
        return self.pls_model.predict_proba(X_transformed, **kwargs)

    # def save(self, path: str = None, verbose=True) -> str:
    #     # Save wrapper logic; the base_model is usually pickled automatically
    #     # by AutoGluon, but ensuring the inner model is handled is good practice.
    #     return super().save(path, verbose)


class FlexiblePipelinePLS(BaseCustomModel):
    """
    AutoGluon model that runs an arbitrary sklearn Pipeline for preprocessing,
    followed by PLS Regression (or PLS-DA for classification).

    Hyperparameter keys:
        - 'proc_pipeline': The sklearn Pipeline object (required).
        - Any key with '__': Treated as a parameter for the 'proc_pipeline'.
        - 'n_components', 'scale': Parameters for the PLS step.
    """

    def _fit(self, X, y, **kwargs):
        # 1. Parse Hyperparameters
        params = self._get_model_params()

        # Extract the pipeline object (must be passed in hyperparameters)
        if "proc_pipeline" not in params:
            raise ValueError(
                "FlexiblePipelinePLS requires a 'proc_pipeline' object in hyperparameters."
            )

        # Clone to ensure thread safety/fresh state during HPO
        self.pipeline = clone(params["proc_pipeline"])

        # Separate Pipeline params (contain '__') from PLS params
        pipeline_params = {}
        pls_params = {"n_components": 10, "max_iter": 500, "scale": True}

        for key, val in params.items():
            if key == "proc_pipeline":
                continue
            if "__" in key:
                # This is a pipeline parameter (e.g. 'filter__window_length')
                pipeline_params[key] = val
            else:
                # This is a PLS parameter
                pls_params[key] = val

        # 2. Configure and Fit Preprocessing
        if pipeline_params:
            self.pipeline.set_params(**pipeline_params)

        # We assume X input is a DataFrame; convert to values for sklearn
        X_values = X.values if hasattr(X, "values") else X

        # Fit_transform the preprocessing pipeline
        # Note: Some preprocessing (like feature selection) might need 'y', others don't.
        # It is safe to pass y to fit_transform for pipelines.
        X_transformed = self.pipeline.fit_transform(X_values, y)

        # 3. Fit PLS (Logic adapted from your original PLSModel)
        n_components = pls_params.get("n_components")
        n_components = min(n_components, X_transformed.shape[1], X_transformed.shape[0] - 1)

        self.model_params_ = pls_params  # Store for saving

        if self.problem_type in ("multiclass", "binary"):
            self._classes = np.unique(y)
            self._n_classes = len(self._classes)
            self._class_to_idx = {c: i for i, c in enumerate(self._classes)}

            # One-hot encode targets for PLS-DA
            y_encoded = np.zeros((len(y), self._n_classes))
            for i, val in enumerate(y):
                y_encoded[i, self._class_to_idx[val]] = 1.0

            self.model = PLSRegression(
                n_components=n_components,
                max_iter=pls_params.get("max_iter"),
                scale=pls_params.get("scale"),
            )
            self.model.fit(X_transformed, y_encoded)
        else:
            self.model = PLSRegression(
                n_components=n_components,
                max_iter=pls_params.get("max_iter"),
                scale=pls_params.get("scale"),
            )
            self.model.fit(X_transformed, y)

    def _predict(self, X, **kwargs):
        X_values = X.values if hasattr(X, "values") else X

        # Apply the frozen preprocessing pipeline
        X_transformed = self.pipeline.transform(X_values)

        y_pred = self.model.predict(X_transformed)

        if self.problem_type in ("multiclass", "binary"):
            indices = np.argmax(y_pred, axis=1)
            return self._classes[indices]
        return y_pred.ravel()

    def _predict_proba(self, X, **kwargs):
        if self.problem_type == "regression":
            # Inline rather than self._predict() to avoid re-entering
            # RamanPreprocessingMixin._predict and applying preprocessing twice.
            X_values = X.values if hasattr(X, "values") else X
            return self.model.predict(self.pipeline.transform(X_values)).ravel()

        X_values = X.values if hasattr(X, "values") else X
        X_transformed = self.pipeline.transform(X_values)
        y_pred = self.model.predict(X_transformed)

        # Softmax conversion
        exp_scores = np.exp(y_pred - np.max(y_pred, axis=1, keepdims=True))
        proba = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        if self.problem_type == "binary":
            return proba[:, 1]
        return proba
