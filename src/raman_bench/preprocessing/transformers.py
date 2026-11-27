"""Raman preprocessing as scikit-learn transformers.

:class:`RamanPreprocessor` applies a preprocessing *recipe*, a set of steps given in the same
schema as a RamanBench recipe config (``{"baseline_correction": True, "snv": True}`` plus optional
``preprocessing_params`` overrides), with exactly the code every benchmark model runs
(:class:`~raman_bench.preprocessing.mixin.RamanPreprocessingMixin`): the stateful steps (MSC/EMSC
reference, standard scaling, GCU, LVSE) are fitted in :meth:`~RamanPreprocessor.fit` on the training
spectra only and replayed in :meth:`~RamanPreprocessor.transform`. The steps of a recipe always run
in the benchmark's fixed order (see the module docstring of :mod:`raman_bench.preprocessing.mixin`)::

    from sklearn.cross_decomposition import PLSRegression
    from sklearn.pipeline import make_pipeline
    from raman_bench.preprocessing import RamanPreprocessor

    model = make_pipeline(
        RamanPreprocessor({"baseline_correction": True, "snv": True, "avgpool": True}),
        PLSRegression(n_components=10),
    )
    model.fit(X_train, y_train)

The single-step transformers below (:class:`SNV`, :class:`MSC`, :class:`AsLSBaseline`, ...) are a
:class:`RamanPreprocessor` with one step switched on and its parameters as plain arguments, so they
can be chained in any order in a :class:`~sklearn.pipeline.Pipeline` and tuned with
:class:`~sklearn.model_selection.GridSearchCV`.

Inputs are 2-D arrays of spectra (rows) by channels (columns). A :class:`pandas.DataFrame` whose
column labels are the Raman shifts in cm^-1 also provides the axis that :class:`PhysicalCrop` needs;
otherwise pass it as ``wavenumbers``. Spectral augmentation, which the benchmark applies to the
training set of the neural-network models, adds rows rather than transforming them and is not a
transformer here; see :func:`raman_bench.preprocessing.augment_spectra`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from raman_bench.config import _normalize_preprocessing_config, _normalize_preprocessing_params
from raman_bench.preprocessing.mixin import (
    _PREP_STEP_DEFINITIONS,
    STEP_ENABLED_PARAMS,
    RamanPreprocessingMixin,
    _output_feature_cols,
)

__all__ = [
    "RamanPreprocessor",
    "recipe_params",
    "CosmicRayRemoval",
    "Crop",
    "PhysicalCrop",
    "AsLSBaseline",
    "AirPLSBaseline",
    "ArPLSBaseline",
    "RubberbandBaseline",
    "MSC",
    "EMSC",
    "SavitzkyGolaySmoothing",
    "WaveletDenoising",
    "SavitzkyGolayDerivative",
    "SNV",
    "VectorNormalization",
    "AveragePooling",
    "Decimation",
    "GCU",
    "LVSE",
]


def recipe_params(preprocessing=None, preprocessing_params: dict | None = None) -> dict:
    """The ``prep_*`` parameters a recipe resolves to, as a benchmark model receives them.

    ``preprocessing`` is a recipe config's ``"preprocessing"`` entry: a dict of step key -> bool
    (steps not listed are off), ``True`` for every step of the ``preprocessing: true`` shorthand,
    or ``None``/``False`` for none. ``preprocessing_params`` overrides single parameters
    (``{"prep_deriv_order": 2}``); it may not switch a step that ``preprocessing`` decides.
    The precedence is that of :func:`raman_bench.prep_hyperparameters.build_prep_model_hyperparameters`.
    Augmentation is always off (it is not a transform).
    """
    cfg = _normalize_preprocessing_config({"preprocessing": preprocessing or False})["preprocessing_config"]
    overrides = _normalize_preprocessing_params({"preprocessing_params": dict(preprocessing_params or {})})[
        "preprocessing_params"
    ]
    params = {p: v for step in _PREP_STEP_DEFINITIONS.values() for p, v in step["defaults"].items()}
    params.update(prep_scaling_enabled=False, prep_ensemble_enabled=False, prep_ensemble_blocks=None)
    decided = set()
    if cfg is not None:
        params["prep_scaling_enabled"] = bool(cfg.get("standard_scaling", False))
        decided.add("prep_scaling_enabled")
        for step_key, enabled_param in STEP_ENABLED_PARAMS.items():
            want = cfg.get(step_key)
            if want is None:
                continue
            params[enabled_param] = bool(want)
            decided.add(enabled_param)
    for name, value in overrides.items():
        if name in decided:
            raise ValueError(f"{name} is decided by `preprocessing`; switch the step there, not in preprocessing_params")
        params[name] = value
    params["prep_aug_enabled"] = False
    return params


class _Host(RamanPreprocessingMixin):
    """The mixin's preprocessing state without a model: the recipe's params and the fitted attributes."""

    def __init__(self, params: dict, wavenumbers: np.ndarray | None):
        self.params = dict(params)
        self._wavenumbers = wavenumbers

    def _get_model_params(self) -> dict:
        return dict(self.params)


def _spectra(X) -> tuple[np.ndarray, np.ndarray | None]:
    """``(X as float64, wavenumbers from numeric DataFrame column labels or None)``."""
    if isinstance(X, pd.DataFrame):
        try:
            wavenumbers = X.columns.to_numpy(dtype=float)
        except (TypeError, ValueError):
            wavenumbers = None
        return X.to_numpy(dtype=np.float64), wavenumbers
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"expected a 2-D array of spectra (n_spectra, n_channels), got shape {X.shape}")
    return X, None


class RamanPreprocessor(TransformerMixin, BaseEstimator):
    """Apply a preprocessing recipe with RamanBench's benchmark code.

    Parameters
    ----------
    preprocessing : dict | bool | None
        Step key -> bool, as in a recipe config (``"baseline_correction"``, ``"airpls"``, ``"arpls"``,
        ``"rubberband"``, ``"cosmic_ray_removal"``, ``"denoising"``, ``"wavelet_denoise"``,
        ``"derivative"``, ``"msc"``, ``"emsc"``, ``"snv"``, ``"vecnorm"``, ``"standard_scaling"``,
        ``"avgpool"``, ``"decimate"``, ``"crop"``, ``"crop_physical"``, ``"gcu"``, ``"lvse"``).
        Steps not listed are off. ``True`` switches on every step of the ``preprocessing: true``
        shorthand; ``None`` applies nothing (unless ``preprocessing_params`` switches steps on).
    preprocessing_params : dict | None
        ``prep_*`` parameter overrides, e.g. ``{"prep_deriv_order": 2, "prep_deriv_wl": 15}``, or the
        preprocessing ensemble ``{"prep_ensemble_enabled": True, "prep_ensemble_blocks": [{"snv": True}, ...]}``.
    wavenumbers : array-like | None
        Raman shift of every input channel (cm^-1), needed only by ``crop_physical`` when ``X`` is
        not a DataFrame with the shifts as column labels.

    Attributes
    ----------
    params_ : dict
        The resolved ``prep_*`` parameters (:func:`recipe_params`).
    n_features_in_, n_features_out_ : int
        Number of channels before and after the recipe (pooling, cropping, GCU, LVSE and the
        ensemble change it).
    """

    def __init__(self, preprocessing=None, preprocessing_params: dict | None = None, wavenumbers=None):
        self.preprocessing = preprocessing
        self.preprocessing_params = preprocessing_params
        self.wavenumbers = wavenumbers

    def _recipe(self) -> tuple[object, dict | None]:
        return self.preprocessing, self.preprocessing_params

    def fit(self, X, y=None):
        X_np, axis = _spectra(X)
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        given = getattr(self, "wavenumbers", None)  # single-step transformers other than PhysicalCrop have none
        wavenumbers = axis if axis is not None else (None if given is None else np.asarray(given, float))
        if wavenumbers is not None and len(wavenumbers) != X_np.shape[1]:
            raise ValueError(f"{len(wavenumbers)} wavenumbers for {X_np.shape[1]} channels")
        self.params_ = recipe_params(*self._recipe())
        self._host = _Host(self.params_, wavenumbers)
        if self._ensemble:
            X_out = self._host._preprocess_fit_ensemble(X_np, self.params_["prep_ensemble_blocks"])
        else:
            X_out = self._host._preprocess_fit(X_np)
        self.n_features_in_ = X_np.shape[1]
        self.n_features_out_ = X_out.shape[1]
        return self

    @property
    def _ensemble(self) -> bool:
        return bool(self.params_.get("prep_ensemble_enabled") and self.params_.get("prep_ensemble_blocks"))

    def transform(self, X):
        check_is_fitted(self, "params_")
        X_np, _ = _spectra(X)
        if X_np.shape[1] != self.n_features_in_:
            raise ValueError(f"fitted on {self.n_features_in_} channels, got {X_np.shape[1]}")
        if self._ensemble:
            return self._host._preprocess_transform_ensemble(X_np)
        return self._host._preprocess_transform(X_np, params=dict(self.params_))

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "params_")
        if input_features is None:
            input_features = getattr(self, "feature_names_in_", [f"x{i}" for i in range(self.n_features_in_)])
        return np.asarray(_output_feature_cols(list(input_features), self.n_features_out_), dtype=object)


class _OneStep(RamanPreprocessor):
    """A :class:`RamanPreprocessor` with one step on; subclasses name the step and map their
    arguments to its ``prep_*`` parameters (``_PARAMS``: argument -> parameter)."""

    _STEP: str = ""
    _PARAMS: dict[str, str] = {}

    def _recipe(self) -> tuple[object, dict | None]:
        return {self._STEP: True}, {p: getattr(self, a) for a, p in self._PARAMS.items()}


class CosmicRayRemoval(_OneStep):
    """Whitaker-Hayes cosmic-ray (spike) removal: channels whose first difference has a modified
    z-score above ``threshold`` are replaced by the mean of the non-spike channels within
    ``kernel_size`` on either side."""

    _STEP, _PARAMS = "cosmic_ray_removal", {"threshold": "prep_crr_threshold", "kernel_size": "prep_crr_kernel_size"}

    def __init__(self, threshold: float = 6, kernel_size: int = 3):
        self.threshold = threshold
        self.kernel_size = kernel_size


class Crop(_OneStep):
    """Keep the channels between the fractions ``start_frac`` and ``end_frac`` of the axis (by index)."""

    _STEP, _PARAMS = "crop", {"start_frac": "prep_crop_start_frac", "end_frac": "prep_crop_end_frac"}

    def __init__(self, start_frac: float = 0.15, end_frac: float = 0.75):
        self.start_frac = start_frac
        self.end_frac = end_frac


class PhysicalCrop(_OneStep):
    """Keep the channels between ``start_cm`` and ``end_cm`` (Raman shift, cm^-1); needs the axis
    (DataFrame column labels or ``wavenumbers``)."""

    _STEP, _PARAMS = "crop_physical", {"start_cm": "prep_crop_physical_start_cm", "end_cm": "prep_crop_physical_end_cm"}

    def __init__(self, start_cm: float = 400.0, end_cm: float = 1800.0, wavenumbers=None):
        self.start_cm = start_cm
        self.end_cm = end_cm
        self.wavenumbers = wavenumbers


class AsLSBaseline(_OneStep):
    """Asymmetric least squares baseline subtraction (smoothness ``lam``, asymmetry ``p``)."""

    _STEP, _PARAMS = "baseline_correction", {"lam": "prep_bl_lam", "p": "prep_bl_p"}

    def __init__(self, lam: float = 1e5, p: float = 0.01):
        self.lam = lam
        self.p = p


class AirPLSBaseline(_OneStep):
    """Adaptive iteratively reweighted penalized least squares (airPLS) baseline subtraction."""

    _STEP, _PARAMS = "airpls", {"lam": "prep_airpls_lam", "max_iter": "prep_airpls_max_iter", "tol": "prep_airpls_tol"}

    def __init__(self, lam: float = 1e5, max_iter: int = 15, tol: float = 1e-3):
        self.lam = lam
        self.max_iter = max_iter
        self.tol = tol


class ArPLSBaseline(_OneStep):
    """Asymmetrically reweighted penalized least squares (arPLS) baseline subtraction."""

    _STEP, _PARAMS = "arpls", {"lam": "prep_arpls_lam", "max_iter": "prep_arpls_max_iter"}

    def __init__(self, lam: float = 1e5, max_iter: int = 15):
        self.lam = lam
        self.max_iter = max_iter


class RubberbandBaseline(_OneStep):
    """Rubberband (lower convex hull) baseline subtraction."""

    _STEP = "rubberband"

    def __init__(self):
        pass


class MSC(_OneStep):
    """Multiplicative scatter correction against the mean training spectrum, with slope and
    offset fitted on the fraction ``start_frac`` to ``end_frac`` of the axis."""

    _STEP, _PARAMS = "msc", {"start_frac": "prep_msc_start_frac", "end_frac": "prep_msc_end_frac"}

    def __init__(self, start_frac: float = 0.0, end_frac: float = 1.0):
        self.start_frac = start_frac
        self.end_frac = end_frac


class EMSC(_OneStep):
    """Extended multiplicative scatter correction against the mean training spectrum, with a
    polynomial baseline of degree ``poly_order``."""

    _STEP, _PARAMS = "emsc", {"poly_order": "prep_emsc_poly_order"}

    def __init__(self, poly_order: int = 4):
        self.poly_order = poly_order


class SavitzkyGolaySmoothing(_OneStep):
    """Savitzky-Golay smoothing (window ``window_length``, polynomial degree ``polyorder``)."""

    _STEP, _PARAMS = "denoising", {"window_length": "prep_denoise_wl", "polyorder": "prep_denoise_po"}

    def __init__(self, window_length: int = 9, polyorder: int = 3):
        self.window_length = window_length
        self.polyorder = polyorder


class WaveletDenoising(_OneStep):
    """Wavelet soft-threshold denoising (``wavelet``, decomposition ``level``); needs PyWavelets."""

    _STEP, _PARAMS = "wavelet_denoise", {"wavelet": "prep_wavelet_name", "level": "prep_wavelet_level"}

    def __init__(self, wavelet: str = "sym7", level: int = 4):
        self.wavelet = wavelet
        self.level = level


class SavitzkyGolayDerivative(_OneStep):
    """Savitzky-Golay derivative of order ``order`` (window ``window_length``, degree ``polyorder``)."""

    _STEP, _PARAMS = "derivative", {
        "order": "prep_deriv_order",
        "window_length": "prep_deriv_wl",
        "polyorder": "prep_deriv_po",
    }

    def __init__(self, order: int = 1, window_length: int = 11, polyorder: int = 2):
        self.order = order
        self.window_length = window_length
        self.polyorder = polyorder


class SNV(_OneStep):
    """Standard normal variate: each spectrum centered on its mean and divided by its standard deviation."""

    _STEP = "snv"

    def __init__(self):
        pass


class VectorNormalization(_OneStep):
    """Each spectrum divided by its Euclidean norm."""

    _STEP = "vecnorm"

    def __init__(self):
        pass


class AveragePooling(_OneStep):
    """Average of each run of ``size`` neighboring channels (width becomes n_channels // size)."""

    _STEP, _PARAMS = "avgpool", {"size": "prep_avgpool_size"}

    def __init__(self, size: int = 5):
        self.size = size


class Decimation(_OneStep):
    """Every ``step``-th channel (width becomes n_channels // step)."""

    _STEP, _PARAMS = "decimate", {"step": "prep_decimate_step"}

    def __init__(self, step: int = 5):
        self.step = step


class GCU(_OneStep):
    """Global compositional unmixing: the spectra replaced by their weights on ``rho`` non-negative
    components fitted on the training spectra (RamanPFN, arXiv:2608.02157)."""

    _STEP, _PARAMS = "gcu", {"rho": "prep_gcu_rho", "random_state": "prep_gcu_random_state"}

    def __init__(self, rho: int = 16, random_state: int | None = 0):
        self.rho = rho
        self.random_state = random_state


class LVSE(_OneStep):
    """Local vibrational subspace encoding: the axis split into ``n_regions`` regions, each standardized
    and projected onto its first ``k_per_region`` singular vectors, fitted on the training spectra
    (RamanPFN, arXiv:2608.02157)."""

    _STEP, _PARAMS = "lvse", {"n_regions": "prep_lvse_n_regions", "k_per_region": "prep_lvse_k_per_region"}

    def __init__(self, n_regions: int = 16, k_per_region: int = 4):
        self.n_regions = n_regions
        self.k_per_region = k_per_region
