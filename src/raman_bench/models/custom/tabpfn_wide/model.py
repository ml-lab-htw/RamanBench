from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator

from raman_bench.preprocessing.bridge_bases import SklearnAutoGluonBridge, _NoAugBase


def _to_numpy(X) -> np.ndarray:
    return X.values if hasattr(X, "values") else np.asarray(X)


def _infer_problem_type(y) -> str:
    arr = np.asarray(y)
    if np.issubdtype(arr.dtype, np.floating):
        return "regression"
    return "binary" if len(np.unique(arr)) == 2 else "multiclass"


def _resolve_device(device: str | None) -> str:
    """Resolve an explicit device, or auto-detect CUDA if ``None``.

    Previously both ``TabPFNWideModel`` and ``_CloneSafeTabPFNWide`` hardcoded
    ``device="cpu"`` as their constructor default -- unlike the upstream
    ``tabpfnwide.classifier.TabPFNWideClassifier`` itself (whose own default is
    ``device="cuda"``) and unlike every other GPU-tagged model in this
    codebase (the from-scratch DL models auto-detect via
    ``BaseRamanEstimator._setup_device()``; TabArena-native foundation models
    like TabICL/Mitra/RealTabPFN thread AutoGluon's allocated ``num_gpus``
    resource through ``AbstractTorchModel``). Because ``SklearnAutoGluonBridge``
    (this model's AutoGluon bridge) never reads or forwards AutoGluon's
    ``num_gpus`` resource kwarg, that hardcoded "cpu" meant TABPFN-WIDE always
    ran on CPU even on a GPU-provisioned cluster node (``info.py`` tags
    ``compute="gpu"``) -- real profiling (``cProfile`` on a local CPU-only
    fit/predict) showed the wall time genuinely dominated by
    ``torch._C._nn.scaled_dot_product_attention`` inside the transformer's
    attention-between-features layers, i.e. legitimate model compute that a
    GPU would substantially speed up, not a Python-level bug. Default is now
    ``None`` ("auto"), matching ``_setup_device()``'s own CUDA-availability
    check -- explicit ``"cpu"``/``"cuda"`` are still accepted for the search
    space / manual overrides.
    """
    if device is not None:
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


class _CloneSafeTabPFNWide(BaseEstimator):
    """sklearn-clone-safe wrapper around TabPFNWideClassifier.

    TabPFNWideClassifier mutates its own params during __init__ (resolves
    model_name → model_path), so get_params() returns both, and re-init
    via sklearn.clone() trips its mutually-exclusive XOR check.
    This wrapper exposes only the user-supplied constructor args, so
    cloning round-trips cleanly. Required by ManyClassClassifier, which
    clones the base estimator per ECOC sub-task.
    """

    def __init__(self, model_name: str = "wide-v2-5k", device: str | None = None):
        self.model_name = model_name
        self.device = device

    def fit(self, X, y):
        from tabpfnwide.classifier import TabPFNWideClassifier

        resolved_device = _resolve_device(self.device)
        self._estimator = TabPFNWideClassifier(model_name=self.model_name, device=resolved_device)
        self._estimator.fit(X, y)
        self.classes_ = self._estimator.classes_
        return self

    def predict(self, X):
        return self._estimator.predict(X)

    def predict_proba(self, X):
        return self._estimator.predict_proba(X)


class TabPFNWideModel(BaseEstimator):
    """TabPFN-Wide for Raman spectra — sklearn-compatible (classification only).

    Built with PriorLabs-TabPFN. TabPFN-Wide targets datasets with many
    features and few samples — a regime Raman spectroscopy frequently
    occupies (2000+ wavenumber columns, often <1000 spectra).

    Requires the ``tabpfnwide`` package.

    Reference:
        TabPFN-Wide: https://github.com/not-a-feature/TabPFN-Wide
        DOI: 10.48550/arXiv.2510.06162
    """

    #: ECOC's own per-sub-model cost multiplies TabPFN-Wide's already-heavy
    #: wide-feature memory footprint by roughly `alphabet_size` fits -- a real
    #: OOM was found combining many-class ECOC with wide Raman spectra even at
    #: a 256G container limit (unlike TabSTAR's simpler, single-fit width cap
    #: at 4000 features, `_TABSTAR_MAX_FEATURES` in wrapped_models.py). Capped
    #: meaningfully lower than that precedent since the failure mode here is
    #: ECOC-specific, not a plain single fit -- unverified against a real OOM
    #: at exactly this width; revisit if one shows up either direction.
    _ECOC_MAX_FEATURES = 2000

    def __init__(
        self,
        model_name: str = "wide-v2-5k",
        device: str | None = None,
        many_class_threshold: int = 10,
    ):
        self.model_name = model_name
        self.device = device
        self.many_class_threshold = many_class_threshold

    def fit(self, X, y):
        try:
            import tabpfnwide.classifier  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "TabPFNWideModel requires tabpfnwide. Install with: pip install tabpfnwide"
            ) from e

        X_arr, y_arr = _to_numpy(X), np.asarray(y)
        self.problem_type_ = _infer_problem_type(y)
        if self.problem_type_ == "regression":
            raise ValueError("TabPFN-Wide does not support regression.")

        self.classes_ = np.unique(y_arr)
        n_features = X_arr.shape[1]
        many_class = len(self.classes_) > self.many_class_threshold

        if many_class and n_features > self._ECOC_MAX_FEATURES:
            # ECOC's per-sub-model cost on top of an already-wide fit OOMed
            # even at 256G -- fail fast rather than risk it here too (matches
            # the prior, pre-ECOC behavior for this specific width x
            # many-class combination; see _ECOC_MAX_FEATURES's docstring).
            raise ValueError(
                f"TabPFN-Wide: {len(self.classes_)} classes exceeds the native limit "
                f"({self.many_class_threshold}) and {n_features} features exceeds the "
                f"ECOC-safe limit ({self._ECOC_MAX_FEATURES}); skipping this dataset."
            )

        self.model_ = _CloneSafeTabPFNWide(model_name=self.model_name, device=self.device)

        if many_class:
            try:
                from tabpfn_extensions.many_class import ManyClassClassifier
            except ImportError as e:
                raise ImportError(
                    f"TabPFN-Wide: {len(self.classes_)} classes exceeds native limit "
                    f"({self.many_class_threshold}). Install tabpfn-extensions: "
                    "pip install tabpfn-extensions"
                ) from e
            self.model_ = ManyClassClassifier(
                estimator=self.model_, alphabet_size=self.many_class_threshold
            )

        self.model_.fit(X_arr, y_arr)
        return self

    def predict(self, X):
        return self.model_.predict(_to_numpy(X))

    def predict_proba(self, X):
        proba = self.model_.predict_proba(_to_numpy(X))
        return proba[:, 1] if self.problem_type_ == "binary" else proba


class _TabPFNWideBridge(SklearnAutoGluonBridge):
    _sklearn_cls = TabPFNWideModel
    ag_key = "TABPFN-WIDE"
    ag_name = "TabPFN-Wide"


class Prep_TABPFN_WIDE(_NoAugBase, _TabPFNWideBridge):  # noqa: N801
    """TabPFN-Wide — classification-only, targets wide datasets (many features, few samples).

    Built with PriorLabs-TabPFN.
    """

    pass
