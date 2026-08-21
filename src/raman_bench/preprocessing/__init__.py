"""Raman spectral preprocessing for RamanBench.

This module provides:

- **Pure NumPy/SciPy implementations** of common Raman preprocessing steps
  (:mod:`raman_bench.preprocessing.raman_preprocessing`) that run without any
  external spectroscopy libraries.

- **AutoGluon HPO mixin** (:class:`~raman_bench.preprocessing.mixin.RamanPreprocessingMixin`)
  that exposes preprocessing parameters as tunable hyperparameters.

- **Preprocessed model wrappers** (:mod:`raman_bench.preprocessing.wrapped_models`)
  — one ``Prep_*`` class per AutoGluon / custom model.

Available preprocessing steps
------------------------------
1. Cosmic-ray removal (Whitaker-Hayes algorithm)
2. Denoising (Savitzky-Golay filter)
3. Denoising (wavelet threshold, requires PyWavelets)
4. Baseline correction (asymmetric least squares / ALS)
5. Baseline correction (airPLS)
6. Baseline correction (arPLS)
7. Baseline correction (rubberband / convex hull)
8. Multiplicative scatter correction (MSC)
9. Extended multiplicative scatter correction (EMSC)
10. Derivative (Savitzky-Golay 1st/2nd derivative)
11. Standard Normal Variate (SNV)
12. Standard scaling (zero-mean / unit-variance per feature)
13. Spectral augmentation (noise + shift + mixup; training only)
14. Fingerprint-region cropping (fractional-index proxy)
15. L2 (Euclidean) vector normalization
16. GCU (Global Compositional Unmixing, representation-replacing) — RamanPFN,
    arXiv:2608.02157
17. LVSE (Local Vibrational Subspace Encoding, representation-replacing) —
    RamanPFN, arXiv:2608.02157

"""

from raman_bench.preprocessing.raman_preprocessing import (
    augment_spectra,
    augment_spectra_torch,
    baseline_correction_airpls,
    baseline_correction_arpls,
    baseline_correction_asls,
    cosmic_ray_removal,
    crop_spectra,
    denoise_savgol,
    emsc_fit,
    emsc_transform,
    gcu_fit,
    gcu_transform,
    lvse_fit,
    lvse_transform,
    multiplicative_scatter_correction_fit,
    multiplicative_scatter_correction_transform,
    rubberband_correction,
    savgol_derivative,
    snv,
    vector_normalize,
    wavelet_denoise,
)

__all__ = [
    "augment_spectra",
    "augment_spectra_torch",
    "baseline_correction_airpls",
    "baseline_correction_arpls",
    "baseline_correction_asls",
    "cosmic_ray_removal",
    "crop_spectra",
    "denoise_savgol",
    "emsc_fit",
    "emsc_transform",
    "gcu_fit",
    "gcu_transform",
    "lvse_fit",
    "lvse_transform",
    "multiplicative_scatter_correction_fit",
    "multiplicative_scatter_correction_transform",
    "rubberband_correction",
    "savgol_derivative",
    "snv",
    "vector_normalize",
    "wavelet_denoise",
]
