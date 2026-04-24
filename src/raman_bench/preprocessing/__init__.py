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
2. Baseline correction (asymmetric least squares / ALS)
3. Multiplicative scatter correction (MSC)
4. Denoising (Savitzky-Golay filter)
5. Standard Normal Variate (SNV)
6. Standard scaling (zero-mean / unit-variance per feature)
7. Spectral augmentation (noise + shift + mixup; training only)
"""

from raman_bench.preprocessing.raman_preprocessing import (
    augment_spectra,
    augment_spectra_torch,
    baseline_correction_asls,
    cosmic_ray_removal,
    denoise_savgol,
    multiplicative_scatter_correction_fit,
    multiplicative_scatter_correction_transform,
    snv,
)

__all__ = [
    "augment_spectra",
    "augment_spectra_torch",
    "baseline_correction_asls",
    "cosmic_ray_removal",
    "denoise_savgol",
    "multiplicative_scatter_correction_fit",
    "multiplicative_scatter_correction_transform",
    "snv",
]
