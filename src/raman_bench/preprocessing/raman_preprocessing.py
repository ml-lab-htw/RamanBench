"""
Pure numpy/scipy implementations of Raman spectral preprocessing steps.

No RamanSPy dependency — these are used inside AutoGluon models as tunable hyperparameters.
"""

import numpy as np
import torch
from scipy.signal import savgol_filter
from scipy.sparse import csc_matrix, diags
from scipy.sparse.linalg import spsolve


def _despike_single(spectrum: np.ndarray, threshold: float, kernel_size: int) -> np.ndarray:
    """Apply Whitaker-Hayes despiking to a single spectrum."""
    Y = spectrum.copy()

    # Z-scores are computed once on the differenced spectrum (length n-1)
    diff = np.diff(Y)
    median_diff = np.median(diff)
    mad = np.median(np.abs(diff - median_diff))

    if mad == 0.0:
        return Y

    z_scores = np.abs(0.6745 * (diff - median_diff) / mad)
    spikes = z_scores > threshold  # length n-1

    while spikes.any():
        changes = False

        for i in range(len(spikes)):
            if spikes[i]:
                neighbours = np.arange(
                    max(0, i - kernel_size), min(len(Y) - 1, i + 1 + kernel_size)
                )
                non_spike_neighbours = neighbours[spikes[neighbours] == 0]
                fixed_value = np.mean(Y[non_spike_neighbours])

                if np.isnan(fixed_value):
                    continue

                Y[i] = fixed_value
                spikes[i] = False
                changes = True

        if not changes:
            break

    return Y


def cosmic_ray_removal(X, threshold=6, kernel_size=3):
    """Remove cosmic-ray spikes using the Whitaker-Hayes algorithm.

    This is a pure-NumPy re-implementation of
    ``ramanspy.preprocessing.despike.WhitakerHayes`` that operates on a
    batch of spectra without requiring the RamanSPy library.

    The algorithm computes modified z-scores of the first-difference of each
    spectrum.  Points whose score exceeds *threshold* are classified as spikes
    and replaced by the mean of their non-spiked neighbours within a window of
    ±*kernel_size* pixels.  The procedure repeats until no further spikes can
    be corrected.

    References
    ----------
    Whitaker, D.A. and Hayes, K., 2018. A simple algorithm for despiking Raman
    spectra. Chemometrics and Intelligent Laboratory Systems, 179, pp.82-84.


    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Input spectra.
    threshold : float
        Modified z-score threshold above which a point is classified as a
        cosmic-ray spike (default 6).  Lower values remove more aggressively.
    kernel_size : int
        Half-width (in pixels) of the neighbourhood used to compute the
        replacement value for a detected spike (default 3).

    Returns
    -------
    np.ndarray, shape (n_samples, n_features)
        Despiked spectra.
    """
    # Use the Whitaker-Hayes algorithm

    return np.array([_despike_single(spectrum, threshold, kernel_size) for spectrum in X])


def baseline_correction_asls(X, lam=1e5, p=0.01, n_iter=10):
    """Asymmetric Least Squares baseline correction.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Input spectra.
    lam : float
        Smoothness parameter (1e3–1e7). Higher = smoother baseline.
    p : float
        Asymmetry parameter (0.001–0.1). Lower = more asymmetric.
    n_iter : int
        Number of iterations.

    Returns
    -------
    np.ndarray, shape (n_samples, n_features)
        Baseline-corrected spectra.
    """
    n = X.shape[1]
    # Second-order difference matrix: D shape (n-2, n), D.T @ D shape (n, n)
    D = diags([1, -2, 1], [0, 1, 2], shape=(n - 2, n))
    H = csc_matrix(lam * D.T.dot(D))

    X_corrected = np.empty_like(X)
    for i in range(X.shape[0]):
        y = X[i]
        w = np.ones(n)
        for _ in range(n_iter):
            W = diags(w, 0, shape=(n, n))
            Z = W + H
            z = spsolve(Z, w * y)
            w = p * (y > z) + (1 - p) * (y <= z)
        X_corrected[i] = y - z

    return X_corrected


def multiplicative_scatter_correction_fit(X):
    """Compute the MSC reference spectrum (mean of training data).

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Training spectra.

    Returns
    -------
    np.ndarray, shape (n_features,)
        Reference spectrum.
    """
    return np.mean(X, axis=0)


def multiplicative_scatter_correction_transform(X, reference, start=0.0, end=1.0):
    """Apply multiplicative scatter correction using a fitted reference.

    The linear regression (slope, intercept) is fitted on a *region* of the
    spectrum defined by ``start`` and ``end`` (fractions of the total length),
    but the correction is applied to the **full** spectrum.  Selecting a
    low-variance region for the regression avoids fitting to analyte-specific
    peaks, producing more robust scatter correction.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Input spectra.
    reference : np.ndarray, shape (n_features,)
        Reference spectrum from fit.
    start : float
        Start of the regression region as a fraction of the spectrum
        (0.0–1.0).  Default 0.0 (beginning).
    end : float
        End of the regression region as a fraction of the spectrum
        (0.0–1.0).  Default 1.0 (full spectrum).

    Returns
    -------
    np.ndarray, shape (n_samples, n_features)
        MSC-corrected spectra.
    """
    n_features = X.shape[1]
    i_start = int(round(start * n_features))
    i_end = int(round(end * n_features))
    # Ensure at least 2 points for polyfit
    i_start = max(0, min(i_start, n_features - 2))
    i_end = max(i_start + 2, min(i_end, n_features))

    ref_region = reference[i_start:i_end]

    X_corrected = np.empty_like(X)
    for i in range(X.shape[0]):
        coef = np.polyfit(ref_region, X[i, i_start:i_end], 1)
        X_corrected[i] = (X[i] - coef[1]) / coef[0]
    return X_corrected


def denoise_savgol(X, window_length=9, polyorder=3):
    """Savitzky-Golay denoising filter.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Input spectra.
    window_length : int
        Filter window length (must be odd). Typical: 5, 7, 9, 11, 13, 15.
    polyorder : int
        Polynomial order. Must be < window_length. Typical: 2, 3, 4.

    Returns
    -------
    np.ndarray, shape (n_samples, n_features)
        Denoised spectra.
    """
    window_length = int(window_length)
    polyorder = int(polyorder)

    # Ensure window_length is odd
    if window_length % 2 == 0:
        window_length += 1

    # Ensure polyorder < window_length
    if polyorder >= window_length:
        polyorder = window_length - 1

    return savgol_filter(X, window_length=window_length, polyorder=polyorder, axis=1)


def snv(X):
    """
    Apply Standard Normal Variate (SNV) to spectra.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Spectral data (each row = one spectrum)

    Returns
    -------
    X_snv : ndarray of same shape
        SNV-transformed spectra
    """
    mean = np.mean(X, axis=1, keepdims=True)
    std = np.std(X, axis=1, keepdims=True)

    X_snv = (X - mean) / std
    return X_snv


def augment_spectra(
    X,
    y,
    noise_sigma=0.01,
    shift_max=0,
    n_augments=1,
    mixup_alpha=0.0,
    label_type="classification",
):
    """Augment spectra with noise, shifts, and linear combinations.

    Only to be applied during training. Generates ``n_augments``
    augmented copies per original spectrum using Gaussian noise and
    random wavenumber shifts. When ``mixup_alpha > 0``, additional
    spectra are created by linearly interpolating pairs of
    same-class samples (mixup).

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Input spectra.
    y : np.ndarray, shape (n_samples,)
        Labels (duplicated for augmented samples).
    noise_sigma : float
        Standard deviation of Gaussian noise relative to the
        global std of X (0.001-0.05).
    shift_max : int
        Maximum pixel shift (0, 1, 2, 3).
    n_augments : int
        Number of augmented copies to create per original spectrum.
        Total output size = (1 + n_augments [+ n_mixup]) * n_samples.
    mixup_alpha : float
        Beta-distribution parameter for mixup. When > 0, generates
        one mixup sample per original sample by blending with a
        random same-class partner. Set to 0 to disable.
    label_type : str
        "classification" or "regression". Determines how mixup labels
         are generated.

    Returns
    -------
    X_out : np.ndarray
        Original + augmented (+ mixup) spectra.
    y_out : np.ndarray
        Corresponding labels.
    """
    n_samples, n_features = X.shape
    n_augments = max(1, int(n_augments))
    shift_max = int(shift_max)
    x_std = np.std(X) if noise_sigma > 0 else 0.0

    aug_X_list = [X]
    aug_y_list = [y]

    for _ in range(n_augments):
        X_copy = X.copy()

        # Gaussian noise
        if noise_sigma > 0:
            noise = np.random.normal(
                0,
                noise_sigma * x_std,
                size=X_copy.shape,
            )
            X_copy = X_copy + noise

        # Random wavenumber shift
        if shift_max > 0:
            for i in range(n_samples):
                shift = np.random.randint(
                    -shift_max,
                    shift_max + 1,
                )
                if shift != 0:
                    X_copy[i] = np.roll(X_copy[i], shift)
                    if shift > 0:
                        X_copy[i, :shift] = X_copy[i, shift]
                    else:
                        X_copy[i, shift:] = X_copy[i, shift - 1]

        aug_X_list.append(X_copy)
        aug_y_list.append(y)

        # Mixup: linear combinations of same-class pairs
    if mixup_alpha > 0 and label_type == "classification":
        X_mixup = np.empty_like(X)
        y_mixup = y.copy()

        for i in range(n_samples):
            # Find same-class samples to mix with
            same_class = np.where(y == y[i])[0]
            j = same_class[np.random.randint(len(same_class))]
            lam = np.random.beta(mixup_alpha, mixup_alpha)
            X_mixup[i] = lam * X[i] + (1 - lam) * X[j]

        aug_X_list.append(X_mixup)
        aug_y_list.append(y_mixup)
    elif mixup_alpha > 0 and label_type == "regression":
        lambdas = np.random.beta(mixup_alpha, mixup_alpha, size=n_samples)
        first_idxs = np.random.randint(n_samples, size=n_samples)
        second_idxs = np.random.randint(n_samples, size=n_samples)
        aug_X_list.append(
            X[first_idxs] * lambdas[:, None] + X[second_idxs] * (1.0 - lambdas[:, None])
        )
        # Broadcast lambdas to match y's shape (works for 1-D and multi-output)
        lam_shape = (n_samples,) + (1,) * (y.ndim - 1)
        lam_broadcast = lambdas.reshape(lam_shape)
        mixed_y = y[first_idxs] * lam_broadcast + y[second_idxs] * (1.0 - lam_broadcast)
        aug_y_list.append(mixed_y)

    X_out = np.vstack(aug_X_list)
    y_out = np.concatenate(aug_y_list)

    return X_out, y_out


def augment_spectra_torch(
    X: torch.Tensor,
    y: torch.Tensor,
    noise_sigma: float = 0.01,
    shift_max: int = 0,
    n_augments: int = 1,
    mixup_alpha: float = 0.0,
    label_type: str = "classification",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Augment spectra with shifts, mixup, and noise (applied last).

    Unlike the numpy version, this function does **not** append augmented
    copies after the originals.  Instead it tiles the input ``n_augments``
    times and applies stochastic transforms to every row, so the output
    length is exactly ``n_augments * n_samples`` (the originals are *not*
    preserved separately).

    Parameters
    ----------
    X : torch.Tensor, shape (n_samples, n_features)
        Input spectra.
    y : torch.Tensor, shape (n_samples,) or (n_samples, n_targets)
        Labels.
    noise_sigma : floatStd of Gaussian noise relative to the global std of X.
    shift_max : int
        Maximum random pixel shift per spectrum.
    n_augments : int
        Number of augmented copies.  Output size = n_augments * n_samples
        (+ n_samples mixup rows when mixup_alpha > 0).
    mixup_alpha : float
        Beta-distribution parameter for mixup.  0 disables mixup.
    label_type : str
        "classification" or "regression".

    Returns
    -------
    X_out : torch.Tensor
    y_out : torch.Tensor
    """
    n_samples, n_features = X.shape
    n_augments = max(1, int(n_augments))
    shift_max = int(shift_max)

    # --- 1. Tile the data n_augments times --------------------------------
    # Each row is an independent augmented sample.
    X_aug = X.repeat(n_augments, 1).clone()  # (n_augments * n_samples, n_features)
    y_aug = y.repeat(n_augments, *([1] * (y.dim() - 1))).clone()

    total = X_aug.shape[0]

    # --- 2. Random wavenumber shifts --------------------------------------
    if shift_max > 0:
        shifts = torch.randint(-shift_max, shift_max + 1, (total,))
        for i in range(total):
            s = shifts[i].item()
            if s != 0:
                X_aug[i] = torch.roll(X_aug[i], s)
                if s > 0:
                    X_aug[i, :s] = X_aug[i, s]
                else:
                    X_aug[i, s:] = X_aug[i, s - 1]

    # --- 3. Mixup ---------------------------------------------------------
    if mixup_alpha > 0:
        # Draw lambdas via numpy (torch has no built-in Beta on CPU without distributions)
        lambdas_np = np.random.beta(mixup_alpha, mixup_alpha, size=total)
        lambdas = torch.tensor(lambdas_np, dtype=X.dtype, device=X.device)

        if label_type == "classification":
            # Same-class mixup: blend each sample with a random same-class partner

            for i in range(total):
                same_class = (y_aug == y_aug[i]).nonzero(as_tuple=True)[0]
                j = same_class[torch.randint(len(same_class), (1,)).item()]
                lam = lambdas[i]
                X_aug[i] = lam * X_aug[i] + (1 - lam) * X_aug[j]
                # Label stays the same (same class)

        elif label_type == "regression":
            first_idxs = torch.randint(total, (total,))
            second_idxs = torch.randint(total, (total,))

            lam_x = lambdas.unsqueeze(1)  # (total, 1)
            X_aug = X_aug[first_idxs] * lam_x + X_aug[second_idxs] * (1.0 - lam_x)

            # Broadcast lambdas for multi-output y
            lam_shape = (total,) + (1,) * (y_aug.dim() - 1)
            lam_y = lambdas.reshape(lam_shape)
            y_aug = y_aug[first_idxs] * lam_y + y_aug[second_idxs] * (1.0 - lam_y)

    # --- 4. Gaussian noise (applied last) ---------------------------------
    if noise_sigma > 0:
        x_std = X_aug.std().item()
        noise = torch.normal(
            mean=0.0,
            std=noise_sigma * x_std,
            size=X_aug.shape,
            dtype=X_aug.dtype,
            device=X_aug.device,
        )
        X_aug = X_aug + noise

    return X_aug, y_aug
