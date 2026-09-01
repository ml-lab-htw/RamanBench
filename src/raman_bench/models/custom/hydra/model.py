"""Hydra transform + closed-form GPU ridge — sklearn-compatible estimator.

Investigates issue #4: sktime's ``RocketClassifier``/``Arsenal`` (both used
elsewhere in this package) wrap ``sklearn.linear_model.RidgeClassifierCV``,
which is CPU-only and slow beyond a few hundred samples. Angus Dempster
(ROCKET/Hydra author) pointed us to a GPU-accelerated alternative: the Hydra
convolutional-kernel transform paired with a custom closed-form ridge solver
designed for large data (Dempster et al., "Highly Scalable Time Series
Classification for Very Large Datasets", AALTD 2024 / ECML PKDD 2024,
reference code at https://github.com/angus924/aaltd2024/blob/main/code/).

``_HydraTransform`` is ported near-verbatim from that repo's ``hydra_gpu.py``
(univariate case only -- Raman spectra are single-channel). ``_ClosedFormRidge``
is a from-scratch, in-memory (non-streaming) simplification of that repo's
``ridge.py``: the original targets very-large, disk-mmapped datasets loaded in
batches, which RamanBench's datasets don't need; this keeps the same two-branch
closed-form algorithm (Gram-matrix + exact LOOCV when ``n < p``, normal
equations + held-out-split lambda selection when ``n >= p``) without the
streaming/``Dataset`` plumbing. Unlike the reference code (classification via
``RidgeClassifier`` only), this also supports regression directly, since one
open question (issue #5) is whether ROCKET/Arsenal's regression gap should be
closed on Hydra instead of on the models being replaced.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.base import BaseEstimator

from raman_bench.preprocessing.bridge_bases import SklearnAutoGluonBridge, _NoAugBase

_EPS = float(np.finfo(np.float32).eps)


class _HydraTransform(torch.nn.Module):
    """Univariate Hydra transform (Dempster et al., AALTD 2024 / hydra_gpu.py)."""

    def __init__(self, input_length: int, k: int = 8, g: int = 64, seed: int | None = None):
        super().__init__()

        if seed is not None:
            torch.manual_seed(seed)

        self.k = k  # kernels per group
        self.g = g  # number of groups

        max_exponent = np.log2((input_length - 1) / (9 - 1))  # kernel length = 9
        self.dilations = 2 ** torch.arange(int(max_exponent) + 1)
        self.num_dilations = len(self.dilations)
        self.paddings = torch.div((9 - 1) * self.dilations, 2, rounding_mode="floor").int()

        self.divisor = min(2, self.g)
        self.h = self.g // self.divisor

        W = torch.randn(self.num_dilations, self.divisor, self.k * self.h, 1, 9)
        W = W - W.mean(-1, keepdims=True)
        W = W / W.abs().sum(-1, keepdims=True)
        self.register_buffer("W", W)

        self.num_features = self.num_dilations * self.divisor * self.k * self.h * 2

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        num_examples = X.shape[0]

        if self.divisor > 1:
            diff_X = torch.diff(X)

        Z = []
        for dilation_index in range(self.num_dilations):
            d = self.dilations[dilation_index].item()
            p = self.paddings[dilation_index].item()

            for diff_index in range(self.divisor):
                _Z = F.conv1d(
                    X if diff_index == 0 else diff_X,
                    self.W[dilation_index, diff_index],
                    dilation=d,
                    padding=p,
                ).view(num_examples, self.h, self.k, -1)

                max_values, max_indices = _Z.max(2)
                count_max = torch.zeros(num_examples, self.h, self.k, device=X.device)
                min_values, min_indices = _Z.min(2)
                count_min = torch.zeros(num_examples, self.h, self.k, device=X.device)

                count_max.scatter_add_(-1, max_indices, max_values)
                count_min.scatter_add_(-1, min_indices, torch.ones_like(min_values))

                Z.append(count_max)
                Z.append(count_min)

        Z = torch.cat(Z, 1).view(num_examples, -1)
        return Z.clamp(0).sqrt()

    def batch(self, X: torch.Tensor, batch_size: int = 256) -> torch.Tensor:
        if X.shape[0] <= batch_size:
            return self(X)
        chunks = torch.arange(X.shape[0]).split(batch_size)
        return torch.cat([self(X[c]) for c in chunks])


def _binarize(y_idx: torch.Tensor, n_classes: int) -> torch.Tensor:
    """One-hot ±1 target encoding (true class = +1, all others = -1)."""
    out = -torch.ones(len(y_idx), n_classes)
    out.scatter_(-1, y_idx[:, None], 1.0)
    return out


class _ClosedFormRidge:
    """In-memory closed-form ridge, adapted from aaltd2024/code/ridge.py.

    ``n < p`` (typical for Raman spectra: thousands of Hydra features, tens to
    low-hundreds of training samples): solves via the n x n Gram matrix and
    picks lambda by *exact* leave-one-out CV (no data wasted on a held-out
    split, which matters most exactly in this small-n regime).

    ``n >= p``: solves the p x p normal equations and picks lambda by a
    held-out validation split (matches the reference code's other branch).
    """

    def __init__(
        self,
        device: torch.device,
        standardize_targets: bool,
        lambdas: torch.Tensor | None = None,
        val_fraction: float = 0.2,
        max_val_size: int = 8_192,
        seed: int | None = None,
    ):
        self.device = device
        self.standardize_targets = standardize_targets
        self.lambdas = lambdas if lambdas is not None else torch.logspace(-6, 6, 21)
        self.val_fraction = val_fraction
        self.max_val_size = max_val_size
        self.seed = seed

    def fit(self, Z: torch.Tensor, Y: torch.Tensor) -> _ClosedFormRidge:
        n, p = Z.shape

        self.x_mean = Z.mean(0)
        self.x_std = Z.std(0) + _EPS * 10
        X0 = (Z - self.x_mean) / self.x_std

        self.y_mean = Y.mean(0)
        self.y_std = (
            (Y.std(0) + _EPS * 10) if self.standardize_targets else torch.ones_like(self.y_mean)
        )
        Y0 = (Y - self.y_mean) / self.y_std

        if n < p:
            S2, U = torch.linalg.eigh(X0 @ X0.T)
            S2 = S2.clip(_EPS)
            S = S2.sqrt()
            V = (X0.T @ U) * (1 / S)
            R = U * S
            R2 = R**2
            RTY = R.T @ Y0

            best_alpha, best_err = None, float("inf")
            for lam in self.lambdas * (n**0.5):
                alpha_hat = RTY / (S2[:, None] + lam)
                Y_hat = R @ alpha_hat
                E = Y0 - Y_hat
                diag_H = (R2 / (S2 + lam)).sum(1)
                E_loocv = E / (1 - diag_H[:, None]).clip(_EPS)
                err = (E_loocv**2).mean().item()
                if err < best_err:
                    best_err, best_alpha = err, alpha_hat
            self.B = V @ best_alpha
        else:
            rng = np.random.RandomState(self.seed)
            val_size = max(min(int(n * self.val_fraction), self.max_val_size), 1)
            perm = rng.permutation(n)
            va_idx, tr_idx = perm[:val_size], perm[val_size:]

            Xtr, Ytr = X0[tr_idx], Y0[tr_idx]
            Xva, Yva = X0[va_idx], Y[va_idx]

            XTX = Xtr.T @ Xtr
            XTY = Xtr.T @ Ytr
            S2, V = torch.linalg.eigh(XTX)
            S2 = S2.clip(_EPS)

            best_B, best_err = None, float("inf")
            for lam in self.lambdas * (len(tr_idx) ** 0.5):
                XTXi = (V * (1 / (S2 + lam))) @ V.T
                B = XTXi @ XTY
                Y_hat_va = (Xva @ B) * self.y_std + self.y_mean
                err = ((Yva - Y_hat_va) ** 2).mean().item()
                if err < best_err:
                    best_err, best_B = err, B
            self.B = best_B
        return self

    def decision(self, Z: torch.Tensor) -> torch.Tensor:
        X0 = (Z - self.x_mean) / self.x_std
        return (X0 @ self.B) * self.y_std + self.y_mean


class HydraModel(BaseEstimator):
    """Hydra transform + closed-form ridge for Raman spectra.

    Regression or classification (binary/multiclass) is inferred from ``y``'s
    dtype at fit time, mirroring ``RidgeModel``'s convention. Uses CUDA if
    available (override via ``device``); falls back to CPU otherwise.

    Reference:
        Dempster, A., Schmidt, D. F., & Webb, G. I. (2023). Hydra: competing
        convolutional kernels for fast and accurate time series
        classification. Data Mining and Knowledge Discovery, 37(5), 1779-1805.
        https://doi.org/10.1007/s10618-023-00939-3

        Dempster, A., Tan, C. W., Miller, L., Foumani, N. M., Schmidt, D. F.,
        & Webb, G. I. (2024). Highly scalable time series classification for
        very large datasets. AALTD 2024 (ECML PKDD 2024 workshop).
    """

    def __init__(self, k=8, g=64, random_state=None, device=None):
        self.k = k
        self.g = g
        self.random_state = random_state
        self.device = device

    def fit(self, X, y):
        X_np = X.values if hasattr(X, "values") else np.asarray(X)
        y_arr = np.asarray(y)

        device = torch.device(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.transform_ = _HydraTransform(
            input_length=X_np.shape[1], k=self.k, g=self.g, seed=self.random_state
        ).to(device)

        Xt = torch.tensor(X_np, dtype=torch.float32, device=device).unsqueeze(1)
        Z = self.transform_.batch(Xt)

        if np.issubdtype(y_arr.dtype, np.floating):
            self.problem_type_ = "regression"
            Y = torch.tensor(y_arr, dtype=torch.float32, device=device)[:, None]
            standardize_targets = True
        else:
            unique = np.unique(y_arr)
            self.problem_type_ = "binary" if len(unique) == 2 else "multiclass"
            self.classes_ = unique
            y_idx = torch.tensor(np.searchsorted(unique, y_arr), dtype=torch.long, device=device)
            Y = _binarize(y_idx.cpu(), len(unique)).to(device)
            standardize_targets = False

        self.ridge_ = _ClosedFormRidge(
            device=device, standardize_targets=standardize_targets, seed=self.random_state
        ).fit(Z, Y)
        return self

    def _decision(self, X):
        X_np = X.values if hasattr(X, "values") else np.asarray(X)
        device = self.transform_.W.device
        Xt = torch.tensor(X_np, dtype=torch.float32, device=device).unsqueeze(1)
        Z = self.transform_.batch(Xt)
        return self.ridge_.decision(Z)

    def predict(self, X):
        scores = self._decision(X)
        if self.problem_type_ == "regression":
            return scores.squeeze(-1).cpu().numpy()
        indices = scores.argmax(-1).cpu().numpy()
        return self.classes_[indices]

    def predict_proba(self, X):
        scores = self._decision(X).cpu().numpy()
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        proba = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        if self.problem_type_ == "binary":
            return proba[:, 1]
        return proba


class _HydraBridge(SklearnAutoGluonBridge):
    _sklearn_cls = HydraModel
    ag_key = "HYDRA"
    ag_name = "Hydra"

    def _get_default_searchspace(self):
        from autogluon.common import space

        return {
            "k": space.Int(lower=4, upper=16),
            "g": space.Categorical(32, 64, 128),
        }


class Prep_HYDRA(_NoAugBase, _HydraBridge):  # noqa: N801
    pass
