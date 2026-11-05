"""Process-parallel helpers for CPU work that would otherwise run on one core (2026-11-05).

GPU pods spend much of a fit's wall time in single-threaded Python loops while the other cores
idle (RamanPreprocessing profile on BHT, 2026-11-05):

* the per-spectrum preprocessing steps (crop, CRR, Savitzky-Golay, wavelet, AsLS/airPLS/arPLS,
  rubberband) loop over rows; for the deep models they run on every augmented training set,
  which is new random data at each call, so no cache can serve it;
* TabICL's default ``norm_methods=["none", "power"]`` fits sklearn's Yeo-Johnson
  :class:`~sklearn.preprocessing.PowerTransformer`, one scalar optimisation per column; on
  ~11,000-channel spectra that loop dominates ``fit``.

Both are independent per row (steps) or per column (Yeo-Johnson lambda), so they are split into
chunks and run in a pool of worker processes. Results equal the serial ones: each worker runs
the same function on the same values. One exception, known from the spectrum cache tests:
``scipy.signal.savgol_filter(axis=1)`` can differ in the last bits depending on which rows share
a call.

Off unless :func:`configure` is called with ``n_workers > 1`` (the batch runner's
``--prep-workers``). Workers are started with ``spawn``, so the pool is safe to use from a
process that has already initialised CUDA.
"""

from __future__ import annotations

import concurrent.futures as cf
import multiprocessing as mp
from types import SimpleNamespace

import numpy as np

_N_WORKERS = 1
_POOL: cf.ProcessPoolExecutor | None = None
# Below these sizes a pool round trip costs more than it saves.
MIN_ROWS = 8
MIN_COLUMNS = 512


def configure(n_workers: int) -> None:
    """Set the pool size for this process; 1 turns the parallel paths off."""
    global _N_WORKERS
    assert n_workers >= 1, n_workers
    _N_WORKERS = n_workers


def n_workers() -> int:
    return _N_WORKERS


def _pool() -> cf.ProcessPoolExecutor:
    global _POOL
    if _POOL is None:
        _POOL = cf.ProcessPoolExecutor(max_workers=_N_WORKERS, mp_context=mp.get_context("spawn"))
    return _POOL


def _chunks(n: int, k: int) -> list[slice]:
    bounds = np.linspace(0, n, k + 1).astype(int)
    return [slice(a, b) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]


# --- per-spectrum steps -------------------------------------------------------------------


def _spectrumwise_worker(X: np.ndarray, params: dict, phase: str, wavenumbers) -> np.ndarray:
    from raman_bench.preprocessing.mixin import RamanPreprocessingMixin

    # The step method reads nothing from the model but ``_wavenumbers``.
    holder = SimpleNamespace(_wavenumbers=wavenumbers)
    return RamanPreprocessingMixin._apply_spectrumwise_steps(holder, X, params, phase)


def spectrumwise_rows(X: np.ndarray, params: dict, phase: str, wavenumbers) -> np.ndarray:
    """``_apply_spectrumwise_steps`` on the rows of ``X``, in row chunks across the pool."""
    parts = _chunks(len(X), _N_WORKERS)
    futures = [_pool().submit(_spectrumwise_worker, X[s], params, phase, wavenumbers) for s in parts]
    return np.concatenate([f.result() for f in futures], axis=0)


def use_rows(n_rows: int) -> bool:
    return _N_WORKERS > 1 and n_rows >= MIN_ROWS


# --- Yeo-Johnson lambdas ------------------------------------------------------------------


def _yeo_johnson_worker(X: np.ndarray) -> np.ndarray:
    from sklearn.preprocessing import PowerTransformer

    pt = PowerTransformer(method="yeo-johnson")
    with np.errstate(invalid="ignore"):
        return np.array([pt._yeo_johnson_optimize(col) for col in X.T], dtype=X.dtype)


def patch_power_transformer() -> None:
    """Make ``PowerTransformer(method="yeo-johnson")`` fit its column lambdas in the pool.

    sklearn's ``_fit`` loops over columns: constant columns get lambda 1, every other column
    gets ``_yeo_johnson_optimize(col)`` on the untouched column, then that column is
    transformed. The patch computes the non-constant columns' lambdas up front in the pool
    (same function, same column values) and lets the original ``_fit`` run with
    ``_yeo_johnson_optimize`` answering from that list in column order, so the fitted state is
    the one the serial loop produces. Idempotent.
    """
    from sklearn.preprocessing import PowerTransformer
    from sklearn.preprocessing._data import _is_constant_feature

    if hasattr(PowerTransformer, "_raman_bench_orig_fit"):
        return
    orig = PowerTransformer._fit

    def _fit(self, X, y=None, force_transform=False):
        if self.method != "yeo-johnson" or _N_WORKERS <= 1:
            return orig(self, X, y, force_transform)
        Xc = self._check_input(X, in_fit=True, check_positive=True)
        if Xc.shape[1] < MIN_COLUMNS:
            return orig(self, X, y, force_transform)
        n = Xc.shape[0]
        mean = np.mean(Xc, axis=0, dtype=np.float64)
        var = np.var(Xc, axis=0, dtype=np.float64)
        keep = [i for i in range(Xc.shape[1]) if not _is_constant_feature(var[i], mean[i], n)]
        cols = np.ascontiguousarray(Xc[:, keep])
        parts = _chunks(len(keep), _N_WORKERS)
        futures = [_pool().submit(_yeo_johnson_worker, cols[:, s]) for s in parts]
        lambdas = iter(np.concatenate([f.result() for f in futures]))
        self._yeo_johnson_optimize = lambda col: next(lambdas)  # instance attribute, this fit only
        try:
            return orig(self, X, y, force_transform)
        finally:
            del self._yeo_johnson_optimize
            assert next(lambdas, None) is None, "Yeo-Johnson: precomputed lambdas not all used"

    PowerTransformer._raman_bench_orig_fit = orig
    PowerTransformer._fit = _fit
