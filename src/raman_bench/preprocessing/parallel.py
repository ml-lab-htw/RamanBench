"""Process-parallel helpers for the per-spectrum preprocessing steps (2026-11-05).

GPU pods spend much of a fit's wall time in single-threaded Python loops over rows
(crop, cosmic-ray removal, Savitzky-Golay, wavelet, AsLS/airPLS/arPLS, rubberband). For the deep
models these run on every augmented training set, which is new random data at each call, so the
spectrum cache cannot serve them.

The steps are independent per row, so the rows are split into chunks and run in a pool of
worker processes. Each worker runs the same function on the same values, so the results equal
the serial ones, except that ``scipy.signal.savgol_filter(axis=1)`` can differ in the last bits
depending on which rows share a call.

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
# Below this many rows a pool round trip costs more than it saves.
MIN_ROWS = 8


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
    futures = [
        _pool().submit(_spectrumwise_worker, X[s], params, phase, wavenumbers) for s in parts
    ]
    return np.concatenate([f.result() for f in futures], axis=0)


def use_rows(n_rows: int) -> bool:
    return _N_WORKERS > 1 and n_rows >= MIN_ROWS
