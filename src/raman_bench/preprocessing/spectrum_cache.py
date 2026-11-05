"""Cache for the per-spectrum preprocessing steps (crop, CRR, smoothing, baseline correction).

These steps (``RamanPreprocessingMixin._apply_spectrumwise_steps``) transform each spectrum on
its own, with nothing fitted on the training fold, so a spectrum's output depends only on the
spectrum itself, the step parameters and the code. A benchmark recomputes the same outputs many
times: for every model, outer fold and bag child, and inside each bag child at fit, at the
training-data transform and at prediction. On large datasets with baseline correction this
dominates the run time of GPU pods.

The cache stores each output row under a digest of the input row, per *step key* (the enabled
steps' parameters, a fingerprint of the preprocessing code, and the wavenumber axis when
physical cropping is on). Rows it has not seen are computed by the normal code and added, so
results are the same with or without it; the tests check this bit for bit.

Off by default. ``RAMAN_BENCH_SPECTRUM_CACHE`` turns it on:

- ``memory``: one in-process cache (useful for a process that runs many jobs, e.g.
  ``scripts/run_experiment_batch.py``);
- a directory: as ``memory``, and on first use of a step key also load the rows precomputed into
  ``<dir>/<step key>/*.npz`` by ``scripts/precompute_spectrum_cache.py`` (shared by all pods).
  Runs only read this directory; only the precompute script writes it.

``RAMAN_BENCH_SPECTRUM_CACHE_MAX_GB`` (default 8) bounds the in-memory size; when it is exceeded
the in-memory tables are dropped and refilled on demand.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import os
from collections.abc import Callable
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

ENV_MODE = "RAMAN_BENCH_SPECTRUM_CACHE"
ENV_MAX_GB = "RAMAN_BENCH_SPECTRUM_CACHE_MAX_GB"
DIGEST_SIZE = 16

# Per-spectrum steps in the order the mixin applies them, each with the parameters it reads.
SPECTRUMWISE_STEPS: dict[str, tuple[str, ...]] = {
    "prep_crop_physical_enabled": ("prep_crop_physical_start_cm", "prep_crop_physical_end_cm"),
    "prep_crop_enabled": ("prep_crop_start_frac", "prep_crop_end_frac"),
    "prep_crr_enabled": ("prep_crr_threshold", "prep_crr_kernel_size"),
    "prep_denoise_enabled": ("prep_denoise_wl", "prep_denoise_po"),
    "prep_wavelet_enabled": ("prep_wavelet_name", "prep_wavelet_level"),
    "prep_bl_enabled": ("prep_bl_lam", "prep_bl_p"),
    "prep_airpls_enabled": ("prep_airpls_lam", "prep_airpls_max_iter", "prep_airpls_tol"),
    "prep_arpls_enabled": ("prep_arpls_lam", "prep_arpls_max_iter"),
    "prep_rubberband_enabled": (),
}

_CODE_FINGERPRINT: str | None = None


def code_fingerprint() -> str:
    """Digest of the code that computes the cached outputs, so a code change never reuses rows
    computed by older code."""
    global _CODE_FINGERPRINT
    if _CODE_FINGERPRINT is None:
        from raman_bench.preprocessing import raman_preprocessing
        from raman_bench.preprocessing.mixin import RamanPreprocessingMixin

        src = inspect.getsource(raman_preprocessing) + inspect.getsource(
            RamanPreprocessingMixin._apply_spectrumwise_steps
        )
        _CODE_FINGERPRINT = hashlib.blake2b(src.encode(), digest_size=8).hexdigest()
    return _CODE_FINGERPRINT


def step_key(params: dict, wavenumbers: np.ndarray | None) -> str | None:
    """Key of the per-spectrum part of a recipe, or ``None`` if that part does nothing."""
    enabled = {
        flag: [params.get(p) for p in names]
        for flag, names in SPECTRUMWISE_STEPS.items()
        if params.get(flag, False)
    }
    if not enabled:
        return None
    payload = {"steps": enabled, "code": code_fingerprint()}
    if "prep_crop_physical_enabled" in enabled:
        wn = np.ascontiguousarray(np.asarray(wavenumbers, dtype=np.float64))
        payload["wavenumbers"] = hashlib.blake2b(wn.tobytes(), digest_size=8).hexdigest()
    text = json.dumps(payload, sort_keys=True, default=repr)
    return hashlib.blake2b(text.encode(), digest_size=12).hexdigest()


def row_digests(X: np.ndarray) -> list[bytes]:
    X = np.ascontiguousarray(X)
    return [hashlib.blake2b(row.tobytes(), digest_size=DIGEST_SIZE).digest() for row in X]


class SpectrumCache:
    def __init__(self, directory: str | Path | None = None, max_bytes: float = 8e9) -> None:
        self.directory = Path(directory) if directory else None
        self.max_bytes = max_bytes
        self._tables: dict[str, dict[bytes, np.ndarray]] = {}
        self._bytes = 0
        self.hits = 0
        self.misses = 0

    def _table(self, key: str) -> dict[bytes, np.ndarray]:
        if key not in self._tables:
            table: dict[bytes, np.ndarray] = {}
            if self.directory is not None and (self.directory / key).is_dir():
                for f in sorted((self.directory / key).glob("*.npz")):
                    with np.load(f) as z:
                        out = z["out"]
                        for d, row in zip(z["digests"], out):
                            table[bytes(d)] = row
                        self._bytes += out.nbytes
                logger.info("Spectrum cache: loaded %d precomputed rows for %s", len(table), key)
            self._tables[key] = table
        return self._tables[key]

    def apply(
        self,
        X: np.ndarray,
        key: str,
        compute: Callable[[np.ndarray], np.ndarray],
        store: bool = True,
    ) -> np.ndarray:
        """Return ``compute(X)``, computing only the rows not cached under ``key``.

        ``store=False`` serves cached rows but keeps the computed ones out of the cache: for
        sets whose new rows are never seen again (augmented training sets: random noise copies
        made fresh at every fit), storing them would only fill memory and, past ``max_bytes``,
        evict the real spectra.
        """
        if self._bytes > self.max_bytes:
            logger.info(
                "Spectrum cache: %.1f GB > limit, dropping in-memory tables", self._bytes / 1e9
            )
            self._tables.clear()
            self._bytes = 0
        table = self._table(key)
        digests = row_digests(X)
        missing = [i for i, d in enumerate(digests) if d not in table]
        self.hits += len(digests) - len(missing)
        self.misses += len(missing)
        if not missing:
            return np.stack([table[d] for d in digests])
        out = compute(np.ascontiguousarray(X)[missing])
        if store:
            for i, row in zip(missing, out):
                table[digests[i]] = row
            self._bytes += out.nbytes
            return np.stack([table[d] for d in digests])
        computed = dict(zip(missing, out))
        return np.stack([computed[i] if i in computed else table[d] for i, d in enumerate(digests)])

    def tables(self) -> dict[str, dict[bytes, np.ndarray]]:
        return self._tables


_CACHE: SpectrumCache | None = None
_CACHE_MODE: str | None = None


def get_cache() -> SpectrumCache | None:
    """The process-wide cache selected by ``RAMAN_BENCH_SPECTRUM_CACHE``, or ``None`` (off)."""
    global _CACHE, _CACHE_MODE
    mode = os.environ.get(ENV_MODE, "").strip()
    if mode in ("", "0", "off"):
        return None
    if _CACHE is None or mode != _CACHE_MODE:
        max_bytes = float(os.environ.get(ENV_MAX_GB, "8")) * 1e9
        _CACHE = SpectrumCache(None if mode == "memory" else mode, max_bytes=max_bytes)
        _CACHE_MODE = mode
        logger.info("Spectrum cache on: %s", mode)
    return _CACHE


def save_table(directory: str | Path, key: str, table: dict[bytes, np.ndarray], name: str) -> Path:
    """Write one table as ``<directory>/<key>/<name>.npz`` (written to a temporary name first, so
    a reader never sees a partial file)."""
    out_dir = Path(directory) / key
    out_dir.mkdir(parents=True, exist_ok=True)
    digests = np.frombuffer(b"".join(table), dtype=np.uint8).reshape(len(table), DIGEST_SIZE)
    path = out_dir / f"{name}.npz"
    tmp = out_dir / f".{name}.{os.getpid()}.tmp.npz"
    np.savez(tmp, digests=digests, out=np.stack(list(table.values())))
    os.replace(tmp, path)
    return path
