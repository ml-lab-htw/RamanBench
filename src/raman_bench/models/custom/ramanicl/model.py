"""RamanICL -- an in-context learning model pretrained on a Raman-flavoured prior.

RamanICL uses the TabICLv2 architecture verbatim and changes only the *prior* it is
pretrained on: generic graph-SCM tasks, a fraction of which carry Raman input
corruption (fluorescence baseline, Poisson shot noise). It is therefore a pretrained
foundation model, not a per-task learner -- ``fit`` caches the in-context training set
and runs no gradient steps.

Two things make this bridge different from the from-scratch custom models here
(``ridge``, ``deepcnn``, ...):

**It needs a checkpoint.** There is no RamanBench-native precedent for a custom model
with pretrained weights, so this follows the TabArena-native foundation models (MITRA,
LIMIX2, Kumo) and resolves weights at first ``fit`` with a HuggingFace Hub download
cached under ``CACHE_DIR``. A plain filesystem path is also accepted, because during
model development the checkpoint usually exists only on the author's training cluster.
See ``_resolve_checkpoint`` -- the resolution order matters, and the error when nothing
resolves names every option rather than failing with a bare ``FileNotFoundError``.

**It is regression-only.** The published checkpoints are built with ``max_classes=0``
and a 999-quantile regression decoder, so there is no classification head to call. On a
classification task this raises rather than silently returning something meaningless, so
a core sweep reports honest partial coverage (38 of 53 targets) instead of a fake score.

Inputs are resampled to a fixed channel count (default 256) because the prior generates
spectra on a window of that width and the model's column embedder was trained in that
regime. This is a property of the current checkpoints, not of the method.
"""

from __future__ import annotations

import os

import numpy as np
from sklearn.base import BaseEstimator

from raman_bench.preprocessing.bridge_bases import (
    SklearnAutoGluonBridge,
    _GPURequiredBridge,
    _NoAugBase,
)

#: Env var holding a filesystem path to a ``.ckpt`` (local runs, or a PVC-resident copy).
CKPT_ENV = "RAMANICL_CKPT"
#: Env var holding a HuggingFace Hub model repo id to fetch the checkpoint from.
HF_REPO_ENV = "RAMANICL_HF_REPO"
#: Env var overriding which file to pull from that repo.
HF_FILE_ENV = "RAMANICL_HF_FILE"

DEFAULT_HF_FILE = "step-20000.ckpt"

#: The tabicl release RamanICL checkpoints are trained against. This is the same version
#: this repo pins, so no extra dependency is introduced -- but an older build (observed
#: with 2.1.1) predates architecture keys the checkpoints carry and fails with an
#: unexpected-keyword TypeError rather than anything self-explanatory; see fit().
TABICL_REQUIRED = "tabicl==2.2.0"


def _resolve_checkpoint(checkpoint: str | None) -> str:
    """Return a local path to the RamanICL checkpoint, downloading it if needed.

    Resolution order, first match wins:

    1. ``checkpoint`` passed as a hyperparameter -- an explicit path always wins.
    2. ``$RAMANICL_CKPT`` -- a filesystem path. This covers local development and a
       checkpoint copied onto the shared PVC.
    3. ``$RAMANICL_HF_REPO`` -- a HuggingFace Hub repo, downloaded via
       ``hf_hub_download`` into ``$CACHE_DIR``. On the k8s cluster ``CACHE_DIR`` is
       PVC-backed and shared between pods, so this downloads once, not once per pod.

    The order exists because of a real deployment gap: a training-cluster path like
    ``/scratch/<user>/...`` does not exist inside a k8s pod, which only mounts the PVC.
    An env var pointing there works locally and hard-fails on cluster submission, so the
    Hub route is the one that works unattended and the path routes are the escape hatch.
    """
    if checkpoint:
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(
                f"RamanICL checkpoint hyperparameter points at {checkpoint!r}, which does "
                f"not exist.",
            )
        return checkpoint

    env_path = os.environ.get(CKPT_ENV)
    if env_path:
        if not os.path.isfile(env_path):
            raise FileNotFoundError(
                f"${CKPT_ENV} is set to {env_path!r}, which does not exist. On the k8s "
                f"cluster only the PVC is mounted -- a path under a training cluster's "
                f"/scratch will not resolve here. Copy the checkpoint onto the PVC, or set "
                f"${HF_REPO_ENV} instead.",
            )
        return env_path

    hf_repo = os.environ.get(HF_REPO_ENV)
    if hf_repo:
        from huggingface_hub import hf_hub_download

        cache_dir = os.environ.get("CACHE_DIR")
        return hf_hub_download(
            repo_id=hf_repo,
            filename=os.environ.get(HF_FILE_ENV, DEFAULT_HF_FILE),
            cache_dir=os.path.join(cache_dir, "huggingface") if cache_dir else None,
        )

    raise RuntimeError(
        "RamanICL needs a pretrained checkpoint and none was found. Provide one of: the "
        f"`checkpoint` hyperparameter, ${CKPT_ENV} (a path to a .ckpt), or ${HF_REPO_ENV} "
        f"(a HuggingFace Hub repo id, optionally with ${HF_FILE_ENV}; default file "
        f"{DEFAULT_HF_FILE!r}).",
    )


def _resample(x: np.ndarray, n_channels: int) -> np.ndarray:
    """Resample each spectrum onto ``n_channels`` uniformly spaced points.

    Index-space, not wavenumber-space: the bridge receives a plain feature matrix and
    AutoGluon does not carry the wavenumber axis through, so there is no axis to
    interpolate against. That matches how the checkpoints were evaluated during
    development and keeps the feature count inside the regime they were trained on.
    """
    n, d = x.shape
    if d == n_channels:
        return np.ascontiguousarray(x, dtype=np.float64)
    src = np.linspace(0.0, 1.0, d)
    dst = np.linspace(0.0, 1.0, n_channels)
    out = np.empty((n, n_channels), dtype=np.float64)
    for i in range(n):
        out[i] = np.interp(dst, src, x[i])
    return out


class RamanICLModel(BaseEstimator):
    """RamanICL in-context regressor -- sklearn-compatible.

    Parameters
    ----------
    checkpoint
        Path to a ``.ckpt``. When ``None`` the checkpoint is resolved from the
        environment; see :func:`_resolve_checkpoint`.
    n_channels
        Resample every spectrum to this many channels before inference.
    max_context
        Cap on the in-context training set. TabICL inference cost grows with context
        size and some RamanBench datasets have thousands of training spectra, so the
        context is randomly subsampled above this size. Subsampling is seeded, so a
        given fold is reproducible.
    device
        ``"cuda"``, ``"cpu"``, or ``None`` to pick CUDA when available.
    random_state
        Seed for the context subsample.
    """

    def __init__(
        self,
        checkpoint=None,
        n_channels=256,
        max_context=1024,
        device=None,
        random_state=0,
    ):
        self.checkpoint = checkpoint
        self.n_channels = n_channels
        self.max_context = max_context
        self.device = device
        self.random_state = random_state

    def _resolve_device(self) -> str:
        if self.device:
            return self.device
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    def fit(self, X, y):
        # Imported lazily so this package's info.py stays importable when tabicl is not
        # installed -- the model registry skips a package whose info module raises
        # ImportError, and a top-level import here would silently drop RamanICL from
        # every registry listing on an environment without tabicl.
        from tabicl import TabICLRegressor

        x_np = X.values if hasattr(X, "values") else np.asarray(X)
        y_arr = np.asarray(y)

        if not np.issubdtype(y_arr.dtype, np.floating):
            raise ValueError(
                "RamanICL is regression-only: its checkpoints are built with max_classes=0 "
                "and a 999-quantile regression decoder, so there is no classification head. "
                f"Got a target of dtype {y_arr.dtype}.",
            )

        x_rs = _resample(np.asarray(x_np, dtype=np.float64), self.n_channels)
        y_arr = y_arr.astype(np.float64)

        if self.max_context and len(x_rs) > self.max_context:
            keep = np.random.default_rng(self.random_state).choice(
                len(x_rs), self.max_context, replace=False
            )
            x_rs, y_arr = x_rs[keep], y_arr[keep]

        self.model_ = TabICLRegressor(
            model_path=_resolve_checkpoint(self.checkpoint),
            allow_auto_download=False,
            device=self._resolve_device(),
        )
        try:
            self.model_.fit(x_rs, y_arr)
        except TypeError as exc:  # pragma: no cover - depends on the installed tabicl
            # A checkpoint carries its architecture config, which is splatted into
            # TabICL.__init__. An older tabicl rejects a key it does not know (observed:
            # "unexpected keyword argument 'zero_init'" against 2.1.1). That reads as a
            # bug in this bridge; it is a version mismatch, so say so rather than let the
            # bare TypeError propagate.
            raise RuntimeError(
                f"Failed to instantiate TabICL from the RamanICL checkpoint ({exc}). This "
                f"usually means the installed tabicl is older than the one the checkpoint "
                f"was trained with; RamanICL checkpoints need {TABICL_REQUIRED} (the "
                f"version this repo pins).",
            ) from exc
        return self

    def predict(self, X):
        x_np = X.values if hasattr(X, "values") else np.asarray(X)
        x_rs = _resample(np.asarray(x_np, dtype=np.float64), self.n_channels)
        return np.asarray(self.model_.predict(x_rs)).ravel()


class _RamanICLBridge(_GPURequiredBridge, SklearnAutoGluonBridge):
    # _GPURequiredBridge must precede SklearnAutoGluonBridge: without it AutoGluon's
    # resource manager allocates gpus=0 and the model measurably runs CPU-only,
    # regardless of the pod's GPU or the GPU-tier tag in cluster/gpu_models.json.
    _sklearn_cls = RamanICLModel
    ag_key = "RAMANICL"
    ag_name = "RamanICL"

    def _get_default_searchspace(self):
        # Deliberately empty. The pretrained checkpoint is the model; there is nothing
        # here worth tuning per task, and searching `n_channels` or `max_context` would
        # tune the harness rather than the method.
        return {}


class Prep_RAMANICL(_NoAugBase, _RamanICLBridge):  # noqa: N801
    pass
