"""Fixed hyperparameter overrides for v1 sweeps (``--hyperparameter-overrides``).

A JSON object of hyperparameters merged into *every* config of a model (the default ``c1`` and
all random ``rK``) through TabArena's ``extra_model_hyperparameters``. It is meant for settings
a sweep holds fixed that are not part of the model's search space -- e.g. switching the Raman DL
models' per-epoch augmentation on, so the ``aug_noise_sigma``/``aug_mixup_alpha`` their search
spaces sample actually take effect, and the fixed pre-fit noise copies off::

    {"per_epoch_augmentation": true, "prep_aug_enabled": false}

TabArena rejects an override key that a model config already sets, so an override can add
settings but never silently replace a sampled hyperparameter. The overrides are part of the
cache directory name (``__hp_<stem>-<digest>``), so runs with and without them -- or with two
different override files -- never share a cache path.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

_RESERVED_KEYS = frozenset({"ag_args", "ag_args_ensemble", "ag_args_fit"})


def load_hyperparameter_overrides(path: str | Path) -> dict:
    """Read and validate an overrides file: a non-empty JSON object of hyperparameter -> value.

    Keys starting with ``_`` (e.g. ``"_comment"``) are documentation and are dropped.
    """
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a JSON object of hyperparameter -> value")
    overrides = {k: v for k, v in raw.items() if not str(k).startswith("_")}
    if not overrides:
        raise ValueError(f"{path}: no hyperparameter overrides given")
    reserved = sorted(_RESERVED_KEYS.intersection(overrides))
    if reserved:
        raise ValueError(f"{path}: {reserved} cannot be overridden here (AutoGluon/TabArena control keys)")
    return overrides


def hyperparameter_overrides_slug(path: str | Path) -> str:
    """Directory-safe name: file stem plus a digest of the overrides themselves, so editing the
    file under the same name can never reuse results computed with the old values."""
    overrides = load_hyperparameter_overrides(path)
    payload = json.dumps(overrides, sort_keys=True, default=str).encode()
    return f"{Path(path).stem}-{hashlib.blake2b(payload, digest_size=4).hexdigest()}"
