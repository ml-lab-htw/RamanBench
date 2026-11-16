"""Opt-in preprocessing search for v1 HPO sweeps (``--prep-search-config``).

The v1 config generators (``gen_<key>``) sample model hyperparameters only. Preprocessing is
fixed per model class, or pinned for a whole job with ``--recipe-config`` (``optimize=False``).
A *prep-search* config instead makes preprocessing part of the random search:

* every random config ``r1..rN`` gets an independently sampled preprocessing recipe over the
  listed steps (their ``search_params`` in ``_PREP_STEP_DEFINITIONS``), merged into the model's
  own sampled hyperparameters, so preprocessing and model are tuned jointly by the same search;
* every *other* per-step ``prep_*_enabled`` flag is pinned off in every config, including the
  manual default ``c1``, so the only preprocessing a model can receive comes from the listed
  steps (``Prep_PLS``/``Prep_LR``'s class-default baseline correction does not leak in);
* ``c1`` otherwise stays the model's default config.

Sampled and pinned flags reach the model as user hyperparameters (``_user_params``), which is
what lets them survive the fit-time gate in ``RamanPreprocessingMixin._fit`` for models that do
not set ``_optimize_preprocessing`` (the tree ensembles, among others).

The prep samples are drawn by a seeded ``LocalRandomSearcher`` (seed derived from the model's
``ag_key``), so config ``rK`` is identical in every job of a sweep -- the same requirement that
already holds for ``--num-random-configs``.

Config file (JSON), either form::

    {"preprocessing": {"cosmic_ray_removal": true, "denoising": true, "snv": true}}
    {"cosmic_ray_removal": true, "denoising": true, "snv": true}

Keys are ``_PREP_STEP_DEFINITIONS`` step keys; steps set to ``false`` or left out are pinned off.
``augmentation`` is never pinned (the mixin gates it per model class anyway).
"""

from __future__ import annotations

import hashlib
import json
import zlib
from pathlib import Path

from raman_bench.preprocessing.mixin import (
    _PREP_STEP_DEFINITIONS,
    STEP_ENABLED_PARAMS,
    build_restricted_searchspace,
)

# Steps a prep-search config never pins off: augmentation is gated per model class by
# ``_supports_augmentation`` and has its own per-model knobs in the DL search spaces.
_NEVER_PINNED = frozenset({"augmentation"})


def load_prep_search_config(path: str | Path) -> dict[str, bool]:
    """Read a prep-search config and return ``{step_key: True}`` for the searched steps."""
    raw = json.loads(Path(path).read_text())
    steps = raw.get("preprocessing", raw) if isinstance(raw, dict) else None
    if not isinstance(steps, dict):
        raise ValueError(f"{path}: expected a JSON object of step_key -> bool")
    unknown = sorted(set(steps) - set(_PREP_STEP_DEFINITIONS))
    if unknown:
        raise ValueError(
            f"{path}: unknown preprocessing step(s) {unknown}; "
            f"known: {sorted(_PREP_STEP_DEFINITIONS)}"
        )
    searched = {k: True for k, v in steps.items() if v}
    if not searched:
        raise ValueError(f"{path}: no preprocessing step enabled for the search")
    return searched


def prep_search_slug(path: str | Path) -> str:
    """Directory-safe name for a prep-search config: file stem plus a digest of its content,
    so editing the file under the same name can never reuse results computed with the old one."""
    steps = load_prep_search_config(path)
    digest = hashlib.blake2b(json.dumps(sorted(steps)).encode(), digest_size=4).hexdigest()
    return f"{Path(path).stem}-{digest}"


def pinned_off_params(steps: dict[str, bool]) -> dict[str, bool]:
    """``prep_*_enabled = False`` for every step that is neither searched nor never-pinned."""
    return {
        flag: False
        for step, flag in STEP_ENABLED_PARAMS.items()
        if step not in steps and step not in _NEVER_PINNED
    }


def prep_search_seed(model_cls) -> int:
    """Stable per-model seed for the prep samples (``zlib.crc32``, not ``hash()``, which is
    salted per process)."""
    return zlib.crc32(f"prep-search:{model_cls.ag_key}".encode())


def sample_prep_configs(steps: dict[str, bool], num_configs: int, seed: int) -> list[dict]:
    """``num_configs`` random preprocessing configs over the searched steps' search space."""
    from autogluon.core.searcher.local_random_searcher import LocalRandomSearcher

    if num_configs <= 0:
        return []
    searcher = LocalRandomSearcher(
        search_space=build_restricted_searchspace(steps),
        first_is_default=False,
        random_seed=seed,
    )
    return [searcher.get_config() for _ in range(num_configs)]


def with_prep_search(gen, steps: dict[str, bool]):
    """Return a copy of config generator ``gen`` whose configs also search ``steps``.

    Works for every TabArena generator flavour: it only relies on ``gen.manual_configs`` and
    ``gen.get_searcher_configs(n)``, and returns a ``CustomAGConfigGenerator`` with the same
    model class and name, so experiment names (``{ag_name}_c1`` / ``_rK``) are unchanged.
    """
    from tabarena.utils.config_utils import CustomAGConfigGenerator

    pins = pinned_off_params(steps)
    seed = prep_search_seed(gen.model_cls)

    def search_space_func(num_configs: int) -> list[dict]:
        model_configs = gen.get_searcher_configs(num_configs)
        prep_configs = sample_prep_configs(steps, len(model_configs), seed)
        return [{**m, **pins, **p} for m, p in zip(model_configs, prep_configs)]

    return CustomAGConfigGenerator(
        model_cls=gen.model_cls,
        name=gen.name,
        search_space_func=search_space_func,
        manual_configs=[{**c, **pins} for c in gen.manual_configs],
    )
