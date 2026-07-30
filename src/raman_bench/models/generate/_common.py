"""Shared helper for building per-model :class:`~tabarena.utils.config_utils.ConfigGenerator`\\ s.

Mirrors TabArena's own ``models/{key}/generate.py`` convention (see e.g.
``tabarena.models.nn_torch.generate``): a module-level ``search_space`` dict of
``autogluon.common.space`` objects plus ``manual_configs=[{}]`` (the single
default config, tagged ``_c1`` by :class:`~tabarena.utils.config_utils.ConfigGenerator`).
"""

from __future__ import annotations

from typing import Type


def bridge_search_space(bridge_cls: Type) -> dict:
    """Extract a Prep_* bridge class's existing ``_get_default_searchspace()``.

    RamanBench's ``_XxxBridge`` classes (``preprocessing/wrapped_models.py``)
    already define Raman-tuned hyperparameter ranges as an override of this
    AutoGluon method for the (retiring) in-fit HPO scheduler. None of these
    overrides reference ``self``, so calling the unbound method is safe and
    lets the new TabArena-style config pool reuse the exact same ranges
    without duplicating them.
    """
    return bridge_cls._get_default_searchspace(bridge_cls)
