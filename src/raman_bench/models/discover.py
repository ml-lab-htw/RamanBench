"""Auto-discovery for the ``models/custom/<key>/info.py`` convention.

Mirrors ``tabarena.models._registry.discover_models()``: walks
``raman_bench.models.custom.<key>`` *packages* (plain modules -- the
not-yet-migrated ``models/custom/<name>.py`` files -- are skipped, so
migrating models one at a time is safe), imports each one's ``info``
submodule, and collects the :class:`~raman_bench.models._model_info.ModelInfo`
instances declared there, keyed by ``ag_key``.

A package whose ``info.py`` fails to import is skipped with a warning rather
than failing the whole scan, matching the existing defensive-import posture
for optional foundation-model dependencies elsewhere in this codebase (see
``preprocessing/wrapped_models.py``'s ``_make_optional_prep_class``).
"""

from __future__ import annotations

import importlib
import logging
import pkgutil

from raman_bench.models._model_info import ModelInfo

logger = logging.getLogger(__name__)

_REGISTRY: dict[str, ModelInfo] | None = None


def discover_custom_models() -> dict[str, ModelInfo]:
    """Return ``{ag_key: ModelInfo}`` for every migrated custom model.

    Cached on first call; re-import this module to force a rescan (mainly
    useful in tests that add/remove a model package at runtime).
    """
    global _REGISTRY
    if _REGISTRY is not None:
        return _REGISTRY

    registry: dict[str, ModelInfo] = {}
    import raman_bench.models.custom as pkg

    for _finder, name, is_pkg in pkgutil.iter_modules(pkg.__path__):
        if not is_pkg or name.startswith("_"):
            continue
        try:
            info_module = importlib.import_module(f"raman_bench.models.custom.{name}.info")
        except ImportError as exc:
            logger.warning(
                "Skipping raman_bench.models.custom.%s in registry: failed to "
                "import its info module (%s: %s).",
                name,
                type(exc).__name__,
                exc,
            )
            continue
        for attr_name in dir(info_module):
            if attr_name.startswith("_"):
                continue
            obj = getattr(info_module, attr_name)
            if not isinstance(obj, ModelInfo):
                continue
            key = obj.ag_key
            if key in registry:
                raise RuntimeError(
                    f"Duplicate ModelInfo ag_key {key!r}: {registry[key]} vs {obj} "
                    f"(from raman_bench.models.custom.{name}.info::{attr_name})",
                )
            registry[key] = obj

    _REGISTRY = registry
    return registry
