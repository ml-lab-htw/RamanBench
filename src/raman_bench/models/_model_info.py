"""Per-model metadata bundle for the ``models/custom/<key>/info.py`` convention.

Mirrors upstream TabArena's own per-model ``info.py`` pattern
(``tabarena.models._model_info.ModelInfo``) at the scope RamanBench actually
needs. TabArena's version requires a ``MethodMetadata`` (suite/cache_root/
S3 bucket bookkeeping for their own hosted, versioned leaderboard artifacts)
that RamanBench has no use for -- results live under ``results/<run>/`` and
are aggregated by ``raman_bench_paper``, not cached to R2/S3 per dated suite.
This is the lightweight subset: identity + the HPO search-space generator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class ModelInfo:
    """Bundles a ``Prep_*`` model class with its HPO search space and identity.

    Attributes
    ----------
    model_cls
        The ``Prep_*`` AutoGluon model class (must have ``ag_key``/``ag_name``
        set -- ``ConfigGenerator`` asserts both).
    search_space
        A :class:`~tabarena.utils.config_utils.ConfigGenerator` (or
        ``CustomAGConfigGenerator``) for this model's HPO config pool.
    display_name
        Human-readable name shown in leaderboards/plots.
    compute
        ``"cpu"`` or ``"gpu"`` -- consumed by ``cluster/submit_job.py`` and
        ``cluster/gpu_models.json`` for resource-tier resolution.
    reference_url
        Optional paper/docs URL for the underlying architecture.
    pip_extra
        Optional extra pip requirement strings beyond the base
        ``raman-bench[autogluon]`` install (e.g. a foundation model needing
        its own package).
    """

    model_cls: type
    search_space: object
    display_name: str
    compute: Literal["cpu", "gpu"] = "cpu"
    reference_url: str | None = None
    pip_extra: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ag_key(self) -> str:
        return self.model_cls.ag_key

    @property
    def ag_name(self) -> str:
        return self.model_cls.ag_name
