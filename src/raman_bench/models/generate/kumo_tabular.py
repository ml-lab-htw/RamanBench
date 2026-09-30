"""KUMO-TABULAR: rebind TabArena's own (empty -- ``can_hpo=False``) search space
onto ``Prep_KUMO_TABULAR`` (the large checkpoint).

``KumoTabularModel`` (NVIDIA, https://huggingface.co/blog/nvidia/kumo-tabular)
lives only in TabArena's own package (``tabarena.models.kumo_tabular.model``) --
there is no AutoGluon-core counterpart, same staging situation as every other
tabular foundation model in ``wrapped_models.py``'s
``_OPTIONAL_TABARENA_MODEL_IMPORTS`` block (see that module for the import and
its ``ag_key`` override, ``"TA-KUMO-TABULAR"`` -> ``"KUMO-TABULAR"``).

PROVISIONAL: sourced from autogluon/tabarena PR #625 (branch ``kumo-tabular``),
still open/unmerged as of this onboarding -- see
``requirements-tabarena-git.txt``'s pin comment.

GPU-only (``default_num_gpus = 1``, ``minimum_num_gpus = 1``), so listed in
``cluster/gpu_models.json``.
"""

from __future__ import annotations

from tabarena.models.kumo_tabular.hpo import gen_kumo_tabular as _upstream

from raman_bench.models.generate._tabarena_adapter import (
    rebind_tabarena_generator,
    require_available,
)
from raman_bench.preprocessing.wrapped_models import Prep_KUMO_TABULAR

gen_kumo_tabular = rebind_tabarena_generator(_upstream, require_available(Prep_KUMO_TABULAR, "KUMO-TABULAR"))
