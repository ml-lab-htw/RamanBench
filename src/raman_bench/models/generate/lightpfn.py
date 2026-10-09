"""LIGHTPFN: rebind TabArena's own generator (one default config, ``can_hpo=False``) onto
``Prep_LIGHTPFN``.

``LightPFNModel`` (https://github.com/GioOtto/LightPFN, Apache-2.0) lives only in
TabArena's package (``tabarena.models.lightpfn.model``, autogluon/tabarena#655, on our pin
as a cherry-pick -- see ``requirements-tabarena-git.txt``). See ``wrapped_models.py`` for
the import, its ``ag_key`` override (``"TA-LIGHTPFN"`` -> ``"LIGHTPFN"``) and why it is
classification-only and capped at 10 classes.

The upstream default config keeps encoded contexts only in the refit model
(``cache_context=False`` for the bagged children, ``True`` on refit). GPU by default, so
listed in ``cluster/gpu_models.json``.
"""

from __future__ import annotations

from tabarena.models.lightpfn.hpo import gen_lightpfn as _upstream

from raman_bench.models.generate._tabarena_adapter import (
    rebind_tabarena_generator,
    require_available,
)
from raman_bench.preprocessing.wrapped_models import Prep_LIGHTPFN

gen_lightpfn = rebind_tabarena_generator(_upstream, require_available(Prep_LIGHTPFN, "LIGHTPFN"))
