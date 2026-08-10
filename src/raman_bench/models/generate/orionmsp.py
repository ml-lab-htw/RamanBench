"""ORIONMSP: rebind TabArena's own (empty -- see upstream's ``can_hpo=False``)
search space onto ``Prep_ORIONMSP``.

``OrionMSPModel`` (Orion-MSP v1.5, https://arxiv.org/abs/2511.02818) lives only
in TabArena's own package (``tabarena.models.orionmsp.model``) -- there is no
AutoGluon-core counterpart. See ``wrapped_models.py``'s
``_OPTIONAL_TABARENA_MODEL_IMPORTS`` block for that import and its ``ag_key``
override (``"TA-ORION-MSP"`` -> ``"ORIONMSP"``).

**Classification-only** (``OrionMSPModel.supported_problem_types() ==
["binary", "multiclass"]``; ``_fit`` raises ``AssertionError`` for regression)
-- ``ORIONMSP`` is listed in ``wrapped_models.CLASSIFICATION_ONLY_MODELS``
alongside ROCKET/ARSENAL/TABPFN-WIDE. A full-benchmark run for this model
should only target ``configs/datasets/classification_all.json``.

No ``max_features``/``max_rows``/``max_classes`` cap (confirmed via
``_get_default_auxiliary_params()`` on the installed build), but its own
``_fit`` docstring/comment flags a real performance caveat for wide feature
spaces: "Needs up to 400GB VRAM for datasets with 1k features" -- for any
``X.shape[1] > 500`` (every Raman spectrum in this benchmark; wavenumber axes
run 500-4000+ points) it force-clamps ``batch_size=1`` as an OOM fallback,
which trades memory for a much slower fit. Not a wiring bug and not something
this Prep_* override should paper over -- upstream's own documented tradeoff,
worth remembering when picking a cluster time-limit for this model.
"""

from __future__ import annotations

from tabarena.models.orionmsp.hpo import gen_orionmsp as _upstream

from raman_bench.models.generate._tabarena_adapter import (
    rebind_tabarena_generator,
    require_available,
)
from raman_bench.preprocessing.wrapped_models import Prep_ORIONMSP

gen_orionmsp = rebind_tabarena_generator(_upstream, require_available(Prep_ORIONMSP, "ORIONMSP"))
