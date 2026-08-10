"""XT (ExtraTrees): rebind TabArena's own search space onto ``Prep_XT``.

Not one of the 15 models named in the original wiring-gap report, but affected by
the exact same gap (``XT`` is listed in ``configs/models/all.json`` and its
``Prep_XT`` class lives in ``preprocessing/wrapped_models.py`` alongside the other
14) -- fixed alongside them rather than left half-done.

See ``_tabarena_adapter.rebind_tabarena_generator`` for why this rebind is needed
instead of using ``tabarena.models.extra_trees.hpo.gen_extratrees`` directly.
"""

from __future__ import annotations

from tabarena.models.extra_trees.hpo import gen_extratrees as _upstream

from raman_bench.models.generate._tabarena_adapter import rebind_tabarena_generator
from raman_bench.preprocessing.wrapped_models import Prep_XT

gen_xt = rebind_tabarena_generator(_upstream, Prep_XT)
