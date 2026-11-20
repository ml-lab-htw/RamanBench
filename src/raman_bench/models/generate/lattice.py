"""LATTICE: rebind TabArena's own generator (one default config) onto ``Prep_LATTICE``.

``LatticeModel`` lives in TabArena's package (``tabarena.models.lattice.model``, upstream
autogluon/tabarena main). ag_key ``"TA-LATTICE"`` -> ``"LATTICE"`` (see ``wrapped_models.py``).
"""

from __future__ import annotations

from tabarena.models.lattice.hpo import gen_lattice as _upstream

from raman_bench.models.generate._tabarena_adapter import (
    rebind_tabarena_generator,
    require_available,
)
from raman_bench.preprocessing.wrapped_models import Prep_LATTICE

gen_lattice = rebind_tabarena_generator(_upstream, require_available(Prep_LATTICE, "LATTICE"))
