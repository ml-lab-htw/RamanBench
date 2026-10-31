"""FTFM: rebind TabArena's own generator (one default config, ``can_hpo=False``) onto
``Prep_FTFM``.

``FTFMModel`` lives only in TabArena's package (``tabarena.models.ftfm.model``,
autogluon/tabarena#679, on our pin as a cherry-pick -- see ``requirements-tabarena-git.txt``).
See ``wrapped_models.py`` for the import, its ``ag_key`` override (``"TA-FTFM"`` -> ``"FTFM"``)
and why it is classification-only.
"""

from __future__ import annotations

from tabarena.models.ftfm.hpo import gen_ftfm as _upstream

from raman_bench.models.generate._tabarena_adapter import (
    rebind_tabarena_generator,
    require_available,
)
from raman_bench.preprocessing.wrapped_models import Prep_FTFM

gen_ftfm = rebind_tabarena_generator(_upstream, require_available(Prep_FTFM, "FTFM"))
