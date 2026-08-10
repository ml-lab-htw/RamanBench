"""TABSWIFT: rebind TabArena's own (empty -- TabSwift has no tunable HPO surface)
search space onto ``Prep_TABSWIFT``.

``TabSwiftModel`` lives only in TabArena's own package
(``tabarena.models.tabswift.model``) -- there is no AutoGluon-core counterpart at
all (confirmed against a real installed build: no ``tabswift`` submodule anywhere
under ``autogluon.tabular.models``). See ``wrapped_models.py``'s
``_OPTIONAL_TABARENA_MODEL_IMPORTS`` block for that import and its ``ag_key``
override (``"TA-TABSWIFT"`` -> ``"TABSWIFT"``).
"""

from __future__ import annotations

from tabarena.models.tabswift.hpo import gen_tabswift as _upstream

from raman_bench.models.generate._tabarena_adapter import (
    rebind_tabarena_generator,
    require_available,
)
from raman_bench.preprocessing.wrapped_models import Prep_TABSWIFT

gen_tabswift = rebind_tabarena_generator(_upstream, require_available(Prep_TABSWIFT, "TABSWIFT"))
