"""TABFM: rebind TabArena's own (empty -- TabFM has no tunable HPO surface, see
upstream's ``can_hpo=False``) search space onto ``Prep_TABFM``.

``TabFMModel`` lives only in TabArena's own package (``tabarena.models.tabfm.model``)
-- unlike Mitra/TabDPT/TabICL/RealMLP, it has not (yet) graduated into AutoGluon
core (confirmed against a real installed build: no ``tabfm`` submodule anywhere
under ``autogluon.tabular.models``), so, unlike ``cat.py``/``mitra.py``,
``Prep_TABFM``'s base class is imported directly from
``tabarena.models.tabfm.model`` rather than via AutoGluon's own namespace -- see
``wrapped_models.py``'s ``_OPTIONAL_TABARENA_MODEL_IMPORTS`` block for that import
and its ``ag_key`` override (``"TA-TABFM"`` -> ``"TABFM"``).
"""

from __future__ import annotations

from tabarena.models.tabfm.hpo import gen_tabfm as _upstream

from raman_bench.models.generate._tabarena_adapter import (
    rebind_tabarena_generator,
    require_available,
)
from raman_bench.preprocessing.wrapped_models import Prep_TABFM

gen_tabfm = rebind_tabarena_generator(_upstream, require_available(Prep_TABFM, "TABFM"))
