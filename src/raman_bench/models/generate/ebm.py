"""EBM (Explainable Boosting Machine): rebind TabArena's own search space onto
``Prep_EBM``.

Unlike TABFM/TABPFN-V3/TABSWIFT/MODERNNCA (batch 1) or PERPETUAL_BOOSTER/XRFM/
CHIMERABOOST (this batch), ``EBMModel`` is a graduated AutoGluon-core class
(``autogluon.tabular.models.EBMModel``), not a TabArena-only one -- so this is a
plain rebind exactly like ``cat.py``/``xt.py``, not one that needs
``require_available`` (see ``wrapped_models.py``'s comment on why ``EBMModel`` is
imported unconditionally alongside CatBoostModel/XGBoostModel).

See ``_tabarena_adapter.rebind_tabarena_generator`` for why this rebind is needed
instead of using ``tabarena.models.ebm.hpo.gen_ebm`` directly.
"""

from __future__ import annotations

from tabarena.models.ebm.hpo import gen_ebm as _upstream

from raman_bench.models.generate._tabarena_adapter import rebind_tabarena_generator
from raman_bench.preprocessing.wrapped_models import Prep_EBM

gen_ebm = rebind_tabarena_generator(_upstream, Prep_EBM)
