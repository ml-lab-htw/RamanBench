"""KNN: bespoke handling -- TabArena's own generator is NOT a safe rebind target.

Unlike the other 14 models here, ``tabarena.models.knn.hpo.gen_knn`` is bound to
TabArena's own enhanced ``KNNNewModel`` (``ag_key="TA-KNN"``), a subclass of
AutoGluon's built-in ``KNNModel`` that adds ``scaler``/``cat_threshold``
preprocessing knobs (see ``tabarena/models/knn/model.py`` and its
``KNNPreprocessor``). RamanBench's ``Prep_KNN`` wraps plain AutoGluon ``KNNModel``
(matching AutoGluon's own ``"KNN"`` ag_key, which is what's registered in
``configs/models/all.json``) -- it has no ``KNNPreprocessor`` and its plain
``_fit`` would forward ``scaler``/``cat_threshold`` straight into sklearn's
``KNeighborsClassifier(**params)``, which doesn't accept them, crashing every fit.
``rebind_tabarena_generator`` (see ``_tabarena_adapter.py``) is therefore not used
here -- it would silently produce broken configs.

Instead this reuses the subset of TabArena's own search space that genuinely is
shared with plain ``KNNModel`` (``n_neighbors``, ``weights``, ``p`` -- identical
``Categorical(...)`` choices, copied verbatim from
``tabarena.models.knn.hpo.search_space``) and drops ``scaler``/``cat_threshold``.

Follow-up (not done here): if RamanBench wants TabArena's fuller KNN tuning
surface, ``Prep_KNN`` would need to be rebuilt on top of ``KNNNewModel`` instead
of plain ``KNNModel`` -- a larger change than this wiring fix, since ``Prep_KNN``
already carries its own ``_fit`` override (small-dataset ``n_neighbors`` clamping)
written against plain ``KNNModel``.
"""

from __future__ import annotations

from autogluon.common.space import Categorical
from tabarena.utils.config_utils import ConfigGenerator

from raman_bench.preprocessing.wrapped_models import Prep_KNN

search_space = {
    "n_neighbors": Categorical(
        20, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 30, 40, 50, 100, 200, 300, 400, 500
    ),
    "weights": Categorical("distance", "uniform"),
    "p": Categorical(2, 1, 1.5),
}

gen_knn = ConfigGenerator(
    model_cls=Prep_KNN,
    manual_configs=[{}],
    search_space=search_space,
)
