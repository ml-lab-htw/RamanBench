"""Rebind a TabArena-native config generator onto a RamanBench ``Prep_*`` model class.

TabArena's own package (``tabarena.models.<key>.hpo``, installed as a dependency --
see ``pyproject.toml``'s ``tabarena[tabicl,ebm,search_spaces,realmlp,tabdpt,tabm]``
extras, in particular the ``search_spaces`` extra) already ships a maintained
``ConfigGenerator``/``CustomAGConfigGenerator`` (``gen_<key>``) for every "built-in"
AutoGluon model RamanBench wraps with Raman preprocessing in
``preprocessing/wrapped_models.py`` (XGB, CAT, RF, XT, LR, NN_TORCH, FASTAI,
REALMLP, MITRA, TABM, TABDPT, TABICL, ...). Those upstream generators are bound to
the *plain* AutoGluon model class (e.g. ``CatBoostModel``), not RamanBench's
``Prep_CAT`` -- so they can't be used as-is: every job resolves its model key via
``raman_bench.models.registry.infer_model_cls`` to the ``Prep_*`` class (the one
with tunable Raman preprocessing hyperparameters), and
``scripts/run_experiment.py::run_one`` asserts the generated experiment's
``model_cls`` matches that resolved class exactly.

Hand-copying 15+ near-identical hyperparameter search spaces into this repo would
both duplicate TabArena's own (actively maintained) tuning knowledge and drift out
of sync with it. Instead, this module rebuilds an equivalent generator that keeps
the exact same manual configs / search space, just retargeted at the ``Prep_*``
class -- see ``raman_bench/models/generate/cat.py`` etc. for the (mostly one-line)
call sites this is used from.

Not every one of these 15 models can go through this adapter unmodified -- see
``raman_bench/models/generate/knn.py`` for a model whose upstream generator turned
out to be bound to a different, incompatible TabArena-only model subclass, and
``dummy.py`` / ``realtabpfn_v2.py`` for models with no upstream TabArena generator
at all. Those get bespoke handling instead; this adapter is only for the (majority)
case where the upstream generator's ``model_cls`` and RamanBench's ``Prep_*`` class
share the exact same hyperparameter surface (RamanBench's is a direct subclass of
the same plain AutoGluon model TabArena's generator targets).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tabarena.utils.config_utils import AGConfigGenerator, ConfigGenerator, CustomAGConfigGenerator

if TYPE_CHECKING:
    from autogluon.core.models import AbstractModel


def require_available(model_cls: type[AbstractModel] | None, key: str) -> type[AbstractModel]:
    """Guard for the optional foundation-model ``Prep_*`` classes.

    ``preprocessing/wrapped_models.py`` sets e.g. ``Prep_REALMLP = None`` (rather
    than raising) when this AutoGluon build is missing the underlying model class
    (see its ``_OPTIONAL_AG_MODEL_NAMES`` handling) -- expected, since different
    AutoGluon releases/prereleases carry different bleeding-edge model classes.
    In the normal run path (``run_experiment.py::run_one``) this is never actually
    reached with ``None``: ``infer_model_cls`` already raises first if the key
    isn't registered. This just turns what would otherwise be a confusing
    ``AttributeError: 'NoneType' object has no attribute 'ag_name'`` (from deep
    inside ``tabarena``'s own ``AGConfigGenerator.__init__``) into a clear one, for
    any other caller that imports this module directly.
    """
    if model_cls is None:
        raise ImportError(
            f"{key} is unavailable in this AutoGluon build (its underlying model "
            "class is missing -- see raman_bench.preprocessing.wrapped_models's "
            "defensive _OPTIONAL_AG_MODEL_NAMES import)."
        )
    return model_cls


def rebind_tabarena_generator(
    upstream: AGConfigGenerator, model_cls: type[AbstractModel]
) -> AGConfigGenerator:
    """Return a copy of ``upstream`` (a TabArena ``gen_<key>``) bound to ``model_cls``.

    ``upstream`` is TabArena's own generator for the plain AutoGluon model (its
    ``model_cls`` is e.g. ``CatBoostModel``); the returned generator has the exact
    same ``manual_configs`` and search space, just retargeted at ``model_cls`` (a
    RamanBench ``Prep_*`` subclass of that same AutoGluon model). Every config that
    TabArena would sample for the plain model is produced unchanged -- only which
    class ultimately gets fit differs.

    Handles both of TabArena's generator flavours (see
    ``tabarena.utils.config_utils``): :class:`ConfigGenerator` (a static
    ``autogluon.common.space`` dict) and :class:`CustomAGConfigGenerator` (a
    ``search_space_func(num_configs)`` callable -- used by models whose search
    space needs ``ConfigSpace`` sampling, e.g. CatBoost/XGBoost/RandomForest/
    ExtraTrees).
    """
    if isinstance(upstream, CustomAGConfigGenerator):
        return CustomAGConfigGenerator(
            model_cls=model_cls,
            search_space_func=upstream.search_space_func,
            manual_configs=upstream.manual_configs,
        )
    if isinstance(upstream, ConfigGenerator):
        return ConfigGenerator(
            model_cls=model_cls,
            search_space=upstream.search_space,
            manual_configs=upstream.manual_configs,
        )
    raise TypeError(
        f"Don't know how to rebind a {type(upstream).__name__} (expected "
        "ConfigGenerator or CustomAGConfigGenerator)."
    )
