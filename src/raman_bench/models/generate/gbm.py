"""Representative example: reusing a TabArena built-in model's own search
space, pointed at RamanBench's preprocessing-aware ``Prep_GBM`` instead of
TabArena's plain ``LGBModel``.

The same pattern (import TabArena's ``generate_configs_*`` function, wrap it
in ``CustomAGConfigGenerator`` with RamanBench's ``Prep_*`` class) extends to
every other AutoGluon-built-in model in
:data:`~raman_bench.preprocessing.wrapped_models.PREPROCESSED_MODELS`
(XGB, CAT, RF, XT, KNN, LR, NN_TORCH, FASTAI, REALMLP, MITRA, TABM, TABDPT,
TABFM, TABICL, the REALTABPFN/TABPFN-V3 variants, ...) by importing the
matching ``tabarena.models.{key}.hpo`` module.
"""

from tabarena.models.lightgbm.hpo import generate_configs_lightgbm
from tabarena.utils.config_utils import CustomAGConfigGenerator

from raman_bench.preprocessing.wrapped_models import Prep_GBM

gen_gbm = CustomAGConfigGenerator(
    model_cls=Prep_GBM,
    search_space_func=generate_configs_lightgbm,
    manual_configs=[{}],
)
