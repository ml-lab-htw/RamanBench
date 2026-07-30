"""RamanBench model registry — extends TabArena's registry with RamanBench's models.

Mirrors :mod:`tabarena.benchmark.models.model_registry`: start from TabArena's
full registry (every model TabArena knows about, including foundation models
RamanBench hasn't given Raman-specific preprocessing hooks) and layer
RamanBench's own :data:`~raman_bench.preprocessing.wrapped_models.PREPROCESSED_MODELS`
on top. Two things happen per entry:

- For model families TabArena already registers (GBM, XGB, CAT, RF, ..., the
  TabPFN/TabICL/TabDPT/... foundation models) RamanBench's ``Prep_*`` version
  *replaces* TabArena's, since it adds Raman preprocessing hyperparameters
  (baseline correction, SNV, augmentation, ...) that TabArena's plain wrapper
  doesn't have. This is an intentional override, not a collision to warn about.
- For RamanBench's own architectures with no TabArena equivalent (PLS, DeepCNN,
  RamanNet, SANet, RamanFormer, RamanTransformer, ReZeroNet, FCResNeXt,
  CoAtNet, ROCKET, Arsenal, TabPFN-Wide) the entry is added fresh.

Any TabArena-only model neither overridden nor replaced above (e.g. TabSTAR,
LimiX, OrionMSP) stays available in :data:`raman_bench_model_registry` as a
plain baseline, satisfying "use TabArena's models as much as possible."
"""

from __future__ import annotations

import copy

from autogluon.tabular.registry import ModelRegistry
from tabarena.benchmark.models.model_registry import tabarena_model_registry

from raman_bench.preprocessing.wrapped_models import PREPROCESSED_MODELS

raman_bench_model_registry: ModelRegistry = copy.deepcopy(tabarena_model_registry)

for _model_cls in PREPROCESSED_MODELS.values():
    _key = _model_cls.ag_key
    if _key in raman_bench_model_registry.keys:
        raman_bench_model_registry.remove(model_cls=raman_bench_model_registry.key_to_cls(key=_key))
    raman_bench_model_registry.add(_model_cls)


def infer_model_cls(model_cls, model_register: ModelRegistry | None = None):
    """Resolve a model key / display name / class name string to its class.

    Mirrors :func:`tabarena.benchmark.models.model_registry.infer_model_cls`,
    defaulting to :data:`raman_bench_model_registry` instead of TabArena's own.
    """
    if model_register is None:
        model_register = raman_bench_model_registry
    if isinstance(model_cls, str):
        if model_cls in model_register.key_to_cls_map():
            model_cls = model_register.key_to_cls(key=model_cls)
        elif model_cls in model_register.name_map().values():
            for real_model_cls in model_register.model_cls_list:
                if real_model_cls.ag_name == model_cls:
                    model_cls = real_model_cls
                    break
        elif model_cls in [str(c.__name__) for c in model_register.model_cls_list]:
            for real_model_cls in model_register.model_cls_list:
                if model_cls == str(real_model_cls.__name__):
                    model_cls = real_model_cls
                    break
        else:
            raise AssertionError(f"Unknown model_cls: {model_cls}")
    return model_cls
