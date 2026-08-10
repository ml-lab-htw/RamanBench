"""MODERNNCA: rebind TabArena's own randomised search space (a
``CustomAGConfigGenerator``) onto ``Prep_MODERNNCA``.

``ModernNCAModel`` lives only in TabArena's own package
(``tabarena.models.modernnca.model``); it subclasses plain AutoGluon
``AbstractModel`` directly (not ``AbstractTorchModel``, unlike TabFM/TabPFN-3/
TabSwift above), but is still torch-backed internally and auto-detects a GPU at
fit time, falling back to CPU when none is available. TabArena registers it
twice upstream (a CPU-tier and a GPU-tier ``ModelInfo``, both sharing this exact
``model_cls`` and search space, differing only in declared ``compute``) --
RamanBench keeps a single ``Prep_MODERNNCA`` entry, listed in
``cluster/gpu_models.json`` for the same "can use a GPU when available, degrades
to CPU" reason RealMLP/NN_TORCH/FASTAI are also GPU-tier there.
"""

from __future__ import annotations

from tabarena.models.modernnca.hpo import gen_modernnca as _upstream

from raman_bench.models.generate._tabarena_adapter import (
    rebind_tabarena_generator,
    require_available,
)
from raman_bench.preprocessing.wrapped_models import Prep_MODERNNCA

gen_modernnca = rebind_tabarena_generator(_upstream, require_available(Prep_MODERNNCA, "MODERNNCA"))
