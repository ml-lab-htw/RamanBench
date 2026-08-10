"""TABSTAR: rebind TabArena's own (``autogluon.common.space``) search space onto
``Prep_TABSTAR``.

``TabSTARModel`` (https://arxiv.org/abs/2505.18125) lives only in TabArena's own
package (``tabarena.models.tabstar.model``) -- there is no AutoGluon-core
counterpart. Unlike every other model wired in this batch (and every
``AbstractTorchModel``-based one in batches 1-2), it subclasses plain AutoGluon
``AbstractModel`` directly -- it's a language-adjacent tabular foundation model
(LoRA-fine-tunes a pretrained language-model backbone over serialized rows/
column names, not a from-scratch tabular architecture), which is also why its
search space below is LoRA hyperparameters, not tree/MLP knobs.

``TabSTARModel.ag_key`` is already the short, unprefixed ``"TABSTAR"`` -- no
override needed (see ``wrapped_models.py``'s batch-3 comment block; the other
five models in this batch all needed one).

Supports binary/multiclass/regression -- no problem-type restriction. Originally
onboarded with no ``max_features``/``max_rows``/``max_classes`` cap (confirmed
via ``_get_default_auxiliary_params()`` on the installed build at the time), but
real batch-3 verification later found ``TabSTARModel`` genuinely cannot handle
RamanBench's widest spectra: it builds a per-column LM text embedding (memory
scales with feature count, not row count, matching upstream's own documented
warning about >200-column datasets), and OOM-killed on an 11,084-feature
dataset even with its own internal batch-size-1 OOM fallback exhausted. Now
capped at ``max_features=4000`` via ``_default_auxiliary_params_extra`` --
see ``wrapped_models.py``'s ``_TABSTAR_MAX_FEATURES``/``MAX_FEATURES_MODELS``
comment block for the full memory characterization and cap justification.

GPU-tier the same "auto-detects a GPU at fit time, falls back to CPU when none
is available" way as XRFM/ModernNCA/RealMLP -- ``TabSTARModel._fit`` only raises
if more GPUs are *requested* than are actually available, not merely because
none are present, so it's listed in ``cluster/gpu_models.json``.
"""

from __future__ import annotations

from tabarena.models.tabstar.hpo import gen_tabstar as _upstream

from raman_bench.models.generate._tabarena_adapter import (
    rebind_tabarena_generator,
    require_available,
)
from raman_bench.preprocessing.wrapped_models import Prep_TABSTAR

gen_tabstar = rebind_tabarena_generator(_upstream, require_available(Prep_TABSTAR, "TABSTAR"))
