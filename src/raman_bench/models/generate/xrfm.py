"""XRFM: rebind TabArena's own (ConfigSpace-sampled) search space onto ``Prep_XRFM``.

``XRFMModel`` lives only in TabArena's own package (``tabarena.models.xrfm.model``)
-- there is no AutoGluon-core counterpart. See ``wrapped_models.py``'s
``_OPTIONAL_TABARENA_MODEL_IMPORTS`` block for that import; ``XRFMModel.ag_key``
is already the short form (``"XRFM"``), so no override is needed there (unlike
PERPETUAL_BOOSTER/CHIMERABOOST).

XRFM auto-detects a GPU at fit time and falls back to CPU when none is available
(``_get_default_resources`` returns ``num_gpus=0`` off a CUDA-less node, and
``_fit`` reads that back into ``device = "cpu" if num_gpus == 0 else "cuda"``) --
same "can use a GPU when available, degrades to CPU" shape as RealMLP/NN_TORCH/
FASTAI/ModernNCA, so it's listed in ``cluster/gpu_models.json`` alongside them.

Upstream's generator (``tabarena.models.xrfm.hpo.gen_xrfm``) is a
``CustomAGConfigGenerator`` (a ``ConfigSpace``-sampled ``search_space_func``, not
an ``autogluon.common.space`` dict) -- confirmed to support at least 200 unique
random configs without exhaustion (its continuous ``Float`` dimensions make
duplicate draws practically impossible), unlike PERPETUAL_BOOSTER's small
all-categorical space.

Memory-estimate caveat (found via real local testing, not a wiring bug):
``XRFMModel._estimate_memory_usage_static`` (see ``tabarena/models/xrfm/model.py``)
scales its estimate with ``2e6 * num_features`` -- fine for typical tabular data
(tens to hundreds of columns) but adds up fast for Raman spectra. Confirmed
against a real 11084-feature dataset (``microgel_synthesis``) on a 19 GB-available
laptop: the estimate came out to ~20.8 GB, tripping AutoGluon's own
``_validate_fit_memory_usage`` safety check (``NotEnoughMemoryError``) before fit
even starts. A narrower 2803-feature dataset (``diabetes_skin_ear_lobe``) fit fine
in ~0.4s. This is XRFM's own (not artificially wrong) memory accounting being
conservative for very wide feature spaces -- unlike the ``max_features`` cap
Mitra/TabDPT/TabICL/RealTabPFN carry (a hard reject with no override), XRFM has
no such cap (confirmed: ``max_rows``/``max_features``/``max_classes`` are all
``None`` in its ``_get_default_auxiliary_params()``) and will fit wide datasets
just fine given enough RAM -- cluster nodes with more memory than a laptop should
clear this without needing AutoGluon's own escape hatch
(``ag_args_fit={"ag.max_memory_usage_ratio": ...}``, not set here). Worth
watching during cluster runs on the widest Raman datasets.
"""

from __future__ import annotations

from tabarena.models.xrfm.hpo import gen_xrfm as _upstream

from raman_bench.models.generate._tabarena_adapter import (
    rebind_tabarena_generator,
    require_available,
)
from raman_bench.preprocessing.wrapped_models import Prep_XRFM

gen_xrfm = rebind_tabarena_generator(_upstream, require_available(Prep_XRFM, "XRFM"))
