"""SAP_RPT_OSS: rebind TabArena's own (empty -- see upstream's ``can_hpo=False``)
search space onto ``Prep_SAP_RPT_OSS``.

``SAPRPTOSSModel`` (ConTextTab, https://github.com/SAP-samples/sap-rpt-1-oss)
lives only in TabArena's own package (``tabarena.models.sap_rpt_oss.model``) --
there is no AutoGluon-core counterpart. See ``wrapped_models.py``'s
``_OPTIONAL_TABARENA_MODEL_IMPORTS`` block for that import and its ``ag_key``
override (``"SAP-RPT-OSS"`` -> ``"SAP_RPT_OSS"``, purely to match this registry's
underscore convention for multi-word keys).

Supports binary/multiclass/regression -- no problem-type restriction, unlike
NORI/ORIONMSP in this same batch.

GPU-tier the same "auto-detects a GPU at fit time, falls back to CPU when none
is available" way as XRFM/ModernNCA/RealMLP -- ``SAPRPTOSSModel._fit`` only
raises if more GPUs are *requested* than are actually available, not merely
because none are present, so it's listed in ``cluster/gpu_models.json``.

The default checkpoint (``prefetch_weights`` / ``SAPRPTOSSModel._fit``,
hardcoded to ``"2025-11-04_sap-rpt-one-oss.pt"`` from the ``SAP/sap-rpt-1-oss``
HF repo) was originally gated (``huggingface_hub.errors.GatedRepoError: 403``)
for the RamanBench HF service account. Access was granted via the repo's
self-serve "Agree and access repository" click and confirmed end to end on
2026-08-10: real weight download, fit, save, and predict all succeed, both
locally and on the HTW cluster (``results/v1/smoke_batch3/data/SAP-RPT-OSS_c1_BAG_L1/``).
"""

from __future__ import annotations

from tabarena.models.sap_rpt_oss.hpo import gen_sap_rpt_oss as _upstream

from raman_bench.models.generate._tabarena_adapter import (
    rebind_tabarena_generator,
    require_available,
)
from raman_bench.preprocessing.wrapped_models import Prep_SAP_RPT_OSS

gen_sap_rpt_oss = rebind_tabarena_generator(
    _upstream, require_available(Prep_SAP_RPT_OSS, "SAP_RPT_OSS")
)
