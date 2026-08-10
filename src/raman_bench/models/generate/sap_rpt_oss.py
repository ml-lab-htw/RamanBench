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

Real end-to-end verification (``run_experiment.py --model SAP_RPT_OSS``) could
NOT be completed locally: the default checkpoint (``prefetch_weights`` /
``SAPRPTOSSModel._fit``, hardcoded to
``"2025-11-04_sap-rpt-one-oss.pt"`` from the ``SAP/sap-rpt-1-oss`` HF repo) is
gated (``huggingface_hub.errors.GatedRepoError: 403``) -- confirmed the
RamanBench HF service account (which already has ``canReadGatedRepos: True``
as a general fine-grained scope, and already has access to gated repos like
TabPFN's) has NOT been granted per-repo access to this specific one, on both
this machine and the HTW cluster login node (same account, same result: no
cached weights, same 403). Gating type is ``"auto"`` (self-serve, not manual
review) per ``HfApi().model_info("SAP/sap-rpt-1-oss").gated`` -- unblocking it
is a one-time, ~30-second "Agree and access repository" click at
https://huggingface.co/SAP/sap-rpt-1-oss by whoever owns that HF account, not
a code fix. Registry/generator wiring (this module, ``Prep_SAP_RPT_OSS``) is
still fully verified via ``tests/test_generate_tabarena_foundation_models_batch3.py``;
only the real weight-loading fit itself is blocked pending that access grant.
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
