"""LIMIX2: rebind TabArena's own (empty -- ``can_hpo=False``) search space onto
``Prep_LIMIX2``.

``LimiX2Model`` (https://arxiv.org/abs/2609.17488, Stable AI's second-generation
tabular foundation model) lives only in TabArena's own package
(``tabarena.models.limix_2.model``) -- there is no AutoGluon-core counterpart,
same staging situation as LIMIX (v1). See ``wrapped_models.py``'s
``_OPTIONAL_TABARENA_MODEL_IMPORTS`` block for that import and its ``ag_key``
override (``"TA-LIMIX-2"`` -> ``"LIMIX2"``).

Supports binary/multiclass/regression, with the same ``max_classes=10`` cap
LIMIX (v1) has -- but declared DECLARATIVELY here (``_default_auxiliary_params_extra``
class attribute), unlike LimiXModel's non-declarative method override, so
``Prep_LIMIX2`` lifts it the normal way (``_make_optional_prep_class``'s
``_default_auxiliary_params_extra=_NO_FOUNDATION_MODEL_FEATURE_CAP`` kwarg) --
see ``wrapped_models.py``'s comment above ``Prep_LIMIX2`` for the full
explanation and how this was confirmed against the installed class.

GPU-tier. Unlike LIMIX (v1), LimiX-2 ships its own no-retrieval default configs
(``cls_default_noretrieval_v2.json`` / ``reg_default_noretrieval_v2.json``) as
its ``ConfigGenerator``'s ``manual_configs=[{}]`` default (confirmed by reading
``tabarena/models/limix_2/model.py`` and ``hpo.py`` directly), so it does not
have LIMIX v1's hard CPU-retrieval crash -- but it is still GPU-tier in
practice (in-context prediction over a 400M-parameter network on every
predict call), so it's still listed in ``cluster/gpu_models.json``.

Needs Python >=3.12: its inference package (``LimiX @ git+https://github.com/limix-ldm-ai/LimiX.git@774aa3e``)
declares ``requires-python = ">=3.12"`` and pins ``torch==2.9.1``. Benchmark v2's image (``Dockerfile.v2``) is
Python 3.12 and installs the package without its torch pin (``requirements-git-nodeps.txt``), so LimiX-2 runs
there like every other model. The Python 3.11 image of benchmark v1 does not have it.
"""

from __future__ import annotations

from tabarena.models.limix_2.hpo import gen_limix_2 as _upstream

from raman_bench.models.generate._tabarena_adapter import (
    rebind_tabarena_generator,
    require_available,
)
from raman_bench.preprocessing.wrapped_models import Prep_LIMIX2

gen_limix2 = rebind_tabarena_generator(_upstream, require_available(Prep_LIMIX2, "LIMIX2"))
