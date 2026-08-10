"""LIMIX: rebind TabArena's own (empty -- see upstream's ``can_hpo=False``)
search space onto ``Prep_LIMIX``.

``LimiXModel`` (https://arxiv.org/abs/2509.03505) lives only in TabArena's own
package (``tabarena.models.limix.model``, inference code vendored under
``tabarena/models/limix/_vendor/``) -- there is no AutoGluon-core counterpart.
See ``wrapped_models.py``'s ``_OPTIONAL_TABARENA_MODEL_IMPORTS`` block for that
import and its ``ag_key`` override (``"TA-LIMIX"`` -> ``"LIMIX"``).

Supports binary/multiclass/regression -- no problem-type restriction. Unlike
every other model in this batch, though, ``LimiXModel`` DOES cap one auxiliary
param: ``max_classes=10`` (its own ``_get_default_auxiliary_params`` override).
Checked against RamanBench's real dataset roster
(``data/precomputed/dataset_stats.json``) rather than assumed away:
``bacteria_identification`` (30 classes), ``pharmaceutical_ingredients`` (32),
``rruff_mineral_raw`` (79), ``mlrod`` (16), and the ``cancer_cell_*`` trio (12
each) all exceed it. ``Prep_LIMIX`` gets
``wrapped_models._NO_FOUNDATION_MODEL_FEATURE_CAP`` applied to lift this, the
same override Mitra/TabDPT/TabICL/RealTabPFN use for their max_features cap.

GPU-tier, but UNLIKE the other five models in this batch (and unlike XRFM/
ModernNCA/RealMLP), the CPU fallback is not graceful. ``LimiXModel._fit``
itself only raises if more GPUs are *requested* than are actually available --
but confirmed via a real local run (``run_experiment.py --model LIMIX``, no
GPU): the bundled default config (``cls_default_16M_retrieval.json`` /
``reg_default_16M_retrieval.json``, both "retrieval"-mode) is rejected outright
on CPU by ``LimiXPredictor.__init__`` itself (``ValueError: Retrieval is not
supported for CPU inference!``), independent of how many GPUs were requested.
LimiX ships a separate ``*_noretrieval.json`` config that presumably *does* run
on CPU, but nothing here selects it -- TabArena's own upstream default
(``can_hpo=False``, ``manual_configs=[{}]``) always resolves to the retrieval
config. So, unlike every other GPU-tier model in this registry, a LIMIX job
genuinely needs a real GPU to run its default config at all; local CPU-only
smoke testing for this one model could not be completed here and needs a
cluster GPU node instead. Still listed in ``cluster/gpu_models.json`` (compute
never falls back to CPU the way it does elsewhere, so it belongs there even
more strictly than the other five).

Separately: the upstream pip-installed ``tabarena`` package is missing its own
vendored config JSONs entirely (``tabarena/models/limix/_vendor/config/*.json``
exist in the GitHub source tree but are absent from the installed wheel/git
package -- a real, reproducible packaging bug, not a RamanBench issue) --
without them, ``LimiXModel._fit`` fails before even reaching the
CPU/retrieval check above (``FileNotFoundError`` on the config path). Worth
filing upstream against ``autogluon/tabarena``; until fixed there, any fresh
environment installing this repo's ``[models]`` extra needs those seven JSON
files copied into ``site-packages/tabarena/models/limix/_vendor/config/``
by hand (or a `pip install` from a full source checkout, which keeps
non-Python files that a wheel build drops) before LIMIX can run at all.
"""

from __future__ import annotations

from tabarena.models.limix.hpo import gen_limix as _upstream

from raman_bench.models.generate._tabarena_adapter import (
    rebind_tabarena_generator,
    require_available,
)
from raman_bench.preprocessing.wrapped_models import Prep_LIMIX

gen_limix = rebind_tabarena_generator(_upstream, require_available(Prep_LIMIX, "LIMIX"))
