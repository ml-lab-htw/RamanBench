"""Mitra-v2, wrapped with RamanBench's tunable Raman preprocessing.

Mitra-v2 (arXiv:2609.04540) is the second-generation Mitra tabular foundation model:
the same 12-layer 2D-attention backbone (now 77M parameters) pretrained on a larger,
more diverse synthetic prior, deployed as a fine-tuned, bagged model -- every bag child
fine-tunes the checkpoint on its fit fold for 50 steps and predicts in context. It
already has a full TabArena integration (``tabarena.models.mitra_v2``,
``ag_key="TA-MITRA-V2"``, GPU-only, an AutoGluon-native ``MitraModel`` subclass), so
this module reuses that ``MitraV2Model`` class directly rather than reimplementing
anything -- matching the ``ta_tabpfn_3``/``ta_exaone_tabular`` precedent for
AutoGluon-native TabArena models (no sklearn bridge needed; the class is already an
AutoGluon ``AbstractModel`` subclass with ``ag_key``/``ag_name`` set).

License: Apache-2.0 (both code and weights -- unlike some other foundation-model
wrappers in this repo, there is no separate non-commercial weight license to flag
here).

Requires a CUDA GPU (``minimum_num_gpus=1``, hard requirement when ``num_gpus>0`` is
requested by the fitting harness); the upstream wrapper itself warns fine-tuning on
CPU is "very slow" but does not forbid it outright when no GPU is requested.
"""

from __future__ import annotations

from tabarena.models.mitra_v2.model import MitraV2Model

from raman_bench.preprocessing.bridge_bases import _NoAugBase


class Prep_MITRA_V2(_NoAugBase, MitraV2Model):  # noqa: N801
    """Mitra-v2 with RamanBench's tunable (default-disabled) preprocessing recipe."""


__all__ = ["MitraV2Model", "Prep_MITRA_V2"]
