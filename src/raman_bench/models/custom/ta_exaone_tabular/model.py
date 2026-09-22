"""EXAONE-Tabular, wrapped with RamanBench's tunable Raman preprocessing.

EXAONE-Tabular (LG AI Research, https://github.com/LGAI-Research/EXAONE-Tabular) is a
Cross-axis Summary Transformer (CAST) in-context-learning tabular foundation model:
~21M parameters, no per-dataset gradient training, one released checkpoint per problem
type (classification / regression). It already has a full TabArena integration
(``tabarena.models.exaone_tabular``, ``ag_key="TA-EXAONE-TABULAR"``, GPU-only), so this
module reuses that ``EXAONETabularModel`` directly rather than reimplementing anything --
matching the ``ta_tabpfn_3``/``gbm`` precedent for AutoGluon-native TabArena models
(no sklearn bridge needed; the class is already an AutoGluon ``AbstractModel``
subclass with ``ag_key``/``ag_name`` set).

License note (flagged per RamanBench's own MIT license): the ``exaonetabular``
inference code is BSD-3-Clause-LG AI Research (commercial use permitted), but the
released *weights*, downloaded from Hugging Face on first fit, are licensed
separately under the EXAONE AI Model License Agreement 1.2-NC -- non-commercial
research/education use only. RamanBench's own use here (academic benchmarking) is
within that license, but anyone deploying ``Prep_EXAONE_TABULAR`` commercially must
either obtain a separate license from LG AI Research for the weights or swap in
their own checkpoint.
"""

from __future__ import annotations

from tabarena.models.exaone_tabular.model import EXAONETabularModel

from raman_bench.preprocessing.bridge_bases import _NoAugBase


class Prep_EXAONE_TABULAR(_NoAugBase, EXAONETabularModel):  # noqa: N801
    """EXAONE-Tabular with RamanBench's tunable (default-disabled) preprocessing recipe."""


__all__ = ["EXAONETabularModel", "Prep_EXAONE_TABULAR"]
