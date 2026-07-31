from __future__ import annotations

from autogluon.tabular.models import LGBModel

from raman_bench.preprocessing.bridge_bases import _NoAugBase


class Prep_GBM(_NoAugBase, LGBModel):  # noqa: N801
    pass
