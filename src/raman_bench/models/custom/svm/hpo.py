from __future__ import annotations

from tabarena.utils.config_utils import ConfigGenerator

from raman_bench.models.custom.svm.model import Prep_SVM, _SVMBridge

gen_svm = ConfigGenerator(
    model_cls=Prep_SVM,
    manual_configs=[{}],
    search_space=_SVMBridge._get_default_searchspace(_SVMBridge),
)
