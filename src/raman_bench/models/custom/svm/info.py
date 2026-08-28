from __future__ import annotations

from raman_bench.models._model_info import ModelInfo
from raman_bench.models.custom.svm.hpo import gen_svm
from raman_bench.models.custom.svm.model import Prep_SVM

svm_info = ModelInfo(
    model_cls=Prep_SVM,
    search_space=gen_svm,
    display_name="SVM",
    compute="cpu",
)
