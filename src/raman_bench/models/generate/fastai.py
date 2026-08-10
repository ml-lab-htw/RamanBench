"""FASTAI (NNFastAiTabularModel): rebind TabArena's search space onto ``Prep_FASTAI``.

See ``_tabarena_adapter.rebind_tabarena_generator`` for why this rebind is needed
instead of using ``tabarena.models.fastai.hpo.gen_fastai`` directly.
"""

from __future__ import annotations

from tabarena.models.fastai.hpo import gen_fastai as _upstream

from raman_bench.models.generate._tabarena_adapter import rebind_tabarena_generator
from raman_bench.preprocessing.wrapped_models import Prep_FASTAI

gen_fastai = rebind_tabarena_generator(_upstream, Prep_FASTAI)
