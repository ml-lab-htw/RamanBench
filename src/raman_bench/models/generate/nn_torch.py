"""NN_TORCH (TabularNeuralNetTorchModel): rebind TabArena's search space onto ``Prep_NN_TORCH``.

See ``_tabarena_adapter.rebind_tabarena_generator`` for why this rebind is needed
instead of using ``tabarena.models.nn_torch.hpo.gen_nn_torch`` directly.
"""

from __future__ import annotations

from tabarena.models.nn_torch.hpo import gen_nn_torch as _upstream

from raman_bench.models.generate._tabarena_adapter import rebind_tabarena_generator
from raman_bench.preprocessing.wrapped_models import Prep_NN_TORCH

gen_nn_torch = rebind_tabarena_generator(_upstream, Prep_NN_TORCH)
